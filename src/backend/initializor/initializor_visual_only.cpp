#include "backend.h"
#include "slam_operations.h"

#include "geometry_epipolar.h"
#include "geometry_triangulation.h"

namespace VIO {

namespace {
    constexpr float kMinValidAverageParallaxForPureVisualSfm = 16.0f;
    constexpr int32_t kMinValidCovisibleFeaturesNumberForPureVisualSfm = 30;
}

bool Backend::PrepareForPureVisualSfmByMonoView() {
    // Find the frame with best corresbondence with the first frame.
    FramesCorresbondence best_corres = FramesCorresbondence{
        .num_of_covisible_features = static_cast<int32_t>(visual_frontend_->options().kMaxStoredFeaturePointsNumber),
        .average_parallax = 0.0f,
    };
    const int32_t ref_frame_id = data_manager_->visual_local_map()->frames().front().id();
    int32_t cur_frame_id = ref_frame_id;
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        cur_frame_id = frame.id();
        CONTINUE_IF(cur_frame_id == ref_frame_id);
        best_corres = data_manager_->GetCorresbondence(ref_frame_id, cur_frame_id);
        BREAK_IF(best_corres.num_of_covisible_features < kMinValidCovisibleFeaturesNumberForPureVisualSfm ||
            best_corres.average_parallax > kMinValidAverageParallaxForPureVisualSfm);
    }

    if (best_corres.average_parallax < kMinValidAverageParallaxForPureVisualSfm) {
        ReportWarn("[Backend] Cannot find a frame with best corresbondence with the first frame." <<
            " Number of covisible features [" << best_corres.num_of_covisible_features << "]," <<
            " average parallax [" << best_corres.average_parallax << "/" << kMinValidAverageParallaxForPureVisualSfm << "].");
        return false;
    } else {
        ReportInfo("[Backend] Find frame " << cur_frame_id << " with best corresbondence with the first frame." <<
            " Number of covisible features [" << best_corres.num_of_covisible_features << "]," <<
            " average parallax [" << best_corres.average_parallax << "/" << kMinValidAverageParallaxForPureVisualSfm << "].");
    }

    // Extract covisible features and observations between these two frames.
    std::vector<FeatureType *> covisible_features;
    if (!data_manager_->visual_local_map()->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        ReportError("[Backend] Failed to get covisible features between frame " << ref_frame_id << " and " << cur_frame_id << ".");
        return false;
    }

    std::vector<Vec2> ref_norm_xy;
    std::vector<Vec2> cur_norm_xy;
    ref_norm_xy.reserve(visual_frontend_->options().kMaxStoredFeaturePointsNumber);
    cur_norm_xy.reserve(visual_frontend_->options().kMaxStoredFeaturePointsNumber);

    for (const auto &feature_ptr : covisible_features) {
        const auto &observe_ref = feature_ptr->observe(ref_frame_id).front().rectified_norm_xy;
        const auto &observe_cur = feature_ptr->observe(cur_frame_id).front().rectified_norm_xy;
        ref_norm_xy.emplace_back(observe_ref);
        cur_norm_xy.emplace_back(observe_cur);
    }

    // Solve relative pose between these two frames.
    using namespace VISION_GEOMETRY;
    EpipolarSolver solver;
    solver.options().kModel = EpipolarSolver::EpipolarModel::kFivePoints;
    solver.options().kMethod = EpipolarSolver::EpipolarMethod::kRansac;
    solver.options().kMaxEpipolarResidual = 1e-2f;

    Mat3 essential = Mat3::Identity();
    Mat3 R_cr = Mat3::Identity();
    Vec3 t_cr = Vec3::Zero();
    std::vector<uint8_t> status;
    solver.EstimateEssential(ref_norm_xy, cur_norm_xy, essential, status);
    solver.RecoverPoseFromEssential(ref_norm_xy, cur_norm_xy, essential, R_cr, t_cr);

    // Set the first frame to be origin.
    auto first_frame = data_manager_->visual_local_map()->frame(ref_frame_id);
    auto corr_frame = data_manager_->visual_local_map()->frame(cur_frame_id);
    first_frame->p_wc().setZero();
    first_frame->q_wc().setIdentity();
    corr_frame->p_wc() = - R_cr * t_cr;
    corr_frame->q_wc() = Quat(R_cr.transpose());

    // Triangulize all features observed by these two frames.
    for (const auto &feature_ptr : covisible_features) {
        std::vector<Quat> all_q_wc = std::vector<Quat>{first_frame->q_wc(), corr_frame->q_wc()};
        std::vector<Vec3> all_p_wc = std::vector<Vec3>{first_frame->p_wc(), corr_frame->p_wc()};
        std::vector<Vec2> all_norm_uv = std::vector<Vec2>{
            feature_ptr->observe(ref_frame_id).front().rectified_norm_xy,
            feature_ptr->observe(cur_frame_id).front().rectified_norm_xy};

        Triangulator solver;
        if (solver.Triangulate(all_q_wc, all_p_wc, all_norm_uv, feature_ptr->param())) {
            feature_ptr->status() = FeatureSolvedStatus::kSolved;
        } else {
            feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
        }
    }

    // Estimate pose of each frame between these two frame.
    for (int32_t frame_id = ref_frame_id + 1; frame_id < cur_frame_id; ++frame_id) {
        const auto prev_frame_ptr = data_manager_->visual_local_map()->frame(frame_id - 1);
        if (!TryToSolveFramePoseByFeaturesObservedByItself(frame_id, prev_frame_ptr->p_wc(), prev_frame_ptr->q_wc())) {
            ReportWarn("[Backend] Backend failed to estimate frame pose between frame [" << ref_frame_id << "] and [" << cur_frame_id << "].");
            return false;
        }
    }

    // Triangulize all features observed in other frames. And estimate pose of other frames.
    for (auto &frame : data_manager_->visual_local_map()->frames()) {
        CONTINUE_IF(frame.id() <= static_cast<uint32_t>(cur_frame_id));

        if (!TryToSolveFramePoseByFeaturesObservedByItself(frame.id())) {
            ReportWarn("[Backend] Backend failed to estimate pose of frame [" << frame.id() << "].");
            return false;
        }

        for (auto &pair : frame.features()) {
            auto feature_ptr = pair.second;
            CONTINUE_IF(feature_ptr->status() == FeatureSolvedStatus::kSolved);

            TryToSolveFeaturePositionByFramesObservingIt(feature_ptr->id(), feature_ptr->first_frame_id(),
                std::min(feature_ptr->final_frame_id(), frame.id()));
        }
    }

    return true;
}

bool Backend::PrepareForPureVisualSfmByMultiView() {
    RETURN_FALSE_IF(data_manager_->camera_extrinsics().size() < 2);

    // Set the first frame to be origin.
    const int32_t ref_frame_id = data_manager_->visual_local_map()->frames().front().id();
    auto first_frame = data_manager_->visual_local_map()->frame(ref_frame_id);
    first_frame->p_wc().setZero();
    first_frame->q_wc().setIdentity();

    // Triangulize all features observed in each frame. And estimate pose of each frame.
    for (auto &frame : data_manager_->visual_local_map()->frames()) {
        for (auto &pair : frame.features()) {
            auto feature_ptr = pair.second;
            CONTINUE_IF(feature_ptr->status() == FeatureSolvedStatus::kSolved);

            TryToSolveFeaturePositionByFramesObservingIt(feature_ptr->id(), feature_ptr->first_frame_id(),
                std::min(feature_ptr->final_frame_id(), frame.id()), true);
        }

        // No need to estimate pose of first frame.
        CONTINUE_IF(frame.id() == static_cast<uint32_t>(ref_frame_id));

        if (!TryToSolveFramePoseByFeaturesObservedByItself(frame.id())) {
            ReportWarn("[Backend] Backend failed to estimate pose of frame [" << frame.id() << "].");
            return false;
        }
    }

    return true;
}

bool Backend::PerformPureVisualBundleAdjustment(const bool use_multi_view) {
    // Clear all vectors of vertices and edges.
    ClearGraph();
    // [Vertices] Camera pose of each frame in local map.
    AddAllCameraPosesInLocalMapToGraph();
    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    const bool add_factors_with_cam_ex = false;
    AddAllFeatureInvdepsAndVisualFactorsToGraph(add_factors_with_cam_ex, use_multi_view);
    // Fix first two camera frame position.
    for (uint32_t i = 0; i < 2; ++i) {
        graph_.vertices.all_frames_p_wc[i]->SetFixed(true);
    }

    // Construct pure visual bundle adjustment problem.
    Graph<DorF> graph_optimization_problem;
    ConstructPureVisualGraphOptimizationProblem(graph_optimization_problem);

    // Construct solver to solve this problem.
    SolverLm<DorF> solver;
    solver.options().kEnableReportEachIteration = false;
    solver.options().kMaxConvergedSquaredStepLength = static_cast<DorF>(1e-4);
    solver.options().kMaxCostTimeInSecond = 0.05f;
    solver.problem() = &graph_optimization_problem;
    solver.Solve(false);

    // Update all states into visual local map.
    for (uint32_t i = 0; i < graph_.vertices.all_frames_p_wc.size(); ++i) {
        auto frame_ptr = data_manager_->visual_local_map()->frame(graph_.vertices.all_frames_id[i]);
        frame_ptr->p_wc() = graph_.vertices.all_frames_p_wc[i]->param().cast<float>();
        const auto param_q = graph_.vertices.all_frames_q_wc[i]->param();
        frame_ptr->q_wc() = Quat(param_q(0), param_q(1), param_q(2), param_q(3));
    }

    for (uint32_t i = 0; i < graph_.vertices.all_features_id.size(); ++i) {
        auto feature_ptr = data_manager_->visual_local_map()->feature(graph_.vertices.all_features_id[i]);
        const auto &frame_ptr = data_manager_->visual_local_map()->frame(feature_ptr->first_frame_id());
        const auto &norm_xy = feature_ptr->observes().front()[0].rectified_norm_xy;
        const float invdep = graph_.vertices.all_features_invdep[i]->param()(0);
        Vec3 p_c = Vec3(norm_xy.x(), norm_xy.y(), 1.0f) / invdep;

        if (std::isnan(p_c.z()) || std::isinf(p_c.z())) {
            p_c = Vec3(norm_xy.x(), norm_xy.y(), 1.0f) * options_.kDefaultFeatureDepthInMeter;
            feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
        } else if (p_c.z() < options_.kMinValidFeatureDepthInMeter) {
            feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
        } else {
            feature_ptr->status() = FeatureSolvedStatus::kSolved;
        }
        feature_ptr->param() = frame_ptr->q_wc() * p_c + frame_ptr->p_wc();
    }

    return true;
}

}
