#include "backend.h"
#include "general_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "solver_lm.h"
#include "solver_dogleg.h"

#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"

namespace VIO {

bool Backend::TryToEstimate() {
    // Clear all vectors of vertices and edges.
    ClearBackendGraph();

    // Generate vertices of states to be optimized.
    // [Vertices] Extrinsics of each camera.
    // [Vertices] Camera pose of each frame.
    ConvertCameraPoseAndExtrinsicToVertices();

    // [Edges] Camera pose prior factor.
    // [Edges] Camera extrinsic prior factor.
    RETURN_FALSE_IF(!AddPriorFactorWhenNoPrior());

    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    RETURN_FALSE_IF(!ConvertFeatureInvdepAndAddVisualFactorForEstimation());

    // [Vertices] Velocity of each new frame.
    ConvertImuMotionStatesToVertices();

    // [Edges] Inerial preintegration factor.
    const uint32_t idx_offset = data_manager_->visual_local_map()->frames().size() - data_manager_->frames_with_bias().size();
    RETURN_FALSE_IF(!AddImuPreintegrationFactorForEstimation(idx_offset));

    // Construct graph problem, add all vertices and edges.
    // Add prior information if valid.
    Graph<DorF> graph_optimization_problem;
    ConstructGraphOptimizationProblem(idx_offset, graph_optimization_problem);
    if (states_.prior.is_valid) {
        ReportInfo("[Backend] Before estimation, prior residual squared norm is " << graph_optimization_problem.prior_residual().squaredNorm());
    }

    // Construct solver to solve this problem.
    SolverLm<DorF> solver;
    solver.options().kEnableReportEachIteration = options_.kEnableReportAllInformation;
    solver.options().kMaxConvergedSquaredStepLength = static_cast<DorF>(1e-3);
    solver.options().kMaxCostTimeInSecond = 0.05f;
    solver.problem() = &graph_optimization_problem;
    solver.Solve(states_.prior.is_valid);

    // Show all vertices and incremental function in optimization problem.
    if (options_.kEnableReportAllInformation) {
        solver.problem()->VerticesInformation();
        ShowMatrixImage("solve hessian", solver.problem()->hessian());
    }

    // Update all states in visual_local_map and frames_with_bias.
    UpdateAllStatesAfterEstimation(graph_optimization_problem, idx_offset);

    return true;
}

void Backend::UpdateAllStatesAfterEstimation(const Graph<DorF> &problem, const uint32_t idx_offset) {
    // Update all camera extrinsics.
    for (uint32_t i = 0; i < graph_.vertices.all_cameras_p_ic.size(); ++i) {
        data_manager_->camera_extrinsics()[i].p_ic = graph_.vertices.all_cameras_p_ic[i]->param().cast<float>();
        data_manager_->camera_extrinsics()[i].q_ic.w() = graph_.vertices.all_cameras_q_ic[i]->param()(0);
        data_manager_->camera_extrinsics()[i].q_ic.x() = graph_.vertices.all_cameras_q_ic[i]->param()(1);
        data_manager_->camera_extrinsics()[i].q_ic.y() = graph_.vertices.all_cameras_q_ic[i]->param()(2);
        data_manager_->camera_extrinsics()[i].q_ic.z() = graph_.vertices.all_cameras_q_ic[i]->param()(3);
    }

    // Update all frame pose in local map.
    const Vec3 &p_ic = data_manager_->camera_extrinsics().front().p_ic;
    const Quat &q_ic = data_manager_->camera_extrinsics().front().q_ic;
    for (uint32_t i = 0; i < graph_.vertices.all_frames_p_wi.size(); ++i) {
        auto frame_ptr = data_manager_->visual_local_map()->frame(graph_.vertices.all_frames_id[i]);
        const Vec3 p_wi = graph_.vertices.all_frames_p_wi[i]->param().cast<float>();
        const Quat q_wi = Quat(graph_.vertices.all_frames_q_wi[i]->param()(0),
                               graph_.vertices.all_frames_q_wi[i]->param()(1),
                               graph_.vertices.all_frames_q_wi[i]->param()(2),
                               graph_.vertices.all_frames_q_wi[i]->param()(3));
        Utility::ComputeTransformTransform(p_wi, q_wi, p_ic, q_ic, frame_ptr->p_wc(), frame_ptr->q_wc());

        if (i >= idx_offset) {
            const uint32_t j = i - idx_offset;
            frame_ptr->v_w() = graph_.vertices.all_new_frames_v_wi[j]->param().cast<float>();
        }
    }

    // Update all feature position in local map.
    uint32_t solved_feature_cnt = 0;
    for (uint32_t i = 0; i < graph_.vertices.all_features_id.size(); ++i) {
        auto feature_ptr = data_manager_->visual_local_map()->feature(graph_.vertices.all_features_id[i]);
        const auto &frame_ptr = data_manager_->visual_local_map()->frame(feature_ptr->first_frame_id());
        const auto &norm_xy = feature_ptr->observes().front()[0].rectified_norm_xy;

        const float invdep = graph_.vertices.all_features_invdep[i]->param()(0);
        Vec3 p_c = Vec3(norm_xy.x(), norm_xy.y(), 1.0f) / invdep;
        if (std::isnan(p_c.z()) || std::isinf(p_c.z()) || p_c.z() < options_.kMinValidFeatureDepthInMeter) {
            p_c = Vec3(norm_xy.x(), norm_xy.y(), 1.0f) * options_.kDefaultFeatureDepthInMeter;
            feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
        } else if (p_c.z() > options_.kMaxValidFeatureDepthInMeter) {
            feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
        } else {
            feature_ptr->status() = FeatureSolvedStatus::kSolved;
            ++solved_feature_cnt;
        }
        feature_ptr->param() = frame_ptr->q_wc() * p_c + frame_ptr->p_wc();
    }
    ReportInfo("[Backend] " << solved_feature_cnt << "/" << graph_.vertices.all_features_id.size() << " features are solved in optimization.");

    // Update imu preintegration.
    uint32_t idx = 0;
    for (auto &frame : data_manager_->frames_with_bias()) {
        frame.imu_preint_block.Reset();
        frame.imu_preint_block.bias_accel() = graph_.vertices.all_new_frames_ba[idx]->param().cast<float>();
        frame.imu_preint_block.bias_gyro() = graph_.vertices.all_new_frames_bg[idx]->param().cast<float>();
        frame.imu_preint_block.SetImuNoiseSigma(imu_model_->options().kAccelNoise,
                                                imu_model_->options().kGyroNoise,
                                                imu_model_->options().kAccelRandomWalk,
                                                imu_model_->options().kGyroRandomWalk);
        ++idx;

        const int32_t max_idx = static_cast<int32_t>(frame.packed_measure->imus.size());
        for (int32_t i = 1; i < max_idx; ++i) {
            frame.imu_preint_block.Propagate(*frame.packed_measure->imus[i - 1], *frame.packed_measure->imus[i]);
        }
    }

    // Update prior information.
    if (states_.prior.is_valid) {
        states_.prior.hessian = problem.prior_hessian();
        states_.prior.bias = problem.prior_bias();
        states_.prior.jacobian_t_inv = problem.prior_jacobian_t_inv();
        states_.prior.residual = problem.prior_residual();

        ReportInfo("[Backend] After estimation, prior residual squared norm is " << problem.prior_residual().squaredNorm());
    }
}

}
