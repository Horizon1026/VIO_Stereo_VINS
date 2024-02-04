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

void Backend::ClearBackendGraph() {
    graph_.vertices.all_cameras_p_ic.clear();
    graph_.vertices.all_cameras_q_ic.clear();
    graph_.vertices.all_frames_id.clear();
    graph_.vertices.all_frames_p_wi.clear();
    graph_.vertices.all_frames_q_wi.clear();
    graph_.vertices.all_features_id.clear();
    graph_.vertices.all_features_invdep.clear();
    graph_.vertices.all_new_frames_v_wi.clear();
    graph_.vertices.all_new_frames_ba.clear();
    graph_.vertices.all_cameras_p_ic.clear();
    graph_.vertices.all_new_frames_bg.clear();

    graph_.edges.all_prior_factors.clear();
    graph_.edges.all_visual_reproj_factors.clear();
    graph_.edges.all_imu_factors.clear();
}

TMat2<DorF> Backend::GetVisualObserveInformationMatrix() {
    const auto &camera_model = visual_frontend_->camera_models().front();
    const DorF residual_in_pixel = 1.0;
    const TVec2<DorF> visual_observe_info_vec = TVec2<DorF>(camera_model->fx() * camera_model->fx(),
        camera_model->fy() * camera_model->fy()) / residual_in_pixel;
    return visual_observe_info_vec.asDiagonal();
}

void Backend::ConvertCameraPoseAndExtrinsicToVertices() {
    // Generate vertices of states to be optimized.
    // [Vertices] Extrinsics of each camera.
    for (const auto &extrinsic : data_manager_->camera_extrinsics()) {
        graph_.vertices.all_cameras_p_ic.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_cameras_p_ic.back()->param() = extrinsic.p_ic.cast<DorF>();
        graph_.vertices.all_cameras_p_ic.back()->name() = std::string("p_ic");
        graph_.vertices.all_cameras_q_ic.emplace_back(std::make_unique<VertexQuat<DorF>>(4, 3));
        graph_.vertices.all_cameras_q_ic.back()->param() << extrinsic.q_ic.w(),
            extrinsic.q_ic.x(), extrinsic.q_ic.y(), extrinsic.q_ic.z();
        graph_.vertices.all_cameras_q_ic.back()->name() = std::string("q_ic");
    }

    // [Vertices] Camera pose of each frame.
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        graph_.vertices.all_frames_id.emplace_back(frame.id());

        Vec3 p_wi = Vec3::Zero();
        Quat q_wi = Quat::Identity();
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(),
            data_manager_->camera_extrinsics().front().p_ic,
            data_manager_->camera_extrinsics().front().q_ic, p_wi, q_wi);

        graph_.vertices.all_frames_p_wi.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_p_wi.back()->param() = p_wi.cast<DorF>();
        graph_.vertices.all_frames_p_wi.back()->name() = std::string("p_wi") + std::to_string(frame.id());
        graph_.vertices.all_frames_q_wi.emplace_back(std::make_unique<VertexQuat<DorF>>(4, 3));
        graph_.vertices.all_frames_q_wi.back()->param() << q_wi.w(), q_wi.x(), q_wi.y(), q_wi.z();
        graph_.vertices.all_frames_q_wi.back()->name() = std::string("q_wi") + std::to_string(frame.id());
    }
}

bool Backend::AddPriorFactorWhenNoPrior() {
    RETURN_TRUE_IF(states_.prior.is_valid);
    RETURN_FALSE_IF(graph_.vertices.all_frames_p_wi.empty() || graph_.vertices.all_frames_q_wi.empty());

    // [Edges] Camera pose prior factor.
    // [Edges] Camera extrinsic prior factor.
    graph_.edges.all_prior_factors.emplace_back(std::make_unique<EdgePriorPose<DorF>>());
    auto &prior_factor = graph_.edges.all_prior_factors.back();
    prior_factor->SetVertex(graph_.vertices.all_frames_p_wi.front().get(), 0);
    prior_factor->SetVertex(graph_.vertices.all_frames_q_wi.front().get(), 1);

    TMat<DorF> obv = TVec7<DorF>::Zero();
    obv.block(0, 0, 3, 1) = graph_.vertices.all_frames_p_wi.front()->param();
    obv.block(3, 0, 4, 1) = graph_.vertices.all_frames_q_wi.front()->param();
    prior_factor->observation() = obv;

    prior_factor->information() = TMat6<DorF>::Identity() * 1e6;
    prior_factor->name() = std::string("prior pose");
    RETURN_FALSE_IF(!prior_factor->SelfCheck());

    for (uint32_t i = 0; i < graph_.vertices.all_cameras_p_ic.size(); ++i) {
        graph_.edges.all_prior_factors.emplace_back(std::make_unique<EdgePriorPose<DorF>>());
        auto &prior_factor = graph_.edges.all_prior_factors.back();
        prior_factor->SetVertex(graph_.vertices.all_cameras_p_ic[i].get(), 0);
        prior_factor->SetVertex(graph_.vertices.all_cameras_q_ic[i].get(), 1);

        TMat<DorF> obv = TVec7<DorF>::Zero();
        obv.block(0, 0, 3, 1) = graph_.vertices.all_cameras_p_ic[i]->param();
        obv.block(3, 0, 4, 1) = graph_.vertices.all_cameras_q_ic[i]->param();
        prior_factor->observation() = obv;

        prior_factor->information() = TMat6<DorF>::Identity() * 1e6;
        prior_factor->name() = std::string("prior extrinsic ") + std::to_string(i);
        RETURN_FALSE_IF(!prior_factor->SelfCheck());
    }

    return true;
}

bool Backend::ConvertFeatureInvdepAndAddVisualFactorForEstimation() {
    // Compute information matrix of visual observation.
    const TMat2<DorF> visual_info_matrix = GetVisualObserveInformationMatrix();

    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;

        // Select features which has at least two observations.
        CONTINUE_IF(feature.observes().size() < 2 && feature.observes().front().size() < 2);
        // Select features which is marginalized successfully.
        CONTINUE_IF(feature.status() == FeatureSolvedStatus::kMarginalized);

        // Compute inverse depth by p_w of this feature.
        const auto &frame = data_manager_->visual_local_map()->frame(feature.first_frame_id());
        const Vec3 p_c = frame->q_wc().inverse() * (feature.param() - frame->p_wc());
        const float invdep = p_c.z() < options_.kMinValidFeatureDepthInMeter ? 1.0f / options_.kDefaultFeatureDepthInMeter : 1.0f / p_c.z();
        CONTINUE_IF(std::isinf(invdep) || std::isnan(invdep));

        // Convert feature invdep to vertices, and add visual factors.
        RETURN_FALSE_IF(!ConvertFeatureInvdepAndAddVisualFactor(feature, invdep, visual_info_matrix, feature.final_frame_id()));
    }

    return true;
}

bool Backend::ConvertFeatureInvdepAndAddVisualFactorForMarginalization() {
    // Compute information matrix of visual observation.
    const TMat2<DorF> visual_info_matrix = GetVisualObserveInformationMatrix();

    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;
        // Select features which has at least two observations.
        CONTINUE_IF(feature.observes().size() < 2 && feature.observes().front().size() < 2);
        // Select features which is first observed in oldest keyframe.
        CONTINUE_IF(feature.first_frame_id() != data_manager_->visual_local_map()->frames().front().id());
        // Select features which is solved successfully.
        CONTINUE_IF(feature.status() != FeatureSolvedStatus::kSolved);

        // Compute inverse depth by p_w of this feature.
        const auto &frame = data_manager_->visual_local_map()->frame(feature.first_frame_id());
        const Vec3 p_c = frame->q_wc().inverse() * (feature.param() - frame->p_wc());
        const float invdep = 1.0f / p_c.z();
        CONTINUE_IF(std::isinf(invdep) || std::isnan(invdep) || p_c.z() < kZero);

        // Convert feature invdep to vertices, and add visual factors.
        RETURN_FALSE_IF(!ConvertFeatureInvdepAndAddVisualFactor(feature, invdep, visual_info_matrix, feature.final_frame_id()));
    }

    return true;
}

bool Backend::ConvertFeatureInvdepAndAddVisualFactor(const FeatureType &feature, const float invdep, const TMat2<DorF> &visual_info_matrix, const uint32_t max_frame_id) {
    // Determine the range of all observations of this feature.
    const uint32_t min_frame_id = feature.first_frame_id();
    const uint32_t idx_offset = min_frame_id - data_manager_->visual_local_map()->frames().front().id() + 1;

    // Add vertex of feature invdep.
    graph_.vertices.all_features_id.emplace_back(feature.id());
    graph_.vertices.all_features_invdep.emplace_back(std::make_unique<Vertex<DorF>>(1, 1));
    graph_.vertices.all_features_invdep.back()->param() = TVec1<DorF>(invdep);
    graph_.vertices.all_features_invdep.back()->name() = std::string("invdep ") + std::to_string(feature.id());

    // Add edges of visual reprojection factor, considering two cameras view one frame.
    const auto &obv_in_ref = feature.observe(min_frame_id);
    Vec4 observe_vector = Vec4::Zero();
    observe_vector.head<2>() = obv_in_ref[0].rectified_norm_xy;
    for (uint32_t i = 1; i < obv_in_ref.size(); ++i) {
        observe_vector.tail<2>() = obv_in_ref[i].rectified_norm_xy;

        // Add edge of visual reprojection factor, considering two camera view one frame.
        graph_.edges.all_visual_reproj_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinOneFramesTwoCamera<DorF>>());
        auto &visual_reproj_factor = graph_.edges.all_visual_reproj_factors.back();
        visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[0].get(), 1);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[0].get(), 2);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[i].get(), 3);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[i].get(), 4);
        visual_reproj_factor->observation() = observe_vector.cast<DorF>();
        visual_reproj_factor->information() = visual_info_matrix;
        visual_reproj_factor->kernel() = std::make_unique<KernelHuber<DorF>>(static_cast<DorF>(0.5));
        visual_reproj_factor->name() = std::string("one frame two cameras");
        RETURN_FALSE_IF(!visual_reproj_factor->SelfCheck());
    }

    // In order to add other edges, iterate all observations of this feature.
    for (uint32_t idx = min_frame_id + 1; idx <= max_frame_id; ++idx) {
        const auto &obv_in_cur = feature.observe(idx);
        observe_vector.tail<2>() = obv_in_cur[0].rectified_norm_xy;

        // Add edges of visual reprojection factor, considering one camera views two frames.
        graph_.edges.all_visual_reproj_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera<DorF>>());
        auto &visual_reproj_factor = graph_.edges.all_visual_reproj_factors.back();
        visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[min_frame_id - idx_offset].get(), 1);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[min_frame_id - idx_offset].get(), 2);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[idx - idx_offset].get(), 3);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[idx - idx_offset].get(), 4);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[0].get(), 5);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[0].get(), 6);
        visual_reproj_factor->observation() = observe_vector.cast<DorF>();
        visual_reproj_factor->information() = visual_info_matrix;
        visual_reproj_factor->kernel() = std::make_unique<KernelHuber<DorF>>(static_cast<DorF>(0.5));
        visual_reproj_factor->name() = std::string("two frames one camera");
        RETURN_FALSE_IF(!visual_reproj_factor->SelfCheck());

        // Add edges of visual reprojection factor, considering two cameras view two frames.
        for (uint32_t i = 1; i < obv_in_cur.size(); ++i) {
            observe_vector.tail<2>() = obv_in_cur[i].rectified_norm_xy;

            graph_.edges.all_visual_reproj_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera<DorF>>());
            auto &visual_reproj_factor = graph_.edges.all_visual_reproj_factors.back();
            visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[min_frame_id - idx_offset].get(), 1);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[min_frame_id - idx_offset].get(), 2);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[idx - idx_offset].get(), 3);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[idx - idx_offset].get(), 4);
            visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[0].get(), 5);
            visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[0].get(), 6);
            visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[i].get(), 7);
            visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[i].get(), 8);
            visual_reproj_factor->observation() = observe_vector.cast<DorF>();
            visual_reproj_factor->information() = visual_info_matrix;
            visual_reproj_factor->kernel() = std::make_unique<KernelHuber<DorF>>(static_cast<DorF>(0.5));
            visual_reproj_factor->name() = std::string("two frames two cameras");
            RETURN_FALSE_IF(!visual_reproj_factor->SelfCheck());
        }
    }

    return true;
}

void Backend::ConvertImuMotionStatesToVertices() {
    // [Vertices] Velocity of each new frame.
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t idx_offset = data_manager_->visual_local_map()->frames().size() - data_manager_->frames_with_bias().size();
    for (uint32_t frame_idx = min_frames_idx + idx_offset; frame_idx <= max_frames_idx; ++frame_idx) {
        graph_.vertices.all_new_frames_v_wi.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_new_frames_v_wi.back()->param() = data_manager_->visual_local_map()->frame(frame_idx)->v_w().cast<DorF>();
        graph_.vertices.all_new_frames_v_wi.back()->name() = std::string("v_wi") + std::to_string(frame_idx);
    }

    // [Vertices] Bias_accel and bias_gyro of each new frame.
    for (const auto &frame : data_manager_->frames_with_bias()) {
        // Add vertex of bias_accel and bias_gyro.
        graph_.vertices.all_new_frames_ba.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_new_frames_ba.back()->param() = frame.imu_preint_block.bias_accel().cast<DorF>();
        graph_.vertices.all_new_frames_ba.back()->name() = std::string("bias_a");
        graph_.vertices.all_new_frames_bg.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_new_frames_bg.back()->param() = frame.imu_preint_block.bias_gyro().cast<DorF>();
        graph_.vertices.all_new_frames_bg.back()->name() = std::string("bias_g");
    }
}

bool Backend::AddImuPreintegrationFactorForEstimation(const uint32_t idx_offset) {
    RETURN_TRUE_IF(data_manager_->frames_with_bias().size() < 2);

    // [Edges] Inerial preintegration factor.
    uint32_t frame_idx = idx_offset;
    uint32_t new_frame_idx = 0;
    for (auto it = std::next(data_manager_->frames_with_bias().begin()); it != data_manager_->frames_with_bias().end(); ++it) {
        // The imu preintegration block combined with the oldest 'new frame with bias' is useless.
        // Add edges of imu preintegration.
        const auto &frame = *it;
        graph_.edges.all_imu_factors.emplace_back(std::make_unique<EdgeImuPreintegrationBetweenRelativePose<DorF>>(
            frame.imu_preint_block, options_.kGravityInWordFrame));
        auto &imu_factor = graph_.edges.all_imu_factors.back();
        imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[frame_idx].get(), 0);
        imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[frame_idx].get(), 1);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_v_wi[new_frame_idx].get(), 2);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_ba[new_frame_idx].get(), 3);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_bg[new_frame_idx].get(), 4);
        imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[frame_idx + 1].get(), 5);
        imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[frame_idx + 1].get(), 6);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_v_wi[new_frame_idx + 1].get(), 7);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_ba[new_frame_idx + 1].get(), 8);
        imu_factor->SetVertex(graph_.vertices.all_new_frames_bg[new_frame_idx + 1].get(), 9);
        imu_factor->name() = std::string("imu factor");
        RETURN_FALSE_IF(!imu_factor->SelfCheck());

        ++frame_idx;
        ++new_frame_idx;
        BREAK_IF(frame_idx > data_manager_->visual_local_map()->frames().back().id());
    }

    return true;
}

bool Backend::AddImuPreintegrationFactorForMarginalization(const uint32_t idx_offset) {
    RETURN_TRUE_IF(data_manager_->frames_with_bias().size() < 2);

    // [Edges] Inerial preintegration factor.
    // The imu preintegration block combined with the oldest 'new frame with bias' is useless.
    // Add edges of imu preintegration.
    const uint32_t frame_idx = idx_offset;
    const uint32_t new_frame_idx = 0;
    const auto &frame = *std::next(data_manager_->frames_with_bias().begin());

    graph_.edges.all_imu_factors.emplace_back(std::make_unique<EdgeImuPreintegrationBetweenRelativePose<DorF>>(
        frame.imu_preint_block, options_.kGravityInWordFrame));
    auto &imu_factor = graph_.edges.all_imu_factors.back();
    imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[frame_idx].get(), 0);
    imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[frame_idx].get(), 1);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_v_wi[new_frame_idx].get(), 2);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_ba[new_frame_idx].get(), 3);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_bg[new_frame_idx].get(), 4);
    imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[frame_idx + 1].get(), 5);
    imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[frame_idx + 1].get(), 6);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_v_wi[new_frame_idx + 1].get(), 7);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_ba[new_frame_idx + 1].get(), 8);
    imu_factor->SetVertex(graph_.vertices.all_new_frames_bg[new_frame_idx + 1].get(), 9);
    imu_factor->name() = std::string("imu factor");
    RETURN_FALSE_IF(!imu_factor->SelfCheck());

    return true;
}

void Backend::ConstructGraphOptimizationProblem(const uint32_t idx_offset, Graph<DorF> &problem) {
    // Add all vertices and edges.
    for (uint32_t i = 0; i < graph_.vertices.all_cameras_p_ic.size(); ++i) {
        problem.AddVertex(graph_.vertices.all_cameras_p_ic[i].get());
        problem.AddVertex(graph_.vertices.all_cameras_q_ic[i].get());
    }
    for (uint32_t i = 0; i < graph_.vertices.all_frames_p_wi.size(); ++i) {
        problem.AddVertex(graph_.vertices.all_frames_p_wi[i].get());
        problem.AddVertex(graph_.vertices.all_frames_q_wi[i].get());
        if (i >= idx_offset) {
            const uint32_t j = i - idx_offset;
            problem.AddVertex(graph_.vertices.all_new_frames_v_wi[j].get());
            problem.AddVertex(graph_.vertices.all_new_frames_ba[j].get());
            problem.AddVertex(graph_.vertices.all_new_frames_bg[j].get());
        }
    }
    for (auto &vertex : graph_.vertices.all_features_invdep) {
        problem.AddVertex(vertex.get(), false);
    }
    for (auto &edge : graph_.edges.all_prior_factors) {
        problem.AddEdge(edge.get());
    }
    for (auto &edge : graph_.edges.all_visual_reproj_factors) {
        problem.AddEdge(edge.get());
    }
    for (auto &edge : graph_.edges.all_imu_factors) {
        problem.AddEdge(edge.get());
    }

    // Report information.
    const std::string prior_str = states_.prior.is_valid ? ", and prior information." : ".";
    ReportInfo("[Backend] Estimator adds " <<
        graph_.vertices.all_cameras_p_ic.size() << " cameras_p_ic, " <<
        graph_.vertices.all_cameras_q_ic.size() << " cameras_q_ic, " <<
        graph_.vertices.all_features_invdep.size() << " features_invdep, " <<
        graph_.vertices.all_frames_p_wi.size() << " frames_p_wi, " <<
        graph_.vertices.all_frames_q_wi.size() << " frames_q_wi, " <<
        graph_.vertices.all_new_frames_v_wi.size() << " new_frames_v_wi, " <<
        graph_.vertices.all_new_frames_ba.size() << " new_frames_ba, " <<
        graph_.vertices.all_new_frames_bg.size() << " new_frames_bg, and " <<

        graph_.edges.all_prior_factors.size() << " prior_factors, " <<
        graph_.edges.all_visual_reproj_factors.size() << " visual_reproj_factors, " <<
        graph_.edges.all_imu_factors.size() << " imu_factors" << prior_str);

    // Add prior information if valid.
    if (states_.prior.is_valid) {
        problem.prior_hessian() = states_.prior.hessian;
        problem.prior_bias() = states_.prior.bias;
        problem.prior_jacobian_t_inv() = states_.prior.jacobian_t_inv;
        problem.prior_residual() = states_.prior.residual;
    }
}

}