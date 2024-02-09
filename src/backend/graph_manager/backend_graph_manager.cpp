#include "backend.h"
#include "general_edges.h"
#include "visual_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "solver_lm.h"
#include "solver_dogleg.h"

#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"

namespace VIO {

void Backend::ClearGraph() {
    graph_.vertices.all_cameras_p_ic.clear();
    graph_.vertices.all_cameras_q_ic.clear();

    graph_.vertices.all_frames_id.clear();
    graph_.vertices.all_frames_p_wi.clear();
    graph_.vertices.all_frames_q_wi.clear();
    graph_.vertices.all_frames_p_wc.clear();
    graph_.vertices.all_frames_q_wc.clear();

    graph_.vertices.all_features_id.clear();
    graph_.vertices.all_features_invdep.clear();

    graph_.vertices.all_frames_v_wi.clear();
    graph_.vertices.all_frames_ba.clear();
    graph_.vertices.all_frames_bg.clear();

    graph_.edges.all_prior_factors.clear();
    graph_.edges.all_visual_factors.clear();
    graph_.edges.all_imu_factors.clear();
}

void Backend::ConstructVioGraphOptimizationProblem(Graph<DorF> &problem) {
    // Add all vertices into graph.
    for (uint32_t i = 0; i < graph_.vertices.all_cameras_p_ic.size(); ++i) {
        problem.AddVertex(graph_.vertices.all_cameras_p_ic[i].get());
        problem.AddVertex(graph_.vertices.all_cameras_q_ic[i].get());
    }

    const uint32_t pose_num = graph_.vertices.all_frames_p_wi.size();
    for (uint32_t i = 0; i < pose_num; ++i) {
        problem.AddVertex(graph_.vertices.all_frames_p_wi[i].get());
        problem.AddVertex(graph_.vertices.all_frames_q_wi[i].get());
        problem.AddVertex(graph_.vertices.all_frames_v_wi[i].get());
        problem.AddVertex(graph_.vertices.all_frames_ba[i].get());
        problem.AddVertex(graph_.vertices.all_frames_bg[i].get());
    }
    for (auto &vertex : graph_.vertices.all_features_invdep) {
        problem.AddVertex(vertex.get(), false);
    }

    // Add all edges into graph.
    for (auto &edge : graph_.edges.all_prior_factors) {
        problem.AddEdge(edge.get());
    }
    for (auto &edge : graph_.edges.all_visual_factors) {
        problem.AddEdge(edge.get());
    }
    for (auto &edge : graph_.edges.all_imu_factors) {
        problem.AddEdge(edge.get());
    }

    // Report information.
    ReportInfo("[Backend] Full vio adds [Vertices] " <<
        graph_.vertices.all_cameras_p_ic.size() << " p_ic, " <<
        graph_.vertices.all_cameras_q_ic.size() << " q_ic, " <<
        graph_.vertices.all_frames_p_wi.size() << " p_wi, " <<
        graph_.vertices.all_frames_q_wi.size() << " q_wi, " <<
        graph_.vertices.all_frames_v_wi.size() << " v_wi, " <<
        graph_.vertices.all_frames_ba.size() << " ba, " <<
        graph_.vertices.all_frames_bg.size() << " bg, " <<
        graph_.vertices.all_features_invdep.size() << " invdep.");
    ReportInfo("[Backend] Full vio adds [Edges] " <<
        graph_.edges.all_prior_factors.size() << " prior pose factors, " <<
        graph_.edges.all_visual_factors.size() << " visual factors, " <<
        graph_.edges.all_imu_factors.size() << " imu factors.");

    // Add prior information if it is valid.
    if (states_.prior.is_valid) {
        problem.prior_hessian() = states_.prior.hessian;
        problem.prior_bias() = states_.prior.bias;
        problem.prior_jacobian_t_inv() = states_.prior.jacobian_t_inv;
        problem.prior_residual() = states_.prior.residual;
        ReportInfo("[Backend] Before estimation, prior residual squared norm is " <<
            problem.prior_residual().squaredNorm());
    }
}

void Backend::ConstructPureVisualGraphOptimizationProblem(Graph<DorF> &problem) {
    // Add all vertices into graph.
    for (uint32_t i = 0; i < graph_.vertices.all_frames_p_wc.size(); ++i) {
        problem.AddVertex(graph_.vertices.all_frames_p_wc[i].get());
        problem.AddVertex(graph_.vertices.all_frames_q_wc[i].get());
    }
    for (auto &vertex : graph_.vertices.all_features_invdep) {
        problem.AddVertex(vertex.get(), false);
    }

    // Add all edges into graph.
    for (auto &edge : graph_.edges.all_visual_factors) {
        problem.AddEdge(edge.get());
    }

    // Report information.
    ReportInfo("[Backend] Pure visual BA adds " <<
        graph_.vertices.all_frames_p_wc.size() << " frames_p_wc, " <<
        graph_.vertices.all_frames_q_wc.size() << " frames_q_wc, " <<
        graph_.vertices.all_features_invdep.size() << " features_invdep, " <<
        graph_.edges.all_visual_factors.size() << " visual_reproj_factors.");
}

void Backend::AddAllCameraExtrinsicsToGraph() {
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
}

void Backend::AddAllCameraPosesInLocalMapToGraph() {
    // [Vertices] Camera pose of each frame.
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        graph_.vertices.all_frames_id.emplace_back(frame.id());

        graph_.vertices.all_frames_p_wc.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_p_wc.back()->param() = frame.p_wc().cast<DorF>();
        graph_.vertices.all_frames_p_wc.back()->name() = std::string("p_wc") + std::to_string(frame.id());

        graph_.vertices.all_frames_q_wc.emplace_back(std::make_unique<VertexQuat<DorF>>(4, 3));
        graph_.vertices.all_frames_q_wc.back()->param() << frame.q_wc().w(), frame.q_wc().x(), frame.q_wc().y(), frame.q_wc().z();
        graph_.vertices.all_frames_q_wc.back()->name() = std::string("q_wc") + std::to_string(frame.id());
    }
}

void Backend::AddAllImuPosesInLocalMapToGraph() {
    // [Vertices] Imu pose of each frame.
    uint32_t frame_id = data_manager_->visual_local_map()->frames().front().id();
    for (const auto &frame_with_bias : data_manager_->frames_with_bias()) {
        graph_.vertices.all_frames_p_wi.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_p_wi.back()->param() = frame_with_bias.p_wi.cast<DorF>();
        graph_.vertices.all_frames_p_wi.back()->name() = std::string("p_wi") + std::to_string(frame_id);

        graph_.vertices.all_frames_q_wi.emplace_back(std::make_unique<VertexQuat<DorF>>(4, 3));
        graph_.vertices.all_frames_q_wi.back()->param() << frame_with_bias.q_wi.w(), frame_with_bias.q_wi.x(), frame_with_bias.q_wi.y(), frame_with_bias.q_wi.z();
        graph_.vertices.all_frames_q_wi.back()->name() = std::string("q_wi") + std::to_string(frame_id);

        ++frame_id;
    }
}

void Backend::AddAllImuMotionStatesInLocalMapToGraph() {
    // [Vertices] Imu velocity of each frame.
    // [Vertices] Imu bias of accel and gyro in each frame.
    uint32_t frame_id = data_manager_->visual_local_map()->frames().front().id();
    for (const auto &frame_with_bias : data_manager_->frames_with_bias()) {
        graph_.vertices.all_frames_v_wi.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_v_wi.back()->param() = frame_with_bias.v_wi.cast<DorF>();
        graph_.vertices.all_frames_v_wi.back()->name() = std::string("v_wi") + std::to_string(frame_id);

        graph_.vertices.all_frames_ba.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_ba.back()->param() = frame_with_bias.imu_preint_block.bias_accel().cast<DorF>();
        graph_.vertices.all_frames_ba.back()->name() = std::string("bias_a") + std::to_string(frame_id);

        graph_.vertices.all_frames_bg.emplace_back(std::make_unique<Vertex<DorF>>(3, 3));
        graph_.vertices.all_frames_bg.back()->param() = frame_with_bias.imu_preint_block.bias_gyro().cast<DorF>();
        graph_.vertices.all_frames_bg.back()->name() = std::string("bias_g") + std::to_string(frame_id);

        ++frame_id;
    }
}

bool Backend::AllFeatureInvdepAndVisualFactorsOfCameraPosesToGraph(const FeatureType &feature,
                                                                   const float invdep,
                                                                   const TMat2<DorF> &visual_info_matrix,
                                                                   const uint32_t max_frame_id,
                                                                   const bool use_multi_view) {
    // Determine the range of all observations of this feature.
    const uint32_t min_frame_id = feature.first_frame_id();
    const uint32_t offset = data_manager_->visual_local_map()->frames().front().id();

    // Add vertex of feature invdep.
    graph_.vertices.all_features_id.emplace_back(feature.id());
    graph_.vertices.all_features_invdep.emplace_back(std::make_unique<Vertex<DorF>>(1, 1));
    graph_.vertices.all_features_invdep.back()->param() = TVec1<DorF>(invdep);
    graph_.vertices.all_features_invdep.back()->name() = std::string("invdep ") + std::to_string(feature.id());

    // Extract observation of this feature in first frame which observes it.
    const auto &obv_in_ref = feature.observe(min_frame_id);
    Vec4 observe_vector = Vec4::Zero();
    observe_vector.head<2>() = obv_in_ref[0].rectified_norm_xy;

    // In order to add other edges, iterate all observations of this feature.
    for (uint32_t idx = min_frame_id + 1; idx <= max_frame_id; ++idx) {
        BREAK_IF(idx > feature.final_frame_id());

        const auto &obv_in_cur = feature.observe(idx);
        observe_vector.tail<2>() = obv_in_cur[0].rectified_norm_xy;

        // Add edges of visual reprojection factor, considering one camera views two frames.
        graph_.edges.all_visual_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlane<DorF>>());
        auto &visual_reproj_factor = graph_.edges.all_visual_factors.back();
        visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wc[min_frame_id - offset].get(), 1);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wc[min_frame_id - offset].get(), 2);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wc[idx - offset].get(), 3);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wc[idx - offset].get(), 4);
        visual_reproj_factor->observation() = observe_vector.cast<DorF>();
        visual_reproj_factor->information() = visual_info_matrix;
        visual_reproj_factor->kernel() = std::make_unique<KernelHuber<DorF>>(static_cast<DorF>(0.5));
        visual_reproj_factor->name() = std::string("pure visual ba");
        RETURN_FALSE_IF(!visual_reproj_factor->SelfCheck());
    }

    return true;
}

bool Backend::AllFeatureInvdepAndVisualFactorsOfImuPosesToGraph(const FeatureType &feature,
                                                                const float invdep,
                                                                const TMat2<DorF> &visual_info_matrix,
                                                                const uint32_t max_frame_id,
                                                                const bool use_multi_view) {
    // Determine the range of all observations of this feature.
    const uint32_t min_frame_id = feature.first_frame_id();

    // Add vertex of feature invdep.
    graph_.vertices.all_features_id.emplace_back(feature.id());
    graph_.vertices.all_features_invdep.emplace_back(std::make_unique<Vertex<DorF>>(1, 1));
    graph_.vertices.all_features_invdep.back()->param() = TVec1<DorF>(invdep);
    graph_.vertices.all_features_invdep.back()->name() = std::string("invdep ") + std::to_string(feature.id());

    // Extract observation of this feature in first frame which observes it.
    const auto &obv_in_ref = feature.observe(min_frame_id);
    Vec4 observe_vector = Vec4::Zero();
    observe_vector.head<2>() = obv_in_ref[0].rectified_norm_xy;

    // Add edges of visual reprojection factor, considering two cameras view one frame.
    for (uint32_t i = 1; i < obv_in_ref.size(); ++i) {
        BREAK_IF(use_multi_view);
        observe_vector.tail<2>() = obv_in_ref[i].rectified_norm_xy;

        // Add edge of visual reprojection factor, considering two camera view one frame.
        graph_.edges.all_visual_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinOneFramesTwoCamera<DorF>>());
        auto &visual_reproj_factor = graph_.edges.all_visual_factors.back();
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
        BREAK_IF(idx > feature.final_frame_id());

        const auto &obv_in_cur = feature.observe(idx);
        observe_vector.tail<2>() = obv_in_cur[0].rectified_norm_xy;

        // Add edges of visual reprojection factor, considering one camera views two frames.
        graph_.edges.all_visual_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera<DorF>>());
        auto &visual_reproj_factor = graph_.edges.all_visual_factors.back();
        visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[min_frame_id - 1].get(), 1);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[min_frame_id - 1].get(), 2);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[idx - 1].get(), 3);
        visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[idx - 1].get(), 4);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_p_ic[0].get(), 5);
        visual_reproj_factor->SetVertex(graph_.vertices.all_cameras_q_ic[0].get(), 6);
        visual_reproj_factor->observation() = observe_vector.cast<DorF>();
        visual_reproj_factor->information() = visual_info_matrix;
        visual_reproj_factor->kernel() = std::make_unique<KernelHuber<DorF>>(static_cast<DorF>(0.5));
        visual_reproj_factor->name() = std::string("two frames one camera");
        RETURN_FALSE_IF(!visual_reproj_factor->SelfCheck());

        CONTINUE_IF(use_multi_view);

        // Add edges of visual reprojection factor, considering two cameras view two frames.
        for (uint32_t i = 1; i < obv_in_cur.size(); ++i) {
            observe_vector.tail<2>() = obv_in_cur[i].rectified_norm_xy;

            graph_.edges.all_visual_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera<DorF>>());
            auto &visual_reproj_factor = graph_.edges.all_visual_factors.back();
            visual_reproj_factor->SetVertex(graph_.vertices.all_features_invdep.back().get(), 0);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[min_frame_id - 1].get(), 1);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[min_frame_id - 1].get(), 2);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_p_wi[idx - 1].get(), 3);
            visual_reproj_factor->SetVertex(graph_.vertices.all_frames_q_wi[idx - 1].get(), 4);
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

bool Backend::AddAllFeatureInvdepsAndVisualFactorsToGraph(const bool add_factors_with_cam_ex, const bool use_multi_view) {
    // Compute information matrix of visual observation.
    const TMat2<DorF> visual_info_matrix = GetVisualObserveInformationMatrix();

    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;
        // Select features which has at least two observations.
        CONTINUE_IF(use_multi_view && feature.observes().size() < 2 && feature.observes().front().size() < 2)
        CONTINUE_IF(!use_multi_view && feature.observes().size() < 2);
        // Select features which is solved successfully.
        CONTINUE_IF(feature.status() != FeatureSolvedStatus::kSolved);

        // Compute inverse depth by p_w of this feature.
        const auto &frame = data_manager_->visual_local_map()->frame(feature.first_frame_id());
        const Vec3 p_c = frame->q_wc().inverse() * (feature.param() - frame->p_wc());
        const float invdep = 1.0f / p_c.z();
        CONTINUE_IF(std::isinf(invdep) || std::isnan(invdep));

        // Convert feature invdep to vertices, and add visual factors.
        if (add_factors_with_cam_ex) {
            RETURN_FALSE_IF(!AllFeatureInvdepAndVisualFactorsOfImuPosesToGraph(feature, invdep, visual_info_matrix, feature.final_frame_id(), use_multi_view));
        } else {
            RETURN_FALSE_IF(!AllFeatureInvdepAndVisualFactorsOfCameraPosesToGraph(feature, invdep, visual_info_matrix, feature.final_frame_id(), use_multi_view));
        }
    }

    return true;
}

bool Backend::AddFeatureFirstObserveInOldestFrameAndVisualFactorsToGraph(const bool use_multi_view) {
    // Compute information matrix of visual observation.
    const TMat2<DorF> visual_info_matrix = GetVisualObserveInformationMatrix();

    // Extract index of the oldest frame in visual_local_map;
    const auto oldest_frame_id = data_manager_->visual_local_map()->frames().front().id();

    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;
        // Select features which is first observed in oldest frame in visual_local_map.
        CONTINUE_IF(feature.first_frame_id() != oldest_frame_id);
        // Select features which has at least two observations.
        CONTINUE_IF(use_multi_view && feature.observes().size() < 2 && feature.observes().front().size() < 2)
        CONTINUE_IF(!use_multi_view && feature.observes().size() < 2);
        // Select features which is solved successfully.
        CONTINUE_IF(feature.status() != FeatureSolvedStatus::kSolved);

        // Compute inverse depth by p_w of this feature.
        const auto &frame = data_manager_->visual_local_map()->frame(feature.first_frame_id());
        const Vec3 p_c = frame->q_wc().inverse() * (feature.param() - frame->p_wc());
        CONTINUE_IF(p_c.z() < options_.kMinValidFeatureDepthInMeter);
        const float invdep = 1.0f / p_c.z();
        CONTINUE_IF(std::isinf(invdep) || std::isnan(invdep));

        // Convert feature invdep to vertices, and add visual factors.
        RETURN_FALSE_IF(!AllFeatureInvdepAndVisualFactorsOfImuPosesToGraph(feature, invdep, visual_info_matrix, feature.final_frame_id(), use_multi_view));
    }

    return true;
}

bool Backend::AddImuFactorsToGraph(const bool only_add_oldest_one) {
    RETURN_TRUE_IF(data_manager_->frames_with_bias().size() < 2);

    // [Edges] Imu preintegration block factors.
    int32_t index = 0;
    for (auto it = std::next(data_manager_->frames_with_bias().begin()); it != data_manager_->frames_with_bias().end(); ++it, ++index) {
        // Add edges of imu preintegration between relative imu pose and motion states.
        const auto &frame_with_bias = *it;

        graph_.edges.all_imu_factors.emplace_back(std::make_unique<EdgeImuPreintegrationBetweenRelativePose<DorF>>(
            frame_with_bias.imu_preint_block, options_.kGravityInWordFrame));
        auto &imu_factor = graph_.edges.all_imu_factors.back();
        imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[index].get(), 0);
        imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[index].get(), 1);
        imu_factor->SetVertex(graph_.vertices.all_frames_v_wi[index].get(), 2);
        imu_factor->SetVertex(graph_.vertices.all_frames_ba[index].get(), 3);
        imu_factor->SetVertex(graph_.vertices.all_frames_bg[index].get(), 4);
        imu_factor->SetVertex(graph_.vertices.all_frames_p_wi[index + 1].get(), 5);
        imu_factor->SetVertex(graph_.vertices.all_frames_q_wi[index + 1].get(), 6);
        imu_factor->SetVertex(graph_.vertices.all_frames_v_wi[index + 1].get(), 7);
        imu_factor->SetVertex(graph_.vertices.all_frames_ba[index + 1].get(), 8);
        imu_factor->SetVertex(graph_.vertices.all_frames_bg[index + 1].get(), 9);
        imu_factor->name() = std::string("imu factor [") + std::to_string(index + 1) + std::string("~") + std::to_string(index + 2) + std::string("]");
        RETURN_FALSE_IF(!imu_factor->SelfCheck());

        BREAK_IF(only_add_oldest_one);
    }

    return true;
}

bool Backend::AddPriorFactorForFirstImuPoseAndCameraExtrinsicsToGraph() {
    RETURN_TRUE_IF(states_.prior.is_valid);
    RETURN_FALSE_IF(graph_.vertices.all_frames_p_wi.empty() || graph_.vertices.all_frames_q_wi.empty());

    // [Edges] Imu pose prior factor. (In order to fix first imu pose)
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

    // [Edges] Camera extrinsic prior factor.
    RETURN_TRUE_IF(data_manager_->camera_extrinsics().empty());
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

bool Backend::SyncGraphVerticesToDataManager(const Graph<DorF> &problem) {
    // Update all camera extrinsics.
    for (uint32_t i = 0; i < graph_.vertices.all_cameras_p_ic.size(); ++i) {
        data_manager_->camera_extrinsics()[i].p_ic = graph_.vertices.all_cameras_p_ic[i]->param().cast<float>();
        data_manager_->camera_extrinsics()[i].q_ic.w() = graph_.vertices.all_cameras_q_ic[i]->param()(0);
        data_manager_->camera_extrinsics()[i].q_ic.x() = graph_.vertices.all_cameras_q_ic[i]->param()(1);
        data_manager_->camera_extrinsics()[i].q_ic.y() = graph_.vertices.all_cameras_q_ic[i]->param()(2);
        data_manager_->camera_extrinsics()[i].q_ic.z() = graph_.vertices.all_cameras_q_ic[i]->param()(3);
    }

    // Update all imu poses and motion states.
    uint32_t index = 0;
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        frame_with_bias.p_wi = graph_.vertices.all_frames_p_wi[index]->param().cast<float>();
        frame_with_bias.q_wi.w() = graph_.vertices.all_frames_q_wi[index]->param()(0);
        frame_with_bias.q_wi.x() = graph_.vertices.all_frames_q_wi[index]->param()(1);
        frame_with_bias.q_wi.y() = graph_.vertices.all_frames_q_wi[index]->param()(2);
        frame_with_bias.q_wi.z() = graph_.vertices.all_frames_q_wi[index]->param()(3);
        frame_with_bias.v_wi = graph_.vertices.all_frames_v_wi[index]->param().cast<float>();

        RecomputeImuPreintegrationBlock(graph_.vertices.all_frames_ba[index]->param().cast<float>(),
            graph_.vertices.all_frames_bg[index]->param().cast<float>(), frame_with_bias);
        ++index;
    }

    // Update all camera poses.
    RETURN_FALSE_IF(!SyncTwiToTwcInLocalMap());

    // Update all feature position.
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

    // Update prior information.
    if (states_.prior.is_valid) {
        states_.prior.hessian = problem.prior_hessian();
        states_.prior.bias = problem.prior_bias();
        states_.prior.jacobian_t_inv = problem.prior_jacobian_t_inv();
        states_.prior.residual = problem.prior_residual();
        ReportInfo("[Backend] After estimation, prior residual squared norm [" <<
            problem.prior_residual().squaredNorm() << "].");
    }

    return true;
}

}
