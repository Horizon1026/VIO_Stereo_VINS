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

void Backend::ConstructGraphOptimizationProblem(Graph<DorF> &problem) {
    // Add all vertices into graph.
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

bool Backend::AllFeatureInvdepAndVisualFactorsToGraph(const FeatureType &feature,
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

bool Backend::AllFeatureInvdepAndVisualFactorsWithCameraExtrinsicsToGraph(const FeatureType &feature,
                                                      const float invdep,
                                                      const TMat2<DorF> &visual_info_matrix,
                                                      const uint32_t max_frame_id,
                                                      const bool use_multi_view) {
    // Determine the range of all observations of this feature.
    const uint32_t min_frame_id = feature.first_frame_id();
    const uint32_t idx_offset = min_frame_id - data_manager_->visual_local_map()->frames().front().id() + 1;

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

        CONTINUE_IF(use_multi_view);

        // Add edges of visual reprojection factor, considering two cameras view two frames.
        for (uint32_t i = 1; i < obv_in_cur.size(); ++i) {
            observe_vector.tail<2>() = obv_in_cur[i].rectified_norm_xy;

            graph_.edges.all_visual_factors.emplace_back(std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera<DorF>>());
            auto &visual_reproj_factor = graph_.edges.all_visual_factors.back();
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
            RETURN_FALSE_IF(!AllFeatureInvdepAndVisualFactorsWithCameraExtrinsicsToGraph(feature, invdep, visual_info_matrix, feature.final_frame_id(), use_multi_view));
        } else {
            RETURN_FALSE_IF(!AllFeatureInvdepAndVisualFactorsToGraph(feature, invdep, visual_info_matrix, feature.final_frame_id(), use_multi_view));
        }
    }

    return true;
}

}
