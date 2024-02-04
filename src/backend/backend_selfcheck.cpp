#include "backend.h"
#include "general_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"

namespace VIO {

bool Backend::CheckGraphOptimizationFactors() {
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

    // Iterate all edges.
    RETURN_FALSE_IF(!CheckGraphOptimizationFactors(graph_.edges.all_prior_factors));
    RETURN_FALSE_IF(!CheckGraphOptimizationFactors(graph_.edges.all_visual_reproj_factors));
    RETURN_FALSE_IF(!CheckGraphOptimizationFactors(graph_.edges.all_imu_factors));

    return true;
}

bool Backend::CheckGraphOptimizationFactors(std::vector<std::unique_ptr<Edge<DorF>>> &edges) {
    bool is_factor_valid = true;
    for (auto &edge : edges) {
        if (!edge->SelfCheckJacobians()) {
            is_factor_valid = false;
        }
    }

    return is_factor_valid;
}

}