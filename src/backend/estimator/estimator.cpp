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

bool Backend::TryToEstimate(const bool use_multi_view) {
    // Clear all vectors of vertices and edges.
    ClearGraph();
    // [Vertices] Camera extrinsics.
    AddAllCameraExtrinsicsToGraph();
    // [Vertices] Imu pose of each frame in local map.
    AddAllImuPosesInLocalMapToGraph();
    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    const bool add_factors_with_cam_ex = true;
    AddAllFeatureInvdepsAndVisualFactorsToGraph(add_factors_with_cam_ex, use_multi_view);

    // Construct full visual-inertial problem.
    // Graph<DorF> graph_optimization_problem;
    // ConstructVioGraphOptimizationProblem(graph_optimization_problem);

    data_manager_->ShowLocalMapInWorldFrame("Estimation result", 30, true);

    return true;
}

}
