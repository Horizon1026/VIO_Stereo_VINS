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
    // [Vertices] Imu velocity of each frame.
    // [Vertices] Imu bias of accel and gyro in each frame.
    AddAllImuMotionStatesInLocalMapToGraph();
    // [Vertices] Inverse depth of each feature.
    // [Edges] Visual reprojection factor.
    const bool add_factors_with_cam_ex = true;
    AddAllFeatureInvdepsAndVisualFactorsToGraph(add_factors_with_cam_ex, use_multi_view);
    // [Edges] Imu pose prior factor. (In order to fix first imu pose)
    // [Edges] Camera extrinsic prior factor.
    RETURN_FALSE_IF(!AddPriorFactorForFirstImuPoseAndCameraExtrinsicsToGraph());
    // [Edges] Imu preintegration block factors.
    RETURN_FALSE_IF(!AddImuFactorsToGraph());

    // Construct full visual-inertial problem.
    Graph<DorF> graph_optimization_problem;
    ConstructVioGraphOptimizationProblem(graph_optimization_problem);

    // Construct solver to solve this problem.
    SolverLm<DorF> solver;
    solver.options().kEnableReportEachIteration = options_.kEnableReportAllInformation;
    solver.options().kMaxConvergedSquaredStepLength = static_cast<DorF>(1e-4);
    solver.options().kMaxCostTimeInSecond = 0.05f;
    solver.problem() = &graph_optimization_problem;
    solver.Solve(states_.prior.is_valid);

    // Update all states in visual_local_map and frames_with_bias.
    RETURN_FALSE_IF(!SyncGraphVerticesToDataManager(graph_optimization_problem));

    data_manager_->ShowLocalMapInWorldFrame("Estimation result", 30, true);
    return true;
}

}
