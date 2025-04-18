#include "backend.h"
#include "general_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "solver_lm.h"
#include "solver_dogleg.h"

#include "slam_log_reporter.h"
#include "tick_tock.h"
#include "slam_basic_math.h"

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
    const bool only_add_oldest_one = false;
    RETURN_FALSE_IF(!AddImuFactorsToGraph(only_add_oldest_one));
    // Construct full visual-inertial problem.
    Graph<DorF> graph_optimization_problem;
    float prior_residual_norm = 0.0f;
    ConstructVioGraphOptimizationProblem(graph_optimization_problem, prior_residual_norm);
    // Record log before bundle adjustment.
    UpdateBackendLogGraph();

    // Construct solver to solve this problem.
    SolverLm<DorF> solver;
    solver.options().kEnableReportEachIteration = false;
    solver.options().kOnlyUseFirstEstimatedJacobian = false;
    solver.options().kEnableDegenerateElimination = true;
    solver.options().kLinearFunctionSolverType = LinearFunctionSolverType::kPcgSolver;
    solver.options().kMaxConvergedSquaredStepLength = static_cast<DorF>(1e-4);
    solver.options().kMaxCostTimeInSecond = options_.kMaxToleranceTimeCostForEstimationInSecond;
    solver.problem() = &graph_optimization_problem;
    solver.Solve(states_.prior.is_valid);
    // Update all states in visual_local_map and imu_based_frames.
    RETURN_FALSE_IF(!SyncGraphVerticesToDataManager(graph_optimization_problem));
    return true;
}

}
