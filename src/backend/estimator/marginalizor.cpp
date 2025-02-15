#include "backend.h"
#include "general_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "slam_log_reporter.h"
#include "tick_tock.h"
#include "slam_basic_math.h"

namespace VIO {

BackendMarginalizeType Backend::DecideMarginalizeType() {
    if (data_manager_->visual_local_map()->frames().size() < data_manager_->options().kMaxStoredKeyFrames) {
        return BackendMarginalizeType::kNotMarginalize;
    }

    // Visual frontend select keyframes.
    if (visual_frontend_->options().kSelfSelectKeyframe) {
        if (std::next(data_manager_->imu_based_frames().rbegin())->visual_measure->is_current_keyframe) {
            return BackendMarginalizeType::kMarginalizeOldestFrame;
        } else {
            return BackendMarginalizeType::kMarginalizeSubnewFrame;
        }
    }

    // Backend select keyframes.
    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    covisible_features.reserve(visual_frontend_->options().kMaxStoredFeaturePointsNumber);
    const uint32_t ref_frame_id = data_manager_->visual_local_map()->frames().back().id() - 2;
    const uint32_t cur_frame_id = ref_frame_id + 1;
    if (!data_manager_->visual_local_map()->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        covisible_features.clear();
    }

    // Decide marginalize type.
    if (covisible_features.size() < visual_frontend_->options().kMinDetectedFeaturePointsNumberInCurrentImage) {
        return BackendMarginalizeType::kMarginalizeOldestFrame;
    } else {
        return BackendMarginalizeType::kMarginalizeSubnewFrame;
    }

    return BackendMarginalizeType::kNotMarginalize;
}

bool Backend::TryToMarginalize(const bool use_multi_view) {
    switch (status_.marginalize_type) {
        case BackendMarginalizeType::kMarginalizeOldestFrame: {
            return MarginalizeOldestFrame(use_multi_view);
        }
        case BackendMarginalizeType::kMarginalizeSubnewFrame: {
            return MarginalizeSubnewFrame(use_multi_view);
        }
        default:
        case BackendMarginalizeType::kNotMarginalize: {
            break;
        }
    }

    return true;
}

bool Backend::MarginalizeOldestFrame(const bool use_multi_view) {
    // Clear all vectors of vertices and edges.
    ClearGraph();
    // [Vertices] Camera extrinsics.
    AddAllCameraExtrinsicsToGraph();
    // [Vertices] Imu pose of each frame in local map.
    AddAllImuPosesInLocalMapToGraph();
    // [Vertices] Imu velocity of each frame.
    // [Vertices] Imu bias of accel and gyro in each frame.
    AddAllImuMotionStatesInLocalMapToGraph();
    // [Vertices] Inverse depth of each feature observed in oldest frame.
    // [Edges] Visual reprojection factor.
    RETURN_FALSE_IF(!AddFeatureFirstObserveInOldestFrameAndVisualFactorsToGraph(use_multi_view));
    // [Edges] Imu pose prior factor. (In order to fix first imu pose)
    // [Edges] Camera extrinsic prior factor.
    RETURN_FALSE_IF(!AddPriorFactorForFirstImuPoseAndCameraExtrinsicsToGraph());
    // [Edges] Imu preintegration block factors.
    const bool only_add_oldest_one = true;
    RETURN_FALSE_IF(!AddImuFactorsToGraph(only_add_oldest_one));
    // [Vertices] Newest and subnew frame is not including in prior information.
    // If not remove them, too much zeros will fill prior information, which will perform bad when marginalize subnew frame.
    RemoveNewestTwoFramesFromGraph();
    // Construct full visual-inertial problem.
    Graph<DorF> graph_optimization_problem;
    float prior_residual_norm = 0.0f;
    ConstructVioGraphOptimizationProblem(graph_optimization_problem, prior_residual_norm);

    // Set vertices to be marged.
    std::vector<Vertex<DorF> *> vertices_to_be_marged = {
        graph_.vertices.all_frames_p_wi.front().get(),
        graph_.vertices.all_frames_q_wi.front().get(),
        graph_.vertices.all_frames_v_wi.front().get(),
        graph_.vertices.all_frames_ba.front().get(),
        graph_.vertices.all_frames_bg.front().get(),
    };

    // Do marginalization.
    Marginalizor<DorF> marger;
    marger.problem() = &graph_optimization_problem;
    marger.options().kSortDirection = SortMargedVerticesDirection::kSortAtFront;
    states_.prior.is_valid = marger.Marginalize(vertices_to_be_marged, states_.prior.is_valid);

    // Store prior information.
    if (states_.prior.is_valid) {
        states_.prior.hessian = marger.problem()->prior_hessian();
        states_.prior.bias = marger.problem()->prior_bias();
        states_.prior.jacobian_t_inv = marger.problem()->prior_jacobian_t_inv();
        states_.prior.residual = marger.problem()->prior_residual();
    }

    // Mark the features that been marginalized. They will not be used in the following estimation.
    for (const auto &id : graph_.vertices.all_features_id) {
        data_manager_->visual_local_map()->feature(id)->status() = FeatureSolvedStatus::kMarginalized;
    }

    return true;
}

bool Backend::MarginalizeSubnewFrame(const bool use_multi_view) {
    if (!states_.prior.is_valid) {
        return true;
    }

    if (data_manager_->visual_local_map()->frames().size() < 3) {
        states_.prior.is_valid = false;
        return true;
    }

    // Compute the size of prior information after discarding.
    const int32_t min_size = (data_manager_->visual_local_map()->frames().size() - 2) * 15 +
        6 * data_manager_->camera_extrinsics().size();
    if (states_.prior.hessian.cols() <= min_size) {
        return true;
    }

    // Discard prior information of subnew frame.
    Marginalizor<DorF> marger;
    // Prior information of frame to be discarded shoule be directly discarded.
    marger.DiscardPriorInformation(states_.prior.hessian, states_.prior.bias, min_size, 15);
    // Prior jacobian_t_inv and residual should be decomposed by hessian and bias.
    marger.DecomposeHessianAndBias(states_.prior.hessian, states_.prior.bias,
        states_.prior.jacobian, states_.prior.residual, states_.prior.jacobian_t_inv);

    return true;
}


}
