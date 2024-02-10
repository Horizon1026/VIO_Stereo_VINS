#include "backend.h"
#include "general_edges.h"
#include "inertial_edges.h"
#include "visual_inertial_edges.h"

#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"

namespace VIO {

BackendMarginalizeType Backend::DecideMarginalizeType() {
    if (data_manager_->visual_local_map()->frames().size() < data_manager_->options().kMaxStoredKeyFrames) {
        return BackendMarginalizeType::kNotMarginalize;
    }

    // Visual frontend select keyframes.
    if (visual_frontend_->options().kSelfSelectKeyframe) {
        ReportInfo("[Backend] Visual frontend select keyframe.");
        if (std::next(data_manager_->frames_with_bias().rbegin())->visual_measure->is_current_keyframe) {
            return BackendMarginalizeType::kMarginalizeOldestFrame;
        } else {
            return BackendMarginalizeType::kMarginalizeSubnewFrame;
        }
    }

    // Backend select keyframes.
    ReportInfo("[Backend] Backend select keyframe.");
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
            ReportColorInfo("[Backend] Backend marginalize oldest frame.");
            return MarginalizeOldestFrame(use_multi_view);
            break;
        }
        case BackendMarginalizeType::kMarginalizeSubnewFrame: {
            ReportColorInfo("[Backend] Backend marginalize subnew frame.");
            return MarginalizeSubnewFrame(use_multi_view);
            break;
        }
        default:
        case BackendMarginalizeType::kNotMarginalize: {
            ReportColorInfo("[Backend] Backend not marginalize any frame.");
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

    // marger.problem()->VerticesInformation();
    // data_manager_->ShowMatrixImage("hessian", marger.problem()->hessian());
    // data_manager_->ShowMatrixImage("reverse hessian", marger.reverse_hessian());
    // data_manager_->ShowMatrixImage("prior hessian", marger.problem()->prior_hessian());
    // data_manager_->ShowLocalMapInWorldFrame("Estimation result", 50, true);

    // Report the change of prior information.
    if (states_.prior.is_valid) {
        ReportInfo("[Backend] Estimation change prior residual squared norm [" << prior_residual_norm <<
            "] -> [" << graph_optimization_problem.prior_residual().squaredNorm() << "]. Prior size [" <<
            marger.problem()->prior_hessian().cols() << "].");
    }

    // Store prior information.
    if (states_.prior.is_valid) {
        states_.prior.hessian = marger.problem()->prior_hessian();
        states_.prior.bias = marger.problem()->prior_bias();
        states_.prior.jacobian_t_inv = marger.problem()->prior_jacobian_t_inv();
        states_.prior.residual = marger.problem()->prior_residual();
    }

    return true;
}

bool Backend::MarginalizeSubnewFrame(const bool use_multi_view) {
    if (!states_.prior.is_valid) {
        ReportInfo("[Backend] Prior information is invalid, no need to discard subnew prior information.");
        return true;
    }

    if (data_manager_->visual_local_map()->frames().size() < 3) {
        ReportInfo("[Backend] Totally discard prior information.");
        states_.prior.is_valid = false;
        return true;
    }

    const float prior_residual_norm = states_.prior.residual.squaredNorm();
    const int32_t size_of_prior = static_cast<int32_t>(states_.prior.hessian.cols());

    // Compute the size of prior information after discarding.
    const int32_t min_size = (data_manager_->visual_local_map()->frames().size() - 2) * 15 +
        6 * data_manager_->camera_extrinsics().size();
    if (states_.prior.hessian.cols() <= min_size) {
        ReportInfo("[Backend] No prior information is discarded.");
        return true;
    }

    // Discard prior information of subnew frame.
    Marginalizor<DorF> marger;
    // Prior information of frame to be discarded shoule be directly discarded.
    marger.DiscardPriorInformation(states_.prior.hessian, states_.prior.bias, min_size, 15);
    // Prior jacobian_t_inv and residual should be decomposed by hessian and bias.
    marger.DecomposeHessianAndBias(states_.prior.hessian, states_.prior.bias,
        states_.prior.jacobian, states_.prior.residual, states_.prior.jacobian_t_inv);

    // Report the change of prior information.
    if (states_.prior.is_valid) {
        ReportInfo("[Backend] Estimation change prior residual squared norm [" << prior_residual_norm <<
            "] -> [" << states_.prior.residual.squaredNorm() << "]. Prior size [" <<
            size_of_prior << "] -> [" << states_.prior.hessian.cols() << "].");
    }

    return true;
}


}
