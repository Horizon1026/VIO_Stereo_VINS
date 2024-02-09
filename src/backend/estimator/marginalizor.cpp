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
            ReportInfo("[Backend] Backend marginalize oldest frame.");
            return MarginalizeOldestFrame(use_multi_view);
            break;
        }
        case BackendMarginalizeType::kMarginalizeSubnewFrame: {
            ReportInfo("[Backend] Backend marginalize subnew frame.");
            return MarginalizeSubnewFrame(use_multi_view);
            break;
        }
        default:
        case BackendMarginalizeType::kNotMarginalize: {
            ReportInfo("[Backend] Backend not marginalize any frame.");
            break;
        }
    }

    return true;
}

bool Backend::MarginalizeOldestFrame(const bool use_multi_view) {
    return true;
}

bool Backend::MarginalizeSubnewFrame(const bool use_multi_view) {
    if (!states_.prior.is_valid) {
        ReportInfo("[Backend] Prior information is invalid, no need to discard subnew prior information.");
        return true;
    }

    return true;
}


}
