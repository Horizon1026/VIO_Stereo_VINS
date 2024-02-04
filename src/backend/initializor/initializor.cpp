#include "backend.h"
#include "log_report.h"

namespace VIO {

namespace {
    const float kMinValidImuAccelVarianceForMonoInitialization = 0.4f;
}

bool Backend::TryToInitialize() {
    if (data_manager_->frames_with_bias().size() < data_manager_->options().kMaxStoredKeyFrames) {
        ReportWarn("[Backend] Backend cannot initialize for lack of new frames.");
        return false;
    }

    // Check if imu motion is enough.
    const float imu_accel_variance = data_manager_->ComputeImuAccelVariance();
    if (imu_accel_variance < kMinValidImuAccelVarianceForMonoInitialization) {
        ReportWarn("[Backend] Backend cannot initialize for lack of imu motion.");
        return false;
    }

    // Convert all frames into a covisible graph.
    if (!data_manager_->ConvertAllFramesWithBiasToLocalMap()) {
        ReportError("[Backend] Backend failed to convert frames to covisible graph.");
        return false;
    }

    if (!PrepareForPureVisualSfm()) {
        ReportError("[Backend] Backend failed to prepare for pure visual SFM.");
        return false;
    }

    // Debug.
    should_quit_ = true;

    return true;
}

}
