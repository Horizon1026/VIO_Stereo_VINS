#include "backend.h"
#include "log_report.h"

namespace VIO {

namespace {
    const float kMinValidImuAccelVarianceForMonoInitialization = 0.4f;
}

bool Backend::TryToInitialize() {
    if (data_manager_->frames_with_bias().size() < data_manager_->options().kMaxStoredKeyFrames) {
        ReportWarn("[Backend] Backend cannot initialize for lack of frames.");
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

    // Compute initialized value of visual local map.
    if (!PrepareForPureVisualSfm()) {
        ReportError("[Backend] Backend failed to prepare for pure visual SFM.");
        return false;
    }

    // Perform pure visual bundle adjustment.
    if (!PerformPureVisualBundleAdjustment()) {
        ReportError("[Backend] Backend failed to perform pure visual bundle adjustment.");
        return false;
    }

    // Debug.
    data_manager_->ShowLocalMapFramesAndFeatures();
    should_quit_ = true;
    data_manager_->ShowLocalMapInWorldFrame(30, true);

    return true;
}

}
