#include "backend.h"
#include "slam_log_reporter.h"

namespace VIO {

namespace {
    const float kMinValidImuAccelVarianceForMonoInitialization = 0.4f;
}

bool Backend::TryToInitialize() {
    if (data_manager_->imu_based_frames().size() < data_manager_->options().kMaxStoredKeyFrames) {
        ReportWarn("[Backend] Backend cannot initialize for lack of frames.");
        return false;
    }

    // If this is mono camera, some movement is neccessary for initialization.
    const float imu_accel_variance = data_manager_->ComputeImuAccelVariance();
    if (imu_accel_variance < kMinValidImuAccelVarianceForMonoInitialization && data_manager_->camera_extrinsics().size() < 2) {
        ReportWarn("[Backend] Backend cannot initialize for lack of imu motion in mono-view.");
        return false;
    }

    // Convert all frames into a covisible graph.
    if (!data_manager_->ConvertAllImuBasedFramesToLocalMap()) {
        ReportError("[Backend] Backend failed to convert frames to covisible graph.");
        return false;
    }

    // Compute initialized value of visual local map.
    if (!PrepareForPureVisualSfmByMonoView()) {
        ReportWarn("[Backend] Backend failted to prepare for pure visual SFM in mono-view, try to use multi-view.");
        if (!PrepareForPureVisualSfmByMultiView()) {
            ReportError("[Backend] Backend failed to prepare for pure visual SFM.");
            return false;
        }
    }

    // Perform pure visual bundle adjustment.
    if (!PerformPureVisualBundleAdjustment(false)) {
        ReportError("[Backend] Backend failed to perform pure visual bundle adjustment.");
        return false;
    }

    // Sync motion states.
    if (!data_manager_->SyncTwcToTwiInLocalMap()) {
        ReportError("[Backend] Backend failed to sync motion states.");
        return false;
    }

    // Estimate bias of gyro by visual frame poses.
    if (!EstimateGyroBias()) {
        ReportError("[Backend] Backend failed to estimate bias of gyro.");
        return false;
    }

    // Estimate velocity of each frame, with gravity vector and scale factor.
    Vec3 gravity_c0 = Vec3::Zero();
    Vec all_v_ii = Vec3::Zero();
    float scale = 0.0f;
    if (!EstimateVelocityGravityScaleIn3Dof(gravity_c0, scale)) {
        ReportError("[Backend] Backend failed to estimate velocity, gravity and scale in 3-dof.");
        return false;
    }
    if (!EstimateVelocityGravityScaleIn2Dof(gravity_c0, all_v_ii)) {
        ReportError("[Backend] Backend failed to estimate velocity, gravity and scale in 2-dof.");
        return false;
    }

    // Sync all states in visual_local_map and imu_based_frames.
    if (!SyncInitializedResult(gravity_c0, all_v_ii, scale)) {
        ReportError("[Backend] Backend failed to sync states after initialization.");
        return false;
    }

    return true;
}

bool Backend::SyncInitializedResult(const Vec3 &gravity_c0, const Vec &all_v_ii, const float &scale) {
    const Vec3 &g_c0 = gravity_c0;
    const Vec3 &g_w = options_.kGravityInWordFrame;

    // Compute rotation matrix from first camera frame to world frame.
    const float norm = (g_c0.cross(g_w)).norm();
    const Vec3 vec = g_c0.cross(g_w) / norm;
    const float theta = std::atan2(norm, g_c0.dot(g_w));
    const Vec3 axis_angle = vec * theta;
    const Quat q_wc0 = Utility::Exponent(axis_angle);

    // Recovery all camera states in visual_local_map.
    for (auto &cam_frame : data_manager_->visual_local_map()->frames()) {
        const Quat q_c0c = cam_frame.q_wc();
        const Vec3 p_c0c = cam_frame.p_wc();
        cam_frame.q_wc() = q_wc0 * q_c0c;
        cam_frame.p_wc() = q_wc0 * p_c0c * scale;
    }

    // Recovery all feature states in visual_local_map.
    for (auto &pair : data_manager_->visual_local_map()->features()) {
        auto &feature = pair.second;
        CONTINUE_IF(feature.status() != FeatureSolvedStatus::kSolved);
        feature.param() = q_wc0 * feature.param() * scale;
    }

    // Recovery all imu states in imu_based_frames.
    RETURN_FALSE_IF(!data_manager_->SyncTwcToTwiInLocalMap());
    uint32_t index = 0;
    for (auto &imu_frame : data_manager_->imu_based_frames()) {
        imu_frame.v_wi = imu_frame.q_wi * all_v_ii.segment(index * 3, 3);
        ++index;
    }

    return true;
}

}
