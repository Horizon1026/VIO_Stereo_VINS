#include "backend.h"
#include "log_report.h"

namespace VIO {

bool Backend::TryToInitialize() {
    if (data_manager_->frames_with_bias().size() < data_manager_->options().kMaxStoredNewFrames) {
        ReportWarn("[Backend] Backend cannot initialize for lack of new frames.");
        return false;
    }

    // Convert the new frames into a covisible graph.
    if (!ConvertNewFramesToCovisibleGraphForInitialization()) {
        ReportError("[Backend] Backend failed to convert new frames to covisible graph.");
        return false;
    }

    // Estiamte bias_g (one-for-all), q_wc (define c0 frame as w frame) of each frame.
    if (!EstimateGyroBiasAndRotationForInitialization()) {
        ReportError("[Backend] Backend failed to estimate gyro bias.");
        return false;
    }

    // Estimate velocity of each frame, and gravity vector based on frame c0(camera).
    Vec3 gravity_i0 = Vec3::Zero();
    if (!EstimateVelocityAndGravityForInitialization(gravity_i0)) {
        ReportError("[Backend] Backend failed to estimate velocity or gravity.");
        return false;
    }

    // Transform all states from frame c0(camera) to frame w(world). The z-axis of frame w must be upward.
    if (!TransformAllStatesToWorldFrameForInitialization(gravity_i0)) {
        ReportError("[Backend] Backend failed to transform from i0 to w frame.");
        return false;
    }

    // Try to triangulize all features of vision.
    if (!TriangulizeAllVisualFeatures()) {
        ReportError("[Backend] Backend failed to triangulize features.");
        return false;
    }

    return true;
}

bool Backend::ConvertNewFramesToCovisibleGraphForInitialization() {
    RETURN_FALSE_IF(data_manager_->visual_local_map() == nullptr);

    auto local_map_ptr = data_manager_->visual_local_map();
    local_map_ptr->Clear();
    int32_t frame_id = 1;
    for (const auto &frame : data_manager_->frames_with_bias()) {
        RETURN_FALSE_IF(frame.visual_measure == nullptr);

        std::vector<MatImg> raw_images;
        if (options_.kEnableLocalMapStoreRawImages && frame.packed_measure != nullptr) {
            if (frame.packed_measure->left_image != nullptr) {
                raw_images.emplace_back(frame.packed_measure->left_image->image);
            }
            if (frame.packed_measure->right_image != nullptr) {
                raw_images.emplace_back(frame.packed_measure->right_image->image);
            }
        }

        local_map_ptr->AddNewFrameWithFeatures(frame.visual_measure->features_id,
                                               frame.visual_measure->observes_per_frame,
                                               frame.time_stamp_s,
                                               frame_id,
                                               raw_images);
        ++frame_id;
    }
    RETURN_FALSE_IF(!local_map_ptr->SelfCheck());

    return true;
}

bool Backend::TransformAllStatesToWorldFrameForInitialization(const Vec3 &gravity_i0) {
    // Compute the gravity vector based on frame c0.
    const Quat q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Vec3 gravity_c0 = q_ic.inverse() * gravity_i0;

    // Compute the rotation from c0 to w.
    const Vec3 gravity_w = options_.kGravityInWordFrame;
    const Vec3 cross_vec = gravity_c0.cross(gravity_w);
    const float norm = cross_vec.norm();
    const Vec3 u = cross_vec / norm;
    const float theta = std::atan2(norm, gravity_c0.dot(gravity_w));
    const Vec3 angle_axis = u * theta;
    const Quat q_wc0 = Utility::Exponent(angle_axis);
    ReportInfo(GREEN "[Backend] Estimated q_wc0 is " << LogQuat(q_wc0) << "." << RESET_COLOR);

    // Iterate all frames, transform all states of them from i0 to w.
    // Determine the scope of all frames.
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();

    // Iterate all frames to propagate states.
    for (uint32_t i = min_frames_idx; i <= max_frames_idx; ++i) {
        auto frame = data_manager_->visual_local_map()->frame(i);
        const Quat q_c0c = frame->q_wc();
        const Vec3 p_c0c = frame->p_wc();
        const Vec3 v_c0c = frame->v_w();

        frame->q_wc() = q_wc0 * q_c0c;
        frame->p_wc() = q_wc0 * p_c0c;
        frame->v_w() = q_wc0 * v_c0c;
    }

    return true;
}

}
