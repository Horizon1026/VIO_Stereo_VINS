#include "data_manager.h"

namespace VIO {

void DataManager::Clear() {
    if (visual_local_map_ != nullptr) {
        visual_local_map_->Clear();
    }
    imu_based_frames_.clear();
    camera_extrinsics_.clear();
}

// Transform packed measurements to a new frame.
bool DataManager::ProcessMeasure(std::unique_ptr<PackedMeasurement> &new_packed_measure,
                                 std::unique_ptr<FrontendOutputData> &new_visual_measure) {
    if (new_packed_measure == nullptr || new_visual_measure == nullptr) {
        ReportError("[DataManager] Input new_packed_measure or new_visual_measure is nullptr.");
        return false;
    }

    imu_based_frames_.emplace_back(ImuBasedFrame{});
    ImuBasedFrame &imu_based_frame = imu_based_frames_.back();
    imu_based_frame.time_stamp_s = new_packed_measure->left_image->time_stamp_s;
    imu_based_frame.packed_measure = std::move(new_packed_measure);
    imu_based_frame.visual_measure = std::move(new_visual_measure);

    return true;
}

bool DataManager::ConvertAllImuBasedFramesToLocalMap() {
    RETURN_FALSE_IF(visual_local_map_ == nullptr);
    visual_local_map_->Clear();

    int32_t frame_id = 1;
    for (const auto &frame : imu_based_frames_) {
        if (frame.visual_measure == nullptr) {
            ReportError("[DataManager] Cannot find visual measurement in imu_based_frames_.");
            return false;
        }

        std::vector<MatImg> raw_images;
        if (frame.packed_measure != nullptr) {
            if (frame.packed_measure->left_image != nullptr) {
                raw_images.emplace_back(frame.packed_measure->left_image->image);
            }
            if (frame.packed_measure->right_image != nullptr) {
                raw_images.emplace_back(frame.packed_measure->right_image->image);
            }
        }

        visual_local_map_->AddNewFrameWithFeatures(frame.visual_measure->features_id,
                                                   frame.visual_measure->observes_per_frame,
                                                   frame.time_stamp_s,
                                                   frame_id, raw_images);
        ++frame_id;
    }

    if (!visual_local_map_->SelfCheck()) {
        ReportError("[DataManager] Visual local map self check failed in ConvertAllImuBasedFramesToLocalMap().");
        return false;
    }

    return true;
}

// Compute imu accel variance.
float DataManager::ComputeImuAccelVariance() {
    if (imu_based_frames_.empty()) {
        return 0.0f;
    }

    // Compute mean accel vector.
    Vec3 mean_accel = Vec3::Zero();
    int32_t sample_cnt = 0;
    for (const auto &frame : imu_based_frames_) {
        CONTINUE_IF(frame.packed_measure == nullptr);
        CONTINUE_IF(frame.packed_measure->imus.empty());
        for (const auto &imu : frame.packed_measure->imus) {
            mean_accel += imu->accel;
            ++sample_cnt;
        }
    }
    if (sample_cnt == 0) {
        ReportError("[DataManager] Frames with bias have no imu measurements.");
        return 0.0f;
    }
    mean_accel /= static_cast<float>(sample_cnt);

    // Compute accel variance.
    float variance = 0.0f;
    for (const auto &frame : imu_based_frames_) {
        CONTINUE_IF(frame.packed_measure == nullptr);
        CONTINUE_IF(frame.packed_measure->imus.empty());
        for (const auto &imu : frame.packed_measure->imus) {
            const Vec3 diff = mean_accel - imu->accel;
            variance += diff.squaredNorm();
        }
    }
    variance /= static_cast<float>(sample_cnt);

    return variance;
}

bool DataManager::SyncTwcToTwiInLocalMap() {
    if (camera_extrinsics_.empty()) {
        ReportError("[DataManager] DataManager failed to sync states bases on imu frame.");
        return false;
    }

    // Extract camera extrinsics.
    const Quat &q_ic = camera_extrinsics_.front().q_ic;
    const Vec3 &p_ic = camera_extrinsics_.front().p_ic;

    // T_wi = T_wc * T_ic.inv.
    auto it = imu_based_frames_.begin();
    for (const auto &frame : visual_local_map_->frames()) {
        RETURN_FALSE_IF(it == imu_based_frames_.end());
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(), p_ic, q_ic, it->p_wi, it->q_wi);
        ++it;
    }

    return true;
}

bool DataManager::SyncTwiToTwcInLocalMap() {
    if (camera_extrinsics_.empty()) {
        ReportError("[DataManager] DataManager failed to sync states bases on camera frame.");
        return false;
    }

    // Extract camera extrinsics.
    const Quat &q_ic = camera_extrinsics_.front().q_ic;
    const Vec3 &p_ic = camera_extrinsics_.front().p_ic;

    // T_wc = T_wi * T_ic
    auto it = imu_based_frames_.cbegin();
    for (auto &frame : visual_local_map_->frames()) {
        RETURN_FALSE_IF(it == imu_based_frames_.cend());
        Utility::ComputeTransformTransform(it->p_wi, it->q_wi, p_ic, q_ic, frame.p_wc(), frame.q_wc());
        ++it;
    }

    return true;
}

// Compute correspondence between two frames.
FramesCorresbondence DataManager::GetCorresbondence(const int32_t frame_id_i, const int32_t frame_id_j) {
    FramesCorresbondence corres;

    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    if (!visual_local_map_->GetCovisibleFeatures(frame_id_i, frame_id_j, covisible_features)) {
        ReportError("[DataManager] Failed to get covisible features between frame " << frame_id_i << " and " << frame_id_j << ".");
        return corres;
    }

    // Compute average parallax.
    for (const auto &feature_ptr : covisible_features) {
        const auto observe_i = feature_ptr->observe(frame_id_i).front().raw_pixel_uv;
        const auto observe_j = feature_ptr->observe(frame_id_j).front().raw_pixel_uv;
        corres.average_parallax += (observe_i - observe_j).norm();
    }
    corres.average_parallax /= static_cast<float>(covisible_features.size());
    corres.num_of_covisible_features = static_cast<float>(covisible_features.size());

    return corres;
}

}
