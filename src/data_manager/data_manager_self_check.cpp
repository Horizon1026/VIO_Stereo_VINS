#include "data_manager.h"

namespace VIO {

// Self check.
bool DataManager::SelfCheckVisualLocalMap() {
    if (!visual_local_map_->SelfCheck()) {
        ReportError("[DataManager] Visual local map self check error.");
        return false;
    }

    // Iterate each frame to check all features.
    for (const auto &frame : visual_local_map_->frames()) {
        const auto frame_id = frame.id();
        for (const auto &pair : frame.features()) {
            const auto feature_id = pair.first;
            const auto feature_ptr = pair.second;
            if (feature_id != feature_ptr->id()) {
                ReportError("[DataManager] Visual local map self check frames, feature id error [" <<
                    feature_id << "] != [" << feature_ptr->id() << "].");
                return false;
            }

            if (feature_ptr->observe(frame_id).front().frame_time_stamp_s != frame.time_stamp_s()) {
                ReportError("[DataManager] Visual local map self check frames, feature observe timestamp error [" <<
                    feature_ptr->observe(frame_id).front().frame_time_stamp_s << "] != [" << frame.time_stamp_s() << "].");
                return false;
            }
        }
    }

    // Iterate each feature to check all observations.
    for (const auto &pair : visual_local_map_->features()) {
        const auto feature_id = pair.first;
        const auto &feature = pair.second;
        if (feature_id != feature.id()) {
            ReportError("[DataManager] Visual local map self check features, feature id error [" <<
                feature_id << "] != [" << feature.id() << "].");
            return false;
        }


        for (auto frame_id = feature.first_frame_id(); frame_id <= feature.final_frame_id(); ++frame_id) {
            const auto time_stamp_from_feature = feature.observe(frame_id).front().frame_time_stamp_s;
            const auto time_stamp_from_frame = visual_local_map_->frame(frame_id)->time_stamp_s();
            if (time_stamp_from_feature != time_stamp_from_frame) {
                ReportError("[DataManager] Visual local map self check features, feature observe timestamp error [" <<
                    time_stamp_from_feature << "] != [" << time_stamp_from_frame << "].");
                return false;
            }
        }
    }

    ReportInfo("[DataManager] Visual local map self check ok.");
    return true;
}

bool DataManager::SelfCheckImuBasedFrames() {
    // Iterate each imu based frame.
    float prev_frame_time_stamp_s = 0.0f;
    for (const auto &imu_based_frame : imu_based_frames_) {
        // Check timestamp of visual observations.
        for (const auto &observes : imu_based_frame.visual_measure->observes_per_frame) {
            const auto time_stamp_0 = imu_based_frame.time_stamp_s;
            for (const auto &observe : observes) {
                const auto time_stamp_1 = observe.frame_time_stamp_s;
                if (time_stamp_0 != time_stamp_1) {
                    ReportError("[DataManager] Frames with bias self check frontend output data, feature observe timestamp error [" <<
                        time_stamp_0 << "] != [" << time_stamp_1 << "].");
                    return false;
                }
            }
        }

        // Check timestamp of imu and images.
        const auto latest_imu_time_stamp_s = imu_based_frame.packed_measure->imus.back()->time_stamp_s;
        if (imu_based_frame.packed_measure->left_image != nullptr) {
            const auto left_image_time_stamp_s = imu_based_frame.packed_measure->left_image->time_stamp_s;
            if (latest_imu_time_stamp_s != left_image_time_stamp_s) {
                ReportError("[DataManager] Frames with bias self check imu and left image, feature observe timestamp error [" <<
                    latest_imu_time_stamp_s << "] != [" << left_image_time_stamp_s << "].");
                return false;
            }
        }
        if (imu_based_frame.packed_measure->right_image != nullptr) {
            const auto left_image_time_stamp_s = imu_based_frame.packed_measure->right_image->time_stamp_s;
            if (latest_imu_time_stamp_s != left_image_time_stamp_s) {
                ReportError("[DataManager] Frames with bias self check imu and right image, feature observe timestamp error [" <<
                    latest_imu_time_stamp_s << "] != [" << left_image_time_stamp_s << "].");
                return false;
            }
        }

        // Check imu preintegration time between frames.
        if (imu_based_frame.time_stamp_s == imu_based_frames_.front().time_stamp_s) {
            prev_frame_time_stamp_s = imu_based_frame.time_stamp_s;
            continue;
        }
        const float imu_dt = imu_based_frame.imu_preint_block.integrate_time_s();
        const float cam_dt = imu_based_frame.time_stamp_s - prev_frame_time_stamp_s;
        prev_frame_time_stamp_s = imu_based_frame.time_stamp_s;
        if (std::fabs(imu_dt - cam_dt) > 0.1f) {
            ReportError("[DataManager] Frames with bias self check imu preintegration time, integrate_time_s error [" <<
                imu_dt << "] != [" << cam_dt << "] == [" << imu_based_frame.time_stamp_s << "] - [" <<
                prev_frame_time_stamp_s << "].");
            return false;
        }
    }

    ReportInfo("[DataManager] Frames with bias self check ok.");
    return true;
}

}
