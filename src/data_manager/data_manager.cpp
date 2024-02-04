#include "data_manager.h"

namespace VIO {

void DataManager::Clear() {
    if (visual_local_map_ != nullptr) {
        visual_local_map_->Clear();
    }
    frames_with_bias_.clear();
    camera_extrinsics_.clear();
}

bool DataManager::Configuration(const std::string &log_file_name) {
    // Register packages for log file.
    if (options_.kEnableRecordBinaryCurveLog) {
        if (!logger_.CreateLogFile(log_file_name)) {
            ReportError("[DataManager] Failed to create log file.");
            options_.kEnableRecordBinaryCurveLog = false;
            return false;
        }

        RegisterLogPackages();
        logger_.PrepareForRecording();
    }

    return true;
}

void DataManager::RegisterLogPackages() {
    using namespace SLAM_DATA_LOG;

}

// Transform packed measurements to a new frame.
bool DataManager::ProcessMeasure(std::unique_ptr<PackedMeasurement> &new_packed_measure,
                                 std::unique_ptr<FrontendOutputData> &new_visual_measure) {
    if (new_packed_measure == nullptr || new_visual_measure == nullptr) {
        ReportError("[DataManager] Input new_packed_measure or new_visual_measure is nullptr.");
        return false;
    }

    frames_with_bias_.emplace_back(FrameWithBias{});
    FrameWithBias &frame_with_bias = frames_with_bias_.back();
    frame_with_bias.time_stamp_s = new_packed_measure->left_image->time_stamp_s;
    frame_with_bias.packed_measure = std::move(new_packed_measure);
    frame_with_bias.visual_measure = std::move(new_visual_measure);

    return true;
}

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

bool DataManager::SelfCheckFramesWithBias() {
    // Iterate each frame with bias.
    for (const auto &frame_with_bias : frames_with_bias_) {
        // Check timestamp of visual observations.
        for (const auto &observes : frame_with_bias.visual_measure->observes_per_frame) {
            const auto time_stamp_0 = frame_with_bias.time_stamp_s;
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
        const auto latest_imu_time_stamp_s = frame_with_bias.packed_measure->imus.back()->time_stamp_s;
        const auto oldest_imu_time_stamp_s = frame_with_bias.packed_measure->imus.front()->time_stamp_s;
        if (latest_imu_time_stamp_s - oldest_imu_time_stamp_s > 0.055f) {
            ReportError("[DataManager] Frames with bias self check imus, imu timestamp error [" <<
                latest_imu_time_stamp_s << "] - [" << oldest_imu_time_stamp_s << "] > 0.055f.");
            return false;
        }
        if (frame_with_bias.packed_measure->left_image != nullptr) {
            const auto left_image_time_stamp_s = frame_with_bias.packed_measure->left_image->time_stamp_s;
            if (latest_imu_time_stamp_s != left_image_time_stamp_s) {
                ReportError("[DataManager] Frames with bias self check imu and left image, feature observe timestamp error [" <<
                    latest_imu_time_stamp_s << "] != [" << left_image_time_stamp_s << "].");
                return false;
            }
        }
        if (frame_with_bias.packed_measure->right_image != nullptr) {
            const auto left_image_time_stamp_s = frame_with_bias.packed_measure->right_image->time_stamp_s;
            if (latest_imu_time_stamp_s != left_image_time_stamp_s) {
                ReportError("[DataManager] Frames with bias self check imu and right image, feature observe timestamp error [" <<
                    latest_imu_time_stamp_s << "] != [" << left_image_time_stamp_s << "].");
                return false;
            }
        }
    }

    ReportInfo("[DataManager] Frames with bias self check ok.");
    return true;
}

}
