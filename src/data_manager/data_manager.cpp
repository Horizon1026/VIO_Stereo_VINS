#include "data_manager.h"

namespace VIO {

namespace {
    constexpr uint32_t kDataManagerLocalMapLogIndex = 0;
    constexpr uint32_t kDataManagerCovisibleGraphLogIndex = 1;
}

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

    std::unique_ptr<PackageInfo> package_ptr = std::make_unique<PackageInfo>();
    package_ptr->id = kDataManagerLocalMapLogIndex;
    package_ptr->name = "local map";
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_solved_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_marginalized_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_unsolved_features"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_features_observed_in_newest_keyframe"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_solved_features_observed_in_newest_keyframe"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_frames"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_keyframes"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_newframes"});
    if (!logger_.RegisterPackage(package_ptr)) {
        ReportError("[DataManager] Failed to register package for data manager log.");
    }

    // Register log for each visual frame in local map.
    uint16_t package_id = kDataManagerCovisibleGraphLogIndex;
    for (uint32_t i = 0; i < options_.kMaxStoredKeyFrames; ++i) {
        std::unique_ptr<PackageInfo> package_ptr = std::make_unique<PackageInfo>();
        package_ptr->id = package_id + i;
        package_ptr->name = std::string("frame ") + std::to_string(i + 1) + std::string(" in local map");
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_observed_features"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_solved_features"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_tracked_features_from_prev_frame"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_solved_tracked_features_from_prev_frame"});

        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "time_stamp_s"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wc_x"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wc_y"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wc_z"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wc_w"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wc_x"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wc_y"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wc_z"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wc_x"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wc_y"});
        package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wc_z"});
        if (!logger_.RegisterPackage(package_ptr)) {
            ReportError("[DataManager] Failed to register package for covisible graph log.");
        }
    }
}

void DataManager::TriggerLogRecording(const float time_stamp_s) {
    RETURN_IF(!options_.kEnableRecordBinaryCurveLog);
    RETURN_IF(visual_local_map_ == nullptr);

    RecordLocalMap(time_stamp_s);
    RecordCovisibleGraph(time_stamp_s);
}

void DataManager::RecordLocalMap(const float time_stamp_s) {
    DataManagerLocalMapLog log_package;

    log_package.num_of_features = visual_local_map_->features().size();
    for (const auto &pair : visual_local_map_->features()) {
        const auto &feature = pair.second;
        switch (feature.status()) {
            case FeatureSolvedStatus::kUnsolved: {
                ++log_package.num_of_unsolved_features;
                break;
            }
            case FeatureSolvedStatus::kSolved: {
                ++log_package.num_of_solved_features;
                break;
            }
            case FeatureSolvedStatus::kMarginalized: {
                ++log_package.num_of_marginalized_features;
                break;
            }
            default: break;
        }
    }

    const auto frame_ptr = visual_local_map_->frame(GetNewestKeyframeId());
    if (frame_ptr != nullptr) {
        for (const auto &pair : frame_ptr->features()) {
            const auto &feature_ptr = pair.second;
            ++log_package.num_of_features_observed_in_newest_keyframe;
            if (feature_ptr->status() == FeatureSolvedStatus::kSolved) {
                ++log_package.num_of_solved_features_observed_in_newest_keyframe;
            }
        }
    }

    log_package.num_of_frames = visual_local_map_->frames().size();
    log_package.num_of_newframes = frames_with_bias_.size();
    log_package.num_of_keyframes = visual_local_map_->frames().empty() ? 0 :
        visual_local_map_->frames().size() - frames_with_bias_.size();

    logger_.RecordPackage(kDataManagerLocalMapLogIndex, reinterpret_cast<const char *>(&log_package), time_stamp_s);
}

void DataManager::RecordCovisibleGraph(const float time_stamp_s) {
    RETURN_IF(visual_local_map_->frames().empty());

    uint16_t package_id = kDataManagerCovisibleGraphLogIndex;
    for (const auto &frame : visual_local_map_->frames()) {
        DataManagerCovisibleGraphLog log_package;

        for (const auto &pair : frame.features()) {
            const auto &feature_ptr = pair.second;
            ++log_package.num_of_observed_features;
            if (feature_ptr->status() == FeatureSolvedStatus::kSolved) {
                ++log_package.num_of_solved_features;
            }
            if (feature_ptr->first_frame_id() < frame.id()) {
                ++log_package.num_of_tracked_features_from_prev_frame;
                if (feature_ptr->status() == FeatureSolvedStatus::kSolved) {
                    ++log_package.num_of_solved_tracked_features_from_prev_frame;
                }
            }
        }

        log_package.time_stamp_s = frame.time_stamp_s();
        log_package.p_wc_x = frame.p_wc().x();
        log_package.p_wc_y = frame.p_wc().y();
        log_package.p_wc_z = frame.p_wc().z();
        log_package.q_wc_w = frame.q_wc().w();
        log_package.q_wc_x = frame.q_wc().x();
        log_package.q_wc_y = frame.q_wc().y();
        log_package.q_wc_z = frame.q_wc().z();
        log_package.v_wc_x = frame.v_w().x();
        log_package.v_wc_y = frame.v_w().y();
        log_package.v_wc_z = frame.v_w().z();

        logger_.RecordPackage(package_id, reinterpret_cast<const char *>(&log_package), time_stamp_s);
        ++package_id;
    }
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

// Get specified frame id.
uint32_t DataManager::GetNewestKeyframeId() {
    return visual_local_map_->frames().front().id() + options_.kMaxStoredKeyFrames - options_.kMaxStoredNewFrames - 1;
}

// Get specified frame timestamp.
float DataManager::GetNewestStateTimeStamp() {
    return frames_with_bias_.empty() ? 0.0f : frames_with_bias_.back().time_stamp_s;
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
