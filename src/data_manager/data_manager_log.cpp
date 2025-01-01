#include "data_manager.h"

namespace VIO {

namespace {
    constexpr uint32_t kDataManagerLocalMapLogIndex = 0;
    constexpr uint32_t kDataManagerPointCloudLogIndex = 1;
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
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_frames"});
    if (!logger_.RegisterPackage(package_ptr)) {
        ReportError("[DataManager] Failed to register package for data manager local map log.");
    }

    std::unique_ptr<PackageInfo> package_point_cloud_ptr = std::make_unique<PackageInfo>();
    package_point_cloud_ptr->id = kDataManagerPointCloudLogIndex;
    package_point_cloud_ptr->name = "point cloud";
    package_point_cloud_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kPointCloud, .name = "solved points"});
    if (!logger_.RegisterPackage(package_point_cloud_ptr)) {
        ReportError("[DataManager] Failed to register package for data manager local map log.");
    }
}

void DataManager::TriggerLogRecording(const float time_stamp_s) {
    RETURN_IF(!options_.kEnableRecordBinaryCurveLog);
    RETURN_IF(visual_local_map_ == nullptr);

    RecordLocalMap(time_stamp_s);
    RecordPointCloud(time_stamp_s);
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

    log_package.num_of_frames = visual_local_map_->frames().size();
    logger_.RecordPackage(kDataManagerLocalMapLogIndex, reinterpret_cast<const char *>(&log_package), time_stamp_s);
}

void DataManager::RecordPointCloud(const float time_stamp_s) {
    RETURN_IF(visual_local_map_->frames().empty());
    RETURN_IF(latest_record_point_cloud_time_s_ == time_stamp_s);

    std::vector<Vec3> points;
    points.reserve(visual_local_map_->features().size());
    for (const auto &pair : visual_local_map_->features()) {
        const auto &feature = pair.second;
        CONTINUE_IF(feature.first_frame_id() != visual_local_map_->frames().front().id());
        switch (feature.status()) {
            case FeatureSolvedStatus::kSolved: {
                points.emplace_back(feature.param());
                break;
            }
            default: break;
        }
    }

    logger_.RecordPackage(kDataManagerPointCloudLogIndex, points, time_stamp_s);
    latest_record_point_cloud_time_s_ = time_stamp_s;
}

}
