#include "data_loader.h"
#include "slam_log_reporter.h"
#include "memory"

namespace VIO {

namespace {
    constexpr uint32_t kDataLoaderLogIndex = 0;
    constexpr uint32_t kImuRawDataLogIndex = 1;
    constexpr uint32_t kLeftImageRawDataLogIndex = 2;
    constexpr uint32_t kRightImageRawDataLogIndex = 3;
}

bool DataLoader::Configuration(const std::string &log_file_name) {
    // Register packages for log file.
    if (options_.kEnableRecordBinaryCurveLog) {
        if (!logger_.CreateLogFile(log_file_name)) {
            ReportError("[DataLoader] Failed to create log file.");
            options_.kEnableRecordBinaryCurveLog = false;
            return false;
        }

        RegisterLogPackages();
        logger_.PrepareForRecording();
    }

    return true;
}

void DataLoader::RegisterLogPackages() {
    using namespace SLAM_DATA_LOG;

    std::unique_ptr<PackageInfo> package_ptr = std::make_unique<PackageInfo>();
    package_ptr->id = kDataLoaderLogIndex;
    package_ptr->name = "data_loader";
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_imu_in_package"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_left_image_valid_in_package"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_right_image_valid_in_package"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_imu_in_buffer"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_left_image_in_buffer"});
    package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_right_image_in_buffer"});
    if (!logger_.RegisterPackage(package_ptr)) {
        ReportError("[DataLoader] Failed to register package for data loader log.");
    }

    if (options_.kEnableRecordRawData) {
        std::unique_ptr<PackageInfo> imu_package_ptr = std::make_unique<PackageInfo>();
        imu_package_ptr->id = kImuRawDataLogIndex;
        imu_package_ptr->name = "imu_raw_data";
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "time_stamp_s"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "accel_x_ms2"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "accel_y_ms2"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "accel_z_ms2"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "gyro_x_rads"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "gyro_y_rads"});
        imu_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "gyro_z_rads"});
        if (!logger_.RegisterPackage(imu_package_ptr)) {
            ReportError("[DataLoader] Failed to register package for imu raw data.");
        }

        std::unique_ptr<PackageInfo> left_image_package_ptr = std::make_unique<PackageInfo>();
        left_image_package_ptr->id = kLeftImageRawDataLogIndex;
        left_image_package_ptr->name = "raw_left_image";
        left_image_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kImage, .name = "left image"});
        if (!logger_.RegisterPackage(left_image_package_ptr)) {
            ReportError("[DataLoader] Failed to register package for left image raw data.");
        }

        std::unique_ptr<PackageInfo> right_image_package_ptr = std::make_unique<PackageInfo>();
        right_image_package_ptr->id = kRightImageRawDataLogIndex;
        right_image_package_ptr->name = "raw_right_image";
        right_image_package_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kImage, .name = "right image"});
        if (!logger_.RegisterPackage(right_image_package_ptr)) {
            ReportError("[DataLoader] Failed to register package for right image raw data.");
        }
    }
}

void DataLoader::RecordPackedMeasurementLog(const PackedMeasurement &measure) {
    // Record data loader pop message log.
    log_package_data_.num_of_imu_in_package = static_cast<uint32_t>(measure.imus.size());
    log_package_data_.is_left_image_valid_in_package = static_cast<uint8_t>(measure.left_image != nullptr);
    log_package_data_.is_right_image_valid_in_package = static_cast<uint8_t>(measure.right_image != nullptr);
    log_package_data_.num_of_imu_in_buffer = static_cast<uint32_t>(imu_buffer_.size());
    log_package_data_.num_of_left_image_in_buffer = static_cast<uint32_t>(left_image_buffer_.size());
    log_package_data_.num_of_right_image_in_buffer = static_cast<uint32_t>(right_image_buffer_.size());
    logger_.RecordPackage(kDataLoaderLogIndex, reinterpret_cast<const char *>(&log_package_data_), measure.left_image->time_stamp_s);

    // Record imu raw data log.
    for (uint32_t i = 0; i < measure.imus.size() - 1; ++i) {
        imu_raw_package_data_.time_stamp_s = measure.imus[i]->time_stamp_s;
        imu_raw_package_data_.accel_x_ms2 = measure.imus[i]->accel.x();
        imu_raw_package_data_.accel_y_ms2 = measure.imus[i]->accel.y();
        imu_raw_package_data_.accel_z_ms2 = measure.imus[i]->accel.z();
        imu_raw_package_data_.gyro_x_ms2 = measure.imus[i]->gyro.x();
        imu_raw_package_data_.gyro_y_ms2 = measure.imus[i]->gyro.y();
        imu_raw_package_data_.gyro_z_ms2 = measure.imus[i]->gyro.z();
        logger_.RecordPackage(kImuRawDataLogIndex, reinterpret_cast<const char *>(&imu_raw_package_data_), imu_raw_package_data_.time_stamp_s);
    }

    // Record image raw data log.
    if (measure.left_image != nullptr) {
        logger_.RecordPackage(kLeftImageRawDataLogIndex, GrayImage(measure.left_image->image), measure.left_image->time_stamp_s);
    }
    if (measure.right_image != nullptr) {
        logger_.RecordPackage(kRightImageRawDataLogIndex, GrayImage(measure.right_image->image), measure.right_image->time_stamp_s);
    }
}

}
