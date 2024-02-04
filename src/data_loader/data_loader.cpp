#include "data_loader.h"
#include "log_report.h"
#include "memory"

namespace VIO {

namespace {
    constexpr uint32_t kDataLoaderLogIndex = 0;
    constexpr uint32_t kImuRawDataLogIndex = 1;
    constexpr uint32_t kLeftImageRawDataLogIndex = 2;
    constexpr uint32_t kRightImageRawDataLogIndex = 3;
}

void DataLoader::Clear() {
    imu_buffer_.clear();
    left_image_buffer_.clear();
    right_image_buffer_.clear();
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

// Push measurements into dataloader.
bool DataLoader::PushImuMeasurement(const Vec3 &accel,
                                    const Vec3 &gyro,
                                    const float &time_stamp_s) {
    if (!imu_buffer_.empty() && imu_buffer_.back()->time_stamp_s > time_stamp_s) {
        ReportWarn("[DataLoader] Imu measurement pushed has invalid timestamp. Latest in buffer is "
            << imu_buffer_.back()->time_stamp_s << " s, but pushed is " << time_stamp_s << " s.");
        return false;
    }

    auto object_ptr = imu_pool_.Get();
    object_ptr->accel = accel;
    object_ptr->gyro = gyro;
    object_ptr->time_stamp_s = time_stamp_s;

    std::unique_lock<std::mutex> lck(imu_mutex_);
    imu_buffer_.emplace_back(std::move(object_ptr));

    return true;
}

bool DataLoader::PushImageMeasurement(uint8_t *image_ptr,
                                      const int32_t image_rows,
                                      const int32_t image_cols,
                                      const float &time_stamp_s,
                                      const bool is_left_image) {
    const auto image_buffer_ptr = is_left_image ? &left_image_buffer_ : &right_image_buffer_;
    auto &image_mutex = is_left_image ? left_image_mutex_ : right_image_mutex_;

    if (!image_buffer_ptr->empty() && image_buffer_ptr->back()->time_stamp_s > time_stamp_s) {
        ReportWarn("[DataLoader] Camera measurement pushed has invalid timestamp. Latest in buffer is "
            << image_buffer_ptr->back()->time_stamp_s << " s, but pushed is " << time_stamp_s << " s.");
        return false;
    }

    auto object_ptr = image_pool_.Get();
    object_ptr->time_stamp_s = time_stamp_s;
    object_ptr->image = Eigen::Map<MatImg>(image_ptr, image_rows, image_cols);

    std::unique_lock<std::mutex> lck(image_mutex);
    image_buffer_ptr->emplace_back(std::move(object_ptr));

    return true;
}

// Pop measurements from dataloader.
bool DataLoader::PopSingleMeasurement(SingleMeasurement &measure) {
    std::unique_lock<std::mutex> lck1(imu_mutex_);
    std::unique_lock<std::mutex> lck2(left_image_mutex_);
    std::unique_lock<std::mutex> lck3(right_image_mutex_);

    const bool imu_buffer_empty = imu_buffer_.empty();
    const bool left_buffer_empty = left_image_buffer_.empty();
    const bool right_buffer_empty = right_image_buffer_.empty();

    // If all buffers are empty, nothing can be popped.
    if (imu_buffer_empty && left_buffer_empty && right_buffer_empty) {
        return false;
    }

    measure.imu = nullptr;
    measure.left_image = nullptr;
    measure.right_image = nullptr;

    if (imu_buffer_empty) {
        // If imu buffer is empty, consider about stereo image.
        if (left_buffer_empty) {
            measure.right_image = std::move(right_image_buffer_.front());
            right_image_buffer_.pop_front();
        } else if (right_buffer_empty) {
            measure.left_image = std::move(left_image_buffer_.front());
            left_image_buffer_.pop_front();
        } else {
            // Pop both left and right image, only if their timestamp is nearby.
            const float oldest_left_timestamp_s = left_image_buffer_.front()->time_stamp_s;
            const float oldest_right_timestamp_s = right_image_buffer_.front()->time_stamp_s;
            if (std::fabs(oldest_left_timestamp_s - oldest_right_timestamp_s) < options_.kMaxToleranceTimeDifferenceOfStereoImageInSeconds) {
                measure.left_image = std::move(left_image_buffer_.front());
                measure.right_image = std::move(right_image_buffer_.front());
                left_image_buffer_.pop_front();
                right_image_buffer_.pop_front();
            } else {
                // Pop the oldest image.
                if (oldest_left_timestamp_s < oldest_right_timestamp_s) {
                    measure.left_image = std::move(left_image_buffer_.front());
                    left_image_buffer_.pop_front();
                } else {
                    measure.right_image = std::move(right_image_buffer_.front());
                    right_image_buffer_.pop_front();
                }
            }
        }
    } else {
        // If imu buffer is not empty, consider of all.
        const float oldest_imu_timestamp_s = imu_buffer_.front()->time_stamp_s;

        if (left_buffer_empty && right_buffer_empty) {
            // Only imu buffer is not empty.
            measure.imu = std::move(imu_buffer_.front());
            imu_buffer_.pop_front();
        } else if (left_buffer_empty) {
            // Imu buffer and right image buffer are not empty.
            const float oldest_right_timestamp_s = right_image_buffer_.front()->time_stamp_s;
            if (oldest_imu_timestamp_s < oldest_right_timestamp_s) {
                measure.imu = std::move(imu_buffer_.front());
                imu_buffer_.pop_front();
            } else {
                measure.right_image = std::move(right_image_buffer_.front());
                right_image_buffer_.pop_front();
            }
        } else if (right_buffer_empty) {
            // Imu buffer and left image buffer are not empty.
            const float oldest_left_timestamp_s = left_image_buffer_.front()->time_stamp_s;
            if (oldest_imu_timestamp_s < oldest_left_timestamp_s) {
                measure.imu = std::move(imu_buffer_.front());
                imu_buffer_.pop_front();
            } else {
                measure.left_image = std::move(left_image_buffer_.front());
                left_image_buffer_.pop_front();
            }
        } else {
            // Imu buffer and left/right image buffer are all not empty.
            const float oldest_left_timestamp_s = left_image_buffer_.front()->time_stamp_s;
            const float oldest_right_timestamp_s = right_image_buffer_.front()->time_stamp_s;

            if (oldest_imu_timestamp_s <= oldest_left_timestamp_s || oldest_imu_timestamp_s <= oldest_right_timestamp_s) {
                // Imu data will be popped priorly.
                measure.imu = std::move(imu_buffer_.front());
                imu_buffer_.pop_front();
            } else {
                // Pop both left and right image, only if their timestamp is nearby.
                if (std::fabs(oldest_left_timestamp_s - oldest_right_timestamp_s) < options_.kMaxToleranceTimeDifferenceOfStereoImageInSeconds) {
                    measure.left_image = std::move(left_image_buffer_.front());
                    measure.right_image = std::move(right_image_buffer_.front());
                    left_image_buffer_.pop_front();
                    right_image_buffer_.pop_front();
                } else {
                    // Pop the oldest image.
                    if (oldest_left_timestamp_s < oldest_right_timestamp_s) {
                        measure.left_image = std::move(left_image_buffer_.front());
                        left_image_buffer_.pop_front();
                    } else {
                        measure.right_image = std::move(right_image_buffer_.front());
                        right_image_buffer_.pop_front();
                    }
                }
            }
        }
    }

    return true;
}

bool DataLoader::PopPackedMeasurement(PackedMeasurement &measure) {
    std::unique_lock<std::mutex> lck1(imu_mutex_);
    std::unique_lock<std::mutex> lck2(left_image_mutex_);
    std::unique_lock<std::mutex> lck3(right_image_mutex_);

    const bool imu_buffer_empty = imu_buffer_.empty();
    const bool left_buffer_empty = left_image_buffer_.empty();
    const bool right_buffer_empty = right_image_buffer_.empty();

    // If imu buffer or left image buffer is empty, nothing can be popped.
    if (imu_buffer_empty || left_buffer_empty) {
        return false;
    }

    // Imu data needs waiting.
    if (imu_buffer_.back()->time_stamp_s <= left_image_buffer_.front()->time_stamp_s) {
        return false;
    }
    if (!right_buffer_empty && imu_buffer_.back()->time_stamp_s <= right_image_buffer_.front()->time_stamp_s) {
        return false;
    }

    // Useless image data need to be discarded.
    if (imu_buffer_.front()->time_stamp_s >= left_image_buffer_.front()->time_stamp_s) {
        left_image_buffer_.pop_front();
        return false;
    }
    if (!right_buffer_empty && imu_buffer_.front()->time_stamp_s >= right_image_buffer_.front()->time_stamp_s) {
        right_image_buffer_.pop_front();
        return false;
    }

    // Clear measurement.
    measure.imus.clear();
    measure.left_image = nullptr;
    measure.right_image = nullptr;

    // Pack mono image or stereo image.
    measure.left_image = std::move(left_image_buffer_.front());
    left_image_buffer_.pop_front();
    if (!right_buffer_empty) {
        const float oldest_left_timestamp_s = measure.left_image->time_stamp_s;
        const float oldest_right_timestamp_s = right_image_buffer_.front()->time_stamp_s;
        if (std::fabs(oldest_left_timestamp_s - oldest_right_timestamp_s) < options_.kMaxToleranceTimeDifferenceOfStereoImageInSeconds) {
            measure.right_image = std::move(right_image_buffer_.front());
            right_image_buffer_.pop_front();
        } else if (oldest_right_timestamp_s < oldest_left_timestamp_s) {
            right_image_buffer_.pop_front();
        }
    }

    // Pack sequence of imu data.
    while (imu_buffer_.front()->time_stamp_s <= measure.left_image->time_stamp_s) {
        measure.imus.emplace_back(std::move(imu_buffer_.front()));
        imu_buffer_.pop_front();
    }

    if (std::fabs(measure.imus.back()->time_stamp_s - measure.left_image->time_stamp_s) > options_.kMaxToleranceTimeDifferenceBetweenImuAndImageInSeconds) {
        // Linear interpolation for imu at the timestamp of left image.
        auto mid = imu_pool_.Get();
        auto prev = measure.imus.back().get();
        auto next = imu_buffer_.front().get();
        const float scale = (mid->time_stamp_s - prev->time_stamp_s) / (next->time_stamp_s - prev->time_stamp_s);
        mid->time_stamp_s = measure.left_image->time_stamp_s;
        mid->gyro = prev->gyro * (1 - scale) + next->gyro * scale;
        mid->accel = prev->accel * (1 - scale) + next->accel * scale;

        auto new_item = imu_pool_.Get();
        new_item->time_stamp_s = mid->time_stamp_s;
        new_item->gyro = mid->gyro;
        new_item->accel = mid->accel;

        measure.imus.emplace_back(std::move(mid));
        imu_buffer_.emplace_front(std::move(new_item));
    } else {
        auto new_item = imu_pool_.Get();
        new_item->time_stamp_s = measure.imus.back()->time_stamp_s;
        new_item->gyro = measure.imus.back()->gyro;
        new_item->accel = measure.imus.back()->accel;
        imu_buffer_.emplace_front(std::move(new_item));
    }

    // Record log.
    if (options().kEnableRecordBinaryCurveLog) {
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

    return true;
}

}
