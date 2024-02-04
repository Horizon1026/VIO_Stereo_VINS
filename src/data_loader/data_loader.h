#ifndef _VIO_STEREO_BASALT_DATA_LOADER_H_
#define _VIO_STEREO_BASALT_DATA_LOADER_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"

#include "object_pool.h"

#include "imu_measurement.h"
#include "camera_measurement.h"

#include "data_loader_log.h"
#include "binary_data_log.h"

#include "deque"
#include "mutex"

namespace VIO {

using namespace SLAM_UTILITY;
using namespace SENSOR_MODEL;

/* Measurement Definition. */
struct SingleMeasurement {
    ObjectPtr<ImuMeasurement> imu = nullptr;
    ObjectPtr<CameraMeasurement> left_image = nullptr;
    ObjectPtr<CameraMeasurement> right_image = nullptr;
};
struct PackedMeasurement {
    std::vector<ObjectPtr<ImuMeasurement>> imus;
    ObjectPtr<CameraMeasurement> left_image = nullptr;
    ObjectPtr<CameraMeasurement> right_image = nullptr;
};

/* Options for Data Loader. */
struct DataLoaderOptions {
    float kMaxToleranceTimeDifferenceOfStereoImageInSeconds = 0.005f;
    float kMaxToleranceTimeDifferenceBetweenImuAndImageInSeconds = 0.001f;
    float kMaxToleranceTimeDelayBetweenImuAndImageInSeconds = 1.0f;
    bool kEnableRecordBinaryCurveLog = true;
    bool kEnableRecordRawData = true;
    uint32_t kMaxSizeOfImuBuffer = 200;
    uint32_t kMaxSizeOfImageBuffer = 20;
};

/* Class Data Loader Declaration. */
class DataLoader final {

public:
    DataLoader() = default;
    ~DataLoader() = default;

    void Clear();
    bool Configuration(const std::string &log_file_name);
    void RegisterLogPackages();

    // Push measurements into dataloader.
    bool PushImuMeasurement(const Vec3 &accel,
                            const Vec3 &gyro,
                            const float &time_stamp_s);
    bool PushImageMeasurement(uint8_t *image_ptr,
                              const int32_t image_rows,
                              const int32_t image_cols,
                              const float &time_stamp_s,
                              const bool is_left_image = true);

    // Pop measurements from dataloader.
    bool PopSingleMeasurement(SingleMeasurement &measure);
    bool PopPackedMeasurement(PackedMeasurement &measure);

    // Sync signal for imu and image buffer.
    bool IsImuBufferFull() const { return imu_buffer_.size() >= options_.kMaxSizeOfImuBuffer; }
    bool IsImageBufferFull() const { return left_image_buffer_.size() >= options_.kMaxSizeOfImageBuffer || right_image_buffer_.size() >= options_.kMaxSizeOfImageBuffer; }

    // Reference for member variables.
    DataLoaderOptions &options() { return options_; }
    std::deque<ObjectPtr<ImuMeasurement>> &imu_buffer() { return imu_buffer_; }
    std::deque<ObjectPtr<CameraMeasurement>> &left_image_buffer() { return left_image_buffer_; }
    std::deque<ObjectPtr<CameraMeasurement>> &right_image_buffer() { return right_image_buffer_; }
    ObjectPool<ImuMeasurement> &imu_pool() { return imu_pool_; }
    ObjectPool<CameraMeasurement> &image_pool() { return image_pool_; }

    // Const reference for member variables.
    const DataLoaderOptions &options() const { return options_; }
    const std::deque<ObjectPtr<ImuMeasurement>> &imu_buffer() const { return imu_buffer_; }
    const std::deque<ObjectPtr<CameraMeasurement>> &left_image_buffer() const { return left_image_buffer_; }
    const std::deque<ObjectPtr<CameraMeasurement>> &right_image_buffer() const { return right_image_buffer_; }
    const ObjectPool<ImuMeasurement> &imu_pool() const { return imu_pool_; }
    const ObjectPool<CameraMeasurement> &image_pool() const { return image_pool_; }

private:
    DataLoaderOptions options_;

    std::deque<ObjectPtr<ImuMeasurement>> imu_buffer_;
    std::deque<ObjectPtr<CameraMeasurement>> left_image_buffer_;
    std::deque<ObjectPtr<CameraMeasurement>> right_image_buffer_;

    ObjectPool<ImuMeasurement> imu_pool_;
    ObjectPool<CameraMeasurement> image_pool_;

    std::mutex imu_mutex_;
    std::mutex left_image_mutex_;
    std::mutex right_image_mutex_;

    // Record log.
    SLAM_DATA_LOG::BinaryDataLog logger_;
    DataLoaderLog log_package_data_;
    ImuRawDataLog imu_raw_package_data_;

};

}

#endif // end of _VIO_STEREO_BASALT_DATA_LOADER_H_
