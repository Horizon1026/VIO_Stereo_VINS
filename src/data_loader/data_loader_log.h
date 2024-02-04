#ifndef _VIO_STEREO_BASALT_DATA_LOADER_LOG_H_
#define _VIO_STEREO_BASALT_DATA_LOADER_LOG_H_

#include "datatype_basic.h"

namespace VIO {

/* Packages of log to be recorded. */
#pragma pack(1)
struct DataLoaderLog {
    uint32_t num_of_imu_in_package = 0;
    uint8_t is_left_image_valid_in_package = 0;
    uint8_t is_right_image_valid_in_package = 0;
    uint32_t num_of_imu_in_buffer = 0;
    uint32_t num_of_left_image_in_buffer = 0;
    uint32_t num_of_right_image_in_buffer = 0;
};
struct ImuRawDataLog {
    float time_stamp_s = 0.0f;
    float accel_x_ms2 = 0.0f;
    float accel_y_ms2 = 0.0f;
    float accel_z_ms2 = 0.0f;
    float gyro_x_ms2 = 0.0f;
    float gyro_y_ms2 = 0.0f;
    float gyro_z_ms2 = 0.0f;
};
#pragma pack()

}

#endif // end of _VIO_STEREO_BASALT_DATA_LOADER_LOG_H_
