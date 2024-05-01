#ifndef _VIO_STEREO_VINS_CONFIG_H_
#define _VIO_STEREO_VINS_CONFIG_H_

#include "datatype_basic.h"
#include "string"

namespace VIO {

struct VioOptionsOfCamera {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    float k1 = 0.0f;
    float k2 = 0.0f;
    float k3 = 0.0f;
    float p1 = 0.0f;
    float p2 = 0.0f;
};

struct VioOptionsOfImu {
    float noise_accel = 0.0f;
    float noise_gyro = 0.0f;
    float random_walk_accel = 0.0f;
    float random_walk_gyro = 0.0f;
};

struct VioOptionsOfFeatureDetector {
    int32_t min_valid_feature_distance = 0;
    int32_t grid_filter_rows = 0;
    int32_t grid_filter_cols = 0;
};

struct VioOptionsOfFeatureTracker {
    int32_t half_row_size_of_patch = 0;
    int32_t half_col_size_of_patch = 0;
    uint32_t max_iterations = 0;
};

struct VioOptionsOfFrontend {
    uint32_t image_rows = 0;
    uint32_t image_cols = 0;

    uint32_t max_feature_number = 0;
    uint32_t min_feature_number = 0;

    VioOptionsOfFeatureDetector feature_detector;
    VioOptionsOfFeatureTracker feature_tracker;

    bool select_keyframe = false;

    bool enable_drawing_track_result = false;
    bool enable_recording_curve_binlog = false;
    bool enable_recording_image_binlog = false;
    std::string log_file_name;
};

struct VioOptionsOfBackend {
    Vec3 gravity_w = Vec3::Zero();
    float max_valid_feature_depth_in_meter = 0.0f;
    float min_valid_feature_depth_in_meter = 0.0f;
    float default_feature_depth_in_meter = 0.0f;

    float max_tolerence_time_for_estimation_in_second = 0.0f;

    bool enable_local_map_store_raw_images = false;
    bool enable_recording_curve_binlog = false;
    std::string log_file_name;
};

struct VioOptionsOfDataLoader {
    uint32_t max_size_of_imu_buffer = 0;
    uint32_t max_size_of_image_buffer = 0;

    bool enable_recording_curve_binlog = false;
    bool enable_recording_raw_data_binlog = false;
    std::string log_file_name;
};

struct VioOptionsOfDataManager {
    std::vector<Mat3> all_R_ic = {};
    std::vector<Vec3> all_t_ic = {};

    uint32_t max_num_of_stored_key_frames = 0;
    float max_time_s_of_imu_preintegration_block = 0.0f;
    bool enable_recording_curve_binlog = false;
    std::string log_file_name;
};

/* Options for vio. */
struct VioOptions {
    float max_tolerence_time_s_for_no_data = 0.0f;
    float heart_beat_period_time_s = 0.0f;

    std::vector<VioOptionsOfCamera> cameras;
    VioOptionsOfImu imu;
    VioOptionsOfFrontend frontend;
    VioOptionsOfBackend backend;
    VioOptionsOfDataLoader data_loader;
    VioOptionsOfDataManager data_manager;

    std::string log_file_root_name;
};

}

#endif // end of _VIO_STEREO_VINS_CONFIG_H_
