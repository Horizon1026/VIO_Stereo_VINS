#ifndef _VIO_STEREO_BASALT_CONFIG_H_
#define _VIO_STEREO_BASALT_CONFIG_H_

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
    float noise_accel = std::sqrt(2.0000e-3f);
    float noise_gyro = std::sqrt(1.6968e-04f);
    float random_walk_accel = std::sqrt(3.0000e-3f);
    float random_walk_gyro = std::sqrt(1.9393e-05f);
};

struct VioOptionsOfFeatureDetector {
    int32_t min_valid_feature_distance = 25;
    int32_t grid_filter_rows = 11;
    int32_t grid_filter_cols = 11;
};

struct VioOptionsOfFeatureTracker {
    int32_t half_row_size_of_patch = 6;
    int32_t half_col_size_of_patch = 6;
    uint32_t max_iterations = 15;
};

struct VioOptionsOfFrontend {
    uint32_t image_rows = 0;
    uint32_t image_cols = 0;
    bool enable_drawing_track_result = false;
    bool select_keyframe = true;
    uint32_t max_feature_number = 121;
    uint32_t min_feature_number = 60;
    VioOptionsOfFeatureDetector feature_detector;
    VioOptionsOfFeatureTracker feature_tracker;
    bool enable_recording_curve_binlog = true;
    bool enable_recording_image_binlog = false;
    std::string log_file_name = "frontend.binlog";
};

struct VioOptionsOfBackend {
    /* Method index explaination: */
    // Method 1: Method in Vins-Mono.
    // Method 2: Robust vio initialization - Heyijia.
    // Method 3: Visual rotation directly estimate gyro bias.
    uint32_t method_index_to_estimate_gyro_bias_for_initialization = 3;

    Vec3 gravity_w = Vec3(0.0f, 0.0f, 9.8f);
    float max_valid_feature_depth_in_meter = 120.0f;
    float min_valid_feature_depth_in_meter = 0.05f;
    float default_feature_depth_in_meter = 1.0f;

    bool enable_report_all_information = false;
    bool enable_local_map_store_raw_images = false;

    bool enable_recording_curve_binlog = true;
    std::string log_file_name = "backend.binlog";
};

struct VioOptionsOfDataLoader {
    uint32_t max_size_of_imu_buffer = 200;
    uint32_t max_size_of_image_buffer = 20;
    bool enable_recording_curve_binlog = true;
    std::string log_file_name = "data_loader.binlog";
    bool enable_recording_raw_data_binlog = true;
};

struct VioOptionsOfDataManager {
    uint32_t max_num_of_stored_key_frames = 8;
    uint32_t max_num_of_stored_new_frames = 3;
    bool enable_recording_curve_binlog = true;
    std::string log_file_name = "data_manager.binlog";
    std::vector<Mat3> all_R_ic = {};
    std::vector<Vec3> all_t_ic = {};
};

/* Options for vio. */
struct VioOptions {
    std::string log_file_root_name = "../output/";
    float max_tolerence_time_s_for_no_data = 2.0f;
    float heart_beat_period_time_s = 1.0f;
    std::vector<VioOptionsOfCamera> cameras;
    VioOptionsOfImu imu;
    VioOptionsOfFrontend frontend;
    VioOptionsOfBackend backend;
    VioOptionsOfDataLoader data_loader;
    VioOptionsOfDataManager data_manager;
};

}

#endif // end of _VIO_STEREO_BASALT_CONFIG_H_
