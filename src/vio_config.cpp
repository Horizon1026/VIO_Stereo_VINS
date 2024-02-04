#include "vio.h"

#include "pinhole.h"

#include "optical_flow_basic_klt.h"

#include "feature_point_detector.h"
#include "feature_fast.h"

#include "slam_operations.h"
#include "log_report.h"

namespace VIO {

bool Vio::ConfigAllComponents() {
    if (!ConfigComponentOfDataManager()) {
        ReportError(RED "[Vio] Failed to configure data manager." RESET_COLOR);
        return false;
    } else {
        ReportInfo(GREEN "[Vio] Data manager configured." RESET_COLOR);
    }

    if (!ConfigComponentOfDataLoader()) {
        ReportError(RED "[Vio] Failed to configure data loader." RESET_COLOR);
        return false;
    } else {
        ReportInfo(GREEN "[Vio] Data loader configured." RESET_COLOR);
    }

    if (!ConfigComponentOfFrontend()) {
        ReportError(RED "[Vio] Failed to configure visual frontend." RESET_COLOR);
        return false;
    } else {
        ReportInfo(GREEN "[Vio] Visual frontend configured." RESET_COLOR);
    }

    if (!ConfigComponentOfBackend()) {
        ReportError(RED "[Vio] Failed to configure backend." RESET_COLOR);
        return false;
    } else {
        ReportInfo(GREEN "[Vio] Backend configured." RESET_COLOR);
    }

    return true;
}

bool Vio::ConfigComponentOfDataManager() {
    // Config data manager.
    data_manager_ = std::make_unique<DataManager>();
    data_manager_->options().kMaxStoredKeyFrames = options_.data_manager.max_num_of_stored_key_frames;
    data_manager_->options().kMaxStoredNewFrames = options_.data_manager.max_num_of_stored_new_frames;

    data_manager_->options().kEnableRecordBinaryCurveLog = options_.data_manager.enable_recording_curve_binlog;
    RETURN_FALSE_IF_FALSE(data_manager_->Configuration(options_.log_file_root_name + options_.data_manager.log_file_name));

    // Config all camera extrinsics.
    if (options_.data_manager.all_R_ic.size() != options_.data_manager.all_t_ic.size()) {
        ReportError("[Vio] Camera extrinsics are not valid.");
        return false;
    }
    const uint32_t max_camera_num = options_.data_manager.all_R_ic.size();
    for (uint32_t i = 0; i < max_camera_num; ++i) {
        data_manager_->camera_extrinsics().emplace_back(CameraExtrinsic{
            .q_ic = Quat(options_.data_manager.all_R_ic[i]),
            .p_ic = options_.data_manager.all_t_ic[i],
        });
    }

    return true;
}

bool Vio::ConfigComponentOfDataLoader() {
    data_loader_ = std::make_unique<DataLoader>();
    data_loader_->options().kMaxToleranceTimeDifferenceOfStereoImageInSeconds = 0.005f;
    data_loader_->options().kMaxToleranceTimeDifferenceBetweenImuAndImageInSeconds = 0.001f;
    data_loader_->options().kMaxToleranceTimeDelayBetweenImuAndImageInSeconds = 1.0f;

    data_loader_->options().kEnableRecordBinaryCurveLog = options_.data_loader.enable_recording_curve_binlog;
    data_loader_->options().kEnableRecordRawData = options_.data_loader.enable_recording_raw_data_binlog;
    RETURN_FALSE_IF_FALSE(data_loader_->Configuration(options_.log_file_root_name + options_.data_loader.log_file_name));

    data_loader_->options().kMaxSizeOfImuBuffer = options_.data_loader.max_size_of_imu_buffer;
    data_loader_->options().kMaxSizeOfImageBuffer = options_.data_loader.max_size_of_image_buffer;

    return true;
}

bool Vio::ConfigComponentOfFrontend() {
    using CameraType = SENSOR_MODEL::Pinhole;
    using FeatureType = FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::FastFeature>;
    using KltType = FEATURE_TRACKER::OpticalFlowBasicKlt;

    // Config visual frontend.
    frontend_ = std::make_unique<VisualFrontend>(options_.frontend.image_rows, options_.frontend.image_cols);
    frontend_->options().kEnableRecordBinaryCurveLog = options_.frontend.enable_recording_curve_binlog;
    frontend_->options().kEnableRecordBinaryImageLog = options_.frontend.enable_recording_image_binlog;
    frontend_->options().kEnableShowVisualizeResult = options_.frontend.enable_drawing_track_result;
    frontend_->options().kSelfSelectKeyframe = options_.frontend.select_keyframe;
    frontend_->options().kMaxStoredFeaturePointsNumber = options_.frontend.max_feature_number;
    frontend_->options().kMinDetectedFeaturePointsNumberInCurrentImage = options_.frontend.min_feature_number;
    RETURN_FALSE_IF_FALSE(frontend_->Initialize(options_.log_file_root_name + options_.frontend.log_file_name));

    // Config camera model.
    frontend_->camera_models().clear();
    for (const auto &camera_options : options_.cameras) {
        frontend_->camera_models().emplace_back(std::make_unique<CameraType>());
        frontend_->camera_models().back()->SetIntrinsicParameter(
            camera_options.fx, camera_options.fy, camera_options.cx, camera_options.cy);
        frontend_->camera_models().back()->SetDistortionParameter(std::vector<float>{
            camera_options.k1, camera_options.k2, camera_options.k3, camera_options.p1, camera_options.p2});
    }

    // Config feature detector.
    frontend_->feature_detector() = std::make_unique<FeatureType>();
    frontend_->feature_detector()->options().kMinFeatureDistance = options_.frontend.feature_detector.min_valid_feature_distance;
    frontend_->feature_detector()->options().kGridFilterRowDivideNumber = options_.frontend.feature_detector.grid_filter_rows;
    frontend_->feature_detector()->options().kGridFilterColDivideNumber = options_.frontend.feature_detector.grid_filter_cols;

    // Config optical flow tracker.
    frontend_->feature_tracker() = std::make_unique<KltType>();
    frontend_->feature_tracker()->options().kMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    frontend_->feature_tracker()->options().kPatchRowHalfSize = options_.frontend.feature_tracker.half_row_size_of_patch;
    frontend_->feature_tracker()->options().kPatchColHalfSize = options_.frontend.feature_tracker.half_col_size_of_patch;
    frontend_->feature_tracker()->options().kMaxIteration = options_.frontend.feature_tracker.max_iterations;

    return true;
}

bool Vio::ConfigComponentOfBackend() {
    // Config backend.
    backend_ = std::make_unique<Backend>();
    backend_->options().kMethodIndexToEstimateGyroBiasForInitialization = options_.backend.method_index_to_estimate_gyro_bias_for_initialization;
    backend_->options().kGravityInWordFrame = options_.backend.gravity_w;
    backend_->options().kMaxValidFeatureDepthInMeter = options_.backend.max_valid_feature_depth_in_meter;
    backend_->options().kMinValidFeatureDepthInMeter = options_.backend.min_valid_feature_depth_in_meter;
    backend_->options().kDefaultFeatureDepthInMeter = options_.backend.default_feature_depth_in_meter;

    backend_->options().kEnableReportAllInformation = options_.backend.enable_report_all_information;
    backend_->options().kEnableLocalMapStoreRawImages = options_.backend.enable_local_map_store_raw_images;

    backend_->options().kEnableRecordBinaryCurveLog = options_.backend.enable_recording_curve_binlog;
    RETURN_FALSE_IF_FALSE(backend_->Configuration(options_.log_file_root_name + options_.backend.log_file_name));

    // Config imu model.
    backend_->imu_model() = std::make_unique<Imu>();
    backend_->imu_model()->options().kAccelNoise = options_.imu.noise_accel;
    backend_->imu_model()->options().kGyroNoise = options_.imu.noise_gyro;
    backend_->imu_model()->options().kAccelRandomWalk = options_.imu.random_walk_accel;
    backend_->imu_model()->options().kGyroRandomWalk = options_.imu.random_walk_gyro;

    // Register components.
    backend_->data_manager() = data_manager_.get();
    backend_->visual_frontend() = frontend_.get();

    return true;
}

}
