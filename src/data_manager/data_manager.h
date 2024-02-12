#ifndef _VIO_STEREO_VINS_DATA_MANAGER_H_
#define _VIO_STEREO_VINS_DATA_MANAGER_H_

#include "datatype_basic.h"
#include "covisible_graph.h"
#include "imu.h"

#include "data_loader.h"
#include "visual_frontend.h"

#include "data_manager_log.h"
#include "binary_data_log.h"

#include "memory"
#include "deque"

namespace VIO {

using namespace SLAM_UTILITY;
using namespace SENSOR_MODEL;
using DorF = float;

/* Options for Data Manager. */
struct DataManagerOptions {
    uint32_t kMaxStoredKeyFrames = 0;
    float kMaxValidImuPreintegrationBlockTimeInSecond = 0.0f;
    bool kEnableRecordBinaryCurveLog = false;
};

/* Definition of Covisible Graph for Local Map. */
using FeatureParameter = Vec3;
using FeatureObserve = std::vector<ObservePerView>; // Use std::vector to store observations of left and right camera.
using FeatureType = VisualFeature<FeatureParameter, FeatureObserve>;
using CovisibleGraphType = CovisibleGraph<FeatureParameter, FeatureObserve>;

/* Definition of Frame and FrameWithBias. */
using FrameType = VisualFrame<FeatureType>;
struct FrameWithBias {
    // Imu bias of accel and gyro is inside imu_preint_block.
    ImuPreintegrateBlock<DorF> imu_preint_block;
    float time_stamp_s = 0.0f;
    // Measurement of raw imu(gyro, acc), raw image(left, right) and visual features.
    std::unique_ptr<PackedMeasurement> packed_measure = nullptr;
    std::unique_ptr<FrontendOutputData> visual_measure = nullptr;
    // States based on imu.
    Vec3 p_wi = Vec3::Zero();
    Quat q_wi = Quat::Identity();
    Vec3 v_wi = Vec3::Zero();
};

/* Definition of Corresbondence between Frames. */
struct FramesCorresbondence {
    int32_t num_of_covisible_features = 0;
    float average_parallax = 0.0f;
};

/* Definition of Camera Extrinsic. */
struct CameraExtrinsic {
    // Rotation and translation between imu and camera frame.
    Vec3 p_ic = Vec3::Zero();
    Quat q_ic = Quat::Identity();
};

/* Definition of Global Map Point and Keyframe. */
struct GlobalMapPoint {
    // Position based on camera.(If multi-view, only based on camera with id 0)
    Vec3 p_c = Vec3::Zero();
};
struct GlobalMapKeyframe {
    // Pose based on camera.(If multi-view, only based on camera with id 0)
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
    // Timestamp of this keyframe.
    float time_stamp_s = 0.0f;
    // Feature points based on this keyframe.
    std::vector<GlobalMapPoint> points;
};

/* Class Data Manager Declaration. */
class DataManager final {

public:
    DataManager() = default;
    ~DataManager() = default;

    void Clear();

    // Record log.
    bool Configuration(const std::string &log_file_name);
    void RegisterLogPackages();
    void TriggerLogRecording(const float time_stamp_s);
    void RecordLocalMap(const float time_stamp_s);

    // Self check.
    bool SelfCheckVisualLocalMap();
    bool SelfCheckFramesWithBias();

    // Transform packed measurements to a new frame.
    bool ProcessMeasure(std::unique_ptr<PackedMeasurement> &new_packed_measure,
                        std::unique_ptr<FrontendOutputData> &new_visual_measure);

    // Convert all frames with bias into visual local map.
    bool ConvertAllFramesWithBiasToLocalMap();

    // Compute imu accel variance.
    float ComputeImuAccelVariance();

    // Compute correspondence between two frames.
    FramesCorresbondence GetCorresbondence(const int32_t frame_id_i, const int32_t frame_id_j);

    // Visualizor of managed data.
    void ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id, const uint32_t cur_frame_id, const int32_t delay_ms = 0);
    void ShowAllFramesWithBias(const int32_t delay_ms = 0);
    void ShowLocalMapFramesAndFeatures(const int32_t feature_id = -1, const int32_t camera_id = 0, const int32_t delay_ms = 0);
    void ShowLocalMapInWorldFrame(const std::string &title, const int32_t delay_ms, const bool block_in_loop = false);
    void ShowLocalAndGlobalMapInWorldFrame(const std::string &title, const int32_t delay_ms, const bool block_in_loop = false);
    void ShowSimpleInformationOfVisualLocalMap();
    void ShowTinyInformationOfVisualLocalMap();
    void ShowMatrixImage(const std::string &title, const Mat &matrix);

    // Reference for member variables.
    DataManagerOptions &options() { return options_; }
    CovisibleGraphType *visual_local_map() { return visual_local_map_.get(); }
    std::deque<FrameWithBias> &frames_with_bias() { return frames_with_bias_; }
    std::vector<GlobalMapKeyframe> &global_map_keyframes() { return global_map_keyframes_; }
    std::vector<CameraExtrinsic> &camera_extrinsics() { return camera_extrinsics_; }

private:
    // Support for visualizor.
    RgbPixel GetFeatureColor(const FeatureType &feature);
    void ShowLocalMapInWorldFrame();
    void ShowGlobalMapInWorldFrame();
    void UpdateVisualizorCameraView();

private:
    // Options for data manager.
    DataManagerOptions options_;

    // All frames and map points in local map.
    std::unique_ptr<CovisibleGraphType> visual_local_map_ = std::make_unique<CovisibleGraphType>();
    // All frames with bias in local map.
    std::deque<FrameWithBias> frames_with_bias_;
    // All keyframes and points in global map.
    std::vector<GlobalMapKeyframe> global_map_keyframes_;
    // Camera extrinsics.
    std::vector<CameraExtrinsic> camera_extrinsics_;

    // Record log.
    SLAM_DATA_LOG::BinaryDataLog logger_;

};

}

#endif // end of _VIO_STEREO_VINS_DATA_MANAGER_H_
