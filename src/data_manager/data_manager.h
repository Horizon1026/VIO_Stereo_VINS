#ifndef _VIO_STEREO_BASALT_DATA_MANAGER_H_
#define _VIO_STEREO_BASALT_DATA_MANAGER_H_

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

/* Options for Data Manager. */
struct DataManagerOptions {
    uint32_t kMaxStoredKeyFrames = 0;
    bool kEnableRecordBinaryCurveLog = false;
};

/* Definition of Feature Points. */
using FeatureParameter = Vec3;

/* Definition of Covisible Graph. */
using FeatureObserve = std::vector<ObservePerView>; // Use std::vector to store observations of left and right camera.
using FeatureType = VisualFeature<FeatureParameter, FeatureObserve>;
using CovisibleGraphType = CovisibleGraph<FeatureParameter, FeatureObserve>;

/* Definition of Camera Extrinsic. */
struct CameraExtrinsic {
    // Rotation and translation between imu and camera frame.
    Quat q_ic = Quat::Identity();
    Vec3 p_ic = Vec3::Zero();
};

/* Definition of Frame and FrameWithBias. */
using FrameType = VisualFrame<FeatureType>;
struct FrameWithBias {
    // Imu bias of accel and gyro is inside imu_preint_block.
    ImuPreintegrateBlock<> imu_preint_block;
    float time_stamp_s = 0.0f;
    // Measurement of raw imu(gyro, acc), raw image(left, right) and visual features.
    std::unique_ptr<PackedMeasurement> packed_measure = nullptr;
    std::unique_ptr<FrontendOutputData> visual_measure = nullptr;
    // States based on imu.
    Vec3 p_wi = Vec3::Zero();
    Quat q_wi = Quat::Identity();
    Vec3 v_wi = Vec3::Zero();
};

/* Class Data Manager Declaration. */
class DataManager final {

public:
    DataManager() = default;
    ~DataManager() = default;

    void Clear();
    bool Configuration(const std::string &log_file_name);
    void RegisterLogPackages();

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

    // Visualizor of managed data.
    RgbPixel GetFeatureColor(const FeatureType &feature);
    void ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id, const uint32_t cur_frame_id, const int32_t delay_ms = 0);
    void ShowAllFramesWithBias(const int32_t delay_ms = 0);
    void ShowLocalMapFramesAndFeatures(const int32_t feature_id = -1, const int32_t camera_id = 0, const int32_t delay_ms = 0);
    void ShowLocalMapInWorldFrame(const int32_t delay_ms, const bool block_in_loop = false);
    void ShowSimpleInformationOfVisualLocalMap();
    void ShowTinyInformationOfVisualLocalMap();
    void ShowMatrixImage(const std::string &title, const Mat &matrix);

    // Reference for member variables.
    DataManagerOptions &options() { return options_; }
    CovisibleGraphType *visual_local_map() { return visual_local_map_.get(); }
    std::deque<FrameWithBias> &frames_with_bias() { return frames_with_bias_; }
    std::vector<CameraExtrinsic> &camera_extrinsics() { return camera_extrinsics_; }

private:
    // Options for data manager.
    DataManagerOptions options_;

    // All keyframes and map points.
    // Keyframes : [ p_wc, q_wc ]
    // Feature Points : [ p_w | invdep ]
    std::unique_ptr<CovisibleGraphType> visual_local_map_ = std::make_unique<CovisibleGraphType>();
    // All non-keyframes with bias.
    std::deque<FrameWithBias> frames_with_bias_;
    // Camera extrinsics.
    std::vector<CameraExtrinsic> camera_extrinsics_;

    // Record log.
    SLAM_DATA_LOG::BinaryDataLog logger_;

};

}

#endif // end of _VIO_STEREO_BASALT_DATA_MANAGER_H_
