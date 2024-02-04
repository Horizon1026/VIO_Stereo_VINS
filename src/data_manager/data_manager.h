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
    uint32_t kMaxStoredNewFrames = 0;
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

    // Record log.
    void TriggerLogRecording(const float time_stamp_s);
    void RecordLocalMap(const float time_stamp_s);
    void RecordCovisibleGraph(const float time_stamp_s);

    // Transform packed measurements to a new frame.
    bool ProcessMeasure(std::unique_ptr<PackedMeasurement> &new_packed_measure,
                        std::unique_ptr<FrontendOutputData> &new_visual_measure);

    // Get specified frame id.
    uint32_t GetNewestKeyframeId();
    // Get specified frame timestamp.
    float GetNewestStateTimeStamp();

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
