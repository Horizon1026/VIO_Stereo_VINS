#ifndef _VIO_STEREO_VINS_BACKEND_H_
#define _VIO_STEREO_VINS_BACKEND_H_

#include "basic_type.h"

#include "data_manager.h"
#include "imu.h"

#include "general_graph_optimizor.h"
#include "visual_frontend.h"

#include "backend_log.h"
#include "binary_data_log.h"

namespace vio {

using namespace slam_utility;
using namespace slam_data_log;
using namespace slam_solver;
using namespace sensor_model;
using DorF = float;

/* Options for Backend. */
struct BackendOptions {
    bool kEnableRecordBinaryCurveLog = true;
    bool kEnableLocalMapStoreRawImages = true;

    Vec3 kGravityInWordFrame = Vec3(0.0f, 0.0f, 0.0f);

    float kMaxValidFeatureDepthInMeter = 0.0f;
    float kMinValidFeatureDepthInMeter = 0.0f;
    float kDefaultFeatureDepthInMeter = 0.0f;

    float kMinParallexAngleOfFeatureToBundleAdjustmentInDegree = 0.0f;
    float kMaxToleranceTimeCostForEstimationInSecond = 0.0f;
    float kMaxToleranceReprojectionErrorInNormPlane = 0.0f;

    bool kEnableUseMultiViewObservation = true;
    bool kFixCameraExtrinsics = true;
};

/* Status of Backend. */
enum class BackendMarginalizeType : uint8_t {
    kNotMarginalize = 0,
    kMarginalizeOldestFrame = 1,
    kMarginalizeSubnewFrame = 2,
};

struct BackendStatus {
    struct {
        uint32_t is_initialized : 1;
        uint32_t reserved : 31;
    };
    BackendMarginalizeType marginalize_type = BackendMarginalizeType::kNotMarginalize;
};

struct BackendSignals {
    struct {
        uint32_t should_quit : 1;  // You can kill all relative threads by checking this flag.
        uint32_t reserved : 31;
    };
};

struct BackendStates {
    // Motion states.
    struct {
        Vec3 p_wi = Vec3::Zero();
        Quat q_wi = Quat::Identity();
        Vec3 v_wi = Vec3::Zero();
        Vec3 ba = Vec3::Zero();
        Vec3 bg = Vec3::Zero();
        float time_stamp_s = 0.0f;
    } motion;

    // Prior information.
    struct {
        bool is_valid = false;
        TMat<DorF> hessian;
        TVec<DorF> bias;
        TMat<DorF> jacobian;
        TMat<DorF> jacobian_t_inv;
        TVec<DorF> residual;
    } prior;
};

struct BackendMapPerFrame {
    float time_stamp_s = 0.0f;
    Vec3 p_wi = Vec3::Zero();
    Quat q_wi = Quat::Identity();
    std::vector<Vec2> all_norm_xy_left;
    std::vector<Vec3> all_p_wf;
};

/* Vertices and edges for estimation and marginalization. */
struct BackendVertices {
    std::vector<std::unique_ptr<Vertex<DorF>>> all_cameras_p_ic;
    std::vector<std::unique_ptr<VertexQuat<DorF>>> all_cameras_q_ic;

    std::vector<uint32_t> all_frames_id;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_p_wi;
    std::vector<std::unique_ptr<VertexQuat<DorF>>> all_frames_q_wi;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_p_wc;
    std::vector<std::unique_ptr<VertexQuat<DorF>>> all_frames_q_wc;

    std::vector<uint32_t> all_features_id;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_features_invdep;

    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_v_wi;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_ba;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_bg;
};

struct BackendEdges {
    std::vector<std::unique_ptr<Edge<DorF>>> all_prior_factors;
    std::vector<std::unique_ptr<Edge<DorF>>> all_visual_factors;
    std::vector<std::unique_ptr<Edge<DorF>>> all_imu_factors;
};

struct BackendGraph {
    BackendVertices vertices;
    BackendEdges edges;
};

/* Class Backend Declaration. */
class Backend final {

public:
    Backend() = default;
    ~Backend() = default;

    // Backend operations.
    bool RunOnce();
    void Reset();
    void ResetToReintialize();

    // Backend log recorder.
    bool Configuration(const std::string &log_file_name);

    // Reference for member variables.
    BackendOptions &options() { return options_; }
    VisualFrontend *&visual_frontend() { return visual_frontend_; }
    DataManager *&data_manager() { return data_manager_; }
    std::unique_ptr<Imu> &imu_model() { return imu_model_; }
    BackendSignals &signals() { return signals_; }

    // Const reference for member variables.
    const BackendOptions &options() const { return options_; }
    const std::unique_ptr<Imu> &imu_model() const { return imu_model_; }
    const BackendStatus &status() const { return status_; }
    const BackendStates &states() const { return states_; }
    const BackendSignals &signals() const { return signals_; }

private:
    // Backend log recorder.
    void RegisterLogPackages();
    void RecordBackendLogStates();
    void UpdateBackendLogPredictStates();
    void RecordBackendLogPredictStates();
    void UpdateBackendLogGraph();
    void RecordBackendLogGraph();
    void RecordBackendLogStatus();
    void RecordBackendLogCostTime();
    void RecordBackendLogPriorInformation();
    void RecordBackendLogPredictionReprojectionError(const std::vector<std::pair<uint32_t, Vec2>> &repro_err_with_feature_id, const float time_stamp_s);
    void RecordBackendLogParallexAngleMap();
    void RecordBackendLogMapOfOldestFrame();

    // Backend data processor.
    bool TryToSolveFramePoseByFeaturesObserved(const int32_t frame_id, const Vec3 &init_p_wc = Vec3::Zero(), const Quat &init_q_wc = Quat::Identity());
    bool TryToSolveFeaturePositionByFramesObservingIt(const int32_t feature_id, const int32_t min_frame_id = -1, const int32_t max_frame_id = kMaxInt32,
                                                      const bool use_multi_view = false);
    bool StatisReprojectionErrorInOneFrame(const int32_t frame_id, std::vector<std::pair<uint32_t, Vec2>> &repro_err_with_feature_id);
    float ComputeMaxParallexAngleOfFeature(const uint32_t feature_id);
    TMat2<DorF> GetVisualObserveInformationMatrix();
    void RecomputeImuPreintegrationBlock(const Vec3 &bias_accel, const Vec3 &bias_gyro, ImuBasedFrame &imu_based_frame);
    bool AddNewestFrameWithStatesPredictionToLocalMap();
    bool ControlSizeOfLocalMap();
    void UpdateBackendStates();

    // Backend graph manager.
    void ClearGraph();
    void ConstructVioGraphOptimizationProblem(Graph<DorF> &problem, float &prior_residual_norm);
    void ConstructPureVisualGraphOptimizationProblem(Graph<DorF> &problem);
    void AddAllCameraExtrinsicsToGraph();
    void AddAllCameraPosesInLocalMapToGraph();
    void AddAllImuPosesInLocalMapToGraph();
    void AddAllImuMotionStatesInLocalMapToGraph();
    void RemoveNewestTwoFramesFromGraph();
    bool AllFeatureInvdepAndVisualFactorsOfCameraPosesToGraph(const FeatureType &feature, const float invdep, const TMat2<DorF> &visual_info_matrix,
                                                              const uint32_t max_frame_id, const bool use_multi_view = false);
    bool AllFeatureInvdepAndVisualFactorsOfImuPosesToGraph(const FeatureType &feature, const float invdep, const TMat2<DorF> &visual_info_matrix,
                                                           const uint32_t max_frame_id, const bool use_multi_view = false);
    bool AddAllFeatureInvdepsAndVisualFactorsToGraph(const bool add_factors_with_cam_ex, const bool use_multi_view = false);
    bool AddFeatureFirstObserveInOldestFrameAndVisualFactorsToGraph(const bool use_multi_view = false);
    bool AddImuFactorsToGraph(const bool only_add_oldest_one = false);
    bool AddPriorFactorForFirstImuPoseAndCameraExtrinsicsToGraph();
    bool SyncGraphVerticesToDataManager(const Graph<DorF> &problem);

    // Backend initializor.
    bool TryToInitialize();
    bool PrepareForPureVisualSfmByMonoView();
    bool PrepareForPureVisualSfmByMultiView();
    bool PerformPureVisualBundleAdjustment(const bool use_multi_view = false);
    bool EstimateGyroBias();
    bool EstimateVelocityGravityScaleIn3Dof(Vec3 &gravity_c0, float &scale);
    bool EstimateVelocityGravityScaleIn2Dof(Vec3 &gravity_c0, Vec &all_v_ii);
    bool SyncInitializedResult(const Vec3 &gravity_c0, const Vec &all_v_ii, const float &scale);

    // Backend estimator.
    bool TryToEstimate(const bool use_multi_view);
    BackendMarginalizeType DecideMarginalizeType();
    bool TryToMarginalize(const bool use_multi_view);
    bool MarginalizeOldestFrame(const bool use_multi_view);
    bool MarginalizeSubnewFrame(const bool use_multi_view);

    // Backend map manager.
    bool LoadMapFromOldestKeyFrame();

private:
    // Options of backend.
    BackendOptions options_;

    // Flags of status of backend.
    BackendStatus status_;
    // Motion and prior states of backend.
    BackendStates states_;
    // Signals of backend.
    BackendSignals signals_;

    // Graph of backend.
    BackendGraph graph_;
    // Map of backend.
    BackendMapPerFrame map_of_marged_frame_;

    // Register some relative components.
    VisualFrontend *visual_frontend_ = nullptr;
    DataManager *data_manager_ = nullptr;
    std::unique_ptr<Imu> imu_model_ = nullptr;

    // Record log.
    BinaryDataLog logger_;
    BackendLogGraph log_package_graph_;
    BackendLogStates log_package_states_;
    BackendLogStates log_package_predict_states_;
    BackendLogStatus log_package_status_;
    BackendLogCostTime log_package_cost_time_;
};

}  // namespace vio

#endif  // end of _VIO_STEREO_VINS_BACKEND_H_
