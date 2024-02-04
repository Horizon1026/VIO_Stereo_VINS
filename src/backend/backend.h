#ifndef _VIO_STEREO_BASALT_BACKEND_H_
#define _VIO_STEREO_BASALT_BACKEND_H_

#include "datatype_basic.h"

#include "data_manager.h"
#include "imu.h"
#include "visual_frontend.h"
#include "general_graph_optimizor.h"

#include "binary_data_log.h"
#include "backend_log.h"

namespace VIO {

using namespace SLAM_UTILITY;
using namespace SLAM_DATA_LOG;
using namespace SLAM_SOLVER;
using namespace SENSOR_MODEL;
using DorF = float;

/* Options for Backend. */
struct BackendOptions {
    bool kEnableRecordBinaryCurveLog = true;
    uint32_t kMethodIndexToEstimateGyroBiasForInitialization = 1;
    Vec3 kGravityInWordFrame = Vec3(0.0f, 0.0f, 0.0f);
    float kMaxValidFeatureDepthInMeter = 0.0f;
    float kMinValidFeatureDepthInMeter = 0.0f;
    float kDefaultFeatureDepthInMeter = 0.0f;
    bool kEnableReportAllInformation = false;
    bool kEnableLocalMapStoreRawImages = true;
};

/* Status of Backend. */
enum class BackendMarginalizeType : uint8_t {
    kNotMarginalize = 0,
    kMarginalizeOldestFrame = 1,
    kMarginalizeSubnewFrame = 2,
};

struct BackendStates {
    // Status bits.
    struct {
        uint32_t is_initialized : 1;
        uint32_t reserved : 31;
    };
    BackendMarginalizeType marginalize_type = BackendMarginalizeType::kNotMarginalize;

    // Prior information.
    struct {
        bool is_valid = false;
        TMat<DorF> hessian;
        TVec<DorF> bias;
        TMat<DorF> jacobian;
        TMat<DorF> jacobian_t_inv;
        TVec<DorF> residual;
    } prior;

    // Motion states.
    struct {
        Vec3 p_wi = Vec3::Zero();
        Quat q_wi = Quat::Identity();
        Vec3 v_wi = Vec3::Zero();
        Vec3 ba = Vec3::Zero();
        Vec3 bg = Vec3::Zero();
        float time_stamp_s = 0.0f;
    } motion;
};

/* Vertices and edges for estimation and marginalization. */
struct BackendVertices {
    std::vector<std::unique_ptr<Vertex<DorF>>> all_cameras_p_ic;
    std::vector<std::unique_ptr<VertexQuat<DorF>>> all_cameras_q_ic;

    std::vector<uint32_t> all_frames_id;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_frames_p_wi;
    std::vector<std::unique_ptr<VertexQuat<DorF>>> all_frames_q_wi;

    std::vector<uint32_t> all_features_id;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_features_invdep;

    std::vector<std::unique_ptr<Vertex<DorF>>> all_new_frames_v_wi;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_new_frames_ba;
    std::vector<std::unique_ptr<Vertex<DorF>>> all_new_frames_bg;
};

struct BackendEdges {
    std::vector<std::unique_ptr<Edge<DorF>>> all_prior_factors;
    std::vector<std::unique_ptr<Edge<DorF>>> all_visual_reproj_factors;
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
    bool Configuration(const std::string &log_file_name);

    // Backend log recorder.
    void RegisterLogPackages();

    // Backend initializor.
    bool TryToInitialize();
    bool ConvertNewFramesToCovisibleGraphForInitialization();
    bool EstimatePureRotationOfCameraFrame(const uint32_t ref_frame_id,
                                           const uint32_t cur_frame_id,
                                           const uint32_t min_frame_id,
                                           std::vector<Vec2> &ref_norm_xy,
                                           std::vector<Vec2> &cur_norm_xy,
                                           Quat &q_cr);
    // Estimate gyro bias for initialization.
    bool EstimateGyroBiasAndRotationForInitialization();
    bool EstimateGyroBiasByMethodOneForInitialization();
    bool EstimateGyroBiasByMethodTwoForInitialization();
    bool EstimateGyroBiasByMethodThreeForInitialization();
    // Estimate velocity and gravity for initialization.
    bool EstimateVelocityAndGravityForInitialization(Vec3 &gravity_i0);
    bool SelectTwoFramesWithMaxParallax(CovisibleGraphType *local_map, const FeatureType &feature, int32_t &frame_id_l, int32_t &frame_id_r);
    bool ComputeImuPreintegrationBasedOnFirstFrameForInitialization(std::vector<ImuPreintegrateBlock<>> &imu_blocks);
    bool ConstructLigtFunction(const std::vector<ImuPreintegrateBlock<>> &imu_blocks, Mat6 &A, Vec6 &b, float &Q);
    bool RefineGravityForInitialization(const Mat &M, const Vec &m, const float Q, const float gravity_mag, Vec &rhs);
    bool PropagateAllBasedOnFirstCameraFrameForInitializaion(const std::vector<ImuPreintegrateBlock<>> &imu_blocks, const Vec3 &v_i0i0, const Vec3 &gravity_i0);
    bool TransformAllStatesToWorldFrameForInitialization(const Vec3 &gravity_i0);

    // Backend estimator.
    bool TryToEstimate();
    TMat2<DorF> GetVisualObserveInformationMatrix();

    // Backend maginalizor.
    BackendMarginalizeType DecideMarginalizeType();
    bool TryToMarginalize();
    bool MarginalizeOldestFrame();
    bool MarginalizeSubnewFrame();

    // Support for backend estimator and marginalizor.
    void ClearBackendGraph();
    void ConvertCameraPoseAndExtrinsicToVertices();
    bool AddPriorFactorWhenNoPrior();
    bool ConvertFeatureInvdepAndAddVisualFactorForEstimation();
    bool ConvertFeatureInvdepAndAddVisualFactorForMarginalization();
    bool ConvertFeatureInvdepAndAddVisualFactor(const FeatureType &feature, const float invdep, const TMat2<DorF> &visual_info_matrix, const uint32_t max_frame_id);
    void ConvertImuMotionStatesToVertices();
    bool AddImuPreintegrationFactorForEstimation(const uint32_t idx_offset);
    bool AddImuPreintegrationFactorForMarginalization(const uint32_t idx_offset);
    void ConstructGraphOptimizationProblem(const uint32_t idx_offset, Graph<DorF> &problem);
    void UpdateAllStatesAfterEstimation(const Graph<DorF> &problem, const uint32_t idx_offset);

    // Backend data processor.
    void RecomputeImuPreintegration();
    bool TriangulizeAllNewVisualFeatures();
    bool TriangulizeAllVisualFeatures();
    bool TriangulizeVisualFeature(std::vector<Quat> &q_wc_vec,
                                  std::vector<Vec3> &p_wc_vec,
                                  std::vector<Vec2> &norm_xy_vec,
                                  FeatureType &feature);
    bool ControlLocalMapDimension();
    void UpdateBackendStates();
    bool AddNewestFrameWithBiasIntoLocalMap();

    // Backend log recorder.
    void RecordBackendLogStates();
    void RecordBackendLogStatus();
    void RecordBackendLogCostTime();
    void RecordBackendLogPriorInformation();

    // Backend selfcheck.
    bool CheckGraphOptimizationFactors();
    bool CheckGraphOptimizationFactors(std::vector<std::unique_ptr<Edge<DorF>>> &edges);

    // Backend visualizor.
    RgbPixel GetFeatureColor(const FeatureType &feature);
    void ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id, const uint32_t cur_frame_id, const bool use_rectify = false, const int32_t delay_ms = 0);
    void ShowAllFramesWithBias(const bool use_rectify = false, const int32_t delay_ms = 0);
    void ShowLocalMapFramesAndFeatures(const int32_t feature_id = -1, const int32_t camera_id = 0, const bool use_rectify = false, const int32_t delay_ms = 0);
    void ShowLocalMapInWorldFrame(const int32_t delay_ms, const bool block_in_loop = false);
    void ShowMatrixImage(const std::string &title, const TMat<DorF> &matrix);
    void ShowSimpleInformationOfVisualLocalMap();
    void ShowTinyInformationOfVisualLocalMap();

    // Reference for member variables.
    BackendOptions &options() { return options_; }
    VisualFrontend *&visual_frontend() { return visual_frontend_; }
    DataManager *&data_manager() { return data_manager_; }
    std::unique_ptr<Imu> &imu_model() { return imu_model_; }
    bool &should_quit() { return should_quit_; }

    // Const reference for member variables.
    const BackendOptions &options() const { return options_; }
    const std::unique_ptr<Imu> &imu_model() const { return imu_model_; }
    const bool &should_quit() const { return should_quit_; }
    const BackendStates &states() const { return states_; }

private:
    // Options of backend.
    BackendOptions options_;

    // Motion and prior states of backend.
    BackendStates states_;

    // Graph of backend.
    BackendGraph graph_;

    // Register some relative components.
    VisualFrontend *visual_frontend_ = nullptr;
    DataManager *data_manager_ = nullptr;
    std::unique_ptr<Imu> imu_model_ = nullptr;

    // Record log.
    BinaryDataLog logger_;
    BackendLogStates log_package_states_;
    BackendLogStatus log_package_status_;
    BackendLogCostTime log_package_cost_time_;

    // Signal flags.
    bool should_quit_ = false;  // You can kill all relative threads by checking this flag.
};

}

#endif // end of _VIO_STEREO_BASALT_BACKEND_H_
