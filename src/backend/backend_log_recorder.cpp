#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

namespace VIO {

constexpr uint32_t kBackendStatesLogIndex = 1;
constexpr uint32_t kBackendStatusLogIndex = 2;
constexpr uint32_t kBackendCostTimeLogIndex = 3;
constexpr uint32_t kBackendPriorHessianLogIndex = 4;

void Backend::RegisterLogPackages() {
    using namespace SLAM_DATA_LOG;

    std::unique_ptr<PackageInfo> package_states_ptr = std::make_unique<PackageInfo>();
    package_states_ptr->id = kBackendStatesLogIndex;
    package_states_ptr->name = "backend states";
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "time_stamp_s"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wi_x"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wi_y"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "p_wi_z"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_w"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_x"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_y"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_z"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_pitch"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_roll"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "q_wi_yaw"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wi_x"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wi_y"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "v_wi_z"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_a_x"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_a_y"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_a_z"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_g_x"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_g_y"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "bias_g_z"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_prior_valid"});
    package_states_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "prior_residual"});
    if (!logger_.RegisterPackage(package_states_ptr)) {
        ReportError("[Backend] Failed to register package for backend states log.");
    }

    std::unique_ptr<PackageInfo> package_status_flags_ptr = std::make_unique<PackageInfo>();
    package_status_flags_ptr->id = kBackendStatusLogIndex;
    package_status_flags_ptr->name = "backend status";
    package_status_flags_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "is_initialized"});
    package_status_flags_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint8, .name = "marginalize_type"});
    package_status_flags_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kUint32, .name = "num_of_valid_loop"});
    if (!logger_.RegisterPackage(package_status_flags_ptr)) {
        ReportError("[Backend] Failed to register package for backend status flags log.");
    }

    std::unique_ptr<PackageInfo> package_cost_time_ptr = std::make_unique<PackageInfo>();
    package_cost_time_ptr->id = kBackendCostTimeLogIndex;
    package_cost_time_ptr->name = "backend cost time";
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "total_loop(ms)"});
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "add_newest_frame_into_local_map(ms)"});
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "triangulize_all_visual_features(ms)"});
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "initialize(ms)"});
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "estimate(ms)"});
    package_cost_time_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kFloat, .name = "marginalize(ms)"});
    if (!logger_.RegisterPackage(package_cost_time_ptr)) {
        ReportError("[Backend] Failed to register package for backend cost time log.");
    }

    std::unique_ptr<PackageInfo> package_prior_hessian_ptr = std::make_unique<PackageInfo>();
    package_prior_hessian_ptr->id = kBackendPriorHessianLogIndex;
    package_prior_hessian_ptr->name = "backend prior";
    package_prior_hessian_ptr->items.emplace_back(PackageItemInfo{.type = ItemType::kMatrix, .name = "hessian"});
    if (!logger_.RegisterPackage(package_prior_hessian_ptr)) {
        ReportError("[Backend] Failed to register package for backend prior hessian log.");
    }
}

void Backend::RecordBackendLogStates() {
    RETURN_IF(!options().kEnableRecordBinaryCurveLog);

    log_package_states_.time_stamp_s = states_.motion.time_stamp_s;

    log_package_states_.p_wi_x = states_.motion.p_wi.x();
    log_package_states_.p_wi_y = states_.motion.p_wi.y();
    log_package_states_.p_wi_z = states_.motion.p_wi.z();

    const Vec3 euler = Utility::QuaternionToEuler(states_.motion.q_wi);
    log_package_states_.q_wi_pitch = euler.x();
    log_package_states_.q_wi_roll = euler.y();
    log_package_states_.q_wi_yaw = euler.z();

    log_package_states_.q_wi_w = states_.motion.q_wi.w();
    log_package_states_.q_wi_x = states_.motion.q_wi.x();
    log_package_states_.q_wi_y = states_.motion.q_wi.y();
    log_package_states_.q_wi_z = states_.motion.q_wi.z();

    log_package_states_.v_wi_x = states_.motion.v_wi.x();
    log_package_states_.v_wi_y = states_.motion.v_wi.y();
    log_package_states_.v_wi_z = states_.motion.v_wi.z();

    log_package_states_.bias_a_x = states_.motion.ba.x();
    log_package_states_.bias_a_y = states_.motion.ba.y();
    log_package_states_.bias_a_z = states_.motion.ba.z();

    log_package_states_.bias_g_x = states_.motion.bg.x();
    log_package_states_.bias_g_y = states_.motion.bg.y();
    log_package_states_.bias_g_z = states_.motion.bg.z();

    log_package_states_.is_prior_valid = static_cast<uint8_t>(states_.prior.is_valid);
    log_package_states_.prior_residual = states_.prior.is_valid ? states_.prior.residual.squaredNorm() : 0.0f;

    // Record log.
    logger_.RecordPackage(kBackendStatesLogIndex, reinterpret_cast<const char *>(&log_package_states_), data_manager_->GetNewestStateTimeStamp());
}

void Backend::RecordBackendLogStatus() {
    RETURN_IF(!options().kEnableRecordBinaryCurveLog);

    log_package_status_.is_initialized = states_.is_initialized;
    log_package_status_.marginalize_type = static_cast<uint8_t>(states_.marginalize_type);
    log_package_status_.num_of_valid_loop = states_.is_initialized ? log_package_status_.num_of_valid_loop + 1 : 0;

    // Record log.
    logger_.RecordPackage(kBackendStatusLogIndex, reinterpret_cast<const char *>(&log_package_status_), data_manager_->GetNewestStateTimeStamp());
}

void Backend::RecordBackendLogCostTime() {
    RETURN_IF(!options().kEnableRecordBinaryCurveLog);

    // Record log.
    logger_.RecordPackage(kBackendCostTimeLogIndex, reinterpret_cast<const char *>(&log_package_cost_time_), data_manager_->GetNewestStateTimeStamp());
}

void Backend::RecordBackendLogPriorInformation() {
    RETURN_IF(!options().kEnableRecordBinaryCurveLog);

    // Record log.
    if (states_.prior.is_valid) {
        logger_.RecordPackage(kBackendPriorHessianLogIndex, states_.prior.hessian.cast<float>(), data_manager_->GetNewestStateTimeStamp());
    }
}

}