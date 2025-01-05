#include "backend.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"

namespace VIO {

bool Backend::LoadMapFromOldestKeyFrame() {
    RETURN_FALSE_IF(status_.marginalize_type != BackendMarginalizeType::kMarginalizeOldestFrame);
    RETURN_FALSE_IF(data_manager_->visual_local_map()->frames().empty());
    const auto &oldest_frame = data_manager_->visual_local_map()->frames().front();
    RETURN_FALSE_IF(data_manager_->imu_based_frames().empty());
    const auto &oldest_imu_based_frame = data_manager_->imu_based_frames().front();
    RETURN_FALSE_IF(oldest_imu_based_frame.time_stamp_s != oldest_frame.time_stamp_s());
    RETURN_FALSE_IF(map_of_marged_frame_.time_stamp_s == oldest_imu_based_frame.time_stamp_s);

    map_of_marged_frame_.time_stamp_s = oldest_frame.time_stamp_s();
    map_of_marged_frame_.p_wi = oldest_imu_based_frame.p_wi;
    map_of_marged_frame_.q_wi = oldest_imu_based_frame.q_wi;
    map_of_marged_frame_.all_norm_xy_left.clear();
    map_of_marged_frame_.all_p_wf.clear();
    map_of_marged_frame_.all_norm_xy_left.reserve(oldest_frame.features().size());
    map_of_marged_frame_.all_p_wf.reserve(oldest_frame.features().size());

    for (const auto &pair : oldest_frame.features()) {
        const auto &feature_id = pair.first;
        const auto &feature_ptr = pair.second;
        CONTINUE_IF(feature_ptr->status() != FeatureSolvedStatus::kMarginalized);
        CONTINUE_IF(ComputeMaxParallexAngleOfFeature(feature_id) < 5.0f * kDegToRad);
        const Vec2 norm_xy = feature_ptr->observe(oldest_frame.id())[0].rectified_norm_xy;
        const Vec3 p_wf = feature_ptr->param();
        map_of_marged_frame_.all_norm_xy_left.emplace_back(norm_xy);
        map_of_marged_frame_.all_p_wf.emplace_back(p_wf);
    }

    return true;
}

}
