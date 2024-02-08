#include "backend.h"

namespace VIO {

bool Backend::EstimateGyroBias() {
    // Preintegrate all imu measurement block.
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        RecomputeImuPreintegrationBlock(Vec3::Zero(), Vec3::Zero(), frame_with_bias);
        // Debug.
        frame_with_bias.imu_preint_block.SimpleInformation();
    }

    // Construct incremental function.
    Mat3 hessian = Mat3::Zero();
    Vec3 bias = Vec3::Zero();
    for (auto it = data_manager_->frames_with_bias().cbegin(); std::next(it) != data_manager_->frames_with_bias().cend(); ++it) {
        const auto &frame_i = *it;
        const auto &frame_j = *std::next(it);
    }

    return true;
}

}
