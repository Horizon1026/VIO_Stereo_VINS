#include "backend.h"

namespace VIO {

bool Backend::EstimateGyroBias() {
    // Preintegrate all imu measurement block.
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        RecomputeImuPreintegrationBlock(Vec3::Zero(), Vec3::Zero(), frame_with_bias);
    }

    // Construct incremental function.
    Mat3 hessian = Mat3::Zero();
    Vec3 bias = Vec3::Zero();
    for (auto it = data_manager_->frames_with_bias().cbegin(); std::next(it) != data_manager_->frames_with_bias().cend(); ++it) {
        const auto &frame_i = *it;
        const auto &frame_j = *std::next(it);
        const Quat &q_wi_i = frame_i.q_wi;
        const Quat &q_wi_j = frame_j.q_wi;
        const Quat &q_ij = frame_j.imu_preint_block.q_ij();

        const Mat3 &jacobian = frame_j.imu_preint_block.dr_dbg();
        const Vec3 residual = 2.0f * (q_ij.inverse() * q_wi_i.inverse() * q_wi_j).vec();
        hessian += jacobian.transpose() * jacobian;
        bias += jacobian.transpose() * residual;
    }

    // Solve incremental function.
    const Vec3 delta_bias_gyro = hessian.ldlt().solve(bias);
    RETURN_FALSE_IF(std::isnan(delta_bias_gyro.sum()));
    ReportColorInfo("[Backend] Estimate bias of gyro " << LogVec(delta_bias_gyro));

    // Update bias of gyro and do preintegration.
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        RecomputeImuPreintegrationBlock(Vec3::Zero(),
            frame_with_bias.imu_preint_block.bias_gyro() + delta_bias_gyro,
            frame_with_bias);
    }

    return true;
}

}
