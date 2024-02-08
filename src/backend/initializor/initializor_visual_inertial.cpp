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

bool Backend::EstimateVelocityGravityScaleIn3Dof() {
    const int32_t size = data_manager_->frames_with_bias().size() * 3 + 3 + 1;
    Mat A = Mat::Zero(size, size);
    Vec b = Vec::Zero(size);

    // Extract camera extrinsics.
    const Vec3 &p_ic = data_manager_->camera_extrinsics().front().p_ic;

    // Construct incremental function.
    uint32_t frame_id_i = data_manager_->visual_local_map()->frames().front().id();
    for (auto it = data_manager_->frames_with_bias().cbegin(); std::next(it) != data_manager_->frames_with_bias().cend(); ++it) {
        const auto &cam_frame_i = data_manager_->visual_local_map()->frame(frame_id_i);
        const auto &cam_frame_j = data_manager_->visual_local_map()->frame(frame_id_i + 1);

        const auto &imu_frame_i = *it;
        const auto &imu_frame_j = *std::next(it);

        Mat6x10 H = Mat6x10::Zero();
        Vec6 z = Vec6::Zero();
        Mat6 Q = Mat6::Identity();

        const float &dt = imu_frame_j.imu_preint_block.integrate_time_s();
        const Vec3 &p_wc_i = cam_frame_i->p_wc();
        const Vec3 &p_wc_j = cam_frame_j->p_wc();
        const Mat3 R_iw_i = imu_frame_i.q_wi.inverse().toRotationMatrix();
        const Mat3 R_wi_j = imu_frame_j.q_wi.toRotationMatrix();

        // Construct sub incremental function.
        H.block<3, 3>(0, 0) = - dt * Mat3::Identity();
        H.block<3, 3>(0, 6) = 0.5f * R_iw_i * dt * dt;
        H.block<3, 1>(0, 9) = R_iw_i * (p_wc_j - p_wc_i);
        H.block<3, 3>(3, 0) = - Mat3::Identity();
        H.block<3, 3>(3, 3) = R_iw_i * R_wi_j;
        H.block<3, 3>(3, 6) = R_iw_i * dt;
        z.block<3, 1>(0, 0) = imu_frame_j.imu_preint_block.p_ij() - p_ic + R_iw_i * R_wi_j * p_ic;
        z.block<3, 1>(3, 0) = imu_frame_j.imu_preint_block.v_ij();
        Mat sub_A = H.transpose() * Q * H;
        Vec sub_b = H.transpose() * Q * z;

        // Construct full incremental function.
        uint32_t index = frame_id_i - data_manager_->visual_local_map()->frames().front().id();
        A.block<6, 6>(index * 3, index * 3) += sub_A.topLeftCorner<6, 6>();
        b.segment<6>(index * 3) += sub_b.head<6>();
        A.bottomRightCorner<4, 4>() += sub_A.bottomRightCorner<4, 4>();
        b.tail<4>() += sub_b.tail<4>();
        A.block<6, 4>(index * 3, size - 4) += sub_A.topRightCorner<6, 4>();
        A.block<4, 6>(size - 4, index * 3) += sub_A.bottomLeftCorner<4, 6>();

        ++frame_id_i;
    }

    // Solve incremental function.
    const Vec x = A.ldlt().solve(b);
    const float scale = x.tail<1>()[0];
    const Vec3 gravity_c0 = x.segment<3>(size - 4);
    const Vec all_v_ii = x.head(size - 4);
    ReportColorInfo("[Backend] Backend estimate scale [" << scale << "]" <<
        ", gravity_c0 " << LogVec(gravity_c0) << " with norm [" << gravity_c0.norm() << "].");

    // Check invalidation.
    RETURN_FALSE_IF(scale < 0 || std::fabs(gravity_c0.norm() - options_.kGravityInWordFrame.norm()) > 1.0f);
    return true;
}

bool Backend::EstimateVelocityGravityScaleIn2Dof() {

    return true;
}

}
