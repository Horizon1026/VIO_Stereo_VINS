#include "backend.h"
#include "log_report.h"
#include "relative_rotation.h"

namespace VIO {

bool Backend::EstimateGyroBiasAndRotationForInitialization() {
    switch (options_.kMethodIndexToEstimateGyroBiasForInitialization) {
        case 1:
            return EstimateGyroBiasByMethodOneForInitialization();
        case 2:
            return EstimateGyroBiasByMethodTwoForInitialization();
        case 3:
        default:
            return EstimateGyroBiasByMethodThreeForInitialization();
    }
}

bool Backend::EstimatePureRotationOfCameraFrame(const uint32_t ref_frame_id,
                                                const uint32_t cur_frame_id,
                                                const uint32_t min_frame_id,
                                                std::vector<Vec2> &ref_norm_xy,
                                                std::vector<Vec2> &cur_norm_xy,
                                                Quat &q_cr) {
    ref_norm_xy.clear();
    cur_norm_xy.clear();

    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    if (!data_manager_->visual_local_map()->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        ReportError("[Backend] Failed to get covisible features between frame " << ref_frame_id << " and " << cur_frame_id << ".");
        return false;
    }
    for (const auto &feature_ptr : covisible_features) {
        ref_norm_xy.emplace_back(feature_ptr->observe(ref_frame_id)[0].rectified_norm_xy);
        cur_norm_xy.emplace_back(feature_ptr->observe(cur_frame_id)[0].rectified_norm_xy);
    }

    // Estimate pure rotation.
    using namespace VISION_GEOMETRY;
    RelativeRotation solver;
    RETURN_FALSE_IF(!solver.EstimateRotationByBnb(ref_norm_xy, cur_norm_xy, q_cr));

    // Update rotation of each frames. The frame w is defined as c0.
    if (ref_frame_id == min_frame_id) {
        data_manager_->visual_local_map()->frame(ref_frame_id)->q_wc() = Quat::Identity();
    }
    data_manager_->visual_local_map()->frame(cur_frame_id)->q_wc() = data_manager_->visual_local_map()->frame(ref_frame_id)->q_wc() * q_cr.inverse();

    return true;
}

bool Backend::EstimateGyroBiasByMethodOneForInitialization() {
    ReportInfo("[Backend] Try to estimate bias of gyro by Method 1.");
    RecomputeImuPreintegration();

    // Determine the scope of all frames.
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();

    // Define some temp variables.
    std::vector<Vec2> ref_norm_xy;
    std::vector<Vec2> cur_norm_xy;
    ref_norm_xy.reserve(200);
    cur_norm_xy.reserve(200);

    // Iterate all frames, estimate q_wc of all frames.
    for (uint32_t i = min_frames_idx; i < max_frames_idx; ++i) {
        Quat q_cr = Quat::Identity();
        RETURN_FALSE_IF(!EstimatePureRotationOfCameraFrame(i, i + 1, min_frames_idx, ref_norm_xy, cur_norm_xy, q_cr));
    }

    // Iterate to estimate bias gyro.
    const uint32_t max_iteration = 2;
    for (uint32_t iter = 0; iter < max_iteration; ++iter) {
        Mat3 hessian = Mat3::Zero();
        Vec3 bias = Vec3::Zero();
        // Iterate all imu preintegration block to construct incremental function.
        auto new_frame_iter = std::next(data_manager_->frames_with_bias().begin());
        for (uint32_t i = min_frames_idx; i < max_frames_idx; ++i) {
            // Localize the frame with bias in 'frames_with_bias_' between frame i and i + 1.
            ImuPreintegrateBlock<> &imu_preint_block = new_frame_iter->imu_preint_block;
            ++new_frame_iter;
            const Mat3 dr_dbg = imu_preint_block.dr_dbg();
            const Quat q_bibj = imu_preint_block.q_ij();

            const Quat q_wi_i = data_manager_->visual_local_map()->frame(i)->q_wc();
            const Quat q_wi_j = data_manager_->visual_local_map()->frame(i + 1)->q_wc();

            const Vec3 residual = 2.0f * (q_bibj.inverse() * q_wi_i.inverse() * q_wi_j).vec();
            hessian += dr_dbg.transpose() * dr_dbg;
            bias += dr_dbg.transpose() * residual;
        }

        const Vec3 delta_bg = hessian.ldlt().solve(bias);
        RETURN_FALSE_IF(Eigen::isnan(delta_bg.array()).any());

        // Recompute imu preintegration block with new bias of gyro.
        for (auto &frame : data_manager_->frames_with_bias()) {
            frame.imu_preint_block.ResetIntegratedStates();
            frame.imu_preint_block.bias_gyro() += delta_bg;
            const int32_t max_idx = static_cast<int32_t>(frame.packed_measure->imus.size());
            for (int32_t i = 1; i < max_idx; ++i) {
                frame.imu_preint_block.Propagate(*frame.packed_measure->imus[i - 1], *frame.packed_measure->imus[i]);
            }
        }

        // Check convergence.
        BREAK_IF(delta_bg.squaredNorm() < 1e-6f);
    }

    // Report result.
    const Vec3 bias_g = data_manager_->frames_with_bias().back().imu_preint_block.bias_gyro();
    ReportInfo(GREEN "[Backend] Estimated bias_gyro is " << LogVec(bias_g) << RESET_COLOR);

    return true;
}

bool Backend::EstimateGyroBiasByMethodTwoForInitialization() {
    ReportInfo("[Backend] Try to estimate bias of gyro by Method 2 (not valid now).");
    RecomputeImuPreintegration();

    // Localize the left camera extrinsic.
    const Quat q_ic = data_manager_->camera_extrinsics().front().q_ic;

    // Determine the scope of all frames.
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();

    // Define some temp variables.
    std::vector<FeatureType *> covisible_features;
    std::vector<Vec2> ref_norm_xy;
    std::vector<Vec2> cur_norm_xy;
    ref_norm_xy.reserve(200);
    cur_norm_xy.reserve(200);

    // Something should be recorded.
    using namespace VISION_GEOMETRY;
    std::vector<SummationTerms> all_summation_terms;
    std::vector<Mat3> all_dr_dbgs;
    all_summation_terms.reserve(max_frames_idx - min_frames_idx + 1);
    all_dr_dbgs.reserve(max_frames_idx - min_frames_idx + 1);

    // Iterate all frames and imu preintegrations.
    auto new_frame_iter = std::next(data_manager_->frames_with_bias().begin());
    for (uint32_t i = min_frames_idx; i < max_frames_idx; ++i) {
        // Get covisible features only in left camera.
        data_manager_->visual_local_map()->GetCovisibleFeatures(i, i + 1, covisible_features);
        ref_norm_xy.clear();
        cur_norm_xy.clear();
        for (const auto &feature_ptr : covisible_features) {
            ref_norm_xy.emplace_back(feature_ptr->observe(i)[0].rectified_norm_xy);
            cur_norm_xy.emplace_back(feature_ptr->observe(i + 1)[0].rectified_norm_xy);
        }

        // Localize the frame with bias in 'frames_with_bias_' between frame i and i + 1.
        ImuPreintegrateBlock<> &imu_preint_block = new_frame_iter->imu_preint_block;
        ++new_frame_iter;
        const Mat3 dr_dbg = imu_preint_block.dr_dbg();
        const Quat imu_q_ij = imu_preint_block.q_ij();
        const Quat q_jc = imu_q_ij.inverse() * q_ic;
        all_dr_dbgs.emplace_back(dr_dbg);

        // Compute summation terms.
        all_summation_terms.emplace_back(SummationTerms{});
        auto &terms = all_summation_terms.back();
        for (uint32_t i = 0; i < ref_norm_xy.size(); ++i) {
            const Vec3 f1 = q_jc * Vec3(ref_norm_xy[i].x(), ref_norm_xy[i].y(), 1).normalized();
            const Vec3 f2 = q_ic * Vec3(cur_norm_xy[i].x(), cur_norm_xy[i].y(), 1).normalized();
            const Mat3 F = f2 * f2.transpose();
            const float weight = 1.0f;
            terms.xx += weight * f1.x() * f1.x() * F;
            terms.yy += weight * f1.y() * f1.y() * F;
            terms.zz += weight * f1.z() * f1.z() * F;
            terms.xy += weight * f1.x() * f1.y() * F;
            terms.yz += weight * f1.y() * f1.z() * F;
            terms.zx += weight * f1.z() * f1.x() * F;
        }
    }
    RETURN_FALSE_IF(all_dr_dbgs.size() != all_summation_terms.size());

    // Try to estimate bias by optimization.
    const int32_t max_iteration = 50;
    Vec3 bias_g = Vec3::Zero();
    for (int32_t iter = 0; iter < max_iteration; ++iter) {
        Mat3 hessian = Mat3::Zero();
        Vec3 bias = Vec3::Zero();

        for (uint32_t j = 0; j < all_summation_terms.size(); ++j) {
            const Vec3 jac_bg = all_dr_dbgs[j] * bias_g;
            const Quat q = Utility::ConvertAngleAxisToQuaternion(jac_bg);
            const Vec3 cayley = Utility::ConvertQuaternionToCayley(q);
            Mat1x3 dlambda_dcayley;
            const float smallest_eigen_value = RelativeRotation::ComputeSmallestEigenValueAndJacobian(
                all_summation_terms[j], cayley, dlambda_dcayley);
            const Mat1x3 jacobian = dlambda_dcayley * all_dr_dbgs[j];

            hessian += jacobian.transpose() * jacobian;
            bias += jacobian.transpose() * smallest_eigen_value;
        }

        const Vec3 temp_bg = hessian.ldlt().solve(bias);
        BREAK_IF(Eigen::isnan(temp_bg.array()).any());
        bias_g = temp_bg;
    }

    // Recompute imu preintegration block with new bias of gyro.
    for (auto &frame : data_manager_->frames_with_bias()) {
        frame.imu_preint_block.ResetIntegratedStates();
        frame.imu_preint_block.bias_gyro() = bias_g;
        const int32_t max_idx = static_cast<int32_t>(frame.packed_measure->imus.size());
        for (int32_t i = 1; i < max_idx; ++i) {
            frame.imu_preint_block.Propagate(*frame.packed_measure->imus[i - 1], *frame.packed_measure->imus[i]);
        }
    }

    // Report result.
    ReportInfo(GREEN "[Backend] Estimated bias_gyro is " << LogVec(bias_g) << RESET_COLOR);

    return true;
}

bool Backend::EstimateGyroBiasByMethodThreeForInitialization() {
    ReportInfo("[Backend] Try to estimate bias of gyro by Method 3.");
    RecomputeImuPreintegration();

    // Determine the scope of all frames.
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();

    // Define some temp variables.
    std::vector<Vec2> ref_norm_xy;
    std::vector<Vec2> cur_norm_xy;
    ref_norm_xy.reserve(200);
    cur_norm_xy.reserve(200);

    // Iterate all frames to construct A and b, which is to estimate bias of gyro.
    const int32_t size = (max_frames_idx - min_frames_idx) * 3;
    Eigen::Matrix<float, Eigen::Dynamic, 3> A;
    Vec b;
    A.resize(size, 3);
    b.resize(size);
    auto new_frame_iter = std::next(data_manager_->frames_with_bias().begin());
    for (uint32_t i = min_frames_idx; i < max_frames_idx; ++i) {
        Quat q_cr = Quat::Identity();
        RETURN_FALSE_IF(!EstimatePureRotationOfCameraFrame(i, i + 1, min_frames_idx, ref_norm_xy, cur_norm_xy, q_cr));

        // Localize the frame with bias in 'frames_with_bias_' between frame i and i + 1.
        ImuPreintegrateBlock<> &imu_preint_block = new_frame_iter->imu_preint_block;
        ++new_frame_iter;
        const Mat3 dr_dbg = imu_preint_block.dr_dbg();
        const Quat imu_q_ij = imu_preint_block.q_ij();

        // Construct Ax = b.
        const Quat q_ij = imu_q_ij.inverse() * q_cr.inverse();
        const Vec3 angle_axis = Utility::ConvertQuaternionToAngleAxis(q_ij);
        A.block((i - min_frames_idx) * 3, 0, 3, 3) = dr_dbg;
        b.segment((i - min_frames_idx) * 3, 3) = angle_axis;
    }

    // Estimate bias of gyro.
    const Vec3 bias_g = A.colPivHouseholderQr().solve(b);
    RETURN_FALSE_IF(Eigen::isnan(bias_g.array()).any());

    // Recompute imu preintegration block with new bias of gyro.
    for (auto &frame : data_manager_->frames_with_bias()) {
        frame.imu_preint_block.ResetIntegratedStates();
        frame.imu_preint_block.bias_gyro() = bias_g;
        const int32_t max_idx = static_cast<int32_t>(frame.packed_measure->imus.size());
        for (int32_t i = 1; i < max_idx; ++i) {
            frame.imu_preint_block.Propagate(*frame.packed_measure->imus[i - 1], *frame.packed_measure->imus[i]);
        }
    }

    // Report result.
    ReportInfo(GREEN "[Backend] Estimated bias_gyro is " << LogVec(bias_g) << RESET_COLOR);

    return true;
}

}
