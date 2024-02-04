#include "backend.h"
#include "log_report.h"
#include "polynomial_solver.h"

namespace VIO {

bool Backend::EstimateVelocityAndGravityForInitialization(Vec3 &gravity_i0) {
    // Compute imu blocks based on the first frame.
    std::vector<ImuPreintegrateBlock<>> imu_blocks;
    if (!ComputeImuPreintegrationBasedOnFirstFrameForInitialization(imu_blocks)) {
        ReportError("[Backend] Backend failed to compute imu preintegration block based on first frame.");
        return false;
    }

    // Construct LIGT function.
    Mat6 A = Mat6::Zero();
    Vec6 b = Vec6::Zero();
    float Q = 0.0f;
    if (!ConstructLigtFunction(imu_blocks, A, b, Q)) {
        ReportError("[Backend] Backend failed to construct LIGT function.");
        return false;
    }

    // Solve rhs(velocity and bias).
    Vec rhs = Vec6::Zero();
    if (!RefineGravityForInitialization(A, -2.0f * b, Q, 1.0f, rhs)) {
        ReportError("[Backend] Backend failed to refine gravity. Try to solve LIGT function with ldlt.");
        rhs = A.ldlt().solve(b);
    }
    const Vec3 v_i0i0 = rhs.head<3>();
    gravity_i0 = rhs.tail<3>().normalized() * options_.kGravityInWordFrame.norm();
    ReportInfo(GREEN "[Backend] Estimated v_i0i0 is " << LogVec(v_i0i0) << ", gravity_i0 is " << LogVec(gravity_i0) <<
        ", gravity norm is " << gravity_i0.norm() << RESET_COLOR);

    // Propagate states of all frames based on frame i0(imu).
    if (!PropagateAllBasedOnFirstCameraFrameForInitializaion(imu_blocks, v_i0i0, gravity_i0)) {
        ReportError("[Backend] Backend failed to propagate states of all frames based on frame i0.");
        return false;
    }

    return true;
}

bool Backend::ComputeImuPreintegrationBasedOnFirstFrameForInitialization(std::vector<ImuPreintegrateBlock<>> &imu_blocks) {
    const int32_t num_of_imu_block = data_manager_->visual_local_map()->frames().size() - 1;
    RETURN_FALSE_IF(num_of_imu_block < 1);

    const auto start_iter = std::next(data_manager_->frames_with_bias().begin());
    RETURN_FALSE_IF(start_iter == data_manager_->frames_with_bias().end());
    auto end_iter = std::next(start_iter);

    imu_blocks.clear();
    imu_blocks.emplace_back(start_iter->imu_preint_block);
    for (int32_t i = 1; i < num_of_imu_block; ++i) {
        ++end_iter;

        ImuPreintegrateBlock<> new_imu_block(imu_blocks.back());
        new_imu_block.ResetIntegratedStates();
        for (auto iter = start_iter; iter != end_iter; ++iter) {
            const int32_t max_idx = static_cast<int32_t>(iter->packed_measure->imus.size());
            for (int32_t j = 1; j < max_idx; ++j) {
                new_imu_block.Propagate(*iter->packed_measure->imus[j - 1], *iter->packed_measure->imus[j]);
            }
        }
        imu_blocks.emplace_back(new_imu_block);
    }

    return true;
}

bool Backend::SelectTwoFramesWithMaxParallax(CovisibleGraphType *local_map,
                                             const FeatureType &feature,
                                             int32_t &frame_id_l,
                                             int32_t &frame_id_r) {
    const int32_t num_of_observes = feature.observes().size();
    RETURN_FALSE_IF(feature.observes().size() < 3);

    // Iterate all pairs of frames, select the pair with max parallex angle.
    float max_parallex_angle = -1.0f;

    for (int32_t i = 0; i < num_of_observes - 1; ++i) {
        for (int32_t j = i + 1; j < num_of_observes; ++j) {
            // Extract observations in frame i/j.
            const auto &observe_i = feature.observes()[i];
            const auto &observe_j = feature.observes()[j];
            RETURN_FALSE_IF(observe_i.empty() || observe_j.empty());
            const Vec3 norm_xyz_i = Vec3(observe_i[0].rectified_norm_xy.x(), observe_i[0].rectified_norm_xy.y(), 1.0f);
            const Vec3 norm_xyz_j = Vec3(observe_j[0].rectified_norm_xy.x(), observe_j[0].rectified_norm_xy.y(), 1.0f);

            // Extract rotation of frame i/j. The frame w is i0.
            const Quat &q_wc_i = local_map->frame(i + feature.first_frame_id())->q_wc();
            const Quat &q_wc_j = local_map->frame(j + feature.first_frame_id())->q_wc();

            // Compute parallex angle.
            const Quat q_cjci = q_wc_j.inverse() * q_wc_i;
            const Vec3 angle_axis = norm_xyz_j.cross(q_cjci * norm_xyz_i);
            const float parallex_angle = angle_axis.norm();

            if (parallex_angle > max_parallex_angle) {
                // Update the pair of frames with max parallex angle.
                max_parallex_angle = parallex_angle;
                frame_id_l = i + feature.first_frame_id();
                frame_id_r = j + feature.first_frame_id();
            }
        }
    }

    return true;
}

bool Backend::ConstructLigtFunction(const std::vector<ImuPreintegrateBlock<>> &imu_blocks, Mat6 &A, Vec6 &b, float &Q) {
    // Compute the norm of gravity vector.
    const float gravity_norm = options_.kGravityInWordFrame.norm();

    // Localize the left camera extrinsic. Use 'b' to represent frame of imu.
    const Quat q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Vec3 t_bc = data_manager_->camera_extrinsics().front().p_ic;
    const Mat3 R_cb = q_ic.toRotationMatrix().transpose();

    // Iterate all feature in visual_local_map to create linear function.
    A.setZero();
    b.setZero();
    Q = 0.0f;
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;

        // Select two frames with max parallex angle.
        int32_t frame_id_l = 0;
        int32_t frame_id_r = 0;
        CONTINUE_IF(!SelectTwoFramesWithMaxParallax(data_manager_->visual_local_map(),
            feature, frame_id_l, frame_id_r));

        // Extract frame l/r.
        const auto frame_ptr_l = data_manager_->visual_local_map()->frame(frame_id_l);
        const auto frame_ptr_r = data_manager_->visual_local_map()->frame(frame_id_r);
        if (frame_ptr_l == nullptr || frame_ptr_r == nullptr) {
            ReportError("[Backend] Backend failed to find frame " << frame_id_l << " and " << frame_id_r << ".");
            return false;
        }
        const Mat3 R_wcl = frame_ptr_l->q_wc().toRotationMatrix();
        const Mat3 R_wcr = frame_ptr_r->q_wc().toRotationMatrix();

        // Extract observations of frame l/r.
        const auto &obv_l = feature.observe(frame_id_l);
        const auto &obv_r = feature.observe(frame_id_r);
        if (obv_l.empty() || obv_r.empty()) {
            ReportError("[Backend] Backend failed to find observations of frame " << frame_id_l << " and " << frame_id_r << ", obv_l.size() is " <<
                obv_l.size() << ", obv_r.size() is " << obv_r.size() << ".");
            return false;
        }
        const Vec3 norm_xyz_l = Vec3(obv_l[0].rectified_norm_xy.x(), obv_l[0].rectified_norm_xy.y(), 1.0f);
        const Vec3 norm_xyz_r = Vec3(obv_r[0].rectified_norm_xy.x(), obv_r[0].rectified_norm_xy.y(), 1.0f);

        // Iterate all observations of this feature.
        const int32_t num_of_observes = feature.observes().size();
        for (int32_t i = 0; i < num_of_observes; ++i) {
            const int32_t frame_id_i = i + feature.first_frame_id();
            CONTINUE_IF(frame_id_i == frame_id_r);

            // Extract frame i.
            const auto frame_ptr_i = data_manager_->visual_local_map()->frame(frame_id_i);
            if (frame_ptr_i == nullptr) {
                ReportError("[Backend] Backend failed to find frame " << frame_id_i << ".");
                return false;
            }
            const Mat3 R_wci = frame_ptr_i->q_wc().toRotationMatrix();

            // Extract observations of frame i/l/r.
            const auto &obv_i = feature.observe(frame_id_i);
            if (obv_i.empty()) {
                ReportError("[Backend] Backend failed to find observation of frame " << frame_id_l << ", obv_i.size() is " << obv_i.size() << ".");
                return false;
            }
            const Vec3 norm_xyz_i = Vec3(obv_i[0].rectified_norm_xy.x(), obv_i[0].rectified_norm_xy.y(), 1.0f);

            // Compute matrice B, C, D.
            const Mat3 skew_norm_xyz_i = Utility::SkewSymmetricMatrix(norm_xyz_i);
            const Mat3 skew_norm_xyz_r = Utility::SkewSymmetricMatrix(norm_xyz_r);
            const Mat3 R_cicl = R_wci.transpose() * R_wcl;
            const Mat3 R_crcl = R_wcr.transpose() * R_wcl;
            const Vec3 a_lr_tmp_t = Utility::SkewSymmetricMatrix(R_crcl * norm_xyz_l) * norm_xyz_r;
            const Mat1x3 a_lr_t = a_lr_tmp_t.transpose() * skew_norm_xyz_r;
            const Vec3 theta_lr_vector = skew_norm_xyz_r * R_crcl * norm_xyz_l;
            const float theta_lr = theta_lr_vector.squaredNorm();

            const Mat3 B = skew_norm_xyz_i * R_cicl * norm_xyz_l * a_lr_t * R_wcr.transpose();
            const Mat3 C = theta_lr * skew_norm_xyz_i * R_wci.transpose();
            const Mat3 D = - B - C;
            const Mat3 B_prime = B * R_cb;
            const Mat3 C_prime = C * R_cb;
            const Mat3 D_prime = D * R_cb;

            // Compute matrice S and time t.
            Vec3 S_1i = Vec3::Zero();
            Vec3 S_1r = Vec3::Zero();
            Vec3 S_1l = Vec3::Zero();
            float t_1i = 0.0f;
            float t_1r = 0.0f;
            float t_1l = 0.0f;

            if (frame_id_i != static_cast<int32_t>(feature.first_frame_id())) {
                const int32_t idx_of_imu = frame_id_i - feature.first_frame_id() - 1;
                S_1i = imu_blocks[idx_of_imu].p_ij() + R_wci * t_bc - t_bc;
                t_1i = imu_blocks[idx_of_imu].integrate_time_s();
            }
            if (frame_id_r != static_cast<int32_t>(feature.first_frame_id())) {
                const int32_t idx_of_imu = frame_id_r - feature.first_frame_id() - 1;
                S_1r = imu_blocks[idx_of_imu].p_ij() + R_wci * t_bc - t_bc;
                t_1r = imu_blocks[idx_of_imu].integrate_time_s();
            }
            if (frame_id_l != static_cast<int32_t>(feature.first_frame_id())) {
                const int32_t idx_of_imu = frame_id_l - feature.first_frame_id() - 1;
                S_1l = imu_blocks[idx_of_imu].p_ij() + R_wci * t_bc - t_bc;
                t_1l = imu_blocks[idx_of_imu].integrate_time_s();
            }

            Mat3x6 A_tmp;
            A_tmp.block<3, 3>(0, 0) = B_prime * t_1r + C_prime * t_1i + D_prime * t_1l;
            A_tmp.block<3, 3>(0, 3) = - (B_prime * t_1r * t_1r + C_prime * t_1i * t_1i + D_prime * t_1l * t_1l) * 0.5f * gravity_norm;
            const Vec3 b_tmp = - B_prime * S_1r - C_prime * S_1i - D_prime * S_1l;

            A += A_tmp.transpose() * A_tmp;
            b += A_tmp.transpose() * b_tmp;
            Q += b_tmp.transpose() * b_tmp;
        }
    }

    // Scale the LIGT function.
    const Mat3 A2tA2 = A.block<3, 3>(3, 3);
    const float mean = (A2tA2(0, 0) + A2tA2(1, 1) + A2tA2(2, 2)) / 3.0f;
    const float scale = 1.0f / mean;
    if (!std::isnan(scale)) {
        A *= scale;
        b *= scale;
        Q *= scale;
    }

    return true;
}

bool Backend::RefineGravityForInitialization(const Mat &M,
                                             const Vec &m,
                                             const float Q,
                                             const float gravity_mag,
                                             Vec &rhs) {

    const int32_t q = M.rows() - 3;
    Mat A = 2.0f * M.block(0, 0, q, q);

    Mat Bt = 2. * M.block(q, 0, 3, q);
    Mat BtAi = Bt * A.inverse();

    Mat3 D = 2. * M.block(q, q, 3, 3);
    Mat3 S = D - BtAi * Bt.transpose();

    Mat3 Sa = S.determinant() * S.inverse();
    Mat3 U = S.trace() * Mat3::Identity() - S;

    Vec3 v1 = BtAi * m.head(q);
    Vec3 m2 = m.tail<3>();

    Mat3 X;
    Vec3 Xm2;

    // X = I
    const float c4 = 16.0f * (v1.dot(v1) - 2.0f * v1.dot(m2) + m2.dot(m2));

    X = U;
    Xm2 = X * m2;
    const float c3 = 16.0f * (v1.dot(X * v1) - 2.0f * v1.dot(Xm2) + m2.dot(Xm2));

    X = 2.0f * Sa + U * U;
    Xm2 = X * m2;
    const float c2 = 4.0f * (v1.dot(X * v1) - 2.0f * v1.dot(Xm2) + m2.dot(Xm2));

    X = Sa * U + U * Sa;
    Xm2 = X * m2;
    const float c1 = 2.0f * (v1.dot(X * v1) - 2.0f * v1.dot(Xm2) + m2.dot(Xm2));

    X = Sa * Sa;
    Xm2 = X * m2;
    const float c0 = (v1.dot(X * v1) - 2.0f * v1.dot(Xm2) + m2.dot(Xm2));

    const float s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
    const float s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

    const float t1 = s00 + s11 + s22;
    const float t2 = s00 * s11 + s00 * s22 + s11 * s22
                        - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
    const float t3 = s00 * s11 * s22 + 2.0f * s01 * s02 * s12
                        - s00 * std::pow(s12, 2) - s11 * std::pow(s02, 2) - s22 * std::pow(s01, 2);

    Vec coeffs(7);
    coeffs << 64.0f,
              64.0f * t1,
              16.0f * (std::pow(t1, 2) + 2.0f * t2),
              16.0f * (t1 * t2 + t3),
              4.0f * (std::pow(t2, 2) + 2.0f * t1 * t3),
              4.0f * t3 * t2,
              std::pow(t3, 2);

    const float G2i = 1.0f / std::pow(gravity_mag, 2);

    coeffs(2) -= c4 * G2i;
    coeffs(3) -= c3 * G2i;
    coeffs(4) -= c2 * G2i;
    coeffs(5) -= c1 * G2i;
    coeffs(6) -= c0 * G2i;

    // Eigen::PolynomialSolver<float, Eigen::Dynamic> polynomial_solver;
    // polynomial_solver.compute(coeffs);
    // const Eigen::PolynomialSolver<float, Eigen::Dynamic>::RootsType &roots = polynomial_solver.roots();

    Eigen::VectorXd real, imag;
    if (!FindPolynomialRootsCompanionMatrix(coeffs.cast<double>(), &real, &imag)) {
        ReportError("[Backend] Backend failed to solve polynomial problem.");
        return false;
    }

    // Extract real roots.
    Vec real_roots = GetRealRoots(real, imag).cast<float>();
    if (real_roots.size() == 0) {
        ReportError("[Backend] Backend failed to find real roots.");
        return false;
    }

    Mat W(M.rows(), M.rows());
    W.setZero();
    W.block<3, 3>(q, q) = Mat3::Identity();

    Vec solution;
    float min_cost = std::numeric_limits<float>::max();
    for (Vec::Index i = 0; i < real_roots.size(); ++i) {
        const float lambda = real_roots(i);

        Eigen::FullPivLU<Mat> lu(2.0f * M + 2.0f * lambda * W);
        const Vec x_ = -lu.inverse() * m;

        float cost = x_.transpose() * M * x_;
        cost += m.transpose() * x_;
        cost += Q;
        if (cost < min_cost) {
            solution = x_;
            min_cost = cost;
        }
    }

    const float constraint = solution.transpose() * W * solution;

    if (constraint < 0.0f || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3f) {
        ReportWarn("[Backend] Constraint is " << constraint << ". Constraint error is " <<
            100.0f * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
    }

    rhs = solution;
    return true;
}

bool Backend::PropagateAllBasedOnFirstCameraFrameForInitializaion(const std::vector<ImuPreintegrateBlock<>> &imu_blocks,
                                                                  const Vec3 &v_i0i0,
                                                                  const Vec3 &gravity_i0) {
    // Localize the left camera extrinsic.
    const Quat q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Quat q_ci = q_ic.inverse();

    // Determine the scope of all frames.
    const uint32_t max_frames_idx = data_manager_->visual_local_map()->frames().back().id();
    const uint32_t min_frames_idx = data_manager_->visual_local_map()->frames().front().id();

    // Set states of first frame.
    auto first_frame = data_manager_->visual_local_map()->frame(min_frames_idx);
    first_frame->v_w() = q_ci * v_i0i0;
    const Vec3 p_c0c0 = first_frame->p_wc();
    const Quat q_c0c0 = first_frame->q_wc();
    const Vec3 v_c0c0 = first_frame->v_w();
    const Vec3 gracity_c0 = q_ci * gravity_i0;

    // Iterate all frames to propagate states based on frame c0.
    uint32_t idx_of_imu_block = 0;
    for (uint32_t i = min_frames_idx + 1; i <= max_frames_idx; ++i) {
        const auto &imu_preint_block = imu_blocks[idx_of_imu_block];
        ++idx_of_imu_block;

        const float &dt = imu_preint_block.integrate_time_s();
        const Quat &imu_q_ij = imu_preint_block.q_ij();
        const Vec3 &imu_p_ij = imu_preint_block.p_ij();
        const Vec3 &imu_v_ij = imu_preint_block.v_ij();

        auto frame = data_manager_->visual_local_map()->frame(i);
        frame->q_wc() = q_c0c0 * imu_q_ij;
        frame->p_wc() = q_ci * imu_p_ij + p_c0c0 + v_c0c0 * dt - 0.5f * gracity_c0 * dt * dt;
        frame->v_w() = q_ci * imu_v_ij + v_c0c0 - gracity_c0 * dt;
    }

    return true;
}

}
