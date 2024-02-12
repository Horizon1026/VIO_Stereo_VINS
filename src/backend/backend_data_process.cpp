#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

#include "geometry_epipolar.h"
#include "geometry_triangulation.h"
#include "geometry_pnp.h"

namespace VIO {

TMat2<DorF> Backend::GetVisualObserveInformationMatrix() {
    const auto &camera_model = visual_frontend_->camera_models().front();
    const DorF residual_in_pixel = 1.0;
    const TVec2<DorF> visual_observe_info_vec = TVec2<DorF>(camera_model->fx() * camera_model->fx(),
        camera_model->fy() * camera_model->fy()) / residual_in_pixel;
    return visual_observe_info_vec.asDiagonal();
}

void Backend::RecomputeImuPreintegrationBlock(const Vec3 &bias_accel,
                                              const Vec3 &bias_gyro,
                                              FrameWithBias &frame_with_bias) {
    frame_with_bias.imu_preint_block.Reset();
    frame_with_bias.imu_preint_block.bias_accel() = bias_accel;
    frame_with_bias.imu_preint_block.bias_gyro() = bias_gyro;
    frame_with_bias.imu_preint_block.SetImuNoiseSigma(imu_model_->options().kAccelNoise,
                                                      imu_model_->options().kGyroNoise,
                                                      imu_model_->options().kAccelRandomWalk,
                                                      imu_model_->options().kGyroRandomWalk);

    const uint32_t max_idx = frame_with_bias.packed_measure->imus.size();
    for (uint32_t i = 1; i < max_idx; ++i) {
        frame_with_bias.imu_preint_block.Propagate(*frame_with_bias.packed_measure->imus[i - 1], *frame_with_bias.packed_measure->imus[i]);
    }
}

bool Backend::SyncTwcToTwiInLocalMap() {
    if (data_manager_->camera_extrinsics().empty()) {
        ReportError("[Backend] Backend failed to sync states bases on imu frame.");
        return false;
    }

    // Extract camera extrinsics.
    const Quat &q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Vec3 &p_ic = data_manager_->camera_extrinsics().front().p_ic;

    // T_wi = T_wc * T_ic.inv.
    auto it = data_manager_->frames_with_bias().begin();
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        RETURN_FALSE_IF(it == data_manager_->frames_with_bias().end());
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(), p_ic, q_ic, it->p_wi, it->q_wi);
        ++it;
    }

    return true;
}

bool Backend::SyncTwiToTwcInLocalMap() {
    if (data_manager_->camera_extrinsics().empty()) {
        ReportError("[Backend] Backend failed to sync states bases on camera frame.");
        return false;
    }

    // Extract camera extrinsics.
    const Quat &q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Vec3 &p_ic = data_manager_->camera_extrinsics().front().p_ic;

    // T_wc = T_wi * T_ic
    auto it = data_manager_->frames_with_bias().cbegin();
    for (auto &frame : data_manager_->visual_local_map()->frames()) {
        RETURN_FALSE_IF(it == data_manager_->frames_with_bias().cend());
        Utility::ComputeTransformTransform(it->p_wi, it->q_wi, p_ic, q_ic, frame.p_wc(), frame.q_wc());
        ++it;
    }

    return true;
}

bool Backend::TryToSolveFramePoseByFeaturesObservedByItself(const int32_t frame_id,
                                                            const Vec3 &init_p_wc,
                                                            const Quat &init_q_wc) {
    auto frame_ptr = data_manager_->visual_local_map()->frame(frame_id);
    RETURN_FALSE_IF(frame_ptr == nullptr);
    RETURN_FALSE_IF(frame_ptr->features().empty());

    // Extract relative parameters.
    std::vector<Vec3> all_p_w;
    std::vector<Vec2> all_norm_xy;
    all_p_w.reserve(frame_ptr->features().size());
    all_norm_xy.reserve(frame_ptr->features().size());
    for (const auto &pair : frame_ptr->features()) {
        const auto &feature_ptr = pair.second;
        CONTINUE_IF(feature_ptr->status() != FeatureSolvedStatus::kSolved)
        all_p_w.emplace_back(feature_ptr->param());
        all_norm_xy.emplace_back(feature_ptr->observe(frame_id).front().rectified_norm_xy);
    }
    RETURN_FALSE_IF(all_p_w.size() < 3);

    // Try to estimate pnp problem.
    Vec3 p_wc = init_p_wc;
    Quat q_wc = init_q_wc;
    std::vector<uint8_t> status;
    using namespace VISION_GEOMETRY;
    PnpSolver solver;
    solver.options().kMethod = PnpSolver::PnpMethod::kHuber;
    RETURN_FALSE_IF(!solver.EstimatePose(all_p_w, all_norm_xy, q_wc, p_wc, status));

    frame_ptr->p_wc() = p_wc;
    frame_ptr->q_wc() = q_wc;

    return true;
}

bool Backend::TryToSolveFeaturePositionByFramesObservingIt(const int32_t feature_id,
                                                           const int32_t min_frame_id,
                                                           const int32_t max_frame_id,
                                                           const bool use_multi_view) {
    auto feature_ptr = data_manager_->visual_local_map()->feature(feature_id);
    RETURN_FALSE_IF(feature_ptr == nullptr);
    RETURN_FALSE_IF(feature_ptr->observes().size() < 2);
    RETURN_FALSE_IF(feature_ptr->observes().size() == 1 && feature_ptr->observes().front().size() < 2);

    using namespace VISION_GEOMETRY;
    Triangulator solver;
    solver.options().kMethod = Triangulator::TriangulationMethod::kAnalytic;

    std::vector<Quat> all_q_wc;
    std::vector<Vec3> all_p_wc;
    std::vector<Vec2> all_norm_xy;
    all_q_wc.reserve(max_frame_id - min_frame_id + 1);
    all_p_wc.reserve(max_frame_id - min_frame_id + 1);
    all_norm_xy.reserve(max_frame_id - min_frame_id + 1);

    // Extract all observations.
    const uint32_t max_observe_num = feature_ptr->observes().size();
    for (uint32_t id = 0; id < max_observe_num; ++id) {
        // Extract states of selected frame.
        const uint32_t frame_id = min_frame_id + id;
        const auto frame_ptr = data_manager_->visual_local_map()->frame(frame_id);
        RETURN_FALSE_IF(frame_ptr == nullptr);
        const Quat q_wc = frame_ptr->q_wc();
        const Vec3 p_wc = frame_ptr->p_wc();

        // Add mono-view observations.
        const auto &obv = feature_ptr->observe(frame_id);
        RETURN_FALSE_IF(obv.empty());
        const Vec2 norm_xy = obv[0].rectified_norm_xy;
        all_q_wc.emplace_back(q_wc);
        all_p_wc.emplace_back(p_wc);
        all_norm_xy.emplace_back(norm_xy);

        // Add multi-view observations.
        CONTINUE_IF(!use_multi_view);
        RETURN_FALSE_IF(data_manager_->camera_extrinsics().size() < obv.size());
        const Vec3 p_ic0 = data_manager_->camera_extrinsics()[0].p_ic;
        const Quat q_ic0 = data_manager_->camera_extrinsics()[0].q_ic;
        const Quat q_wi = q_wc * q_ic0.inverse();
        for (uint32_t i = 1; i < obv.size(); ++i) {
            const Vec3 p_ici = data_manager_->camera_extrinsics()[i].p_ic;
            const Quat q_ici = data_manager_->camera_extrinsics()[i].q_ic;
            // T_wci = T_wc0 * T_ic0.inv * T_ici.
            /*  [R_wci  t_wci] = [R_wc0  t_wc0] * [R_ic0.t  -R_ic0.t * t_ic0] * [R_ici  t_ici]
                [  0      1  ]   [  0      1  ]   [   0              1      ]   [  0      1  ]
                                = [R_wc0 * R_ic0.t  -R_wc0 * R_ic0.t * t_ic0 + t_wc0] * [R_ici  t_ici]
                                    [       0                        1                ]   [  0      1  ]
                                = [R_wc0 * R_ic0.t * R_ici  R_wc0 * R_ic0.t * t_ici - R_wc0 * R_ic0.t * t_ic0 + t_wc0]
                                    [           0                                          1                           ] */
            const Quat q_wci = q_wi * q_ici;
            const Vec3 p_wci = q_wi * p_ici - q_wi * p_ic0 + p_wc;
            const Vec2 norm_xy_i = obv[i].rectified_norm_xy;
            all_q_wc.emplace_back(q_wci);
            all_p_wc.emplace_back(p_wci);
            all_norm_xy.emplace_back(norm_xy_i);
        }
    }

    // Triangulize feature.
    Vec3 p_w = Vec3::Zero();
    if (solver.Triangulate(all_q_wc, all_p_wc, all_norm_xy, p_w)) {
        feature_ptr->param() = p_w;
        feature_ptr->status() = FeatureSolvedStatus::kSolved;
    } else {
        feature_ptr->status() = FeatureSolvedStatus::kUnsolved;
    }

    return true;
}

bool Backend::AddNewestFrameWithStatesPredictionToLocalMap() {
    // Check validation. Frames_with_bias must have one more frame than visual_local_map.
    if (data_manager_->visual_local_map()->frames().size() + 1 != data_manager_->frames_with_bias().size()) {
        ReportError("[Backend] Size of frames in local map and in frames_with_bias is not match. [" <<
            data_manager_->visual_local_map()->frames().size() + 1 << "] != [" <<
            data_manager_->frames_with_bias().size() << "].");
        return false;
    }

    // Extract newest frame with bias.
    auto &newest_frame_with_bias = data_manager_->frames_with_bias().back();
    if (newest_frame_with_bias.visual_measure == nullptr) {
        ReportError("[Backend] Backend find newest_frame_with_bias.visual_measure to be nullptr.");
        return false;
    }

    // Preintegrate newest imu measurements.
    auto it = std::next(data_manager_->frames_with_bias().rbegin());
    if (it == data_manager_->frames_with_bias().rend()) {
        ReportError("[Backend] Backend failed to extract subnew frame with bias.");
        return false;
    }
    auto &subnew_frame_with_bias = *it;
    const Vec3 &bias_accel = subnew_frame_with_bias.imu_preint_block.bias_accel();
    const Vec3 &bias_gyro = subnew_frame_with_bias.imu_preint_block.bias_gyro();
    RecomputeImuPreintegrationBlock(bias_accel, bias_gyro, newest_frame_with_bias);

    // Predict pose and velocity of newest frame based on imu frame.
    const float dt = newest_frame_with_bias.imu_preint_block.integrate_time_s();
    newest_frame_with_bias.p_wi = subnew_frame_with_bias.q_wi * newest_frame_with_bias.imu_preint_block.p_ij() +
        subnew_frame_with_bias.p_wi + subnew_frame_with_bias.v_wi * dt - 0.5f * options_.kGravityInWordFrame * dt * dt;
    newest_frame_with_bias.q_wi = subnew_frame_with_bias.q_wi * newest_frame_with_bias.imu_preint_block.q_ij();
    newest_frame_with_bias.v_wi = subnew_frame_with_bias.q_wi * newest_frame_with_bias.imu_preint_block.v_ij() +
        subnew_frame_with_bias.v_wi - options_.kGravityInWordFrame * dt;

    // Add new frame into visual_local_map.
    std::vector<MatImg> raw_images;
    if (options_.kEnableLocalMapStoreRawImages && newest_frame_with_bias.packed_measure != nullptr) {
        if (newest_frame_with_bias.packed_measure->left_image != nullptr) {
            raw_images.emplace_back(newest_frame_with_bias.packed_measure->left_image->image);
        }
        if (newest_frame_with_bias.packed_measure->right_image != nullptr) {
            raw_images.emplace_back(newest_frame_with_bias.packed_measure->right_image->image);
        }
    }
    const auto &newest_cam_frame_id = data_manager_->visual_local_map()->frames().back().id() + 1;
    data_manager_->visual_local_map()->AddNewFrameWithFeatures(newest_frame_with_bias.visual_measure->features_id,
                                                               newest_frame_with_bias.visual_measure->observes_per_frame,
                                                               newest_frame_with_bias.time_stamp_s,
                                                               newest_cam_frame_id, raw_images);

    // Sync imu pose to camera pose.
    const Quat &q_ic = data_manager_->camera_extrinsics().front().q_ic;
    const Vec3 &p_ic = data_manager_->camera_extrinsics().front().p_ic;
    auto &newest_cam_frame = data_manager_->visual_local_map()->frames().back();
    Utility::ComputeTransformTransform(newest_frame_with_bias.p_wi, newest_frame_with_bias.q_wi,
        p_ic, q_ic, newest_cam_frame.p_wc(), newest_cam_frame.q_wc());

    // Try to triangulize newest frame for better pose estimation.
    // Eliminate the drift of prediction from only imu.
    TryToSolveFramePoseByFeaturesObservedByItself(newest_cam_frame.id(), newest_cam_frame.p_wc(), newest_cam_frame.q_wc());

    // Try to triangulize all new features observed in newest frame.
    for (auto &pair : newest_cam_frame.features()) {
        const auto &feature_id = pair.first;
        const auto &feature_ptr = pair.second;
        CONTINUE_IF(feature_ptr->status() == FeatureSolvedStatus::kSolved);
        TryToSolveFeaturePositionByFramesObservingIt(feature_id, feature_ptr->first_frame_id(),
            feature_ptr->final_frame_id(), true);
    }

    return true;
}

bool Backend::ControlSizeOfLocalMap() {
    RETURN_TRUE_IF(!status_.is_initialized);

    // Remove frames according to marginalization type.
    switch (status_.marginalize_type) {
        case BackendMarginalizeType::kMarginalizeOldestFrame: {
            // Remove oldest frame in visual_local_map and frames_with_bias.
            const auto oldest_frame_id = data_manager_->visual_local_map()->frames().front().id();
            data_manager_->visual_local_map()->RemoveFrame(oldest_frame_id);
            data_manager_->frames_with_bias().pop_front();
            break;
        }

        case BackendMarginalizeType::kMarginalizeSubnewFrame: {
            // Merge imu measurements relative to newest and subnew frame.
            auto it = data_manager_->frames_with_bias().rbegin();
            auto &newest_frame = *it;
            ++it;
            auto &subnew_frame = *it;
            // Make subnew_frame to be neweset_frame.
            subnew_frame.time_stamp_s = newest_frame.time_stamp_s;
            subnew_frame.p_wi = newest_frame.p_wi;
            subnew_frame.q_wi = newest_frame.q_wi;
            subnew_frame.v_wi = newest_frame.v_wi;
            subnew_frame.visual_measure = std::move(newest_frame.visual_measure);
            subnew_frame.packed_measure->left_image = std::move(newest_frame.packed_measure->left_image);
            subnew_frame.packed_measure->right_image = std::move(newest_frame.packed_measure->right_image);
            // Process imu measurements.
            const uint32_t max_idx = newest_frame.packed_measure->imus.size();
            if (subnew_frame.imu_preint_block.integrate_time_s() + newest_frame.imu_preint_block.integrate_time_s() >=
                data_manager_->options().kMaxValidImuPreintegrationBlockTimeInSecond) {
                // If integration time is too long, only integrate the incremental part.
                for (uint32_t i = 1; i < max_idx; ++i) {
                    // Only integration the new part of imu measurements.
                    subnew_frame.imu_preint_block.Propagate(*newest_frame.packed_measure->imus[i - 1], *newest_frame.packed_measure->imus[i]);
                }
                for (uint32_t i = 1; i < max_idx; ++i) {
                    // Merge imu measurements. Skip the first imu measurements since it is the same.
                    subnew_frame.packed_measure->imus.emplace_back(std::move(newest_frame.packed_measure->imus[i]));
                }
            } else {
                // If integration time is short, integrate totally for better performance.
                for (uint32_t i = 1; i < max_idx; ++i) {
                    subnew_frame.packed_measure->imus.emplace_back(std::move(newest_frame.packed_measure->imus[i]));
                }
                RecomputeImuPreintegrationBlock(newest_frame.imu_preint_block.bias_accel(),
                                                newest_frame.imu_preint_block.bias_gyro(),
                                                subnew_frame);
            }

            // Remove subnew frame in visual_local_map and frames_with_bias.
            const auto subnew_frame_id = data_manager_->visual_local_map()->frames().back().id() - 1;
            data_manager_->visual_local_map()->RemoveFrame(subnew_frame_id);
            data_manager_->frames_with_bias().pop_back();

            break;
        }

        default:
        case BackendMarginalizeType::kNotMarginalize: {
            break;
        }
    }

    // Remove useless features.
    std::vector<uint32_t> features_id;
    features_id.reserve(100);
    const auto &newest_keyframe_id = data_manager_->visual_local_map()->frames().back().id() - 2;
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
        const auto &feature = pair.second;
        // Remove features that has no observations.
        if (feature.observes().empty()) {
            features_id.emplace_back(feature.id());
        }
        // Remove features that cannot has more observations.
        if (feature.observes().size() < 2 && feature.final_frame_id() < newest_keyframe_id) {
            features_id.emplace_back(feature.id());
        }
        // Remove features that has been marginalized.
        if (feature.status() == FeatureSolvedStatus::kMarginalized) {
            features_id.emplace_back(feature.id());
        }
    }
    // Remove selected features.
    for (const auto &id : features_id) {
        data_manager_->visual_local_map()->RemoveFeature(id);
    }

    // Check validation of visual local map.
    RETURN_FALSE_IF(!data_manager_->visual_local_map()->SelfCheck());

    return true;
}

void Backend::UpdateBackendStates() {
    states_.motion.time_stamp_s = data_manager_->frames_with_bias().empty() ? 0.0f :
        data_manager_->frames_with_bias().back().time_stamp_s;

    if (!status_.is_initialized) {
        states_.prior.is_valid = false;
        states_.motion.p_wi.setZero();
        states_.motion.q_wi.setIdentity();
        states_.motion.v_wi.setZero();
        states_.motion.ba.setZero();
        states_.motion.bg.setZero();
        return;
    }

    const auto &newest_frame_with_bias = data_manager_->frames_with_bias().back();
    states_.motion.p_wi = newest_frame_with_bias.p_wi;
    states_.motion.q_wi = newest_frame_with_bias.q_wi;
    states_.motion.v_wi = newest_frame_with_bias.v_wi;
    states_.motion.ba = newest_frame_with_bias.imu_preint_block.bias_accel();
    states_.motion.bg = newest_frame_with_bias.imu_preint_block.bias_gyro();
}

bool Backend::SelectKeyframesIntoGlobalMap() {
    RETURN_TRUE_IF(!status_.is_initialized);
    RETURN_TRUE_IF(!options_.kEnableRecordGlobalMap);
    RETURN_TRUE_IF(data_manager_->visual_local_map()->frames().empty());

    bool add_oldest_one_into_global_map = data_manager_->global_map_keyframes().empty();
    if (!add_oldest_one_into_global_map) {
        add_oldest_one_into_global_map = (data_manager_->visual_local_map()->frames().back().p_wc() -
            data_manager_->global_map_keyframes().back().p_wc).norm() > 3.0f;
    }
    RETURN_TRUE_IF(!add_oldest_one_into_global_map);

    const auto &newest_frame = data_manager_->visual_local_map()->frames().back();
    data_manager_->global_map_keyframes().emplace_back(GlobalMapKeyframe{});
    auto &new_keyframe = data_manager_->global_map_keyframes().back();
    new_keyframe.time_stamp_s = newest_frame.time_stamp_s();
    new_keyframe.p_wc = newest_frame.p_wc();
    new_keyframe.q_wc = newest_frame.q_wc();

    for (const auto &pair : newest_frame.features()) {
        const auto &feature_ptr = pair.second;
        CONTINUE_IF(feature_ptr->status() != FeatureSolvedStatus::kSolved);
        new_keyframe.points.emplace_back(GlobalMapPoint{
            .p_c = newest_frame.q_wc().inverse() * (feature_ptr->param() - newest_frame.p_wc()),
        });
    }

    ReportColorInfo("[Backend] New keyframe with " << new_keyframe.points.size() << " points is added into global map.");
    return true;
}

}
