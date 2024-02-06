#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

#include "geometry_epipolar.h"
#include "geometry_triangulation.h"
#include "geometry_pnp.h"

namespace VIO {

bool Backend::TryToSolveFramePoseByFeaturesObservedByItself(const int32_t frame_id,
                                                            const Vec3 init_p_wc,
                                                            const Quat init_q_wc) {
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

}
