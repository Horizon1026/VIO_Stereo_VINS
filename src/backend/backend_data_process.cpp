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
    solver.options().kMethod = PnpSolver::PnpMethod::kRansac;
    RETURN_FALSE_IF(!solver.EstimatePose(all_p_w, all_norm_xy, q_wc, p_wc, status));

    frame_ptr->p_wc() = p_wc;
    frame_ptr->q_wc() = q_wc;

    return true;
}

bool Backend::TryToSolveFeaturePositionByFramesObservingIt(const int32_t feature_id,
                                                           const int32_t min_frame_id,
                                                           const int32_t max_frame_id,
                                                           const bool use_multi_view) {

    return true;
}

}
