#ifndef _VISUAL_VERTICES_H_
#define _VISUAL_VERTICES_H_

#include "vertex.h"

#include "algorithm"

namespace vio {

/* Optimize log(invdep), with invdep = exp(log(invdep)). */
template <typename Scalar>
class VertexFeatureLogInvdep: public slam_solver::Vertex<Scalar> {

public:
    explicit VertexFeatureLogInvdep(const Scalar min_depth): slam_solver::Vertex<Scalar>(1, 1), max_log_invdep_(-std::log(min_depth)) {}
    virtual ~VertexFeatureLogInvdep() = default;

    virtual void UpdateParam(const TVec<Scalar> &delta_param) override {
        // Enforce minimum depth without imposing a maximum depth.
        const Scalar log_invdep = std::min(max_log_invdep_, this->param()(0) + delta_param(0));
        const Scalar invdep = std::exp(log_invdep);
        if (invdep > static_cast<Scalar>(0) && std::isfinite(static_cast<Scalar>(1) / invdep)) {
            this->param()(0) = log_invdep;
        }
    }

private:
    Scalar max_log_invdep_;
};

}  // namespace vio

#endif  // end of _VISUAL_VERTICES_H_
