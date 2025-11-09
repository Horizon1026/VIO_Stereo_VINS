#ifndef _GENERAL_EDGES_H_
#define _GENERAL_EDGES_H_

#include "basic_type.h"
#include "slam_basic_math.h"

#include "edge.h"
#include "kernel.h"
#include "kernel_cauchy.h"
#include "kernel_huber.h"
#include "kernel_tukey.h"
#include "vertex.h"
#include "vertex_quaternion.h"

namespace VIO {

/* Class Edge pose prior. This can be used to fix a pose with specified weight. */
template <typename Scalar>
class EdgePriorPose : public Edge<Scalar> {
    // Vertices are [position, p_wc]
    //              [rotation, q_wc]

public:
    EdgePriorPose()
        : Edge<Scalar>(6, 2) {}
    virtual ~EdgePriorPose() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        p_wc_ = this->GetVertex(0)->param();
        const TVec4<Scalar> &param = this->GetVertex(1)->param();
        q_wc_ = TQuat<Scalar>(param(0), param(1), param(2), param(3));

        // Get observation.
        obv_p_wc_ = this->observation().block(0, 0, 3, 1);
        const TVec4<Scalar> param_obv = this->observation().block(3, 0, 4, 1);
        obv_q_wc_ = TQuat<Scalar>(param_obv(0), param_obv(1), param_obv(2), param_obv(3));

        // Compute residual.
        this->residual().setZero(6);
        this->residual().head(3) = p_wc_ - obv_p_wc_;
        this->residual().tail(3) = static_cast<Scalar>(2) * (obv_q_wc_.inverse() * q_wc_).vec();
    }

    virtual void ComputeJacobians() override {
        TMat<Scalar> jacobian_p = TMat<Scalar>::Zero(6, 3);
        jacobian_p.block(0, 0, 3, 3).setIdentity();

        TMat<Scalar> jacobian_q = TMat<Scalar>::Zero(6, 3);
        jacobian_q.block(3, 0, 3, 3) = Utility::Qleft(obv_q_wc_.inverse() * q_wc_).template bottomRightCorner<3, 3>();

        this->GetJacobian(0) = jacobian_p;
        this->GetJacobian(1) = jacobian_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();

    TVec3<Scalar> obv_p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> obv_q_wc_ = TQuat<Scalar>::Identity();
};

}  // namespace VIO

#endif  // end of _GENERAL_EDGES_H_