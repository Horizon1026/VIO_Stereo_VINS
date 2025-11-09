#ifndef _VISUAL_INERTIAL_EDGES_H_
#define _VISUAL_INERTIAL_EDGES_H_

#include "basic_type.h"
#include "slam_basic_math.h"

#include "edge.h"
#include "kernel.h"
#include "vertex.h"
#include "vertex_quaternion.h"

namespace VIO {

/* Class Edge reprojection. Project feature 1-dof invdep on visual norm plane via imu pose. */
template <typename Scalar>
class EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera : public Edge<Scalar> {
    // Vertices are [feature, invdep]
    //              [first imu pose, p_wi0]
    //              [first imu pose, q_wi0]
    //              [imu pose, p_wi]
    //              [imu pose, q_wi]
    //              [extrinsic, p_ic]
    //              [extrinsic, q_ic]

public:
    EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera()
        : Edge<Scalar>(2, 7) {}
    virtual ~EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract states.
        inv_depth0_ = this->GetVertex(0)->param()(0);
        p_wi0_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i0 = this->GetVertex(2)->param();
        q_wi0_ = TQuat<Scalar>(param_i0(0), param_i0(1), param_i0(2), param_i0(3));
        p_wi_ = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(4)->param();
        q_wi_ = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_ic_ = this->GetVertex(5)->param();
        const TVec4<Scalar> &param_ic = this->GetVertex(6)->param();
        q_ic_ = TQuat<Scalar>(param_ic(0), param_ic(1), param_ic(2), param_ic(3));

        // Extract observations.
        norm_xy0_ = this->observation().block(0, 0, 2, 1);
        norm_xy_ = this->observation().block(2, 0, 2, 1);

        // Compute projection.
        p_c0_ = TVec3<Scalar>(norm_xy0_(0), norm_xy0_(1), static_cast<Scalar>(1)) / inv_depth0_;
        p_i0_ = q_ic_ * p_c0_ + p_ic_;
        p_w_ = q_wi0_ * p_i0_ + p_wi0_;
        p_i_ = q_wi_.inverse() * (p_w_ - p_wi_);
        p_c_ = q_ic_.inverse() * (p_i_ - p_ic_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        if (std::isinf(inv_depth_) || std::isnan(inv_depth_) || inv_depth_ < kZeroFloat) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_) && inv_depth_ > kZeroFloat) {
            const Scalar inv_depth_2 = inv_depth_ * inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, -p_c_(0) * inv_depth_2, 0, inv_depth_, -p_c_(1) * inv_depth_2;
        }

        const TQuat<Scalar> q_ci = q_ic_.inverse();
        const TQuat<Scalar> q_cw = q_ci * q_wi_.inverse();
        const TQuat<Scalar> q_ci0 = q_cw * q_wi0_;
        const TQuat<Scalar> q_cc0 = q_ci0 * q_ic_;

        const TMat3<Scalar> R_ci = q_ci.toRotationMatrix();
        const TMat3<Scalar> R_cw = q_cw.toRotationMatrix();
        const TMat3<Scalar> R_ci0 = q_ci0.toRotationMatrix();
        const TMat3<Scalar> R_cc0 = q_cc0.toRotationMatrix();

        const TMat3<Scalar> jacobian_cam0_p = R_cw;
        const TMat3<Scalar> jacobian_cam0_q = -R_ci0 * Utility::SkewSymmetricMatrix(p_i0_);

        const TMat3<Scalar> jacobian_cam_p = -R_cw;
        const TMat3<Scalar> jacobian_cam_q = R_ci * Utility::SkewSymmetricMatrix(p_i_);

        const TVec3<Scalar> jacobian_invdep = -R_cc0 * TVec3<Scalar>(norm_xy0_.x(), norm_xy0_.y(), static_cast<Scalar>(1)) / (inv_depth0_ * inv_depth0_);

        const TMat3<Scalar> jacobian_ex_p = R_ci * ((q_wi_.inverse() * q_wi0_).matrix() - TMat3<Scalar>::Identity());
        const TMat3<Scalar> jacobian_ex_q = -R_cc0 * Utility::SkewSymmetricMatrix(p_c0_) + Utility::SkewSymmetricMatrix(R_cc0 * p_c0_) +
                                            Utility::SkewSymmetricMatrix(q_ic_.inverse() * (q_wi_.inverse() * (q_wi0_ * p_ic_ + p_wi0_ - p_wi_) - p_ic_));

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cam0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cam0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_cam_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_cam_q;
        this->GetJacobian(5) = jacobian_2d_3d * jacobian_ex_p;
        this->GetJacobian(6) = jacobian_2d_3d * jacobian_ex_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc0_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi0_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy0_ = TVec2<Scalar>::Zero();
    Scalar inv_depth0_ = 0;
    TVec3<Scalar> p_i0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c0_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy_ = TVec2<Scalar>::Zero();
    Scalar inv_depth_ = 0;
    TVec3<Scalar> p_i_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_ic_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic_ = TQuat<Scalar>::Identity();
};

/* Class Edge reprojection. Project feature 1-dof invdep on visual norm plane via imu pose. */
template <typename Scalar>
class EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera : public Edge<Scalar> {
    // Vertices are [feature, invdep]
    //              [first imu pose, p_wi0]
    //              [first imu pose, q_wi0]
    //              [imu pose, p_wi]
    //              [imu pose, q_wi]
    //              [extrinsic, p_ic0]
    //              [extrinsic, q_ic0]
    //              [extrinsic, p_ic]
    //              [extrinsic, q_ic]

public:
    EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera()
        : Edge<Scalar>(2, 9) {}
    virtual ~EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesTwoCamera() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract states.
        inv_depth0_ = this->GetVertex(0)->param()(0);
        p_wi0_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i0 = this->GetVertex(2)->param();
        q_wi0_ = TQuat<Scalar>(param_i0(0), param_i0(1), param_i0(2), param_i0(3));
        p_wi_ = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(4)->param();
        q_wi_ = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_ic0_ = this->GetVertex(5)->param();
        const TVec4<Scalar> &param_ic0 = this->GetVertex(6)->param();
        q_ic0_ = TQuat<Scalar>(param_ic0(0), param_ic0(1), param_ic0(2), param_ic0(3));
        p_ic_ = this->GetVertex(7)->param();
        const TVec4<Scalar> &param_ic = this->GetVertex(8)->param();
        q_ic_ = TQuat<Scalar>(param_ic(0), param_ic(1), param_ic(2), param_ic(3));

        // Extract observations.
        norm_xy0_ = this->observation().block(0, 0, 2, 1);
        norm_xy_ = this->observation().block(2, 0, 2, 1);

        // Compute projection.
        p_c0_ = TVec3<Scalar>(norm_xy0_(0), norm_xy0_(1), static_cast<Scalar>(1)) / inv_depth0_;
        p_i0_ = q_ic0_ * p_c0_ + p_ic0_;
        p_w_ = q_wi0_ * p_i0_ + p_wi0_;
        p_i_ = q_wi_.inverse() * (p_w_ - p_wi_);
        p_c_ = q_ic_.inverse() * (p_i_ - p_ic_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        if (std::isinf(inv_depth_) || std::isnan(inv_depth_) || inv_depth_ < kZeroFloat) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_) && inv_depth_ > kZeroFloat) {
            const Scalar inv_depth_2 = inv_depth_ * inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, -p_c_(0) * inv_depth_2, 0, inv_depth_, -p_c_(1) * inv_depth_2;
        }

        const TQuat<Scalar> q_ci = q_ic_.inverse();
        const TQuat<Scalar> q_cw = q_ci * q_wi_.inverse();
        const TQuat<Scalar> q_ci0 = q_cw * q_wi0_;
        const TQuat<Scalar> q_cc0 = q_ci0 * q_ic0_;
        const TMat3<Scalar> R_ci = q_ci.toRotationMatrix();
        const TMat3<Scalar> R_cw = q_cw.toRotationMatrix();
        const TMat3<Scalar> R_ci0 = q_ci0.toRotationMatrix();
        const TMat3<Scalar> R_cc0 = q_cc0.toRotationMatrix();

        const TMat3<Scalar> jacobian_cam0_p = R_cw;
        const TMat3<Scalar> jacobian_cam0_q = -R_ci0 * Utility::SkewSymmetricMatrix(p_i0_);

        const TMat3<Scalar> jacobian_cam_p = -R_cw;
        const TMat3<Scalar> jacobian_cam_q = R_ci * Utility::SkewSymmetricMatrix(p_i_);

        const TVec3<Scalar> jacobian_invdep = -R_cc0 * TVec3<Scalar>(norm_xy0_.x(), norm_xy0_.y(), static_cast<Scalar>(1)) / (inv_depth0_ * inv_depth0_);

        const TMat3<Scalar> jacobian_ex0_p = R_ci0;
        const TMat3<Scalar> jacobian_ex0_q = -R_cc0 * Utility::SkewSymmetricMatrix(p_c0_);

        const TMat3<Scalar> jacobian_ex_p = -R_ci;
        const TMat3<Scalar> jacobian_ex_q = Utility::SkewSymmetricMatrix(p_c_);

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cam0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cam0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_cam_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_cam_q;
        this->GetJacobian(5) = jacobian_2d_3d * jacobian_ex0_p;
        this->GetJacobian(6) = jacobian_2d_3d * jacobian_ex0_q;
        this->GetJacobian(7) = jacobian_2d_3d * jacobian_ex_p;
        this->GetJacobian(8) = jacobian_2d_3d * jacobian_ex_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc0_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi0_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy0_ = TVec2<Scalar>::Zero();
    Scalar inv_depth0_ = 0;
    TVec3<Scalar> p_i0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_ic0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic0_ = TQuat<Scalar>::Identity();

    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy_ = TVec2<Scalar>::Zero();
    Scalar inv_depth_ = 0;
    TVec3<Scalar> p_i_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_ic_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic_ = TQuat<Scalar>::Identity();

    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();
};

/* Class Edge reprojection. Project feature 1-dof invdep on visual norm plane via imu pose. */
template <typename Scalar>
class EdgeFeatureInvdepToNormPlaneViaImuWithinOneFramesTwoCamera : public Edge<Scalar> {
    // Vertices are [feature, invdep]
    //              [extrinsic, p_ic0]
    //              [extrinsic, q_ic0]
    //              [extrinsic, p_ic]
    //              [extrinsic, q_ic]

public:
    EdgeFeatureInvdepToNormPlaneViaImuWithinOneFramesTwoCamera()
        : Edge<Scalar>(2, 5) {}
    virtual ~EdgeFeatureInvdepToNormPlaneViaImuWithinOneFramesTwoCamera() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract states.
        inv_depth0_ = this->GetVertex(0)->param()(0);
        p_ic0_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_ic0 = this->GetVertex(2)->param();
        q_ic0_ = TQuat<Scalar>(param_ic0(0), param_ic0(1), param_ic0(2), param_ic0(3));
        p_ic_ = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_ic = this->GetVertex(4)->param();
        q_ic_ = TQuat<Scalar>(param_ic(0), param_ic(1), param_ic(2), param_ic(3));

        // Extract observations.
        norm_xy0_ = this->observation().block(0, 0, 2, 1);
        norm_xy_ = this->observation().block(2, 0, 2, 1);

        // Compute projection.
        p_c0_ = TVec3<Scalar>(norm_xy0_(0), norm_xy0_(1), static_cast<Scalar>(1)) / inv_depth0_;
        p_i_ = q_ic0_ * p_c0_ + p_ic0_;
        p_c_ = q_ic_.inverse() * (p_i_ - p_ic_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        if (std::isinf(inv_depth_) || std::isnan(inv_depth_) || inv_depth_ < kZeroFloat) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_) && inv_depth_ > kZeroFloat) {
            const Scalar inv_depth_2 = inv_depth_ * inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, -p_c_(0) * inv_depth_2, 0, inv_depth_, -p_c_(1) * inv_depth_2;
        }

        const TQuat<Scalar> q_ci = q_ic_.inverse();
        const TQuat<Scalar> q_cc0 = q_ci * q_ic0_;
        const TMat3<Scalar> R_ci = q_ci.toRotationMatrix();
        const TMat3<Scalar> R_cc0 = q_cc0.toRotationMatrix();

        const TVec3<Scalar> jacobian_invdep = -R_cc0 * TVec3<Scalar>(norm_xy0_.x(), norm_xy0_.y(), static_cast<Scalar>(1)) / (inv_depth0_ * inv_depth0_);

        const TMat3<Scalar> jacobian_ex0_p = R_ci;
        const TMat3<Scalar> jacobian_ex0_q = -R_cc0 * Utility::SkewSymmetricMatrix(p_c0_);

        const TMat3<Scalar> jacobian_ex_p = -R_ci;
        const TMat3<Scalar> jacobian_ex_q = Utility::SkewSymmetricMatrix(p_c_);

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_ex0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_ex0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_ex_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_ex_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec2<Scalar> norm_xy0_ = TVec2<Scalar>::Zero();
    Scalar inv_depth0_ = 0;
    TVec3<Scalar> p_c0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_ic0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic0_ = TQuat<Scalar>::Identity();

    TVec2<Scalar> norm_xy_ = TVec2<Scalar>::Zero();
    Scalar inv_depth_ = 0;
    TVec3<Scalar> p_i_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_ic_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic_ = TQuat<Scalar>::Identity();
};

}  // namespace VIO

#endif  // end of _VISUAL_INERTIAL_EDGES_H_
