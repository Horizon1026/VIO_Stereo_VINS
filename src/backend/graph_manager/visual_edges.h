#ifndef _VISUAL_EDGES_H_
#define _VISUAL_EDGES_H_

#include "basic_type.h"
#include "edge.h"
#include "slam_basic_math.h"
#include "vertex.h"
#include "vertex_quaternion.h"

namespace vio {

/* Class Edge reprojection. Project feature 3-dof position on visual norm plane. */
template <typename Scalar>
class EdgeFeaturePosToNormPlane: public slam_solver::Edge<Scalar> {
    // Vertices are [feature, p_w]
    //              [camera, p_wc]
    //              [camera, q_wc]

public:
    EdgeFeaturePosToNormPlane(): slam_solver::Edge<Scalar>(2, 3) {}
    virtual ~EdgeFeaturePosToNormPlane() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Compute prediction.
        p_w_ = this->GetVertex(0)->param();
        p_wc_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param = this->GetVertex(2)->param();
        q_wc_ = TQuat<Scalar>(param(0), param(1), param(2), param(3));
        p_c_ = q_wc_.inverse() * (p_w_ - p_wc_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        // Get observation.
        pixel_norm_xy_ = this->observation();

        // Compute residual.
        if (std::isinf(inv_depth_) || std::isnan(inv_depth_)) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - pixel_norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_)) {
            const Scalar inv_depth_2 = inv_depth_ * inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, -p_c_(0) * inv_depth_2, 0, inv_depth_, -p_c_(1) * inv_depth_2;
        }

        this->GetJacobian(0) = jacobian_2d_3d * (q_wc_.inverse().matrix());
        this->GetJacobian(1) = -this->GetJacobian(0);
        this->GetJacobian(2) = jacobian_2d_3d * slam_utility::Utility::Utility::SkewSymmetricMatrix(p_c_);
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> pixel_norm_xy_ = TVec2<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();
    Scalar inv_depth_ = 0;
};


/* Class Edge reprojection. Project feature 3-dof position on visual unit sphere. */
template <typename Scalar>
class EdgeFeaturePosToUnitSphere: public slam_solver::Edge<Scalar> {
    // Vertices are [feature, p_w]
    //              [camera, p_wc]
    //              [camera, q_wc]

public:
    EdgeFeaturePosToUnitSphere(): slam_solver::Edge<Scalar>(2, 3) {}
    virtual ~EdgeFeaturePosToUnitSphere() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Compute prediction.
        p_w_ = this->GetVertex(0)->param();
        p_wc_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param = this->GetVertex(2)->param();
        q_wc_ = TQuat<Scalar>(param(0), param(1), param(2), param(3));
        p_c_ = q_wc_.inverse() * (p_w_ - p_wc_);

        // Get observation.
        obv_norm_xy_ = this->observation();
        const TVec3<Scalar> obv_p_c = TVec3<Scalar>(obv_norm_xy_.x(), obv_norm_xy_.y(), 1.0f);

        // Compute residual.
        this->residual() = tangent_base_transpose_ * (p_c_.normalized() - obv_p_c.normalized());
    }

    virtual void ComputeJacobians() override {
        const Scalar p_c_norm = p_c_.norm();
        const Scalar p_c_norm3 = p_c_norm * p_c_norm * p_c_norm;
        TMat3<Scalar> jacobian_norm = TMat3<Scalar>::Zero();
        jacobian_norm << 1.0 / p_c_norm - p_c_.x() * p_c_.x() / p_c_norm3, -p_c_.x() * p_c_.y() / p_c_norm3, -p_c_.x() * p_c_.z() / p_c_norm3,
            -p_c_.x() * p_c_.y() / p_c_norm3, 1.0 / p_c_norm - p_c_.y() * p_c_.y() / p_c_norm3, -p_c_.y() * p_c_.z() / p_c_norm3,
            -p_c_.x() * p_c_.z() / p_c_norm3, -p_c_.y() * p_c_.z() / p_c_norm3, 1.0 / p_c_norm - p_c_.z() * p_c_.z() / p_c_norm3;

        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        jacobian_2d_3d = tangent_base_transpose_ * jacobian_norm;

        this->GetJacobian(0) = jacobian_2d_3d * (q_wc_.inverse().matrix());
        this->GetJacobian(1) = -this->GetJacobian(0);
        this->GetJacobian(2) = jacobian_2d_3d * slam_utility::Utility::SkewSymmetricMatrix(p_c_);
    }

    // Set tangent base.
    void SetTrangetBase(const TVec3<Scalar> &vec) { tangent_base_transpose_ = Utility::TangentBase(vec).transpose(); }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> obv_norm_xy_ = TVec2<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();
    TMat2x3<Scalar> tangent_base_transpose_;
};

/* Class Edge reprojection. Project feature 1-dof invdep on visual norm plane. */
template <typename Scalar>
class EdgeFeatureInvdepToNormPlane: public slam_solver::Edge<Scalar> {
    // Vertices are [feature, log(invdep)]
    //              [first camera, p_wc0]
    //              [first camera, q_wc0]
    //              [camera, p_wc]
    //              [camera, q_wc]

public:
    EdgeFeatureInvdepToNormPlane(): slam_solver::Edge<Scalar>(2, 5) {}
    virtual ~EdgeFeatureInvdepToNormPlane() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        inv_depth0_ = std::exp(this->GetVertex(0)->param()(0));
        p_wc0_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(2)->param();
        q_wc0_ = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_wc_ = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_j = this->GetVertex(4)->param();
        q_wc_ = TQuat<Scalar>(param_j(0), param_j(1), param_j(2), param_j(3));

        norm_xy0_ = this->observation().block(0, 0, 2, 1);
        norm_xy_ = this->observation().block(2, 0, 2, 1);

        // Scale all 3D coordinates by anchor inverse depth to avoid overflow for distant points.
        p_c0_ = TVec3<Scalar>(norm_xy0_(0), norm_xy0_(1), static_cast<Scalar>(1));
        p_w_ = q_wc0_ * p_c0_ + inv_depth0_ * p_wc0_;
        p_c_ = q_wc_.inverse() * (p_w_ - inv_depth0_ * p_wc_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        if (std::isinf(inv_depth_) || std::isnan(inv_depth_)) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_)) {
            const Scalar inv_depth_2 = inv_depth_ * inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, -p_c_(0) * inv_depth_2, 0, inv_depth_, -p_c_(1) * inv_depth_2;
        }

        const TMat3<Scalar> R_cw = q_wc_.toRotationMatrix().transpose();
        const TMat3<Scalar> R_cc0 = R_cw * q_wc0_.matrix();

        const TMat3<Scalar> jacobian_cam0_q = -R_cc0 * Utility::SkewSymmetricMatrix(p_c0_);
        const TMat3<Scalar> jacobian_cam0_p = inv_depth0_ * R_cw;

        const TMat3<Scalar> jacobian_cam_q = Utility::SkewSymmetricMatrix(p_c_);
        const TMat3<Scalar> jacobian_cam_p = -inv_depth0_ * R_cw;

        const TVec3<Scalar> jacobian_invdep = inv_depth0_ * R_cw * (p_wc0_ - p_wc_);

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cam0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cam0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_cam_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_cam_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc0_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy0_ = TVec2<Scalar>::Zero();
    Scalar inv_depth0_ = 0;
    TVec3<Scalar> p_c0_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy_ = TVec2<Scalar>::Zero();
    Scalar inv_depth_ = 0;
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();
};

/* Class Edge relative camera translation expressed in the reference camera frame. */
template <typename Scalar>
class EdgeCameraRelativePosition: public slam_solver::Edge<Scalar> {
    // Vertices are [extrinsic, p_ic0], [extrinsic, q_ic0], [extrinsic, p_ic1].

public:
    EdgeCameraRelativePosition(): slam_solver::Edge<Scalar>(3, 3) {}
    virtual ~EdgeCameraRelativePosition() = default;

    virtual void ComputeResidual() override {
        const TVec4<Scalar> &param = this->GetVertex(1)->param();
        const TQuat<Scalar> q_ic0(param(0), param(1), param(2), param(3));
        R_c0i_ = q_ic0.toRotationMatrix().transpose();
        p_c0c1_ = R_c0i_ * (this->GetVertex(2)->param() - this->GetVertex(0)->param());
        this->residual() = p_c0c1_ - this->observation();
    }

    virtual void ComputeJacobians() override {
        this->GetJacobian(0) = -R_c0i_;
        this->GetJacobian(1) = Utility::SkewSymmetricMatrix(p_c0c1_);
        this->GetJacobian(2) = R_c0i_;
    }

private:
    TMat3<Scalar> R_c0i_ = TMat3<Scalar>::Identity();
    TVec3<Scalar> p_c0c1_ = TVec3<Scalar>::Zero();
};

/* Class Edge stereo epipolar distance with the calibrated baseline direction. */
template <typename Scalar>
class EdgeCameraRelativeRotation: public slam_solver::Edge<Scalar> {
    // Vertices are [extrinsic, q_ic0], [extrinsic, q_ic1].

public:
    explicit EdgeCameraRelativeRotation(const TVec3<Scalar> &p_c0c1): slam_solver::Edge<Scalar>(1, 2) {
        skew_baseline_ = Utility::SkewSymmetricMatrix(p_c0c1.normalized());
    }
    virtual ~EdgeCameraRelativeRotation() = default;

    virtual void ComputeResidual() override {
        const TVec4<Scalar> &param0 = this->GetVertex(0)->param();
        const TVec4<Scalar> &param1 = this->GetVertex(1)->param();
        const TQuat<Scalar> q_ic0(param0(0), param0(1), param0(2), param0(3));
        const TQuat<Scalar> q_ic1(param1(0), param1(1), param1(2), param1(3));
        R_c0c1_ = (q_ic0.inverse() * q_ic1).toRotationMatrix();
        const auto &obv = this->observation();
        norm_xyz0_ = TVec3<Scalar>(obv(0, 0), obv(1, 0), static_cast<Scalar>(1));
        norm_xyz1_ = TVec3<Scalar>(obv(2, 0), obv(3, 0), static_cast<Scalar>(1));
        rotated_ray_ = R_c0c1_ * norm_xyz1_;
        epipolar_line_ = skew_baseline_ * rotated_ray_;
        line_norm_ = std::sqrt(epipolar_line_.template head<2>().squaredNorm() + static_cast<Scalar>(1e-12));
        this->residual()(0) = norm_xyz0_.dot(epipolar_line_) / line_norm_;
    }

    virtual void ComputeJacobians() override {
        TVec3<Scalar> dr_dline = norm_xyz0_ / line_norm_;
        dr_dline.template head<2>() -= this->residual()(0) * epipolar_line_.template head<2>() / (line_norm_ * line_norm_);
        const TMat<Scalar> dr_dray = dr_dline.transpose() * skew_baseline_;
        this->GetJacobian(0) = dr_dray * Utility::SkewSymmetricMatrix(rotated_ray_);
        this->GetJacobian(1) = -dr_dray * R_c0c1_ * Utility::SkewSymmetricMatrix(norm_xyz1_);
    }

private:
    TMat3<Scalar> skew_baseline_ = TMat3<Scalar>::Zero();
    TMat3<Scalar> R_c0c1_ = TMat3<Scalar>::Identity();
    TVec3<Scalar> norm_xyz0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> norm_xyz1_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> rotated_ray_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> epipolar_line_ = TVec3<Scalar>::Zero();
    Scalar line_norm_ = 0;
};


}  // namespace vio

#endif  // end of _VISUAL_EDGES_H_
