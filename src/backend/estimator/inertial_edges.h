#ifndef _INERTIAL_EDGES_H_
#define _INERTIAL_EDGES_H_

#include "datatype_basic.h"
#include "math_kinematics.h"

#include "imu_state.h"
#include "imu_preintegrate.h"

#include "vertex.h"
#include "vertex_quaternion.h"
#include "edge.h"
#include "kernel.h"
#include "kernel_huber.h"
#include "kernel_cauchy.h"
#include "kernel_tukey.h"

namespace VIO {

/* Class Edge reprojection. Align imu preintegration between two imu pose. */
template <typename Scalar>
class EdgeImuPreintegrationBetweenRelativePose : public Edge<Scalar> {
// Vertices are [imu pose 0, p_wi0]
//              [imu pose 0, q_wi0]
//              [imu vel 0, v_wi0]
//              [imu bias_a 0, bias_a0]
//              [imu bias_g 0, bias_g0]
//              [imu pose 1, p_wi1]
//              [imu pose 1, q_wi1]
//              [imu vel 1, v_wi1]
//              [imu bias_a 1, bias_a1]
//              [imu bias_g 1, bias_g1]

public:
    EdgeImuPreintegrationBetweenRelativePose(const ImuPreintegrateBlock<> &imu_block,
                                             const Vec3 &gravity_w) : Edge<Scalar>(15, 10) {
        linear_point_.p_ij = imu_block.p_ij().cast<Scalar>();
        linear_point_.q_ij = imu_block.q_ij().cast<Scalar>();
        linear_point_.v_ij = imu_block.v_ij().cast<Scalar>();
        linear_point_.bias_a = imu_block.bias_accel().cast<Scalar>();
        linear_point_.bias_g = imu_block.bias_gyro().cast<Scalar>();
        imu_jacobians_.dp_dba = imu_block.dp_dba().cast<Scalar>();
        imu_jacobians_.dp_dbg = imu_block.dp_dbg().cast<Scalar>();
        imu_jacobians_.dr_dbg = imu_block.dr_dbg().cast<Scalar>();
        imu_jacobians_.dv_dba = imu_block.dv_dba().cast<Scalar>();
        imu_jacobians_.dv_dbg = imu_block.dv_dbg().cast<Scalar>();
        gravity_w_ = gravity_w.cast<Scalar>();
        integrate_time_s_ = static_cast<Scalar>(imu_block.integrate_time_s());
        this->information() = imu_block.covariance().inverse().cast<Scalar>();
    }
    virtual ~EdgeImuPreintegrationBetweenRelativePose() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract states.
        p_wi0_ = this->GetVertex(0)->param();
        const TVec4<Scalar> &param_i0 = this->GetVertex(1)->param();
        q_wi0_ = TQuat<Scalar>(param_i0(0), param_i0(1), param_i0(2), param_i0(3));
        v_wi0_ = this->GetVertex(2)->param();
        bias_a0_ = this->GetVertex(3)->param();
        bias_g0_ = this->GetVertex(4)->param();
        p_wi1_ = this->GetVertex(5)->param();
        const TVec4<Scalar> &param_i1 = this->GetVertex(6)->param();
        q_wi1_ = TQuat<Scalar>(param_i1(0), param_i1(1), param_i1(2), param_i1(3));
        v_wi1_ = this->GetVertex(7)->param();
        bias_a1_ = this->GetVertex(8)->param();
        bias_g1_ = this->GetVertex(9)->param();

        // Extract observations.
        const Scalar &dt = integrate_time_s_;
        CorrectObservation(bias_a0_, bias_g0_);

        // Compute residual.
        this->residual().setZero(15);
        this->residual().template block<3, 1>(ImuIndex::kPosition, 0) = q_wi0_.inverse() * (p_wi1_ - p_wi0_ - v_wi0_ * dt + static_cast<Scalar>(0.5) * gravity_w_ * dt * dt) - p_ij_;
        this->residual().template block<3, 1>(ImuIndex::kRotation, 0) = static_cast<Scalar>(2) * (q_ij_.inverse() * (q_wi0_.inverse() * q_wi1_)).vec();
        this->residual().template block<3, 1>(ImuIndex::kVelocity, 0) = q_wi0_.inverse() * (v_wi1_ - v_wi0_ + gravity_w_ * dt) - v_ij_;
        this->residual().template block<3, 1>(ImuIndex::kBiasAccel, 0) = bias_a1_ - bias_a0_;
        this->residual().template block<3, 1>(ImuIndex::kBiasGyro, 0) = bias_g1_ - bias_g0_;
    }

    virtual void ComputeJacobians() override {
        const Scalar &dt = integrate_time_s_;
        const Scalar dt2 = dt * dt;
        const TMat3<Scalar> R_i0w = q_wi0_.inverse().toRotationMatrix();

        // Compute jacobian dr dp0.
        TMat15x3<Scalar> dr_dp0 = TMat15x3<Scalar>::Zero();
        dr_dp0.template block<3, 3>(ImuIndex::kPosition, 0) = - R_i0w;
        TMat15x3<Scalar> dr_dq0 = TMat15x3<Scalar>::Zero();
        dr_dq0.template block<3, 3>(ImuIndex::kPosition, 0) = Utility::SkewSymmetricMatrix(q_wi0_.inverse() * (static_cast<Scalar>(0.5) * gravity_w_ * dt2 + p_wi1_ - p_wi0_ - v_wi0_ * dt));
        dr_dq0.template block<3, 3>(ImuIndex::kRotation, 0) = - (Utility::Qleft(q_wi1_.inverse() * q_wi0_) * Utility::Qright(q_ij_)).template bottomRightCorner<3, 3>();
        dr_dq0.template block<3, 3>(ImuIndex::kVelocity, 0) = Utility::SkewSymmetricMatrix(q_wi0_.inverse() * (gravity_w_ * dt + v_wi1_ - v_wi0_));
        TMat15x3<Scalar> dr_dv0 = TMat15x3<Scalar>::Zero();
        dr_dv0.template block<3, 3>(ImuIndex::kPosition, 0) = - R_i0w * dt;
        dr_dv0.template block<3, 3>(ImuIndex::kVelocity, 0) = - R_i0w;
        TMat15x3<Scalar> dr_dba0 = TMat15x3<Scalar>::Zero();
        dr_dba0.template block<3, 3>(ImuIndex::kPosition, 0) = - imu_jacobians_.dp_dba;
        dr_dba0.template block<3, 3>(ImuIndex::kVelocity, 0) = - imu_jacobians_.dv_dba;
        dr_dba0.template block<3, 3>(ImuIndex::kBiasAccel, 0) = - TMat3<Scalar>::Identity();
        TMat15x3<Scalar> dr_dbg0 = TMat15x3<Scalar>::Zero();
        dr_dbg0.template block<3, 3>(ImuIndex::kPosition, 0) = - imu_jacobians_.dp_dbg;
        dr_dbg0.template block<3, 3>(ImuIndex::kRotation, 0) = - Utility::Qleft(q_wi1_.inverse() * q_wi0_ * linear_point_.q_ij).template bottomRightCorner<3, 3>() * imu_jacobians_.dr_dbg;
        dr_dbg0.template block<3, 3>(ImuIndex::kVelocity, 0) = - imu_jacobians_.dv_dbg;
        dr_dbg0.template block<3, 3>(ImuIndex::kBiasGyro, 0) = - TMat3<Scalar>::Identity();

        // Compute jacobian dr dp1.
        TMat15x3<Scalar> dr_dp1 = TMat15x3<Scalar>::Zero();
        dr_dp1.template block<3, 3>(ImuIndex::kPosition, 0) = R_i0w;
        TMat15x3<Scalar> dr_dq1 = TMat15x3<Scalar>::Zero();
        dr_dq1.template block<3, 3>(ImuIndex::kRotation, 0) = Utility::Qleft(q_ij_.inverse() * q_wi0_.inverse() * q_wi1_).template bottomRightCorner<3, 3>();
        TMat15x3<Scalar> dr_dv1 = TMat15x3<Scalar>::Zero();
        dr_dv1.template block<3, 3>(ImuIndex::kVelocity, 0) = R_i0w;
        TMat15x3<Scalar> dr_dba1 = TMat15x3<Scalar>::Zero();
        dr_dba1.template block<3, 3>(ImuIndex::kBiasAccel, 0) = TMat3<Scalar>::Identity();
        TMat15x3<Scalar> dr_dbg1 = TMat15x3<Scalar>::Zero();
        dr_dbg1.template block<3, 3>(ImuIndex::kBiasGyro, 0) = TMat3<Scalar>::Identity();

        this->GetJacobian(0) = dr_dp0;
        this->GetJacobian(1) = dr_dq0;
        this->GetJacobian(2) = dr_dv0;
        this->GetJacobian(3) = dr_dba0;
        this->GetJacobian(4) = dr_dbg0;

        this->GetJacobian(5) = dr_dp1;
        this->GetJacobian(6) = dr_dq1;
        this->GetJacobian(7) = dr_dv1;
        this->GetJacobian(8) = dr_dba1;
        this->GetJacobian(9) = dr_dbg1;
    }

    void CorrectObservation(const TVec3<Scalar> &bias_a,
                            const TVec3<Scalar> &bias_g) {
        const TVec3<Scalar> dba = bias_a - linear_point_.bias_a;
        const TVec3<Scalar> dbg = bias_g - linear_point_.bias_g;
        p_ij_ = linear_point_.p_ij + imu_jacobians_.dp_dba * dba + imu_jacobians_.dp_dbg * dbg;
        v_ij_ = linear_point_.v_ij + imu_jacobians_.dv_dba * dba + imu_jacobians_.dv_dbg * dbg;

        const TVec3<Scalar> omega = imu_jacobians_.dr_dbg * dbg;
        q_ij_ = linear_point_.q_ij * Utility::Exponent(omega);
        q_ij_.normalize();
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wi0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi0_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> v_wi0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> bias_a0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> bias_g0_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_wi1_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi1_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> v_wi1_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> bias_a1_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> bias_g1_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_ij_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ij_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> v_ij_ = TVec3<Scalar>::Zero();

    // Imu observations.
    struct LineralizedPoint {
        TVec3<Scalar> p_ij = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_ij = TQuat<Scalar>::Identity();
        TVec3<Scalar> v_ij = TVec3<Scalar>::Zero();
        TVec3<Scalar> bias_a = TVec3<Scalar>::Zero();
        TVec3<Scalar> bias_g = TVec3<Scalar>::Zero();
    } linear_point_;
    struct ImuJacobians {
        TMat3<Scalar> dp_dbg = TMat3<Scalar>::Identity();
        TMat3<Scalar> dp_dba = TMat3<Scalar>::Identity();
        TMat3<Scalar> dr_dbg = TMat3<Scalar>::Identity();
        TMat3<Scalar> dv_dbg = TMat3<Scalar>::Identity();
        TMat3<Scalar> dv_dba = TMat3<Scalar>::Identity();
    } imu_jacobians_;
    TVec3<Scalar> gravity_w_ = TVec3<Scalar>(0, 0, 9.8);
    Scalar integrate_time_s_ = static_cast<Scalar>(0);

};

}

#endif // end of _INERTIAL_EDGES_H_
