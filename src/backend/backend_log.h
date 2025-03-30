#ifndef _VIO_STEREO_VINS_BACKEND_LOG_H_
#define _VIO_STEREO_VINS_BACKEND_LOG_H_

#include "basic_type.h"

namespace VIO {

/* Packages of log to be recorded. */
#pragma pack(1)
struct BackendLogStates {
    float time_stamp_s = 0.0f;
    float p_wi_x = 0.0f;
    float p_wi_y = 0.0f;
    float p_wi_z = 0.0f;
    float q_wi_w = 0.0f;
    float q_wi_x = 0.0f;
    float q_wi_y = 0.0f;
    float q_wi_z = 0.0f;
    float v_wi_x = 0.0f;
    float v_wi_y = 0.0f;
    float v_wi_z = 0.0f;
    float bias_a_x = 0.0f;
    float bias_a_y = 0.0f;
    float bias_a_z = 0.0f;
    float bias_g_x = 0.0f;
    float bias_g_y = 0.0f;
    float bias_g_z = 0.0f;
};

struct BackendLogGraph {
    uint32_t num_of_p_ic = 0;
    uint32_t num_of_q_ic = 0;
    uint32_t num_of_p_wi = 0;
    uint32_t num_of_q_wi = 0;
    uint32_t num_of_v_wi = 0;
    uint32_t num_of_bias_a = 0;
    uint32_t num_of_bias_g = 0;
    uint32_t num_of_p_wc = 0;
    uint32_t num_of_q_wc = 0;
    uint32_t num_of_feature_invdep = 0;

    uint32_t num_of_prior_factor = 0;
    uint32_t num_of_visual_factor = 0;
    uint32_t num_of_imu_factor = 0;

    uint8_t is_prior_valid = 0;
    float prior_residual = 0.0f;
};

struct BackendLogStatus {
    uint8_t is_initialized = 0;
    uint8_t marginalize_type = 0;
    uint32_t valid_loop_count = 0;
};

struct BackendLogCostTime {
    float total_loop = 0.0f;
    float add_new_frame = 0.0f;
    float initialize = 0.0f;
    float estimate = 0.0f;
    float marginalize = 0.0f;
    float update_state = 0.0f;
    float record_log = 0.0f;
};
#pragma pack()

}

#endif // end of _VIO_STEREO_VINS_BACKEND_LOG_H_