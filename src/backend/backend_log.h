#ifndef _VIO_STEREO_BASALT_BACKEND_LOG_H_
#define _VIO_STEREO_BASALT_BACKEND_LOG_H_

#include "datatype_basic.h"
#include "data_manager.h"

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
    float q_wi_pitch = 0.0f;
    float q_wi_roll = 0.0f;
    float q_wi_yaw = 0.0f;
    float v_wi_x = 0.0f;
    float v_wi_y = 0.0f;
    float v_wi_z = 0.0f;
    float bias_a_x = 0.0f;
    float bias_a_y = 0.0f;
    float bias_a_z = 0.0f;
    float bias_g_x = 0.0f;
    float bias_g_y = 0.0f;
    float bias_g_z = 0.0f;

    uint8_t is_prior_valid = 0;
    float prior_residual = 0.0f;
};

struct BackendLogStatus {
    uint8_t is_initialized = 0;
    uint8_t marginalize_type = 0;
    uint32_t num_of_valid_loop = 0;
};

struct BackendLogCostTime {
    float total_loop = 0.0f;
    float add_newest_frame_into_local_map = 0.0f;
    float triangulize_all_visual_features = 0.0f;
    float initialize = 0.0f;
    float estimate = 0.0f;
    float marginalize = 0.0f;
};
#pragma pack()

}

#endif // end of _VIO_STEREO_BASALT_BACKEND_LOG_H_