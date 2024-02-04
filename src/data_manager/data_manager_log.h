#ifndef _VIO_STEREO_BASALT_DATA_MANAGER_LOG_H_
#define _VIO_STEREO_BASALT_DATA_MANAGER_LOG_H_

#include "datatype_basic.h"

namespace VIO {

/* Packages of log to be recorded. */
#pragma pack(1)
struct DataManagerLocalMapLog {
    uint32_t num_of_features = 0;
    uint32_t num_of_solved_features = 0;
    uint32_t num_of_marginalized_features = 0;
    uint32_t num_of_unsolved_features = 0;
    uint32_t num_of_features_observed_in_newest_keyframe = 0;
    uint32_t num_of_solved_features_observed_in_newest_keyframe = 0;

    uint32_t num_of_frames = 0;
    uint32_t num_of_keyframes = 0;
    uint32_t num_of_newframes = 0;
};

struct DataManagerCovisibleGraphLog {
    uint32_t num_of_observed_features = 0;
    uint32_t num_of_solved_features = 0;
    uint32_t num_of_tracked_features_from_prev_frame = 0;
    uint32_t num_of_solved_tracked_features_from_prev_frame = 0;

    float time_stamp_s = 0.0f;
    float p_wc_x = 0.0f;
    float p_wc_y = 0.0f;
    float p_wc_z = 0.0f;
    float q_wc_w = 0.0f;
    float q_wc_x = 0.0f;
    float q_wc_y = 0.0f;
    float q_wc_z = 0.0f;
    float v_wc_x = 0.0f;
    float v_wc_y = 0.0f;
    float v_wc_z = 0.0f;
};
#pragma pack()

}

#endif // end of _VIO_STEREO_BASALT_DATA_MANAGER_LOG_H_
