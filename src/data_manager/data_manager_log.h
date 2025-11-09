#ifndef _VIO_STEREO_VINS_DATA_MANAGER_LOG_H_
#define _VIO_STEREO_VINS_DATA_MANAGER_LOG_H_

#include "basic_type.h"

namespace vio {

/* Packages of log to be recorded. */
#pragma pack(1)
struct DataManagerLocalMapLog {
    uint32_t num_of_features = 0;
    uint32_t num_of_solved_features = 0;
    uint32_t num_of_marginalized_features = 0;
    uint32_t num_of_unsolved_features = 0;

    uint32_t num_of_frames = 0;
};
#pragma pack()

}  // namespace vio

#endif  // end of _VIO_STEREO_VINS_DATA_MANAGER_LOG_H_
