#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

namespace VIO {

bool Backend::RunOnce() {
    ReportInfo(MAGENTA "[Backend] Backend is triggerred to run once." RESET_COLOR);
    TickTock timer;
    TickTock total_timer;

    // Process newest frame into visual local map.
    log_package_cost_time_.add_new_frame_into_local_map = 0.0f;
    if (status_.is_initialized) {
        // Process newest visual and imu measurements.
        timer.TockTickInMillisecond();
        const bool res = AddNewestFrameWithStatesPredictionToLocalMap();
        log_package_cost_time_.add_new_frame_into_local_map = timer.TockTickInMillisecond();
        if (res) {
            ReportColorInfo("[Backend] Backend succeed to add newest frame with states prediction within " <<
                log_package_cost_time_.add_new_frame_into_local_map << " ms.");
        } else {
            ResetToReintialize();
            ReportColorError("[Backend] Backend failed to add newest frame and do states prediction.");
        }
    }

    // Part of initialization.
    log_package_cost_time_.initialize = 0.0f;
    if (!status_.is_initialized) {
        // Try to initialize vio if not initialized.
        timer.TockTickInMillisecond();
        const bool res = TryToInitialize();
        log_package_cost_time_.initialize = timer.TockTickInMillisecond();
        if (res) {
            status_.is_initialized = true;
            ReportColorInfo("[Backend] Backend succeed to initialize within " << log_package_cost_time_.initialize << " ms.");
        } else {
            ResetToReintialize();
            ReportColorWarn("[Backend] Backend failed to initialize. All states will be reset for reinitialization.");
        }
    }

    // Part of estimation and marginalization.
    log_package_cost_time_.estimate = 0.0f;
    log_package_cost_time_.marginalize = 0.0f;
    if (status_.is_initialized) {
        // Try to do graph optimization if vio has initialized.
        timer.TockTickInMillisecond();
        const bool estimate_res = TryToEstimate(true);
        log_package_cost_time_.estimate = timer.TockTickInMillisecond();
        if (estimate_res) {
            ReportColorInfo("[Backend] Backend succeed to estimate within " << log_package_cost_time_.estimate << " ms.");
        } else {
            ResetToReintialize();
            ReportColorWarn("[Backend] Backend failed to estimate. All states will be reset for reinitialization.");
        }

        // Decide marginalization type.
        status_.marginalize_type = DecideMarginalizeType();

        // Try to do marginalization if neccessary.
        timer.TockTickInMillisecond();
        const bool marginalize_res = TryToMarginalize(true);
        log_package_cost_time_.marginalize = timer.TockTickInMillisecond();
        if (marginalize_res) {
            ReportColorInfo("[Backend] Backend succeed to marginalize within " << log_package_cost_time_.marginalize << " ms.");
        } else {
            ResetToReintialize();
            ReportColorWarn("[Backend] Backend failed to marginalize. All states will be reset for reinitialization.");
        }
    }

    // Add new keyframe into global map if needed.
    RETURN_FALSE_IF(!SelectKeyframesIntoGlobalMap());
    // Control the dimension of local map.
    RETURN_FALSE_IF(!ControlSizeOfLocalMap());
    // Update backend states for output.
    UpdateBackendStates();

    // Record logs.
    log_package_cost_time_.total_loop = total_timer.TockTickInMillisecond();
    RecordBackendLogStates();
    RecordBackendLogStatus();
    RecordBackendLogCostTime();
    RecordBackendLogPriorInformation();
    data_manager_->TriggerLogRecording(states_.motion.time_stamp_s);

    return true;
}

void Backend::Reset() {
    // Clear data manager.
    data_manager_->visual_local_map()->Clear();
    data_manager_->frames_with_bias().clear();

    // Clear states flag.
    status_.is_initialized = false;
    states_.prior.is_valid = false;
}

void Backend::ResetToReintialize() {
    // Clear data manager.
    data_manager_->visual_local_map()->Clear();
    while (data_manager_->frames_with_bias().size() >= data_manager_->options().kMaxStoredKeyFrames) {
        data_manager_->frames_with_bias().pop_front();
    }

    // Clear states flag.
    status_.is_initialized = false;
    states_.prior.is_valid = false;
}

}
