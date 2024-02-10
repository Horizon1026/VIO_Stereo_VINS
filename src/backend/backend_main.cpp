#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

namespace VIO {

bool Backend::RunOnce() {
    ReportInfo(MAGENTA "[Backend] Backend is triggerred to run once." RESET_COLOR);
    if (data_manager_ == nullptr) {
        ReportError("[Backend] Backend cannot link with data manager.");
        return false;
    }
    TickTock timer;

    if (status_.is_initialized) {
        // Process newest visual and imu measurements.
        if (!AddNewestFrameWithStatesPredictionToLocalMap()) {
            ReportError("[Backend] Backend failed to add newest frame and do states prediction.");
            ResetToReintialize();
        }
    }

    if (!status_.is_initialized) {
        // Try to initialize vio if not initialized.
        timer.TockTickInMillisecond();
        const bool res = TryToInitialize();
        if (res) {
            status_.is_initialized = true;
            ReportColorInfo("[Backend] Backend succeed to initialize within " << timer.TockTickInMillisecond() << " ms.");
        } else {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to initialize. All states will be reset for reinitialization.");
        }
    }

    if (status_.is_initialized) {
        // Try to do graph optimization if vio has initialized.
        timer.TockTickInMillisecond();
        const bool estimate_res = TryToEstimate(true);
        if (estimate_res) {
            ReportColorInfo("[Backend] Backend succeed to estimate within " << timer.TockTickInMillisecond() << " ms.");
        } else {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to estimate. All states will be reset for reinitialization.");
        }

        // Decide marginalization type.
        status_.marginalize_type = DecideMarginalizeType();

        // Try to do marginalization if neccessary.
        timer.TockTickInMillisecond();
        const bool marginalize_res = TryToMarginalize(true);
        if (marginalize_res) {
            ReportColorInfo("[Backend] Backend succeed to marginalize within " << timer.TockTickInMillisecond() << " ms.");
        } else {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to marginalize. All states will be reset for reinitialization.");
        }
    }

    // Show information of visual local map.
    data_manager_->ShowTinyInformationOfVisualLocalMap();

    // Check data manager components.
    data_manager_->SelfCheckVisualLocalMap();
    data_manager_->SelfCheckFramesWithBias();

    // Debug.
    data_manager_->ShowLocalMapInWorldFrame("Estimation result", 1, false);

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
