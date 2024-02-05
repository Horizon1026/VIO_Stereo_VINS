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

    if (!states_.is_initialized) {
        timer.TockTickInMillisecond();
        const bool res = TryToInitialize();
        data_manager_->ShowTinyInformationOfVisualLocalMap();
        if (res) {
            states_.is_initialized = true;
            ReportInfo(GREEN "[Backend] Backend succeed to initialize within " << timer.TockTickInMillisecond() << " ms." RESET_COLOR);
        } else {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to initialize. All states will be reset for reinitialization.");
        }
    }

    // Check data manager components.
    data_manager_->SelfCheckVisualLocalMap();
    data_manager_->SelfCheckFramesWithBias();

    return true;
}

void Backend::Reset() {

}

void Backend::ResetToReintialize() {
    data_manager_->visual_local_map()->Clear();

    while (data_manager_->frames_with_bias().size() >= data_manager_->options().kMaxStoredKeyFrames) {
        data_manager_->frames_with_bias().pop_front();
    }
}

}
