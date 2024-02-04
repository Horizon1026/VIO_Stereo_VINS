#include "backend.h"
#include "log_report.h"

namespace VIO {

bool Backend::TryToInitialize() {
    if (data_manager_->frames_with_bias().size() < data_manager_->options().kMaxStoredKeyFrames) {
        ReportWarn("[Backend] Backend cannot initialize for lack of new frames.");
        return false;
    }

    // Convert all frames into a covisible graph.
    if (!data_manager_->ConvertAllFramesWithBiasToLocalMap()) {
        ReportError("[Backend] Backend failed to convert frames to covisible graph.");
        return false;
    }

    return true;
}

}
