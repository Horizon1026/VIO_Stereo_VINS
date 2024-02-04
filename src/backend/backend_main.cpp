#include "backend.h"
#include "log_report.h"
#include "tick_tock.h"

namespace VIO {

bool Backend::Configuration(const std::string &log_file_name) {
    if (options_.kEnableRecordBinaryCurveLog) {
        if (!logger_.CreateLogFile(log_file_name)) {
            ReportError("[Backend] Failed to create log file.");
            options_.kEnableRecordBinaryCurveLog = false;
            return false;
        }

        RegisterLogPackages();
        logger_.PrepareForRecording();
    }

    return true;
}

bool Backend::RunOnce() {
    ReportInfo(MAGENTA "[Backend] Backend is triggerred to run once." RESET_COLOR);
    if (data_manager_ == nullptr) {
        ReportError("[Backend] Backend cannot link with data manager.");
        return false;
    }
    TickTock timer;
    TickTock total_timer;

    // Add newest frame_with_bias into visual_local_map.
    if (states_.is_initialized) {
        if (!AddNewestFrameWithBiasIntoLocalMap()) {
            ReportError("[Backend] Backend failed to add newest frame_with_bias into local map.");
            return false;
        }
        log_package_cost_time_.add_newest_frame_into_local_map = timer.TockTickInMillisecond();

        if (!TriangulizeAllNewVisualFeatures()) {
            ReportError("[Backend] Backend failed to triangulize features in local map.");
            return false;
        }
        log_package_cost_time_.triangulize_all_visual_features = timer.TockTickInMillisecond();
    }

    // If backend is not initialized, try to initialize.
    if (!states_.is_initialized) {
        if (TryToInitialize()) {
            states_.is_initialized = true;
            log_package_cost_time_.initialize = timer.TockTickInMillisecond();

            if (options_.kEnableReportAllInformation) {
                ShowTinyInformationOfVisualLocalMap();
            }
            ReportInfo(GREEN "[Backend] Backend succeed to initialize within " << log_package_cost_time_.initialize << " ms." RESET_COLOR);
        } else {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to initialize. All states will be reset for reinitialization.");
        }
    } else {
        log_package_cost_time_.initialize = 0.0f;
    }

    // If backend is initialized.
    if (states_.is_initialized) {
        // Check visual-inertial factors and report error.
        // CheckGraphOptimizationFactors();

        // Check data manager components.
        // data_manager_->SelfCheckVisualLocalMap();
        // data_manager_->SelfCheckFramesWithBias();

        // Try to estimate states.
        timer.TockTickInMillisecond();
        if (!TryToEstimate()) {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to estimate.");
            return true;
        } else {
            log_package_cost_time_.estimate = timer.TockTickInMillisecond();
            ReportInfo(GREEN "[Backend] Backend succeed to estimate states within " << log_package_cost_time_.estimate << " ms." RESET_COLOR);
        }

        if (options_.kEnableReportAllInformation) {
            // Show information of visual local map if neccessary.
            ShowTinyInformationOfVisualLocalMap();
            // Show all frames and features in local map.
            ShowLocalMapFramesAndFeatures(0, false, 1);
            // Show all frames with bias.
            ShowAllFramesWithBias();
            // Show covisible features between keyframe and non-keyframe.
            ShowFeaturePairsBetweenTwoFrames(data_manager_->GetNewestKeyframeId(), data_manager_->GetNewestKeyframeId() + 1, true, 1);
        }

        // Decide marginalization type.
        states_.marginalize_type = DecideMarginalizeType();

        // Try to marginalize if necessary.
        timer.TockTickInMillisecond();
        if (!TryToMarginalize()) {
            ResetToReintialize();
            ReportWarn("[Backend] Backend failed to marginalize.");
            return true;
        } else {
            log_package_cost_time_.marginalize = timer.TockTickInMillisecond();
            ReportInfo(GREEN "[Backend] Backend succeed to marginalize states within " << log_package_cost_time_.marginalize << " ms." RESET_COLOR);
        }
    }

    // Trigger log recording of data manager.
    data_manager_->TriggerLogRecording(data_manager_->GetNewestStateTimeStamp());

    // Control the dimension of local map.
    RETURN_FALSE_IF(!ControlLocalMapDimension());

    // Update states.
    UpdateBackendStates();
    log_package_cost_time_.total_loop = total_timer.TockTickInMillisecond();

    // Record logs of backend.
    RecordBackendLogStates();
    RecordBackendLogStatus();
    RecordBackendLogCostTime();
    RecordBackendLogPriorInformation();

    return true;
}

void Backend::Reset() {
    // Clear stored states in data_manager.
    data_manager_->frames_with_bias().clear();
    data_manager_->visual_local_map()->Clear();

    // Reset status.
    states_.is_initialized = false;
}

void Backend::ResetToReintialize() {
    // Clear stored states in data_manager.
    while (data_manager_->frames_with_bias().size() >= data_manager_->options().kMaxStoredNewFrames) {
        data_manager_->frames_with_bias().pop_front();
    }
    data_manager_->visual_local_map()->Clear();

    // Reset status.
    states_.is_initialized = false;
}

}
