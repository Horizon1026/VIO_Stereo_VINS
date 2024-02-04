#include "vio.h"

#include "slam_operations.h"
#include "log_report.h"

namespace VIO {

bool Vio::RunOnce() {
    // Try to load packed measurements.
    std::unique_ptr<PackedMeasurement> packed_measure = std::make_unique<PackedMeasurement>();
    const bool res = data_loader_->PopPackedMeasurement(*packed_measure);
    if (!res) {
        const float time_s_for_no_data = measure_invalid_timer_.TockInSecond();
        if (time_s_for_no_data > options_.max_tolerence_time_s_for_no_data) {
            ReportInfo("[Vio] Failed to load packed measures for " << time_s_for_no_data << " s. Skip this tick.");
        }
        return false;
    }
    measure_invalid_timer_.TockTickInSecond();

    // Check integrity of the packed measurements.
    if (packed_measure->imus.empty() || packed_measure->left_image == nullptr || packed_measure->right_image == nullptr) {
        ReportWarn("[Vio] Packed measurements is not valid at " << vio_sys_timer_.TockInSecond() << " s.");
        return false;
    }

    // Transform image measurement to be features measurement.
    if (!frontend_->RunOnce(GrayImage(packed_measure->left_image->image),
                            GrayImage(packed_measure->right_image->image),
                            packed_measure->imus.back()->time_stamp_s)) {
        ReportWarn("[Vio] Visual frontend failed to run once at " << vio_sys_timer_.TockInSecond() << " s.");
        return false;
    }

    // Store feature and imu measurements into data_manager_->frames_with_bias().
    std::unique_ptr<FrontendOutputData> visual_measure = std::make_unique<FrontendOutputData>(frontend_->output_data());
    if (!data_manager_->ProcessMeasure(packed_measure, visual_measure)) {
        ReportWarn("[Vio] Data manager failed to store feature and imu measurements at " << vio_sys_timer_.TockInSecond() << " s.");
        return false;
    }

    // Process feature and imu measurements.
    if (!backend_->RunOnce()) {
        ReportWarn("[Vio] Backend failed to process feature and imu measurements at " << vio_sys_timer_.TockInSecond() << " s.");
        return false;
    }

    HeartBeat();

    return true;
}

void Vio::HeartBeat() {
    if (vio_heart_beat_timer_.TockInSecond() > options_.heart_beat_period_time_s) {
        ReportInfo("[Vio] Heart beat for " << vio_heart_beat_timer_.TockTickInSecond() << " s. Vio has running for " <<
            vio_sys_timer_.TockInSecond() << " s.");
    }
}

}
