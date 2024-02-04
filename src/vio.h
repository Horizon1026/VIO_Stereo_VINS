#ifndef _VIO_STEREO_BASALT_H_
#define _VIO_STEREO_BASALT_H_

#include "vio_config.h"

#include "tick_tock.h"
#include "data_manager.h"
#include "data_loader.h"
#include "visual_frontend.h"
#include "backend.h"

#include "memory"

namespace VIO {

/* Class Vio Declaration. */
class Vio final {

public:
    Vio() = default;
    virtual ~Vio() = default;

    // Run once without loop.
    bool RunOnce();
    // Config all components of vio.
    bool ConfigAllComponents();

    // Reference for member variables.
    VioOptions &options() { return options_; }
    std::unique_ptr<DataManager> &data_manager() { return data_manager_; }
    std::unique_ptr<DataLoader> &data_loader() { return data_loader_; }
    std::unique_ptr<VisualFrontend> &frontend() { return frontend_; }
    std::unique_ptr<Backend> &backend() { return backend_; }

    // Const reference for member variables.
    const VioOptions &options() const { return options_; }
    const std::unique_ptr<DataManager> &data_manager() const { return data_manager_; }
    const std::unique_ptr<DataLoader> &data_loader() const { return data_loader_; }
    const std::unique_ptr<VisualFrontend> &frontend() const { return frontend_; }
    const std::unique_ptr<Backend> &backend() const { return backend_; }

private:
    // Basic methods.
    void HeartBeat();

    // Config all components of vio.
    bool ConfigComponentOfDataManager();
    bool ConfigComponentOfDataLoader();
    bool ConfigComponentOfFrontend();
    bool ConfigComponentOfBackend();

private:
    // Options for vio.
    VioOptions options_;

    // All components.
    std::unique_ptr<DataManager> data_manager_ = nullptr;
    std::unique_ptr<DataLoader> data_loader_ = nullptr;
    std::unique_ptr<VisualFrontend> frontend_ = nullptr;
    std::unique_ptr<Backend> backend_ = nullptr;

    // Vio timers.
    TickTock vio_sys_timer_;
    TickTock vio_heart_beat_timer_;
    TickTock measure_invalid_timer_;

};

}

#endif
