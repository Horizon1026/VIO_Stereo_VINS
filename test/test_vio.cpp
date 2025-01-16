#include "string"
#include "fstream"
#include "sstream"
#include "unistd.h"
#include "thread"

#include "visualizor_2d.h"
#include "visualizor_3d.h"
#include "vio.h"
#include "tick_tock.h"

#include "enable_stack_backward.h"

using namespace SLAM_VISUALIZOR;

VIO::Vio vio;
std::mutex g_mutex_load_image;
double time_stamp_offset = 1403636579.0;

void PublishImuData(const std::string &csv_file_path,
                    const float period_ms) {
    std::ifstream file(csv_file_path.c_str());
    if (!file.is_open()) {
        ReportError("Failed to load imu data file " << csv_file_path);
        return;
    }

    // Print header of data file.
    std::string one_line;
    std::getline(file, one_line);
    if (one_line.empty()) {
        ReportError("Imu data file is empty. " << csv_file_path);
        return;
    } else {
        ReportInfo("Imu data file header is [ " << one_line << " ].");
    }

    // Publish each line of data file.
    while (std::getline(file, one_line) && !one_line.empty()) {
        TickTock timer;

        std::istringstream imu_data(one_line);
        std::string one_item;
        uint32_t i = 0;
        double temp[7] = {};
        while (std::getline(imu_data, one_item, ',') && !one_item.empty()) {
            std::istringstream item_data(one_item);
            item_data >> temp[i];
            ++i;
        }

        // Send data to dataloader of vio.
        const double time_stamp_s = temp[0];
        const Vec3 accel = Vec3(temp[4], temp[5], temp[6]);
        const Vec3 gyro = Vec3(temp[1], temp[2], temp[3]);
        vio.data_loader()->PushImuMeasurement(accel.cast<float>(), gyro.cast<float>(), static_cast<float>(time_stamp_s * 1e-9 - time_stamp_offset));

        // Waiting for next timestamp.
        while (timer.TockInMillisecond() < period_ms || vio.data_loader()->IsImuBufferFull()) {
            usleep(100);
            BREAK_IF(vio.backend()->signals().should_quit);
        }

        BREAK_IF(vio.backend()->signals().should_quit);
    }

    file.close();
}

void PublishCameraData(const std::string &csv_file_path,
                       const std::string &image_file_root,
                       const float period_ms,
                       const bool is_left_camera) {
    std::ifstream file(csv_file_path.c_str());
    if (!file.is_open()) {
        ReportError("Failed to load camera data file " << csv_file_path);
        return;
    }

    // Print header of data file.
    std::string one_line;
    std::getline(file, one_line);
    if (one_line.empty()) {
        ReportError("Camera data file is empty. " << csv_file_path);
        return;
    } else {
        ReportInfo("Camera data file header is [ " << one_line << " ].");
    }

    // Publish each line of data file.
    double time_stamp_s = 0.0;
    std::string image_file_name;
    while (std::getline(file, one_line) && !one_line.empty()) {
        TickTock timer;

        std::istringstream camera_data(one_line);
        camera_data >> time_stamp_s >> image_file_name;
        image_file_name.erase(std::remove(image_file_name.begin(), image_file_name.end(), ','), image_file_name.end());

        GrayImage image; {
            // This mutex is important for multi-thread loading image. Because Visualizor2D::LoadImage is a static function.
            std::unique_lock<std::mutex> lck(g_mutex_load_image);
            Visualizor2D::LoadImage(image_file_root + image_file_name, image);
        }
        image.memory_owner() = false;
        if (image.data() == nullptr) {
            ReportError("Failed to load image file.");
            return;
        }

        // Send data to dataloader of vio.
        vio.data_loader()->PushImageMeasurement(image.data(), image.rows(), image.cols(), static_cast<float>(time_stamp_s * 1e-9 - time_stamp_offset), is_left_camera);

        // Waiting for next timestamp.
        while (timer.TockInMillisecond() < period_ms || vio.data_loader()->IsImageBufferFull()) {
            usleep(100);
            BREAK_IF(vio.backend()->signals().should_quit);
        }

        BREAK_IF(vio.backend()->signals().should_quit);
    }

    file.close();
}


void TestRunVio(const uint32_t max_wait_ticks) {
    uint32_t cnt = max_wait_ticks;
    const uint32_t max_valid_steps = 3682;
    uint32_t valid_steps = 0;
    while (cnt) {
        const bool res = vio.RunOnce();
        if (res) {
            ++valid_steps;
        }
        if (valid_steps > max_valid_steps) {
            vio.backend()->signals().should_quit = true;
        }

        if (vio.backend()->signals().should_quit) {
            break;
        } else {
            vio.data_manager()->ShowLocalMapInWorldFrame("Vio 3d local map", 1, false);
        }

        if (!res) {
            usleep(1000);
            --cnt;
            continue;
        }
        cnt = max_wait_ticks;
    }

    vio.data_manager()->ShowLocalMapInWorldFrame("Vio 3d local map", 20, true);
}

void ConfigAllComponentsOfVio()
{
    /* VioOptionsOfCamera */
    // Fill left and right camera intrinsics.
    const VIO::VioOptionsOfCamera left_camera_intrinsics {
        .fx = 458.654f,
        .fy = 457.296f,
        .cx = 367.215f,
        .cy = 248.375f,
        .k1 = -0.28340811f,
        .k2 = 0.07395907f,
        .k3 = 0.0f,
        .p1 = 0.00019359f,
        .p2 = 1.76187114e-05f,
    };
    vio.options().cameras.emplace_back(left_camera_intrinsics);
    const VIO::VioOptionsOfCamera right_camera_intrinsics {
        .fx = 457.587f,
        .fy = 456.134f,
        .cx = 379.999,
        .cy = 255.238,
        .k1 = -0.28368365f,
        .k2 = 0.07451284f,
        .k3 = 0.0f,
        .p1 = -0.00010473f,
        .p2 = -3.55590700e-05f,
    };
    vio.options().cameras.emplace_back(right_camera_intrinsics);

    /* VioOptionsOfImu */
    // Fill imu noise sigma.
    vio.options().imu.noise_accel = std::sqrt(2.0000e-3f);
    vio.options().imu.noise_gyro = std::sqrt(1.6968e-04f);
    vio.options().imu.random_walk_accel = std::sqrt(3.0000e-3f);
    vio.options().imu.random_walk_gyro = std::sqrt(1.9393e-05f);

    /* VioOptionsOfFrontend */
    // Fill options of visual frontend.
    vio.options().frontend.image_rows = 480;
    vio.options().frontend.image_cols = 752;
    vio.options().frontend.max_feature_number = 144;
    vio.options().frontend.min_feature_number = 60;
    vio.options().frontend.select_keyframe = false;
    vio.options().frontend.enable_drawing_track_result = false;
    vio.options().frontend.enable_recording_curve_binlog = true;
    vio.options().frontend.enable_recording_image_binlog = false;
    vio.options().frontend.log_file_name = "frontend.binlog";
    // Fill options of feature detector.
    vio.options().frontend.feature_detector.min_valid_feature_distance = 25;
    vio.options().frontend.feature_detector.grid_filter_rows = 12;
    vio.options().frontend.feature_detector.grid_filter_cols = 12;
    // Fill options of feature tracker.
    vio.options().frontend.feature_tracker.half_row_size_of_patch = 6;
    vio.options().frontend.feature_tracker.half_col_size_of_patch = 6;
    vio.options().frontend.feature_tracker.max_iterations = 15;

    /* VioOptionsOfBackend */
    // Fill options of backend.
    vio.options().backend.gravity_w = Vec3(0.0f, 0.0f, 9.8f);
    vio.options().backend.max_valid_feature_depth_in_meter = 30.0f;
    vio.options().backend.min_valid_feature_depth_in_meter = 0.05f;
    vio.options().backend.default_feature_depth_in_meter = 2.0f;
    vio.options().backend.min_parallex_angle_of_feature_for_local_ba = 3.0f;
    vio.options().backend.max_tolerence_time_for_estimation_in_second = 0.03f;
    vio.options().backend.use_multi_view_observation = true;
    vio.options().backend.fix_camera_extrinsics = true;
    vio.options().backend.max_tolerence_reprojection_error_in_normal_plane = 0.03f;
    vio.options().backend.enable_local_map_store_raw_images = false;
    vio.options().backend.enable_recording_curve_binlog = true;
    vio.options().backend.log_file_name = "backend.binlog";

    /* VioOptionsOfDataLoader */
    // Fill options of data loader.
    vio.options().data_loader.max_size_of_imu_buffer = 200;
    vio.options().data_loader.max_size_of_image_buffer = 20;
    vio.options().data_loader.enable_recording_curve_binlog = true;
    vio.options().data_loader.enable_recording_raw_data_binlog = true;
    vio.options().data_loader.log_file_name = "data_loader.binlog";

    /* VioOptionsOfDataManager */
    // Fill left and right camera extrinsics.
    Mat3 R_i_cl;
    R_i_cl << 0.0148655429818,  -0.999880929698,  0.00414029679422,
              0.999557249008,   0.0149672133247,  0.025715529948,
              -0.0257744366974, 0.00375618835797, 0.999660727178;
    const Vec3 p_i_cl = Vec3(-0.0216401454975,-0.064676986768, 0.00981073058949);
    vio.options().data_manager.all_R_ic.emplace_back(R_i_cl);
    vio.options().data_manager.all_t_ic.emplace_back(p_i_cl);
    Mat3 R_i_cr;
    R_i_cr << 0.0125552670891,  -0.999755099723, 0.0182237714554,
              0.999598781151,   0.0130119051815, 0.0251588363115,
              -0.0253898008918, 0.0179005838253, 0.999517347078;
    const Vec3 p_i_cr = Vec3(-0.0198435579556, 0.0453689425024, 0.00786212447038);
    vio.options().data_manager.all_R_ic.emplace_back(R_i_cr);
    vio.options().data_manager.all_t_ic.emplace_back(p_i_cr);
    vio.options().data_manager.max_num_of_stored_key_frames = 6;
    vio.options().data_manager.max_time_s_of_imu_preintegration_block = 3.0f;
    vio.options().data_manager.enable_recording_curve_binlog = true;
    vio.options().data_manager.log_file_name = "data_manager.binlog";

    /* VioOptions */
    // Fill options of vio.
    vio.options().max_tolerence_time_s_for_no_data = 2.0f;
    vio.options().heart_beat_period_time_s = 1.0f;
    vio.options().log_file_root_name = "../../Workspace/output/";

    // Config vio.
    vio.ConfigAllComponents();
    LogFixPercision(4);
}

static std::ofstream g_txt_log("../output/vio_log.txt");
int main(int argc, char **argv) {
    // Root direction of Euroc dataset.
    std::string dataset_root_dir = "D:/My_Github/Datasets/Euroc/MH_01_easy/";
    if (argc == 2) {
        dataset_root_dir = argv[1];
    }

    // Fill configuration of vio.
    ReportInfo(YELLOW ">> Test vio on " << dataset_root_dir << "." RESET_COLOR);
    ConfigAllComponentsOfVio();

    // Config visualizor 3d.
    Visualizor3D::camera_view().q_wc = Quat(1.0, -1.0, 0, 0).normalized();
    Visualizor3D::camera_view().p_wc.y() = -3.0f;

    // Start threads for data pipeline and vio node.
    const float imu_timeout_ms = 0.5f;
    const float image_timeout_ms = 10.0f;
    std::thread thread_pub_imu_data{PublishImuData, dataset_root_dir + "mav0/imu0/data.csv", imu_timeout_ms};
    std::thread thread_pub_cam_left_data(PublishCameraData, dataset_root_dir + "mav0/cam0/data.csv", dataset_root_dir + "mav0/cam0/data/", image_timeout_ms, true);
    std::thread thread_pub_cam_right_data(PublishCameraData, dataset_root_dir + "mav0/cam1/data.csv", dataset_root_dir + "mav0/cam1/data/", image_timeout_ms, false);
    std::thread thread_test_vio(TestRunVio, 1500);

    // Waiting for the end of the threads. Recovery their resources.
    thread_pub_imu_data.join();
    thread_pub_cam_left_data.join();
    thread_pub_cam_right_data.join();
    thread_test_vio.join();

    return 0;
}
