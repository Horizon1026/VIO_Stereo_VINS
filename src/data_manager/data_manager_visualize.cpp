#include "data_manager.h"
#include "image_painter.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "visualizor_2d.h"
#include "visualizor_3d.h"

using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

namespace VIO {

namespace {
    constexpr int32_t kMaxImageNumInOneRow = 3;
}

RgbPixel DataManager::GetFeatureColor(const FeatureType &feature) {
    RgbPixel pixel_color = RgbColor::kLightSkyBlue;
    switch (feature.status()) {
        case FeatureSolvedStatus::kSolved:
            if (feature.final_frame_id() == visual_local_map_->frames().back().id()) {
                pixel_color = RgbColor::kYellow;
            } else {
                pixel_color = RgbColor::kGreen;
            }
            break;
        case FeatureSolvedStatus::kMarginalized:
            pixel_color = RgbColor::kBlue;
            break;
        default:
        case FeatureSolvedStatus::kUnsolved:
            pixel_color = RgbColor::kRed;
            break;
    }

    return pixel_color;
}

void DataManager::ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id, const uint32_t cur_frame_id, const int32_t delay_ms) {
    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    if (!visual_local_map_->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        ReportError("[DataManager] Failed to get covisible features between frame " << ref_frame_id << " and " << cur_frame_id << ".");
        return;
    }

    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    for (const auto &feature_ptr: covisible_features) {
        ref_pixel_uv.emplace_back(feature_ptr->observe(ref_frame_id)[0].raw_pixel_uv);
        cur_pixel_uv.emplace_back(feature_ptr->observe(cur_frame_id)[0].raw_pixel_uv);
    }

    // Create gray image of ref and cur image.
    const GrayImage ref_image(visual_local_map_->frame(ref_frame_id)->raw_images()[0]);
    const GrayImage cur_image(visual_local_map_->frame(cur_frame_id)->raw_images()[0]);

    // Draw tracking results.
    const std::vector<uint8_t> tracked_status(ref_pixel_uv.size(), 1);
    Visualizor2D::ShowImageWithTrackedFeatures(std::string("Raw image [ ") + std::to_string(ref_frame_id) + std::string(" | ") + std::to_string(cur_frame_id) +
                                                   std::string(" ] covisible features"),
                                               ref_image, cur_image, ref_pixel_uv, cur_pixel_uv, tracked_status);

    Visualizor2D::WaitKey(delay_ms);
}

void DataManager::ShowLocalMapFramesAndFeatures(const int32_t feature_id, const int32_t camera_id, const int32_t delay_ms) {
    RETURN_IF(visual_local_map_->frames().empty());
    RETURN_IF(visual_local_map_->frames().front().raw_images().empty());

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images =
        options_.kMaxStoredKeyFrames % cols_of_images == 0 ? options_.kMaxStoredKeyFrames / cols_of_images : options_.kMaxStoredKeyFrames / cols_of_images + 1;
    const int32_t image_cols = visual_local_map_->frames().front().raw_images()[camera_id].cols();
    const int32_t image_rows = visual_local_map_->frames().front().raw_images()[camera_id].rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (auto &frame: visual_local_map_->frames()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Load image.
        show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = frame.raw_images()[camera_id];
        // Accumulate index.
        ++frame_id;
    }
    GrayImage gray_show_image(show_image_mat);
    uint8_t *show_image_buf = (uint8_t *)SlamMemory::Malloc(show_image_rows * show_image_cols * 3 * sizeof(uint8_t));
    RgbImage show_image(show_image_buf, show_image_rows, show_image_cols, true);
    ImagePainter::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (const auto &frame: visual_local_map_->frames()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color =
            frame_id >= static_cast<int32_t>(visual_local_map_->frames().size() - options_.kMaxStoredKeyFrames) ? RgbColor::kRed : RgbColor::kGreen;
        ImagePainter::DrawString(
            show_image, std::string("[ ") + std::to_string(frame.id()) + std::string(" | ") + std::to_string(frame.time_stamp_s()) + std::string("s ]"),
            col_offset, row_offset, info_color, font_size);
        // Draw all observed features in this frame and this camera image.
        for (auto &pair: frame.features()) {
            auto &feature = pair.second;
            auto &observe = feature->observe(frame.id());
            CONTINUE_IF(feature_id > 0 && static_cast<uint32_t>(feature_id) != feature->id());
            CONTINUE_IF(static_cast<int32_t>(observe.size()) <= camera_id);

            // Draw feature in rgb image.
            Vec2 pixel_uv = observe[camera_id].raw_pixel_uv;
            const RgbPixel pixel_color = GetFeatureColor(*feature);
            const std::string feature_text =
                feature->first_frame_id() == frame.id() ? std::to_string(feature->id()) + std::string("+") : std::to_string(feature->id());
            ImagePainter::DrawSolidCircle(show_image, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, 3, pixel_color);
            ImagePainter::DrawString(show_image, feature_text, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, pixel_color);
        }
        // Accumulate index.
        ++frame_id;
    }

    const std::vector<std::string> camera_name = {"left", "right"};
    Visualizor2D::ShowImage(std::string("Local map [") + camera_name[camera_id] + std::string("] <distorted>"), show_image);
    Visualizor2D::WaitKey(delay_ms);
}

void DataManager::ShowAllImuBasedFrames(const int32_t delay_ms) {
    RETURN_IF(imu_based_frames_.empty());
    RETURN_IF(imu_based_frames_.front().packed_measure->left_image == nullptr);

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images =
        options_.kMaxStoredKeyFrames % cols_of_images == 0 ? options_.kMaxStoredKeyFrames / cols_of_images : options_.kMaxStoredKeyFrames / cols_of_images + 1;
    const int32_t image_cols = imu_based_frames_.front().packed_measure->left_image->image.cols();
    const int32_t image_rows = imu_based_frames_.front().packed_measure->left_image->image.rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (const auto &imu_based_frame: imu_based_frames_) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Load image.
        show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = imu_based_frame.packed_measure->left_image->image;
        // Accumulate index.
        ++frame_id;
    }
    GrayImage gray_show_image(show_image_mat);
    uint8_t *show_image_buf = (uint8_t *)SlamMemory::Malloc(show_image_rows * show_image_cols * 3 * sizeof(uint8_t));
    RgbImage show_image(show_image_buf, show_image_rows, show_image_cols, true);
    ImagePainter::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (const auto &imu_based_frame: imu_based_frames_) {
        CONTINUE_IF(imu_based_frame.packed_measure == nullptr || imu_based_frame.visual_measure == nullptr);
        CONTINUE_IF((imu_based_frame.packed_measure->left_image == nullptr));

        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color =
            frame_id >= static_cast<int32_t>(options_.kMaxStoredKeyFrames - options_.kMaxStoredKeyFrames) ? RgbColor::kRed : RgbColor::kGreen;
        ImagePainter::DrawString(show_image, std::string("[ ") + std::to_string(imu_based_frame.time_stamp_s) + std::string("s ]"), col_offset, row_offset,
                                 info_color, font_size);

        // Draw all observed features in this frame and this camera image.
        for (uint32_t i = 0; i < imu_based_frame.visual_measure->features_id.size(); ++i) {
            const Vec2 pixel_uv = imu_based_frame.visual_measure->observes_per_frame[i][0].raw_pixel_uv + Vec2(col_offset, row_offset);
            const RgbPixel pixel_color = RgbColor::kDeepSkyBlue;
            ImagePainter::DrawSolidCircle(show_image, pixel_uv.x(), pixel_uv.y(), 3, pixel_color);
            ImagePainter::DrawString(show_image, std::to_string(imu_based_frame.visual_measure->features_id[i]), pixel_uv.x(), pixel_uv.y(), pixel_color);
        }

        // Accumulate index.
        ++frame_id;
    }

    Visualizor2D::ShowImage(std::string("New frames with bias [left] <distorted>"), show_image);
    Visualizor2D::WaitKey(delay_ms);
}

void DataManager::ShowLocalMapInWorldFrame() {
    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType {
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });

    RETURN_IF(visual_local_map_->features().empty());
    RETURN_IF(visual_local_map_->frames().empty());

    // Add all features in locap map.
    for (const auto &pair: visual_local_map_->features()) {
        const auto &feature = pair.second;
        Visualizor3D::points().emplace_back(PointType {
            .p_w = feature.param(),
            .color = GetFeatureColor(feature),
            .radius = 2,
        });
    }

    // Add all frames in locap map.
    bool is_p_wi0_valid = false;
    Vec3 p_wi0 = Vec3::Zero();
    Vec3 p_wi = Vec3::Zero();
    Quat q_wi = Quat::Identity();
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
    for (const auto &frame: visual_local_map_->frames()) {
        // Add imu frame in local map.
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(), camera_extrinsics_.front().p_ic, camera_extrinsics_.front().q_ic, p_wi, q_wi);
        Visualizor3D::poses().emplace_back(PoseType {.p_wb = p_wi, .q_wb = q_wi, .scale = 0.02f});

        // Link relative imu pose.
        if (is_p_wi0_valid) {
            Visualizor3D::lines().emplace_back(LineType {.p_w_i = p_wi0, .p_w_j = p_wi, .color = RgbColor::kWhite});
        }
        p_wi0 = p_wi;
        is_p_wi0_valid = true;

        if (frame.id() == visual_local_map_->frames().back().id()) {
            // Add camera frames in local map for newest frame.
            for (const auto &extrinsic: camera_extrinsics_) {
                Utility::ComputeTransformTransform(p_wi, q_wi, extrinsic.p_ic, extrinsic.q_ic, p_wc, q_wc);
                Visualizor3D::camera_poses().emplace_back(CameraPoseType {.p_wc = p_wc, .q_wc = q_wc, .scale = 0.03f});
            }

            // Link newest frame origin to world frame origin.
            Visualizor3D::dashed_lines().emplace_back(DashedLineType {.p_w_i = p_wi, .p_w_j = Vec3::Zero(), .dot_step = 10, .color = RgbColor::kSlateGray});
        }
    }
}

void DataManager::ShowInformationWithStringsInVisualizor() {
    Visualizor3D::strings().emplace_back(std::string("<local map> ") + std::to_string(visual_local_map_->frames().size()) + std::string(" frames, ") +
                                         std::to_string(visual_local_map_->features().size()) + std::string(" features."));
    RETURN_IF(visual_local_map_->frames().empty());

    Visualizor3D::strings().emplace_back(std::string("<time stamp> ") + std::to_string(visual_local_map_->frames().back().time_stamp_s()) + std::string("s."));
}

void DataManager::UpdateVisualizorCameraView() {
    RETURN_IF(visual_local_map_->frames().empty());

    // Set visualizor camera view by newest frame.
    Vec3 euler = Utility::QuaternionToEuler(Visualizor3D::camera_view().q_wc);
    euler.x() = -90.0f;
    euler.y() = 0.0f;
    Visualizor3D::camera_view().q_wc = Utility::EulerToQuaternion(euler);

    const Vec3 p_c = Vec3(0, 0, 3);
    const Vec3 p_w = Visualizor3D::camera_view().q_wc * p_c + Visualizor3D::camera_view().p_wc;
    Visualizor3D::camera_view().p_wc = visual_local_map_->frames().back().p_wc() - p_w + Visualizor3D::camera_view().p_wc;
}

void DataManager::ShowLocalMapInWorldFrame(const std::string &title, const int32_t delay_ms, const bool block_in_loop) {
    Visualizor3D::Clear();
    ShowLocalMapInWorldFrame();
    ShowInformationWithStringsInVisualizor();

    // Refresh screen.
    UpdateVisualizorCameraView();
    const int32_t delay = delay_ms < 1 ? 0 : delay_ms;
    do {
        Visualizor3D::Refresh(title, delay);
    } while (!Visualizor3D::ShouldQuit() && block_in_loop);
}

void DataManager::ShowMatrixImage(const std::string &title, const Mat &matrix) {
    const uint32_t scale = 3;
    uint8_t *buf = (uint8_t *)malloc(matrix.rows() * matrix.cols() * scale * scale * sizeof(uint8_t));
    GrayImage image_matrix(buf, matrix.rows() * scale, matrix.cols() * scale, true);
    ImagePainter::ConvertMatrixToImage<float>(matrix, image_matrix, 100.0f, scale);
    Visualizor2D::ShowImage(title, image_matrix);
    Visualizor2D::WaitKey(1);
}

void DataManager::ShowSimpleInformationOfVisualLocalMap() {
    for (const auto &frame: imu_based_frames_) {
        ReportInfo(" - imu based frame timestamp_s is " << frame.time_stamp_s);
        frame.imu_preint_block.SimpleInformation();
    }
    for (const auto &frame: visual_local_map_->frames()) {
        frame.SimpleInformation();
    }
}

void DataManager::ShowTinyInformationOfVisualLocalMap() {
    ReportInfo("[DataManager] Visual local map:");
    for (const auto &frame: visual_local_map_->frames()) {
        ReportInfo(" - cam frame " << frame.id() << " at " << frame.time_stamp_s() << "s" << ", q_wc " << LogQuat(frame.q_wc()) << ", p_wc "
                                   << LogVec(frame.p_wc()));
    }
    for (const auto &frame: imu_based_frames_) {
        const auto &imus_vector = frame.packed_measure->imus;
        ReportInfo(" - imu frame at " << frame.time_stamp_s << "s, " << "imu [" << imus_vector.front()->time_stamp_s << " ~ "
                                      << imus_vector.back()->time_stamp_s << "]s" << ", q_wi " << LogQuat(frame.q_wi) << ", p_wi " << LogVec(frame.p_wi)
                                      << ", v_wi " << LogVec(frame.v_wi) << ", bias_a " << LogVec(frame.imu_preint_block.bias_accel()) << ", bias_g "
                                      << LogVec(frame.imu_preint_block.bias_gyro()));
    }
}

}  // namespace VIO
