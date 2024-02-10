#include "data_manager.h"
#include "log_report.h"
#include "slam_memory.h"
#include "math_kinematics.h"
#include "visualizor.h"
#include "visualizor_3d.h"

using namespace SLAM_VISUALIZOR;

namespace VIO {

namespace {
    constexpr int32_t kMaxImageNumInOneRow = 3;
}

RgbPixel DataManager::GetFeatureColor(const FeatureType &feature) {
    RgbPixel pixel_color = RgbPixel{.r = 255, .g = 255, .b = 0};
    switch (feature.status()) {
        case FeatureSolvedStatus::kSolved:
            if (feature.observes().size() > 1) {
                // If this feature is observed in different frame.
                pixel_color = RgbPixel{.r = 0, .g = 255, .b = 0};
            } else {
                // If this feature is only observed in one frame but has stereo view.
                pixel_color = RgbPixel{.r = 255, .g = 255, .b = 0};
            }
            break;
        case FeatureSolvedStatus::kMarginalized:
            pixel_color = RgbPixel{.r = 0, .g = 0, .b = 255};
            break;
        default:
        case FeatureSolvedStatus::kUnsolved:
            pixel_color = RgbPixel{.r = 255, .g = 0, .b = 0};
            break;
    }

    if (feature.observes().size() == 1 && feature.observes().front().size() == 1) {
        pixel_color = RgbPixel{.r = 255, .g = 255, .b = 255};
    }

    return pixel_color;
}

void DataManager::ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id,
                                                   const uint32_t cur_frame_id,
                                                   const int32_t delay_ms) {
    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    if (!visual_local_map_->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        ReportError("[DataManager] Failed to get covisible features between frame " << ref_frame_id << " and " << cur_frame_id << ".");
        return;
    }

    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    for (const auto &feature_ptr : covisible_features) {
        ref_pixel_uv.emplace_back(feature_ptr->observe(ref_frame_id)[0].raw_pixel_uv);
        cur_pixel_uv.emplace_back(feature_ptr->observe(cur_frame_id)[0].raw_pixel_uv);
    }

    // Create gray image of ref and cur image.
    const GrayImage ref_image(visual_local_map_->frame(ref_frame_id)->raw_images()[0]);
    const GrayImage cur_image(visual_local_map_->frame(cur_frame_id)->raw_images()[0]);

    // Draw tracking results.
    const std::vector<uint8_t> tracked_status(ref_pixel_uv.size(), 1);
    Visualizor::ShowImageWithTrackedFeatures(std::string("Raw image [ ") + std::to_string(ref_frame_id) + std::string(" | ") +
        std::to_string(cur_frame_id) + std::string(" ] covisible features"), ref_image, cur_image,
        ref_pixel_uv, cur_pixel_uv, tracked_status);

    Visualizor::WaitKey(delay_ms);
}

void DataManager::ShowLocalMapFramesAndFeatures(const int32_t feature_id, const int32_t camera_id, const int32_t delay_ms) {
    RETURN_IF(visual_local_map_->frames().empty());
    RETURN_IF(visual_local_map_->frames().front().raw_images().empty());

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images = options_.kMaxStoredKeyFrames % cols_of_images == 0 ?
        options_.kMaxStoredKeyFrames / cols_of_images :
        options_.kMaxStoredKeyFrames / cols_of_images + 1;
    const int32_t image_cols = visual_local_map_->frames().front().raw_images()[camera_id].cols();
    const int32_t image_rows = visual_local_map_->frames().front().raw_images()[camera_id].rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (auto &frame : visual_local_map_->frames()) {
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
    Visualizor::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (const auto &frame : visual_local_map_->frames()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color = frame_id >= static_cast<int32_t>(visual_local_map_->frames().size() - options_.kMaxStoredKeyFrames) ?
            RgbPixel{.r = 255, .g = 0, .b = 0} : RgbPixel{.r = 0, .g = 255, .b = 0};
        Visualizor::DrawString(show_image, std::string("[ ") + std::to_string(frame.id()) + std::string(" | ") + std::to_string(frame.time_stamp_s()) + std::string("s ]"),
            col_offset, row_offset, info_color, font_size);
        // Draw all observed features in this frame and this camera image.
        for (auto &pair : frame.features()) {
            auto &feature = pair.second;
            auto &observe = feature->observe(frame.id());
            CONTINUE_IF(feature_id > 0 && static_cast<uint32_t>(feature_id) != feature->id());
            CONTINUE_IF(static_cast<int32_t>(observe.size()) <= camera_id);

            // Draw feature in rgb image.
            Vec2 pixel_uv = observe[camera_id].raw_pixel_uv;
            const RgbPixel pixel_color = GetFeatureColor(*feature);
            const std::string feature_text = feature->first_frame_id() == frame.id() ? std::to_string(feature->id()) + std::string("+") : std::to_string(feature->id());
            Visualizor::DrawSolidCircle(show_image, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, 3, pixel_color);
            Visualizor::DrawString(show_image, feature_text, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, pixel_color);
        }
        // Accumulate index.
        ++frame_id;
    }

    const std::vector<std::string> camera_name = {"left", "right"};
    Visualizor::ShowImage(std::string("Local map [") + camera_name[camera_id] + std::string("] <distorted>"), show_image);
    Visualizor::WaitKey(delay_ms);
}

void DataManager::ShowAllFramesWithBias(const int32_t delay_ms) {
    RETURN_IF(frames_with_bias_.empty());
    RETURN_IF(frames_with_bias_.front().packed_measure->left_image == nullptr);

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images = options_.kMaxStoredKeyFrames % cols_of_images == 0 ?
        options_.kMaxStoredKeyFrames / cols_of_images :
        options_.kMaxStoredKeyFrames / cols_of_images + 1;
    const int32_t image_cols = frames_with_bias_.front().packed_measure->left_image->image.cols();
    const int32_t image_rows = frames_with_bias_.front().packed_measure->left_image->image.rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (auto &frame_with_bias : frames_with_bias_) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Load image.
        show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = frame_with_bias.packed_measure->left_image->image;
        // Accumulate index.
        ++frame_id;
    }
    GrayImage gray_show_image(show_image_mat);
    uint8_t *show_image_buf = (uint8_t *)SlamMemory::Malloc(show_image_rows * show_image_cols * 3 * sizeof(uint8_t));
    RgbImage show_image(show_image_buf, show_image_rows, show_image_cols, true);
    Visualizor::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (auto &frame_with_bias : frames_with_bias_) {
        CONTINUE_IF(frame_with_bias.packed_measure == nullptr || frame_with_bias.visual_measure == nullptr);
        CONTINUE_IF((frame_with_bias.packed_measure->left_image == nullptr));

        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color = frame_id >= static_cast<int32_t>(options_.kMaxStoredKeyFrames - options_.kMaxStoredKeyFrames) ?
            RgbPixel{.r = 255, .g = 0, .b = 0} : RgbPixel{.r = 0, .g = 255, .b = 0};
        Visualizor::DrawString(show_image, std::string("[ ") + std::to_string(frame_with_bias.time_stamp_s) + std::string("s ]"),
            col_offset, row_offset, info_color, font_size);

        // Draw all observed features in this frame and this camera image.
        for (uint32_t i = 0; i < frame_with_bias.visual_measure->features_id.size(); ++i) {
            const Vec2 pixel_uv = frame_with_bias.visual_measure->observes_per_frame[i][0].raw_pixel_uv + Vec2(col_offset, row_offset);
            const RgbPixel pixel_color = RgbPixel{.r = 0, .g = 255, .b = 127};
            Visualizor::DrawSolidCircle(show_image, pixel_uv.x(), pixel_uv.y(), 3, pixel_color);
            Visualizor::DrawString(show_image, std::to_string(frame_with_bias.visual_measure->features_id[i]),
                pixel_uv.x(), pixel_uv.y(), pixel_color);
        }

        // Accumulate index.
        ++frame_id;
    }

    Visualizor::ShowImage(std::string("New frames with bias [left] <distorted>"), show_image);
    Visualizor::WaitKey(delay_ms);
}

void DataManager::ShowLocalMapInWorldFrame(const std::string &title, const int32_t delay_ms, const bool block_in_loop) {
    Visualizor3D::Clear();

    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });

    RETURN_IF(visual_local_map_->features().empty());
    RETURN_IF(visual_local_map_->frames().empty());

    // Add all features in locap map.
    for (const auto &pair : visual_local_map_->features()) {
        const auto &feature = pair.second;
        Visualizor3D::points().emplace_back(PointType{
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
    for (const auto &frame : visual_local_map_->frames()) {
        // Add imu frame in local map.
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(),
            camera_extrinsics_.front().p_ic,
            camera_extrinsics_.front().q_ic, p_wi, q_wi);
        Visualizor3D::poses().emplace_back(PoseType{ .p_wb = p_wi, .q_wb = q_wi, .scale = 0.02f });

        // Link relative imu pose.
        if (is_p_wi0_valid) {
            Visualizor3D::lines().emplace_back(LineType{ .p_w_i = p_wi0, .p_w_j = p_wi, .color = RgbPixel{.r = 255, .g = 255, .b = 255} });
        }
        p_wi0 = p_wi;
        is_p_wi0_valid = true;

        // Add camera frames in local map for newest frame.
        if (frame.id() == visual_local_map_->frames().back().id()) {
            Visualizor3D::poses().back().scale = 0.1f;
            for (const auto &extrinsic : camera_extrinsics_) {
                Utility::ComputeTransformTransform(p_wi, q_wi, extrinsic.p_ic, extrinsic.q_ic, p_wc, q_wc);
                Visualizor3D::poses().emplace_back(PoseType{ .p_wb = p_wc, .q_wb = q_wc, .scale = 0.01f });
            }
        }
    }

    // Set visualizor camera view by newest frame.
    const Vec3 p_c = Vec3(0, 0, 1);
    const Vec3 p_w = Visualizor3D::camera_view().q_wc * p_c + Visualizor3D::camera_view().p_wc;
    Visualizor3D::camera_view().p_wc = visual_local_map_->frames().back().p_wc() - p_w + Visualizor3D::camera_view().p_wc;

    // Refresh screen.
    const int32_t delay = delay_ms < 1 ? 0 : delay_ms;
    do {
        Visualizor3D::Refresh(title, delay);
    } while (!Visualizor3D::ShouldQuit() && block_in_loop);
}

void DataManager::ShowMatrixImage(const std::string &title, const Mat &matrix) {
    const uint32_t scale = 3;
    uint8_t *buf = (uint8_t *)malloc(matrix.rows() * matrix.cols() * scale * scale * sizeof(uint8_t));
    GrayImage image_matrix(buf, matrix.rows() * scale, matrix.cols() * scale, true);
    Visualizor::ConvertMatrixToImage<float>(matrix, image_matrix, 100.0f, scale);
    Visualizor::ShowImage(title, image_matrix);
    Visualizor::WaitKey(1);
}

void DataManager::ShowSimpleInformationOfVisualLocalMap() {
    for (const auto &frame : frames_with_bias_) {
        ReportInfo(" - Frame with bias timestamp_s is " << frame.time_stamp_s);
        frame.imu_preint_block.SimpleInformation();
    }
    for (const auto &frame : visual_local_map_->frames()) {
        frame.SimpleInformation();
    }
}

void DataManager::ShowTinyInformationOfVisualLocalMap() {
    ReportInfo("[DataManager] Visual local map:");
    for (const auto &frame : visual_local_map_->frames()) {
        ReportInfo(" - cam frame " << frame.id() <<
            " at " << frame.time_stamp_s() << "s" <<
            ", q_wc " << LogQuat(frame.q_wc()) <<
            ", p_wc " << LogVec(frame.p_wc()) <<
            ", v_wc " << LogVec(frame.v_w()));
    }
    for (const auto &frame : frames_with_bias_) {
        const auto &imus_vector = frame.packed_measure->imus;
        ReportInfo(" - imu frame at " << frame.time_stamp_s << "s, " <<
            "imu [" << imus_vector.front()->time_stamp_s << " ~ " << imus_vector.back()->time_stamp_s << "]s" <<
            ", q_wi " << LogQuat(frame.q_wi) <<
            ", p_wi " << LogVec(frame.p_wi) <<
            ", v_wi " << LogVec(frame.v_wi) <<
            ", bias_a " << LogVec(frame.imu_preint_block.bias_accel()) <<
            ", bias_g " << LogVec(frame.imu_preint_block.bias_gyro()));
    }
}

}
