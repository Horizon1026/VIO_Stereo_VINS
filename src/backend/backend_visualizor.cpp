#include "backend.h"
#include "log_report.h"
#include "slam_memory.h"
#include "math_kinematics.h"
#include "visualizor.h"
#include "visualizor_3d.h"

using namespace SLAM_VISUALIZOR;

namespace VIO {

namespace {
    constexpr int32_t kMaxImageNumInOneRow = 4;
}

RgbPixel Backend::GetFeatureColor(const FeatureType &feature) {
    RgbPixel pixel_color = RgbPixel{.r = 255, .g = 255, .b = 0};
    switch (feature.status()) {
        case FeatureSolvedStatus::kSolved:
            if (feature.observes().size() > 1) {
                // If this feature is observed in different frame.
                pixel_color = RgbPixel{.r = 0, .g = 255, .b = 0};
                // If this feature is observed in newset keyframe.
                if (feature.first_frame_id() == data_manager_->GetNewestKeyframeId()) {
                    pixel_color = RgbPixel{.r = 255, .g = 0, .b = 255};
                }
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

void Backend::ShowFeaturePairsBetweenTwoFrames(const uint32_t ref_frame_id,
                                               const uint32_t cur_frame_id,
                                               const bool use_rectify,
                                               const int32_t delay_ms) {
    // Get covisible features only in left camera.
    std::vector<FeatureType *> covisible_features;
    if (!data_manager_->visual_local_map()->GetCovisibleFeatures(ref_frame_id, cur_frame_id, covisible_features)) {
        ReportError("[Backend] Failed to get covisible features between frame " << ref_frame_id << " and " << cur_frame_id << ".");
        return;
    }

    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    if (use_rectify) {
        Vec2 rectify_pixel_uv;
        for (const auto &feature_ptr : covisible_features) {
            visual_frontend_->camera_models()[0]->LiftFromNormalizedPlaneToImagePlane(feature_ptr->observe(ref_frame_id)[0].rectified_norm_xy, rectify_pixel_uv);
            ref_pixel_uv.emplace_back(rectify_pixel_uv);
            visual_frontend_->camera_models()[0]->LiftFromNormalizedPlaneToImagePlane(feature_ptr->observe(cur_frame_id)[0].rectified_norm_xy, rectify_pixel_uv);
            cur_pixel_uv.emplace_back(rectify_pixel_uv);
        }
    } else {
        for (const auto &feature_ptr : covisible_features) {
            ref_pixel_uv.emplace_back(feature_ptr->observe(ref_frame_id)[0].raw_pixel_uv);
            cur_pixel_uv.emplace_back(feature_ptr->observe(cur_frame_id)[0].raw_pixel_uv);
        }
    }

    // Create gray image of ref and cur image.
    const GrayImage ref_image(data_manager_->visual_local_map()->frame(ref_frame_id)->raw_images()[0]);
    const GrayImage cur_image(data_manager_->visual_local_map()->frame(cur_frame_id)->raw_images()[0]);

    // Draw tracking results.
    const std::vector<uint8_t> tracked_status(ref_pixel_uv.size(), 1);
    if (use_rectify) {
        // Correct distorted image.
        MatImg ref_mat_image, cur_mat_image;
        ref_mat_image.resize(ref_image.rows(), ref_image.cols());
        cur_mat_image.resize(ref_image.rows(), ref_image.cols());
        GrayImage ref_rectify_image(ref_mat_image);
        GrayImage cur_rectify_image(cur_mat_image);
        visual_frontend_->camera_models()[0]->CorrectDistortedImage(ref_image, ref_rectify_image);
        visual_frontend_->camera_models()[0]->CorrectDistortedImage(cur_image, cur_rectify_image);

        Visualizor::ShowImageWithTrackedFeatures(std::string("Recify image [ ") + std::to_string(ref_frame_id) + std::string(" | ") +
            std::to_string(cur_frame_id) + std::string(" ] covisible features"), ref_rectify_image, cur_rectify_image,
            ref_pixel_uv, cur_pixel_uv, tracked_status);
    } else {
        Visualizor::ShowImageWithTrackedFeatures(std::string("Raw image [ ") + std::to_string(ref_frame_id) + std::string(" | ") +
            std::to_string(cur_frame_id) + std::string(" ] covisible features"), ref_image, cur_image,
            ref_pixel_uv, cur_pixel_uv, tracked_status);
    }

    Visualizor::WaitKey(delay_ms);
}

void Backend::ShowMatrixImage(const std::string &title, const TMat<DorF> &matrix) {
    const uint32_t scale = 3;
    uint8_t *buf = (uint8_t *)malloc(matrix.rows() * matrix.cols() * scale * scale * sizeof(uint8_t));
    GrayImage image_matrix(buf, matrix.rows() * scale, matrix.cols() * scale, true);
    Visualizor::ConvertMatrixToImage<DorF>(matrix, image_matrix, 100.0f, scale);
    Visualizor::ShowImage(title, image_matrix);
    Visualizor::WaitKey(1);
}

void Backend::ShowLocalMapFramesAndFeatures(const int32_t feature_id, const int32_t camera_id, const bool use_rectify, const int32_t delay_ms) {
    RETURN_IF(data_manager_->visual_local_map()->frames().empty());
    RETURN_IF(data_manager_->visual_local_map()->frames().front().raw_images().empty());

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images = data_manager_->options().kMaxStoredKeyFrames % cols_of_images == 0 ?
        data_manager_->options().kMaxStoredKeyFrames / cols_of_images :
        data_manager_->options().kMaxStoredKeyFrames / cols_of_images + 1;
    const int32_t image_cols = data_manager_->visual_local_map()->frames().front().raw_images()[camera_id].cols();
    const int32_t image_rows = data_manager_->visual_local_map()->frames().front().raw_images()[camera_id].rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (auto &frame : data_manager_->visual_local_map()->frames()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Load image.
        if (use_rectify) {
            GrayImage raw_gray_image(frame.raw_images()[camera_id]);
            MatImg rectify_image_mat = MatImg::Zero(raw_gray_image.rows(), raw_gray_image.cols());
            GrayImage rectify_gray_image(rectify_image_mat);
            visual_frontend_->camera_models()[camera_id]->CorrectDistortedImage(raw_gray_image, rectify_gray_image);
            show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = rectify_image_mat;
        } else {
            show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = frame.raw_images()[camera_id];
        }
        // Accumulate index.
        ++frame_id;
    }
    GrayImage gray_show_image(show_image_mat);
    uint8_t *show_image_buf = (uint8_t *)SlamMemory::Malloc(show_image_rows * show_image_cols * 3 * sizeof(uint8_t));
    RgbImage show_image(show_image_buf, show_image_rows, show_image_cols, true);
    Visualizor::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color = frame_id >= static_cast<int32_t>(data_manager_->visual_local_map()->frames().size() - data_manager_->options().kMaxStoredNewFrames) ?
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
            if (use_rectify) {
                visual_frontend_->camera_models()[camera_id]->LiftFromNormalizedPlaneToImagePlane(observe[camera_id].rectified_norm_xy, pixel_uv);
                CONTINUE_IF(pixel_uv.x() < 0 || pixel_uv.x() > image_cols || pixel_uv.y() < 0 || pixel_uv.y() > image_rows);
            }
            const RgbPixel pixel_color = GetFeatureColor(*feature);
            const std::string feature_text = feature->first_frame_id() == frame.id() ? std::to_string(feature->id()) + std::string("+") : std::to_string(feature->id());
            Visualizor::DrawSolidCircle(show_image, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, 3, pixel_color);
            Visualizor::DrawString(show_image, feature_text, pixel_uv.x() + col_offset, pixel_uv.y() + row_offset, pixel_color);
        }
        // Accumulate index.
        ++frame_id;
    }

    const std::string status_of_distortion = use_rectify ? "rectify" : "distorted";
    const std::vector<std::string> camera_name = {"left", "right"};
    Visualizor::ShowImage(std::string("Local map [") + camera_name[camera_id] + std::string("] <") + status_of_distortion + std::string(">"), show_image);
    Visualizor::WaitKey(delay_ms);
}

void Backend::ShowAllFramesWithBias(const bool use_rectify, const int32_t delay_ms) {
    RETURN_IF(data_manager_->frames_with_bias().empty());
    RETURN_IF(data_manager_->frames_with_bias().front().packed_measure->left_image == nullptr);

    // Memory allocation.
    const int32_t cols_of_images = kMaxImageNumInOneRow;
    const int32_t rows_of_images = data_manager_->options().kMaxStoredNewFrames % cols_of_images == 0 ?
        data_manager_->options().kMaxStoredNewFrames / cols_of_images :
        data_manager_->options().kMaxStoredNewFrames / cols_of_images + 1;
    const int32_t image_cols = data_manager_->frames_with_bias().front().packed_measure->left_image->image.cols();
    const int32_t image_rows = data_manager_->frames_with_bias().front().packed_measure->left_image->image.rows();
    const int32_t show_image_cols = image_cols * cols_of_images;
    const int32_t show_image_rows = image_rows * rows_of_images;

    // Load all frame images.
    int32_t frame_id = 0;
    MatImg show_image_mat = MatImg::Zero(show_image_rows, show_image_cols);
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Load image.
        if (use_rectify) {
            GrayImage raw_gray_image(frame_with_bias.packed_measure->left_image->image);
            MatImg rectify_image_mat = MatImg::Zero(raw_gray_image.rows(), raw_gray_image.cols());
            GrayImage rectify_gray_image(rectify_image_mat);
            visual_frontend_->camera_models()[0]->CorrectDistortedImage(raw_gray_image, rectify_gray_image);
            show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = rectify_image_mat;
        } else {
            show_image_mat.block(row_offset, col_offset, image_rows, image_cols) = frame_with_bias.packed_measure->left_image->image;
        }
        // Accumulate index.
        ++frame_id;
    }
    GrayImage gray_show_image(show_image_mat);
    uint8_t *show_image_buf = (uint8_t *)SlamMemory::Malloc(show_image_rows * show_image_cols * 3 * sizeof(uint8_t));
    RgbImage show_image(show_image_buf, show_image_rows, show_image_cols, true);
    Visualizor::ConvertUint8ToRgb(gray_show_image.data(), show_image.data(), gray_show_image.rows() * gray_show_image.cols());

    // Iterate all frames in local map.
    frame_id = 0;
    for (auto &frame_with_bias : data_manager_->frames_with_bias()) {
        CONTINUE_IF(frame_with_bias.packed_measure == nullptr || frame_with_bias.visual_measure == nullptr);
        CONTINUE_IF((frame_with_bias.packed_measure->left_image == nullptr));

        // Compute location offset.
        const int32_t row_offset = image_rows * (frame_id / cols_of_images);
        const int32_t col_offset = image_cols * (frame_id % cols_of_images);
        // Type basic information of each frame.
        const int32_t font_size = 16;
        const RgbPixel info_color = frame_id >= static_cast<int32_t>(data_manager_->options().kMaxStoredKeyFrames - data_manager_->options().kMaxStoredNewFrames) ?
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

    const std::string status_of_distortion = use_rectify ? "rectify" : "distorted";
    Visualizor::ShowImage(std::string("New frames with bias [left] <") + status_of_distortion + std::string(">"), show_image);
    Visualizor::WaitKey(delay_ms);
}

void Backend::ShowLocalMapInWorldFrame(const int32_t delay_ms, const bool block_in_loop) {
    Visualizor3D::Clear();

    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });

    RETURN_IF(data_manager_->visual_local_map()->features().empty());
    RETURN_IF(data_manager_->visual_local_map()->frames().empty());

    // Add all features in locap map.
    for (const auto &pair : data_manager_->visual_local_map()->features()) {
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
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        // Add imu frame in local map.
        Utility::ComputeTransformTransformInverse(frame.p_wc(), frame.q_wc(),
            data_manager_->camera_extrinsics().front().p_ic,
            data_manager_->camera_extrinsics().front().q_ic, p_wi, q_wi);
        Visualizor3D::poses().emplace_back(PoseType{ .p_wb = p_wi, .q_wb = q_wi, .scale = 0.02f });

        // Link relative imu pose.
        if (is_p_wi0_valid) {
            Visualizor3D::lines().emplace_back(LineType{ .p_w_i = p_wi0, .p_w_j = p_wi, .color = RgbPixel{.r = 255, .g = 255, .b = 255} });
        }
        p_wi0 = p_wi;
        is_p_wi0_valid = true;

        // Add camera frames in local map for newest frame.
        if (frame.id() == data_manager_->visual_local_map()->frames().back().id()) {
            Visualizor3D::poses().back().scale = 0.1f;
            for (const auto &extrinsic : data_manager_->camera_extrinsics()) {
                Utility::ComputeTransformTransform(p_wi, q_wi, extrinsic.p_ic, extrinsic.q_ic, p_wc, q_wc);
                Visualizor3D::poses().emplace_back(PoseType{ .p_wb = p_wc, .q_wb = q_wc, .scale = 0.01f });
            }
        }
    }

    // Set visualizor camera view by newest frame.
    const Vec3 p_c = Vec3(0, 0, 0.3);
    const Vec3 p_w = Visualizor3D::camera_view().q_wc * p_c + Visualizor3D::camera_view().p_wc;
    Visualizor3D::camera_view().p_wc = data_manager_->visual_local_map()->frames().back().p_wc() - p_w + Visualizor3D::camera_view().p_wc;

    // Refresh screen.
    const int32_t delay = delay_ms < 1 ? 0 : delay_ms;
    do {
        Visualizor3D::Refresh("Visualizor 3D", delay);
    } while (!Visualizor3D::ShouldQuit() && block_in_loop);
}

void Backend::ShowSimpleInformationOfVisualLocalMap() {
    for (const auto &frame : data_manager_->frames_with_bias()) {
        ReportInfo(" - Frame with bias timestamp_s is " << frame.time_stamp_s);
        frame.imu_preint_block.SimpleInformation();
    }
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        frame.SimpleInformation();
    }
}

void Backend::ShowTinyInformationOfVisualLocalMap() {
    ReportInfo("[Backend] Visual local map:");
    for (const auto &frame : data_manager_->visual_local_map()->frames()) {
        ReportInfo(" - frame " << frame.id() << " at " << frame.time_stamp_s() << "s, " <<
            " q_wc " << LogQuat(frame.q_wc()) << ", p_wc " << LogVec(frame.p_wc()) <<
            ", v_w " << LogVec(frame.v_w()));
    }
    for (const auto &frame : data_manager_->frames_with_bias()) {
        ReportInfo(" - frame with bias at " << frame.time_stamp_s << "s, " <<
            "bias a " << LogVec(frame.imu_preint_block.bias_accel()) << ", bias g " <<
            LogVec(frame.imu_preint_block.bias_gyro()));
    }
}

}
