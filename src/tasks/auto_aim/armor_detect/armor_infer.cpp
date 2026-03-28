#include "armor_infer.hpp"
#include <optional>
namespace awakening::auto_aim::armor_infer {
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};
[[nodiscard]] static inline std::vector<GridAndStride>
generate_grids_and_stride(int target_w, int target_h, const std::vector<int>& strides) noexcept {
    std::vector<GridAndStride> grid_strides;
    for (int stride: strides) {
        const int num_w = target_w / stride;
        const int num_h = target_h / stride;
        grid_strides.reserve(grid_strides.size() + num_w * num_h);
        for (int gy = 0; gy < num_h; ++gy) {
            for (int gx = 0; gx < num_w; ++gx) {
                grid_strides.push_back(GridAndStride { gx, gy, stride });
            }
        }
    }
    return grid_strides;
}
std::vector<ArmorObject> ArmorInfer::postProcessTUP_impl(const cv::Mat& out) const {
    static std::optional<std::vector<GridAndStride>> _grid_strides;
    if (!_grid_strides) {
        _grid_strides = generate_grids_and_stride(inputW(), inputH(), { 8, 16, 32 });
    }
    const auto& grid_strides = _grid_strides.value();
    std::vector<ArmorObject> out_objs;
    const int num_anchors =
        static_cast<int>(std::min<size_t>(grid_strides.size(), static_cast<size_t>(out.rows)));
    for (int a = 0; a < num_anchors; ++a) {
        const float confidence = out.at<float>(a, 8);
        if (confidence < conf_threshold_)
            continue;

        const auto& gs = grid_strides[a];
        const int gx = gs.grid0, gy = gs.grid1, stride = gs.stride;

        // color & class
        const int color_offset = 9;
        const int num_colors = ModelTraits<Mode::TUP>::NUM_COLORS;
        const int num_classes = ModelTraits<Mode::TUP>::NUM_CLASSES;

        cv::Mat color_scores = out.row(a).colRange(color_offset, color_offset + num_colors);
        cv::Mat class_scores =
            out.row(a).colRange(color_offset + num_colors, color_offset + num_colors + num_classes);

        double max_color, max_class;
        cv::Point color_id, class_id;
        cv::minMaxLoc(color_scores, nullptr, &max_color, nullptr, &color_id);
        cv::minMaxLoc(class_scores, nullptr, &max_class, nullptr, &class_id);

        const float x1 = (out.at<float>(a, 0) + gx) * stride;
        const float y1 = (out.at<float>(a, 1) + gy) * stride;
        const float x2 = (out.at<float>(a, 2) + gx) * stride;
        const float y2 = (out.at<float>(a, 3) + gy) * stride;
        const float x3 = (out.at<float>(a, 4) + gx) * stride;
        const float y3 = (out.at<float>(a, 5) + gy) * stride;
        const float x4 = (out.at<float>(a, 6) + gx) * stride;
        const float y4 = (out.at<float>(a, 7) + gy) * stride;

        ArmorObject obj;
        obj.pts = { cv::Point2f(x1, y1),
                    cv::Point2f(x2, y2),
                    cv::Point2f(x3, y3),
                    cv::Point2f(x4, y4) };
        obj.box = cv::boundingRect(obj.pts);
        obj.color = static_cast<ArmorColor>(color_id.x);
        obj.number = static_cast<ArmorNumber>(class_id.x);
        obj.confidence = confidence;
        out_objs.push_back(std::move(obj));
    }
    return topKAndNms(out_objs, top_k_, nms_threshold_);
}

std::vector<ArmorObject> ArmorInfer::postProcessRP_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;
    const int rows = out.rows;
    const int color_offset = 9;
    const int num_colors = ModelTraits<Mode::RP>::NUM_COLORS;
    const int num_classes = ModelTraits<Mode::RP>::NUM_CLASSES;

    for (int r = 0; r < rows; ++r) {
        float conf_raw = out.at<float>(r, 8);
        const float confidence = static_cast<float>(sigmoid(conf_raw));
        if (confidence < conf_threshold_)
            continue;

        cv::Mat color_scores = out.row(r).colRange(color_offset, color_offset + num_colors);
        cv::Mat class_scores =
            out.row(r).colRange(color_offset + num_colors, color_offset + num_colors + num_classes);

        double max_color_score, max_class_score;
        cv::Point color_id, class_id;
        cv::minMaxLoc(color_scores, nullptr, &max_color_score, nullptr, &color_id);
        cv::minMaxLoc(class_scores, nullptr, &max_class_score, nullptr, &class_id);

        ArmorObject obj;
        obj.pts.resize(4);
        for (int k = 0; k < 4; ++k) {
            const float x = out.at<float>(r, 0 + k * 2);
            const float y = out.at<float>(r, 1 + k * 2);
            obj.pts[k] = cv::Point2f(x, y);
        }
        obj.box = cv::boundingRect(obj.pts);

        obj.color = static_cast<ArmorColor>(color_id.x);

        obj.number = static_cast<ArmorNumber>(class_id.x);
        obj.confidence = confidence;
        out_objs.push_back(std::move(obj));
    }

    return topKAndNms(out_objs, top_k_, nms_threshold_);
}

std::vector<ArmorObject> ArmorInfer::postProcessAT_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;

    constexpr int nkpt = ModelTraits<Mode::AT>::NUM_KPTS;
    constexpr int nk = nkpt * 2; // keypoints flattened
    auto max_det = out.rows;
    auto det_dim = out.cols;
    auto output_ptr = out.ptr<float>();
    for (int i = 0; i < max_det; ++i) {
        const float* row = output_ptr + i * det_dim;
        float conf = row[4];
        if (!std::isfinite(conf) || conf < conf_threshold_)
            continue;
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];
        int cls = static_cast<int>(row[5]);

        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)
            || x2 <= x1 || y2 <= y1)
            continue;

        ArmorObject obj;
        obj.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        obj.confidence = conf;
        auto color_num = ModelTraits<Mode::AT>::CLASSES[cls];
        obj.color = color_num.first;
        obj.number = color_num.second;

        obj.pts.reserve(nkpt);
        for (int k = 0; k < nkpt; ++k) {
            float kx = row[6 + 2 * k];
            float ky = row[6 + 2 * k + 1];
            obj.pts.emplace_back(kx, ky);
        }

        out_objs.emplace_back(std::move(obj));
    }

    return out_objs;
}
std::vector<ArmorObject> ArmorInfer::postProcessBOX_impl(const cv::Mat& out) const {
    std::vector<ArmorObject> out_objs;
    auto max_det = out.rows;
    auto det_dim = out.cols;
    auto output_ptr = out.ptr<float>();
    for (int i = 0; i < max_det; ++i) {
        const float* row = output_ptr + i * det_dim;
        float conf = row[4];
        if (!std::isfinite(conf) || conf < conf_threshold_)
            continue;
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];
        int cls = static_cast<int>(row[5]);

        if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)
            || x2 <= x1 || y2 <= y1)
            continue;

        ArmorObject obj;
        obj.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        obj.confidence = conf;
        auto color_num = ModelTraits<Mode::BOX>::CLASSES[cls];
        obj.color = color_num.first;
        obj.number = color_num.second;
        std::vector<cv::Point2f> pts;
        pts.resize(4);
        pts[0] = cv::Point2f(obj.box.x + obj.box.width, obj.box.y + obj.box.height); // 右下
        pts[1] = cv::Point2f(obj.box.x + obj.box.width, obj.box.y); // 右上
        pts[2] = cv::Point2f(obj.box.x, obj.box.y); // 左上
        pts[3] = cv::Point2f(obj.box.x, obj.box.y + obj.box.height); // 左下
        obj.pts = std::move(pts);

        out_objs.emplace_back(std::move(obj));
    }
    return out_objs;
}
} // namespace awakening::auto_aim::armor_infer