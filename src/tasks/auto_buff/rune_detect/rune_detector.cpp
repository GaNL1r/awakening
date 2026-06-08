#include "rune_detector.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/image.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>
namespace awakening::auto_buff {
struct RuneDetector::Impl {
    struct Params {
        int bin_threshold;
        int color_diff_thresh;
        double rune_r_min_area;
        double rune_r_max_area;
        double rune_r_1x1ratio_tol;
        double rune_r_fill_ratio_min;
        double rune_pan_min_area;
        double rune_pan_max_area;
        double rune_pan_cluster_radius;
        double rune_pan_max_square_ratio;
        void load(const YAML::Node& config) {
            bin_threshold = config["bin_threshold"].as<int>();
            color_diff_thresh = config["color_diff_thresh"].as<int>();
            rune_r_min_area = config["rune_r_min_area"].as<double>();
            rune_r_max_area = config["rune_r_max_area"].as<double>();
            rune_r_1x1ratio_tol = config["rune_r_1x1ratio_tol"].as<double>();
            rune_r_fill_ratio_min = config["rune_r_fill_ratio_min"].as<double>();
            rune_pan_min_area = config["rune_pan_min_area"].as<double>();
            rune_pan_max_area = config["rune_pan_max_area"].as<double>();
            rune_pan_cluster_radius = config["rune_pan_cluster_radius"].as<double>();
            rune_pan_max_square_ratio = config["rune_pan_max_square_ratio"].as<double>();
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
    }
    cv::Mat preprocess(const cv::Mat& src, PixelFormat format) const noexcept {
        cv::Mat bin;
        if (format == PixelFormat::RGB) {
            cv::cvtColor(src, bin, cv::COLOR_RGB2GRAY);
        } else if (format == PixelFormat::BGR) {
            cv::cvtColor(src, bin, cv::COLOR_BGR2GRAY);
        } else {
            bin = src;
        }
        cv::threshold(bin, bin, params_.bin_threshold, 255, cv::THRESH_BINARY);

        return bin;
    };
    void color_filter(
        const cv::Mat& color,
        PixelFormat format,
        const std::vector<std::vector<cv::Point>>& contours,
        std::vector<bool>& used_flags,
        EnemyColor enemy_color
    ) const noexcept {
        bool need_red = enemy_color == EnemyColor::BLUE;
        for (int i = 0; i < contours.size(); i++) {
            cv::Rect2f r = cv::boundingRect(contours[i]);
            if (r.width < 5 || r.height < 5)
                continue;

            cv::Rect2f rr = r & cv::Rect2f(0, 0, color.cols, color.rows);
            if (rr.width < 2 || rr.height < 2)
                continue;

            const cv::Mat roi = color(rr);
            const cv::Scalar avg = cv::mean(roi);
            double B, G, R;
            switch (format) {
                case PixelFormat::BGR:
                    B = avg[0];
                    G = avg[1];
                    R = avg[2];
                    break;
                case PixelFormat::RGB:
                    R = avg[0];
                    G = avg[1];
                    B = avg[2];
                    break;
                default:
                    B = avg[0];
                    G = avg[1];
                    R = avg[2];
                    break;
            }

            const double diff_RB = R - B;
            const double diff_BR = B - R;

            const bool is_red = (diff_RB > params_.color_diff_thresh);
            const bool is_blue = (diff_BR > params_.color_diff_thresh);

            bool invalid = false;

            if (!need_red) {
                if (is_red)
                    invalid = true;
            } else {
                if (is_blue)
                    invalid = true;
            }

            used_flags[i] = !invalid;

            if (!used_flags[i]) {
            }
        }
    }

    std::vector<RuneR> get_rune_rs(
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<cv::Vec4i>& hierarchy,
        std::vector<bool>& used_flags
    ) const noexcept {
        std::vector<RuneR> result;

        for (int i = 0; i < contours.size(); i++) {
            if (used_flags[i])
                continue;
            if (hierarchy[i][3] != -1)
                continue;

            double area = cv::contourArea(contours[i]);
            if (area < params_.rune_r_min_area || area > params_.rune_r_max_area)
                continue;

            cv::RotatedRect rr = cv::minAreaRect(contours[i]);
            float w = rr.size.width;
            float h = rr.size.height;

            if (w < 5 || h < 5)
                continue;

            double ratio = (w > h ? w / h : h / w);
            if (ratio - 1.0 > params_.rune_r_1x1ratio_tol)
                continue;

            double rect_area = w * h;
            if (rect_area <= 1e-5)
                continue;

            double fill_ratio = area / rect_area;
            if (fill_ratio < params_.rune_r_fill_ratio_min)
                continue;

            result.emplace_back(RuneR { .rr = rr });
        }

        return result;
    }

    inline int find_top_parent(int idx, const std::vector<cv::Vec4i>& hierarchy) const noexcept {
        int p = hierarchy[idx][3]; // parent
        while (p != -1 && hierarchy[p][3] != -1) {
            p = hierarchy[p][3]; // 一直追溯到最顶层 parent
        }
        return p; // 若 p == -1 表示 contour 本身就是顶层轮廓
    }
    inline void
    mark_parent(int idx, const std::vector<cv::Vec4i>& hierarchy, std::vector<bool>& used_flags)
        const noexcept {
        int p = hierarchy[idx][3]; // parent
        while (p != -1 && hierarchy[p][3] != -1) {
            p = hierarchy[p][3]; // 一直追溯到最顶层 parent
            used_flags[p] = true;
        }
    }
    std::vector<RunePan> get_rune_pans(
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<cv::Vec4i>& hierarchy,
        std::vector<bool>& used_flags
    ) const noexcept {
        std::vector<RunePan> results;
        if (hierarchy.empty())
            return results;

        struct Node {
            int idx;
            cv::Point2f center;
            int parent_top_id;
        };
        std::vector<Node> candidates;
        for (int i = 0; i < contours.size(); i++) {
            if (used_flags[i])
                continue;

            const auto& cnt = contours[i];

            double contour_area = cv::contourArea(cnt);
            if (contour_area < params_.rune_pan_min_area
                || contour_area > params_.rune_pan_max_area)
                continue;

            cv::Moments m = cv::moments(cnt);
            if (m.m00 == 0)
                continue;

            cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
            int top_parent = find_top_parent(i, hierarchy);
            candidates.push_back({ i, center, top_parent });
        }

        if (candidates.size() < 3)
            return results;

        std::unordered_map<int, std::vector<int>> groups;
        for (int i = 0; i < candidates.size(); i++) {
            groups[candidates[i].parent_top_id].push_back(i);
        }

        for (auto& [parent_top_id, idx_list]: groups) {
            int M = idx_list.size();
            if (M < 3 || M > 7)
                continue;

            std::vector<int> cluster_id(M, -1);
            int cluster_count = 0;

            for (int i = 0; i < M; i++) {
                if (cluster_id[i] != -1)
                    continue;

                cluster_id[i] = cluster_count;

                std::queue<int> q;
                q.push(i);

                while (!q.empty()) {
                    int u = q.front();
                    q.pop();

                    for (int v = 0; v < M; v++) {
                        if (cluster_id[v] != -1)
                            continue;

                        auto& cu = candidates[idx_list[u]].center;
                        auto& cv = candidates[idx_list[v]].center;

                        double dx = cu.x - cv.x;
                        double dy = cu.y - cv.y;
                        double dist = std::sqrt(dx * dx + dy * dy);

                        if (dist <= params_.rune_pan_cluster_radius) {
                            cluster_id[v] = cluster_count;
                            q.push(v);
                        }
                    }
                }
                cluster_count++;
            }

            std::vector<int> cluster_size(cluster_count, 0);
            for (int id: cluster_id)
                cluster_size[id]++;

            std::vector<std::vector<cv::Point2f>> cluster_points(cluster_count);

            for (int i = 0; i < M; i++) {
                int cid = cluster_id[i];

                if (cluster_size[cid] >= 3) {
                    int contour_index = candidates[idx_list[i]].idx;
                    used_flags[contour_index] = true;
                    mark_parent(contour_index, hierarchy, used_flags);
                    cluster_points[cid].push_back(candidates[idx_list[i]].center);
                }
            }

            for (int cid = 0; cid < cluster_count; cid++) {
                if (cluster_points[cid].size() < 3)
                    continue;

                cv::RotatedRect rr = cv::minAreaRect(cluster_points[cid]);
                double w = rr.size.width;
                double h = rr.size.height;

                if (w < 1 || h < 1)
                    continue;

                double ratio = (w > h ? w / h : h / w);
                if (ratio > params_.rune_pan_max_square_ratio)
                    continue;

                std::vector<std::pair<double, cv::Point2f>> dist_list;
                dist_list.reserve(cluster_points[cid].size());

                for (auto& p: cluster_points[cid]) {
                    double dx = p.x - rr.center.x;
                    double dy = p.y - rr.center.y;
                    double dist = dx * dx + dy * dy;
                    dist_list.emplace_back(dist, p);
                }

                std::sort(dist_list.begin(), dist_list.end(), [](auto& a, auto& b) {
                    return a.first > b.first;
                });
                if (dist_list.size() >= 4) {
                    RunePan pan;
                    pan.center = rr.center;
                    for (int i = 0; i < 4; i++) {
                        pan.corners[i] = dist_list[i].second;
                    }
                    results.push_back(pan);
                }
            }
        }
        return results;
    }
    RuneDetection detect(const CommonFrame& frame, EnemyColor enemy_color) const noexcept {
        RuneDetection result;
        cv::Mat roi = frame.img_frame.src_img(frame.expanded);
        auto bin = preprocess(roi, frame.img_frame.format);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bin, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        std::vector<bool> used_flags;
        used_flags.assign(contours.size(), false);
        color_filter(roi, frame.img_frame.format, contours, used_flags, enemy_color);
        result.pans = get_rune_pans(contours, hierarchy, used_flags);
        result.r_tags = get_rune_rs(contours, hierarchy, used_flags);
        result.add_offset(frame.expanded.tl());

        return result;
    }
};
RuneDetector::RuneDetector(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
RuneDetector::~RuneDetector() noexcept {
    _impl.reset();
}
RuneDetection RuneDetector::detect(const CommonFrame& frame, EnemyColor enemy_color) {
    return _impl->detect(frame, enemy_color);
}
} // namespace awakening::auto_buff