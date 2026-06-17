#include "rune_detector.hpp"
#include "rune_infer.hpp"
#include "tasks/auto_buff/type.hpp"
#include "utils/common/image.hpp"
#include <cstdlib>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#if USE_OPENVINO
    #include "utils/net_detector/openvino/net_detector_openvino.hpp"
#endif
#ifdef USE_TRT
    #include "utils/net_detector/tensorrt/net_detector_tensorrt.hpp"
#endif

namespace awakening::auto_buff {
struct RuneDetector::Impl {
    static constexpr const char* OPENVINO = "openvino";
    static constexpr const char* TENSORRT = "tensorrt";
    struct Params {
        struct CvParams {
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
        } cv_params;

        void load(const YAML::Node& config) {
            cv_params.load(config["cv"]);
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
        auto backend = config["backend"].as<std::string>();
        if (backend != "opencv") {
            rune_infer_ = RuneInfer::create(config["net_detector"]["rune_infer"]);

            const double scale = rune_infer_->use_norm() ? 1.0 / 255.0f : 1.0f;
            auto format = rune_infer_->target_format();
            auto net_cfg = utils::NetDetectorBase::Config {
                .target_format = format,
                .preprocess_scale = scale,
                .target_w = rune_infer_->input_w(),
                .target_h = rune_infer_->input_h(),
            };
            bool backend_valid = false;
#ifdef USE_OPENVINO
            if (backend == OPENVINO) {
                backend_valid = true;
                net_detector_ = std::make_unique<utils::NetDetectorOpenVINO>(
                    config["net_detector"][OPENVINO],
                    net_cfg
                );
            }
#endif
#ifdef USE_TRT
            if (backend == TENSORRT) {
                backend_valid = true;
                net_detector_ = std::make_unique<utils::NetDetectorTensorrt>(
                    config["net_detector"][TENSORRT],
                    net_cfg
                );
            }
#endif
            if (!backend_valid) {
                throw std::runtime_error("Invalid backend");
            }
        }
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
        cv::threshold(bin, bin, params_.cv_params.bin_threshold, 255, cv::THRESH_BINARY);

        return bin;
    };
    RuneColor get_color(const cv::Mat& src, const cv::Rect& rect, PixelFormat format) const {
        if (rect.area() <= 0) {
            return RuneColor::NONE;
        }
        if (rect.x < 0 || rect.y < 0 || rect.x >= src.cols || rect.y >= src.rows) {
            return RuneColor::NONE;
        }

        int x2 = rect.x + rect.width;
        int y2 = rect.y + rect.height;

        if (x2 <= 0 || y2 <= 0 || x2 > src.cols || y2 > src.rows) {
            return RuneColor::NONE;
        }
        const cv::Mat roi = src(rect);
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

        // if (std::abs(R - B) < params_.cv_params.color_diff_thresh) {
        //     return RuneColor::NONE;
        // }
        if (R > B) {
            return RuneColor::RED;
        } else {
            return RuneColor::BLUE;
        }
    }
    void color_filter(
        const cv::Mat& src,
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

            cv::Rect2f rr = r & cv::Rect2f(0, 0, src.cols, src.rows);
            if (rr.width < 2 || rr.height < 2)
                continue;

            auto color = get_color(src, rr, format);
            bool invalid = false;

            if (need_red) {
                if (color != RuneColor::RED) {
                    used_flags[i] = true;
                }
            } else {
                if (color != RuneColor::BLUE) {
                    used_flags[i] = true;
                }
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
            if (area < params_.cv_params.rune_r_min_area
                || area > params_.cv_params.rune_r_max_area)
                continue;
            cv::RotatedRect rr = cv::minAreaRect(contours[i]);
            float w = rr.size.width;
            float h = rr.size.height;

            double ratio = (w > h ? w / h : h / w);
            if (ratio - 1.0 > params_.cv_params.rune_r_1x1ratio_tol)
                continue;

            double rect_area = w * h;

            double fill_ratio = area / rect_area;
            if (fill_ratio < params_.cv_params.rune_r_fill_ratio_min)
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
    std::vector<RuneFanTarget> get_rune_fan_targets(
        const std::vector<std::vector<cv::Point>>& contours,
        const std::vector<cv::Vec4i>& hierarchy,
        std::vector<bool>& used_flags
    ) const noexcept {
        std::vector<RuneFanTarget> results;
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
            if (contour_area < params_.cv_params.rune_pan_min_area
                || contour_area > params_.cv_params.rune_pan_max_area)
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

                        const auto& cu = candidates[idx_list[u]].center;
                        const auto& cv = candidates[idx_list[v]].center;

                        double dx = cu.x - cv.x;
                        double dy = cu.y - cv.y;
                        double dist = std::sqrt(dx * dx + dy * dy);

                        if (dist <= params_.cv_params.rune_pan_cluster_radius) {
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
                if (cid < 0)
                    continue;

                if (cluster_size[cid] >= 3) {
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
                if (ratio > params_.cv_params.rune_pan_max_square_ratio)
                    continue;

                std::vector<std::pair<double, cv::Point2f>> dist_list;
                dist_list.reserve(cluster_points[cid].size());

                for (auto& p: cluster_points[cid]) {
                    double dx = p.x - rr.center.x;
                    double dy = p.y - rr.center.y;
                    double dist = dx * dx + dy * dy;
                    dist_list.emplace_back(dist, p);
                }

                std::sort(dist_list.begin(), dist_list.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

                if (dist_list.size() < 4)
                    continue;

                RuneFanTarget fan_target;
                fan_target.center = rr.center;
                fan_target.rr = rr;

                for (int i = 0; i < 4; i++) {
                    fan_target.corners[i] = dist_list[i].second;
                }

                results.push_back(fan_target);

                for (int i = 0; i < M; i++) {
                    int cid2 = cluster_id[i];
                    if (cid2 < 0)
                        continue;

                    if (cluster_size[cid2] >= 3) {
                        int contour_index = candidates[idx_list[i]].idx;
                        used_flags[contour_index] = true;
                        mark_parent(contour_index, hierarchy, used_flags);
                    }
                }
            }
        }

        return results;
    }
    RuneDetection
    detect(const CommonFrame& frame, const cv::Rect& focus, EnemyColor enemy_color) const noexcept {
        RuneDetection result;
        cv::Mat roi = frame.img_frame.src_img(focus);
        if (net_detector_) {
            utils::NetDetectorBase::OutPut net_output;
            net_output = net_detector_->detect(roi, frame.img_frame.format);
            result.fan_blades = rune_infer_->process(net_output.output);
            for (auto& fan_blade: result.fan_blades) {
                fan_blade.transform(net_output.transform_matrix);
                fan_blade.add_offset(focus.tl());
            }
        }

        auto bin = preprocess(roi, frame.img_frame.format);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(bin, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        std::vector<bool> used_flags;
        used_flags.assign(contours.size(), false);
        // color_filter(roi, frame.img_frame.format, contours, used_flags, enemy_color);
        result.fan_targets = get_rune_fan_targets(contours, hierarchy, used_flags);
        result.rune_rs = get_rune_rs(contours, hierarchy, used_flags);
        for (auto& rune_r: result.rune_rs) {
            rune_r.color = get_color(roi, rune_r.rr.boundingRect2f(), frame.img_frame.format);
            rune_r.add_offset(focus.tl());
        }
        for (auto& fan_target: result.fan_targets) {
            fan_target.color =
                get_color(roi, fan_target.rr.boundingRect2f(), frame.img_frame.format);
            fan_target.add_offset(focus.tl());
        }
        return result;
    }
    utils::NetDetectorBase::Ptr net_detector_;
    RuneInfer::Ptr rune_infer_;
};
RuneDetector::RuneDetector(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
RuneDetector::~RuneDetector() noexcept {
    _impl.reset();
}
RuneDetection
RuneDetector::detect(const CommonFrame& frame, const cv::Rect& focus, EnemyColor enemy_color) {
    return _impl->detect(frame, focus, enemy_color);
}
} // namespace awakening::auto_buff