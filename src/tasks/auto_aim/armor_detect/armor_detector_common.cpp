// Copyright 2025 Xiaojian Wu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "tasks/auto_aim/armor_detect/armor_detector_common.hpp"
#include "common.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector_base.hpp"
#include "tasks/auto_aim/armor_detect/number_classifier/factory.hpp"
#include "utils/utils.hpp"
#include "wust_vl/common/utils/timer.hpp"
namespace awakening {
namespace auto_aim {
    struct ArmorDetectorCommon::Impl {
    public:
        Impl(const YAML::Node& config) {
            params_.load(config);
            number_classifier_ = NumberClassifierFactory::createNumberClassifier(
                params_.classify_backend,
                params_.classify_model_path,
                params_.classify_label_path
            );
        }

        bool extractNetImage(const cv::Mat& src, ArmorObject& armor) const noexcept {
            constexpr int light_length = 12;
            constexpr int warp_height = 28;
            constexpr int small_armor_width = 32;
            constexpr int large_armor_width = 54;
            const cv::Size roi_size(20, 28);

            if (src.empty() || src.cols < 10 || src.rows < 10) {
                std::cerr << "[extractNetImage] input src is empty or too small!" << std::endl;
                return false;
            }

            const auto ordered = armor.sortCorners(armor.pts);

            const cv::Point2f& p0 = ordered[0];
            const cv::Point2f& p1 = ordered[1];
            const cv::Point2f& p2 = ordered[2];
            const cv::Point2f& p3 = ordered[3];

            const float l1_len = cv::norm(p1 - p0);
            const float l2_len = cv::norm(p2 - p3);
            const cv::Point2f c1 = (p0 + p1) * 0.5f;
            const cv::Point2f c2 = (p2 + p3) * 0.5f;

            const float avg_light_len = 0.5f * (l1_len + l2_len);
            const float center_dist =
                avg_light_len > 1e-3f ? cv::norm(c1 - c2) / avg_light_len : 0.f;

            const bool is_large = center_dist > params_.armor_params.min_large_center_distance;

            const cv::Rect bbox = cv::boundingRect(armor.pts);
            if (bbox.width <= 0 || bbox.height <= 0)
                return false;

            if (bbox.width > src.cols || bbox.height > src.rows)
                return false;
            const int dw = static_cast<int>(bbox.width * (params_.expand_ratio_w - 1.f));
            const int dh = static_cast<int>(bbox.height * (params_.expand_ratio_h - 1.f));

            int new_x = bbox.x - (dw >> 1);
            int new_y = bbox.y - (dh >> 1);
            new_x = std::max(new_x, 0);
            new_y = std::max(new_y, 0);

            int new_w = std::min(bbox.width + dw, src.cols - new_x);
            int new_h = std::min(bbox.height + dh, src.rows - new_y);

            if (new_w <= 0 || new_h <= 0)
                return false;

            const cv::Rect expanded_rect(new_x, new_y, new_w, new_h);

            cv::Mat litroi_color = src(expanded_rect);
            if (litroi_color.empty())
                return false;

            cv::Mat litroi_gray;
            try {
                cv::cvtColor(litroi_color, litroi_gray, cv::COLOR_BGR2GRAY);
            } catch (...) {
                return false;
            }

            armor.whole_gray_img = litroi_gray;
            if (params_.enable_cv) {
                cv::Mat litroi_binary;
                try {
                    cv::threshold(
                        litroi_gray,
                        litroi_binary,
                        params_.light_params.binary_thres,
                        255,
                        cv::THRESH_BINARY
                    );
                    armor.whole_binary_img = litroi_binary;
                } catch (...) {
                    return false;
                }
            }
            const cv::Point2f offset(static_cast<float>(new_x), static_cast<float>(new_y));

            if (params_.enable_classify) {
                cv::Point2f src_vertices[4] = { armor.pts[1] - offset,
                                                armor.pts[0] - offset,
                                                armor.pts[3] - offset,
                                                armor.pts[2] - offset };

                const int warp_width = is_large ? large_armor_width : small_armor_width;
                const int top_light_y = (warp_height - light_length) / 2 - 1;
                const int bottom_light_y = top_light_y + light_length;
                if (warp_width <= 0 || warp_height <= 0)
                    return false;
                cv::Point2f dst_vertices[4] = {
                    { 0.f, static_cast<float>(bottom_light_y) },
                    { 0.f, static_cast<float>(top_light_y) },
                    { static_cast<float>(warp_width - 1), static_cast<float>(top_light_y) },
                    { static_cast<float>(warp_width - 1), static_cast<float>(bottom_light_y) }
                };

                const cv::Mat warp_mat = cv::getPerspectiveTransform(src_vertices, dst_vertices);

                cv::Mat number_image;
                cv::warpPerspective(
                    litroi_gray,
                    number_image,
                    warp_mat,
                    cv::Size(warp_width, warp_height),
                    cv::INTER_LINEAR,
                    cv::BORDER_CONSTANT,
                    0
                );

                const int roi_x = (warp_width - roi_size.width) >> 1;
                const cv::Rect num_roi(roi_x, 0, roi_size.width, roi_size.height);

                if ((num_roi & cv::Rect(0, 0, warp_width, warp_height)) != num_roi)
                    return false;

                cv::Mat num_crop = number_image(num_roi);

                cv::threshold(
                    num_crop,
                    armor.number_img,
                    0,
                    255,
                    cv::THRESH_BINARY | cv::THRESH_OTSU
                );
            }

            armor.whole_rgb_img = litroi_color;
            armor.local_offset = offset;

            return true;
        }

        bool refineLightsFromArmorPts(ArmorObject& armor) const noexcept {
            armor.center = (armor.pts[0] + armor.pts[1] + armor.pts[2] + armor.pts[3]) * 0.25f;

            const int n_lights = static_cast<int>(armor.lights.size());
            if (n_lights < 2)
                return false;

            const auto ordered = armor.sortCorners(armor.pts);

            const cv::Point2f ref_centers[2] = { (ordered[0] + ordered[1]) * 0.5f,
                                                 (ordered[2] + ordered[3]) * 0.5f };

            int best0 = -1, best1 = -1;
            float best0_d2 = std::numeric_limits<float>::max();
            float best1_d2 = std::numeric_limits<float>::max();

            for (int i = 0; i < n_lights; ++i) {
                const cv::Point2f& c = armor.lights[i].center;

                const cv::Point2f d0 = c - ref_centers[0];
                const float dist0 = d0.dot(d0);
                if (dist0 < best0_d2) {
                    best0_d2 = dist0;
                    best0 = i;
                }

                const cv::Point2f d1 = c - ref_centers[1];
                const float dist1 = d1.dot(d1);
                if (dist1 < best1_d2) {
                    best1_d2 = dist1;
                    best1 = i;
                }
            }

            if (best0 == best1) {
                best1 = -1;
                best1_d2 = std::numeric_limits<float>::max();

                for (int i = 0; i < n_lights; ++i) {
                    if (i == best0)
                        continue;

                    const cv::Point2f d = armor.lights[i].center - ref_centers[1];
                    const float dist = d.dot(d);
                    if (dist < best1_d2) {
                        best1_d2 = dist;
                        best1 = i;
                    }
                }
            }

            if (best0 < 0 || best1 < 0)
                return false;

            const auto& l0 = armor.lights[best0];
            const auto& l1 = armor.lights[best1];

            if (l0.center.x < l1.center.x) {
                armor.lights[0] = l0;
                armor.lights[1] = l1;
            } else {
                armor.lights[0] = l1;
                armor.lights[1] = l0;
            }

            return true;
        }

        std::vector<Light>
        findLights(const cv::Mat& color_img, const cv::Mat& binary_img, ArmorObject& armor)
            const noexcept {
            std::vector<std::vector<cv::Point>> contours;
            contours.reserve(64);

            cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<Light> all_lights;
            all_lights.reserve(contours.size());

            for (const auto& contour: contours) {
                const int n = static_cast<int>(contour.size());
                if (n < 6)
                    continue;

                Light light(contour);
                if (!isLight(light))
                    continue;

                int sum_r = 0;
                int sum_b = 0;

                for (const auto& pt: contour) {
                    const cv::Vec3b* row = color_img.ptr<cv::Vec3b>(pt.y);
                    const cv::Vec3b& pix = row[pt.x];
                    sum_r += pix[0];
                    sum_b += pix[2];
                }

                const int avg_diff = std::abs(sum_r - sum_b) / n;
                if (avg_diff <= params_.light_params.color_diff_thresh)
                    continue;

                light.color = (sum_r > sum_b) ? EnemyColor::RED : EnemyColor::BLUE; // 0=红, 1=蓝
                all_lights.emplace_back(std::move(light));
            }

            std::sort(all_lights.begin(), all_lights.end(), [](const Light& a, const Light& b) {
                return a.center.x < b.center.x;
            });

            armor.lights = all_lights;
            return all_lights;
        }

        bool isLight(const Light& light) const noexcept {
            // width / length 比例
            const float ratio = light.width / light.length;

            if (ratio <= params_.light_params.min_ratio || ratio >= params_.light_params.max_ratio)
                return false;

            if (light.tilt_angle >= params_.light_params.max_angle)
                return false;

            return true;
        }
        bool isArmor(const Light& l1, const Light& l2) const noexcept {
            const float len1 = l1.length;
            const float len2 = l2.length;
            if (len1 <= 1e-3f || len2 <= 1e-3f)
                return false;

            const float min_len = (len1 < len2) ? len1 : len2;
            const float max_len = (len1 < len2) ? len2 : len1;
            if (min_len / max_len <= params_.armor_params.min_light_ratio)
                return false;

            const cv::Point2f d = l1.center - l2.center;
            const float dist2 = d.dot(d);

            const float avg_len = 0.5f * (len1 + len2);

            const float min_small = params_.armor_params.min_small_center_distance * avg_len;
            const float max_small = params_.armor_params.max_small_center_distance * avg_len;
            const float min_large = params_.armor_params.min_large_center_distance * avg_len;
            const float max_large = params_.armor_params.max_large_center_distance * avg_len;

            const float min_small2 = min_small * min_small;
            const float max_small2 = max_small * max_small;
            const float min_large2 = min_large * min_large;
            const float max_large2 = max_large * max_large;

            const bool small_ok = (dist2 >= min_small2 && dist2 < max_small2);
            const bool large_ok = (dist2 >= min_large2 && dist2 < max_large2);

            if (!(small_ok || large_ok))
                return false;

            static const float tan_max_angle =
                std::tan(params_.armor_params.max_angle * CV_PI / 180.0f);

            if (std::abs(d.y) >= std::abs(d.x) * tan_max_angle)
                return false;

            if (l1.color != l2.color)
                return false;

            return true;
        }

        std::vector<ArmorObject> detectNet(
            const cv::Mat& src_img,
            std::vector<ArmorObject>& objs_result,
            Eigen::Matrix3f transform_matrix,
            EnemyColor detect_color,
            const std::optional<ArmorNumber>& target_number
        ) const noexcept {
            std::vector<ArmorObject> armors;

            if (!src_img.data || src_img.empty()) {
                std::cout << "img data nullptr or empty" << std::endl;
                return armors;
            }

            if (objs_result.empty()) {
                return armors;
            }

            for (auto& armor_in: objs_result) {
                ArmorObject armor = armor_in;

                if ((detect_color == EnemyColor::RED && armor.color == ArmorColor::BLUE)
                    || (detect_color == EnemyColor::BLUE && armor.color == ArmorColor::RED))
                {
                    continue;
                }
                if (params_.enable_classify || params_.enable_cv) {
                    bool ok = false;

                    ok = extractNetImage(src_img, armor);

                    if (!ok)
                        continue;
                }

                if (params_.enable_classify) {
                    number_classifier_->classifyNumber(armor);
                    if (armor.confidence < params_.classifier_threshold)
                        continue;
                }

                if (target_number.has_value()) {
                    if (!isSameTarget(target_number.value(), armor.number)) {
                        continue;
                    }
                }

                if (armor.color == ArmorColor::NONE || armor.color == ArmorColor::PURPLE) {
                    armor.is_ok = false;
                    armor.transform(transform_matrix);
                    armors.emplace_back(armor);
                    continue;
                }
                if (params_.enable_cv) {
                    findLights(armor.whole_rgb_img, armor.whole_binary_img, armor);

                    if (refineLightsFromArmorPts(armor)) {
                        if (isArmor(armor.lights[0], armor.lights[1])) {
                            armor.is_ok = true;
                            for (auto& light: armor.lights) {
                                light.addOffset(armor.local_offset);
                            }
                        }
                    }

                    if (armor.is_ok) {
                        armor.is_ok = armor.checkOkptsRight(params_.max_pts_error);
                    }
                }

                if (!armor.is_ok) {
                    auto ordered = armor.sortCorners(armor.pts);
                    Light l1, l2;
                    l1.length = cv::norm(ordered[1] - ordered[0]);
                    l1.center = (ordered[0] + ordered[1]) / 2.0;
                    l2.length = cv::norm(ordered[2] - ordered[3]);
                    l2.center = (ordered[2] + ordered[3]) / 2.0;
                    if (!isArmor(l1, l2)) {
                        continue;
                    }
                }

                armor.transform(transform_matrix);

                armors.emplace_back(armor);
            }

            return armors;
        }
        std::unique_ptr<NumberClassifierBase> number_classifier_;
        struct ArmorDetectCommonParams {
            std::string classify_backend = "opencv";
            std::string classify_model_path;
            std::string classify_label_path;
            double classifier_threshold = 0.5;
            LightParams light_params;
            ArmorParams armor_params;
            float expand_ratio_w = 1.1f;
            float expand_ratio_h = 1.1f;
            double max_pts_error = 20.0;
            bool enable_cv = false;
            bool enable_classify = true;
            void load(const YAML::Node& config) {
                expand_ratio_w = config["cv"]["light"]["expand_ratio_w"].as<float>(1.1);
                expand_ratio_h = config["cv"]["light"]["expand_ratio_h"].as<float>(1.1);
                max_pts_error = config["cv"]["light"]["max_pts_error"].as<double>(20.0);
                enable_cv = config["cv"]["enable"].as<bool>();
                light_params.load(config["cv"]["light"]);
                armor_params.load(config["cv"]["armor"]);
                enable_classify = config["classify"]["enable"].as<bool>();
                classify_model_path =
                    utils::expandEnv(config["classify"]["model_path"].as<std::string>());
                classify_label_path =
                    utils::expandEnv(config["classify"]["label_path"].as<std::string>());
                classify_backend = config["classify"]["backend"].as<std::string>();
                classifier_threshold = config["classify"]["threshold"].as<double>();
            }
        } params_;
    };
    ArmorDetectorCommon::ArmorDetectorCommon(const YAML::Node& config) {
        _impl = std::make_unique<Impl>(config);
    }
    ArmorDetectorCommon::~ArmorDetectorCommon() {
        _impl.reset();
    }
    std::vector<ArmorObject> ArmorDetectorCommon::detectNet(
        const cv::Mat& src_img,
        std::vector<ArmorObject>& objs_result,
        Eigen::Matrix3f transform_matrix,
        EnemyColor detect_color,
        const std::optional<ArmorNumber>& target_number
    ) {
        return _impl
            ->detectNet(src_img, objs_result, transform_matrix, detect_color, target_number);
    }
} // namespace auto_aim
} // namespace awakening