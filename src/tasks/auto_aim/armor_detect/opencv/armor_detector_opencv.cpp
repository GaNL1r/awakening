// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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

#include "tasks/auto_aim/armor_detect/opencv/armor_detector_opencv.hpp"
#include "common.hpp"
#include "tasks/auto_aim/armor_detect/number_classifier/number_classifier.hpp"
namespace awakening {
namespace auto_aim {
    struct ArmorDetectorOpenCV::Impl {
    public:
        Impl(const YAML::Node& config) {
            auto classify_model_path =
                utils::expandEnv(config["classify"]["model_path"].as<std::string>());
            auto classify_label_path =
                utils::expandEnv(config["classify"]["label_path"].as<std::string>());
            double classify_threshold = config["classify"]["threshold"].as<double>();
            number_classifier_ =
                std::make_unique<NumberClassifier>(classify_model_path, classify_label_path);
            light_params_.load(config["light"]);
            armor_params_.load(config["armor"]);
        }

        std::vector<ArmorObject> detect(
            const cv::Mat& input,
            EnemyColor detect_color,
            const std::optional<ArmorNumber>& target_number
        ) noexcept {
            if (input.empty())
                return {};

            std::vector<Light> lights_;

            cv::Mat binary_img, gray_img;
            std::tie(binary_img, gray_img) = preprocessImage(input);
            lights_ = findLights(input, binary_img);
            std::vector<ArmorObject> armors = matchLights(lights_, detect_color);
            std::vector<ArmorObject> valid_armors;

            for (auto& armor: armors) {
                try {
                    armor.number_img = extractNumber(gray_img, armor);
                    if (armor.number_img.empty())
                        continue;

                    if (!number_classifier_->classifyNumber(armor))
                        continue;
                    if (target_number.has_value()) {
                        if (!isSameTarget(target_number.value(), armor.number)) {
                            continue;
                        }
                    }
                    if (armor.confidence < classifier_threshold_)
                        continue;

                    if (armor.number != ArmorNumber::NO1 && armor.number != ArmorNumber::BASE
                        && armor.type == ArmorType::LARGE)
                    {
                        continue;
                    }

                    valid_armors.push_back(armor);

                } catch (const std::exception& e) {
                    std::cerr << "[detect] Exception: " << e.what() << std::endl;
                }
            }
            return valid_armors;
        }

        std::tuple<cv::Mat, cv::Mat> preprocessImage(const cv::Mat& img) noexcept {
            cv::Mat gray_img;
            cv::Mat binary_img;

            if (img.empty()) {
                return { binary_img, gray_img }; // 空图直接返回空
            }

            if (img.channels() == 3) {
                cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);
            } else if (img.channels() == 1) {
                cv::cvtColor(img, gray_img, cv::COLOR_BayerRG2GRAY);
            } else {
                return { binary_img, gray_img };
            }

            cv::threshold(gray_img, binary_img, light_params_.binary_thres, 255, cv::THRESH_BINARY);

            return { binary_img, gray_img };
        }

        std::vector<Light> findLights(const cv::Mat& img, const cv::Mat& binary_img) noexcept {
            std::vector<std::vector<cv::Point>> contours;
            contours.reserve(64);
            cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat color_img;
            if (img.channels() == 3) {
                color_img = img;
            } else if (img.channels() == 1) {
                cv::cvtColor(img, color_img, cv::COLOR_BayerRG2BGR);
            } else {
                return {};
            }
            std::vector<Light> lights;
            lights.reserve(contours.size());

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
                    const cv::Vec3b& pix = color_img.at<cv::Vec3b>(pt);
                    sum_r += pix[0];
                    sum_b += pix[2];
                }

                const int avg_diff = std::abs(sum_r - sum_b) / n;
                if (avg_diff <= light_params_.color_diff_thresh)
                    continue;

                light.color = (sum_r > sum_b) ? EnemyColor::RED : EnemyColor::BLUE;
                lights.emplace_back(std::move(light));
            }

            std::sort(lights.begin(), lights.end(), [](const Light& a, const Light& b) {
                return a.center.x < b.center.x;
            });

            return lights;
        }

        bool isLight(const Light& light) noexcept {
            // width / length 比例
            const float ratio = light.width / light.length;

            if (ratio <= light_params_.min_ratio || ratio >= light_params_.max_ratio)
                return false;

            if (light.tilt_angle >= light_params_.max_angle)
                return false;

            return true;
        }

        std::vector<ArmorObject>
        matchLights(const std::vector<Light>& lights, EnemyColor detect_color) noexcept {
            const int n = static_cast<int>(lights.size());
            std::vector<ArmorObject> armors;
            armors.reserve(n);

            for (int i = 0; i < n; ++i) {
                const Light& l1 = lights[i];
                if (l1.color != detect_color)
                    continue;

                const float max_dx = l1.length * armor_params_.max_large_center_distance;

                for (int j = i + 1; j < n; ++j) {
                    const Light& l2 = lights[j];

                    if (l2.color != detect_color)
                        continue;

                    const float dx = l2.center.x - l1.center.x;
                    if (dx > max_dx)
                        break;

                    ArmorType type = isArmor(l1, l2);
                    if (type == ArmorType::INVALID)
                        continue;

                    if (containLight(i, j, lights))
                        continue;

                    ArmorObject armor(l1, l2);
                    armor.type = type;
                    armor.color =
                        (detect_color == EnemyColor::RED) ? ArmorColor::RED : ArmorColor::BLUE;

                    armors.emplace_back(std::move(armor));
                }
            }

            return armors;
        }

        // Check if there is another light in the boundingRect formed by the 2 lights
        bool containLight(const int i, const int j, const std::vector<Light>& lights) noexcept {
            const Light& l1 = lights[i];
            const Light& l2 = lights[j];

            float min_x = std::min({ l1.top.x, l1.bottom.x, l2.top.x, l2.bottom.x });
            float max_x = std::max({ l1.top.x, l1.bottom.x, l2.top.x, l2.bottom.x });
            float min_y = std::min({ l1.top.y, l1.bottom.y, l2.top.y, l2.bottom.y });
            float max_y = std::max({ l1.top.y, l1.bottom.y, l2.top.y, l2.bottom.y });

            const float avg_len = 0.5f * (l1.length + l2.length);
            const float avg_wid = 0.5f * (l1.width + l2.width);

            for (int k = i + 1; k < j; ++k) {
                const Light& t = lights[k];

                if (t.width > 2.0f * avg_wid)
                    continue;
                if (t.length < 0.5f * avg_len)
                    continue;

                const cv::Point2f& c = t.center;
                if (c.x >= min_x && c.x <= max_x && c.y >= min_y && c.y <= max_y) {
                    return true;
                }
            }
            return false;
        }

        ArmorType isArmor(const Light& l1, const Light& l2) noexcept {
            const float len1 = l1.length;
            const float len2 = l2.length;
            if (len1 <= 1e-3f || len2 <= 1e-3f)
                return ArmorType::INVALID;

            const float min_len = (len1 < len2) ? len1 : len2;
            const float max_len = (len1 < len2) ? len2 : len1;
            if (min_len / max_len <= armor_params_.min_light_ratio)
                return ArmorType::INVALID;
            const cv::Point2f d = l1.center - l2.center;
            const float dist2 = d.dot(d);

            const float avg_len = 0.5f * (len1 + len2);

            const float min_small = armor_params_.min_small_center_distance * avg_len;
            const float max_small = armor_params_.max_small_center_distance * avg_len;
            const float min_large = armor_params_.min_large_center_distance * avg_len;
            const float max_large = armor_params_.max_large_center_distance * avg_len;

            const float min_small2 = min_small * min_small;
            const float max_small2 = max_small * max_small;
            const float min_large2 = min_large * min_large;
            const float max_large2 = max_large * max_large;

            const bool small_ok = (dist2 >= min_small2 && dist2 < max_small2);
            const bool large_ok = (dist2 >= min_large2 && dist2 < max_large2);

            if (!(small_ok || large_ok))
                return ArmorType::INVALID;

            static const float tan_max_angle = std::tan(armor_params_.max_angle * CV_PI / 180.0f);

            if (std::abs(d.y) >= std::abs(d.x) * tan_max_angle)
                return ArmorType::INVALID;

            float delta_angle = std::fabs(l1.angle - l2.angle);
            if (delta_angle > 90.0f)
                delta_angle = 180.0f - delta_angle;

            if (delta_angle >= light_params_.max_angle_diff)
                return ArmorType::INVALID;

            return large_ok ? ArmorType::LARGE : ArmorType::SMALL;
        }

        cv::Mat extractNumber(const cv::Mat& src, const ArmorObject& armor) const noexcept {
            constexpr int light_length = 12;
            constexpr int warp_height = 28;
            constexpr int small_armor_width = 32;
            constexpr int large_armor_width = 54;
            const cv::Size roi_size(20, 28);

            cv::Point2f src_pts[4] = { armor.lights[0].bottom,
                                       armor.lights[0].top,
                                       armor.lights[1].top,
                                       armor.lights[1].bottom };

            const int warp_width =
                (armor.type == ArmorType::SMALL) ? small_armor_width : large_armor_width;

            const int top_y = (warp_height - light_length) / 2 - 1;
            const int bottom_y = top_y + light_length;

            cv::Point2f dst_pts[4] = {
                { 0.f, static_cast<float>(bottom_y) },
                { 0.f, static_cast<float>(top_y) },
                { static_cast<float>(warp_width - 1), static_cast<float>(top_y) },
                { static_cast<float>(warp_width - 1), static_cast<float>(bottom_y) }
            };

            cv::Mat warp_mat = cv::getPerspectiveTransform(src_pts, dst_pts);

            cv::Mat warped;
            cv::warpPerspective(
                src,
                warped,
                warp_mat,
                cv::Size(warp_width, warp_height),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                0
            );

            const int roi_x = (warp_width - roi_size.width) >> 1;
            if (roi_x < 0 || roi_x + roi_size.width > warp_width)
                return cv::Mat();

            cv::Mat number = warped(cv::Rect(roi_x, 0, roi_size.width, roi_size.height));

            cv::threshold(number, number, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            return number;
        }

        void setCallback(DetectorCallback callback) {
            this->infer_callback_ = callback;
        }
        void toPts(ArmorObject& armor) {
            if (armor.lights.size() != 2) {
                armor.is_ok = false;
                return;
            }
            armor.pts[0] = armor.lights[0].top;
            armor.pts[1] = armor.lights[0].bottom;
            armor.pts[2] = armor.lights[1].bottom;
            armor.pts[3] = armor.lights[1].top;
        }

        void pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) {
            frame.id = current_id_++;
            std::vector<ArmorObject> objs_result;
            auto roi = frame.img_frame.src_img(frame.expanded);

            objs_result = detect(roi, frame.detect_color, target_number);

            if (this->infer_callback_) {
                this->infer_callback_(objs_result, frame);
                return;
            }
            return;
        }

        LightParams light_params_;
        ArmorParams armor_params_;
        double classifier_threshold_ = 0.5;
        std::unique_ptr<NumberClassifier> number_classifier_;
        DetectorCallback infer_callback_;
        int current_id_ = 0;
    };
    ArmorDetectorOpenCV::ArmorDetectorOpenCV(const YAML::Node& config) {
        _impl = std::make_unique<Impl>(config);
    }
    ArmorDetectorOpenCV::~ArmorDetectorOpenCV() {
        _impl.reset();
    }
    void ArmorDetectorOpenCV::setCallback(DetectorCallback callback) {
        _impl->setCallback(callback);
    }
    void ArmorDetectorOpenCV::pushInput(
        CommonFrame& frame,
        const std::optional<ArmorNumber>& target_number
    ) {
        _impl->pushInput(frame, target_number);
    }
} // namespace auto_aim
} // namespace awakening