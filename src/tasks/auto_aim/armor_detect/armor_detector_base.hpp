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
#pragma once
#include "tasks/auto_aim/type.hpp"
#include <optional>
namespace awakening {
namespace auto_aim {
    struct LightParams {
        // width / height
        double min_ratio;
        double max_ratio;
        // vertical angle
        double max_angle;
        // judge color
        int color_diff_thresh;
        double max_angle_diff;
        int binary_thres;
        void load(const YAML::Node& config) {
            binary_thres = config["binary_thres"].as<int>();
            min_ratio = config["min_ratio"].as<double>();
            max_ratio = config["max_ratio"].as<double>();
            max_angle = config["max_angle"].as<double>();
            max_angle_diff = config["max_angle_diff"].as<double>();
            color_diff_thresh = config["color_diff_thresh"].as<int>();
        }
    };
    struct ArmorParams {
        double min_light_ratio;
        // light pairs distance
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        // horizontal angle
        double max_angle;
        void load(const YAML::Node& config) {
            min_light_ratio = config["min_light_ratio"].as<double>();
            min_small_center_distance = config["min_small_center_distance"].as<double>();
            max_small_center_distance = config["max_small_center_distance"].as<double>();
            min_large_center_distance = config["min_large_center_distance"].as<double>();
            max_large_center_distance = config["max_large_center_distance"].as<double>();
            max_angle = config["max_angle"].as<double>();
        }
    };
    class ArmorDetectorBase {
    public:
        using Ptr = std::unique_ptr<ArmorDetectorBase>;
        virtual ~ArmorDetectorBase() = default;

        virtual void
        pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) = 0;

        using DetectorCallback =
            std::function<void(const std::vector<ArmorObject>&, const CommonFrame&)>;

        virtual void setCallback(DetectorCallback cb) = 0;
    };
} // namespace auto_aim
} // namespace awakening
