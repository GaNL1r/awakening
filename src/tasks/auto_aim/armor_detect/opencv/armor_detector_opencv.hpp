// Copyright Chen Jun 2023. Licensed under the MIT License.
//
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0.
//
// Copyright (C) FYT Vision Group. All rights reserved.
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
#include "tasks/auto_aim/armor_detect/armor_detector_base.hpp"
#include <optional>
namespace awakening {
namespace auto_aim {
    class ArmorDetectorOpenCV: public ArmorDetectorBase {
    public:
        using Ptr = std::unique_ptr<ArmorDetectorOpenCV>;
        explicit ArmorDetectorOpenCV(const YAML::Node& config);
        static Ptr create(const YAML::Node& config) {
            return std::make_unique<ArmorDetectorOpenCV>(config);
        }
        ~ArmorDetectorOpenCV();
        void
        pushInput(CommonFrame& frame, const std::optional<ArmorNumber>& target_number) override;
        void setCallback(DetectorCallback callback) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
    };
} // namespace auto_aim
} // namespace awakening