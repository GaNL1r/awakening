// Copyright Chen Jun 2023. Licensed under the MIT License.
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
#include "base.hpp"
#include "tasks/auto_aim/type.hpp"
namespace awakening {
namespace auto_aim {
    class NumberClassifier: public NumberClassifierBase {
    public:
        NumberClassifier(
            const std::string& classify_model_path,
            const std::string& classify_label_path
        );
        void initNumberClassifier() override;
        bool classifyNumber(ArmorObject& armor) override;

    private:
        std::vector<std::string> class_names_;
        std::string classify_model_path_;
        std::string classify_label_path_;
    };
} // namespace auto_aim
} // namespace awakening