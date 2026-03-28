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

#include "number_classifier.hpp"
#include <wust_vl/common/utils/logger.hpp>
namespace awakening {
namespace auto_aim {
    NumberClassifier::NumberClassifier(
        const std::string& classify_model_path,
        const std::string& classify_label_path
    ):
        classify_model_path_(classify_model_path),
        classify_label_path_(classify_label_path) {
        initNumberClassifier();
    }
    void NumberClassifier::initNumberClassifier() {
        const std::string model_path = classify_model_path_;
        std::unique_ptr<cv::dnn::Net> number_net_ =
            std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(model_path));

        if (number_net_->empty()) {
            WUST_ERROR("number_classifier")
                << "Failed to load number classifier model from " << model_path;
            std::exit(EXIT_FAILURE);
        } else {
            WUST_INFO("number_classifier")
                << "Successfully loaded number classifier model from " << model_path;
        }

        const std::string label_path = classify_label_path_;
        std::ifstream label_file(label_path);
        std::string line;

        class_names_.clear();

        while (std::getline(label_file, line)) {
            class_names_.push_back(line);
        }

        if (class_names_.empty()) {
            WUST_ERROR("number_classifier") << "Failed to load labels from " << label_path;
            std::exit(EXIT_FAILURE);
        } else {
            WUST_INFO("number_classifier")
                << "Successfully loaded " << class_names_.size() << " labels from " << label_path;
        }
        number_net_.reset();
    }
    bool NumberClassifier::classifyNumber(ArmorObject& armor) {
        static thread_local std::unique_ptr<cv::dnn::Net> thread_net;
        if (armor.number_img.empty()) {
            return false;
        }

        if (!thread_net) {
            thread_net =
                std::make_unique<cv::dnn::Net>(cv::dnn::readNetFromONNX(classify_model_path_));
            WUST_DEBUG("number_classifier") << "Loaded number classifier model for this thread";
            if (thread_net->empty()) {
                WUST_ERROR("number_classifier")
                    << "Failed to load thread-local number classifier model.";
                return false;
            }
        }

        const cv::Mat image = armor.number_img;

        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0 / 255.0);

        thread_net->setInput(blob);
        cv::Mat outputs = thread_net->forward();
        double max_val;
        cv::minMaxLoc(outputs, nullptr, &max_val);

        cv::Mat prob;
        cv::exp(outputs - max_val, prob);
        prob /= cv::sum(prob)[0];

        double confidence;
        cv::Point class_id;
        cv::minMaxLoc(prob, nullptr, &confidence, nullptr, &class_id);

        const int label_id = class_id.x;
        const double raw_conf = armor.confidence;
        armor.confidence = confidence;

        static const std::map<int, ArmorNumber> label_to_armor_number = {
            { 0, ArmorNumber::NO1 },    { 1, ArmorNumber::NO2 }, { 2, ArmorNumber::NO3 },
            { 3, ArmorNumber::NO4 },    { 4, ArmorNumber::NO5 }, { 5, ArmorNumber::OUTPOST },
            { 6, ArmorNumber::SENTRY }, { 7, ArmorNumber::BASE }
        };

        if (label_id < 8 && label_to_armor_number.find(label_id) != label_to_armor_number.end()) {
            armor.number = label_to_armor_number.at(label_id);

            return true;
        } else {
            armor.confidence = raw_conf;
            return false;
        }
    }
} // namespace auto_aim
} // namespace awakening