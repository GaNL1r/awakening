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
#ifdef USE_TRT
    #include "number_classifier_trt.hpp"
    #include <wust_vl/common/utils/logger.hpp>
namespace awakening {
namespace auto_aim {
    NumberClassifierTRT::NumberClassifierTRT(
        const std::string& classify_model_path,
        const std::string& classify_label_path
    ):
        classify_model_path_(classify_model_path),
        classify_label_path_(classify_label_path) {
        initNumberClassifier();
    }
    void NumberClassifierTRT::initNumberClassifier() {
        const std::string model_path = classify_model_path_;
        trt_net_ = std::make_unique<wust_vl::ml_net::TensorRTNet>();
        wust_vl::ml_net::TensorRTNet::Params trt_params;
        trt_params.model_path = model_path;
        trt_params.input_dims = nvinfer1::Dims4 { 1, 1, 20, 28 };
        trt_net_->init(trt_params);
        const auto input_output_dims = trt_net_->getInputOutputDims();
        input_dims_ = std::get<0>(input_output_dims);
        output_dims_ = std::get<1>(input_output_dims);

        const std::string label_path = classify_label_path_;
        std::ifstream label_file(label_path);
        std::string line;

        class_names_.clear();

        while (std::getline(label_file, line)) {
            class_names_.push_back(line);
        }

        if (class_names_.empty()) {
            WUST_ERROR("number_classifier_trt") << "Failed to load labels from " << label_path;
            std::exit(EXIT_FAILURE);
        } else {
            WUST_INFO("number_classifier_trt")
                << "Successfully loaded " << class_names_.size() << " labels from " << label_path;
        }
    }
    bool NumberClassifierTRT::classifyNumber(ArmorObject& armor) {
        static thread_local std::unique_ptr<nvinfer1::IExecutionContext> ctx;
        if (armor.number_img.empty()) {
            return false;
        }

        if (!ctx) {
            auto c = trt_net_->getAContext();
            ctx = std::unique_ptr<nvinfer1::IExecutionContext>(c);
            WUST_DEBUG("number_classifier_trt") << "Loaded number classifier model for this thread";
            if (!ctx) {
                WUST_ERROR("number_classifier_trt")
                    << "Failed to load thread-local number classifier model.";
                return false;
            }
        }

        const cv::Mat image = armor.number_img;
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(28, 20));
        trt_net_->input2Device(blob.ptr<float>());
        void* input_tensor_ptr = trt_net_->getInputTensorPtr();
        trt_net_->infer(input_tensor_ptr, ctx.get());

        const float* out = static_cast<float*>(trt_net_->output2Host());

        cv::Mat outputs(1, 9, CV_32F);
        std::memcpy(outputs.data, out, 9 * sizeof(float));

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
#endif