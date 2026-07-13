#pragma once
#include "tasks/vslam/type.hpp"
#include <algorithm>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>
#include <yaml-cpp/node/node.h>
namespace awakening::vslam {
class Orb {
public:
    Orb(const YAML::Node& config) {
        detector_ = cv::ORB::create();
        extractor_ = cv::ORB::create();
        matcher_ = cv::BFMatcher::create();
    }
    void detect(const cv::Mat& src, Feature& feature) {
        cv::Mat gray;
        if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = src;
        }
        feature.detected.emplace();
        detector_->detect(gray, feature.detected.value().keypoints);
        extractor_->compute(
            gray,
            feature.detected.value().keypoints,
            feature.detected.value().descriptors
        );
    }
    Match match(const Feature& feature1, const Feature& feature2) {
        Match match;
        std::vector<cv::DMatch> raw_matches;
        matcher_->match(
            feature1.detected.value().descriptors,
            feature2.detected.value().descriptors,
            raw_matches
        );
        auto min_max = std::minmax_element(
            raw_matches.begin(),
            raw_matches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; }
        );
        for (int i = 0; i < feature1.detected.value().descriptors.rows; i++) {
            if (raw_matches[i].distance <= std::max(min_max.second->distance * 2.0, 30.0)) {
                match.matches.push_back(raw_matches[i]);
            }
        }

        return std::move(match);
    }

    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};
} // namespace awakening::vslam