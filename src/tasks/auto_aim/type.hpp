#pragma once
#include "common.hpp"
#include "utils/utils.hpp"
namespace awakening {

namespace auto_aim {
    constexpr double SMALL_ARMOR_WIDTH = 133.0 / 1000.0; // 135
    constexpr double SMALL_ARMOR_HEIGHT = 50.0 / 1000.0; // 55
    constexpr double LARGE_ARMOR_WIDTH = 225.0 / 1000.0;
    constexpr double LARGE_ARMOR_HEIGHT = 50.0 / 1000.0; // 55

    constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;
    struct Light: public cv::RotatedRect {
        Light() = default;

        explicit Light(const std::vector<cv::Point>& contour);

        void addOffset(const cv::Point2f& offset) noexcept;
        void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept;

        cv::Point2f top, bottom;
        EnemyColor color;
        cv::Point2f axis;
        double length = 0;
        double width = 0;
        float tilt_angle = 0;
    };

    enum class ArmorColor : int { BLUE = 0, RED, NONE, PURPLE };

    int formArmorColor(const ArmorColor& color) noexcept;
    enum class ArmorNumber : int { SENTRY = 0, NO1, NO2, NO3, NO4, NO5, OUTPOST, BASE, UNKNOWN };
    std::ostream& operator<<(std::ostream& os, const ArmorNumber& number) noexcept;

    int formArmorNumber(const ArmorNumber& number) noexcept;
    std::string armorNumberToString(const ArmorNumber& num) noexcept;
    ArmorNumber armorNumberFromString(const std::string& s) noexcept;
    int retypetotracker(const ArmorNumber& a) noexcept;
    bool isSameTarget(const ArmorNumber& a, const ArmorNumber& b) noexcept;
    enum class ArmorsNum { NORMAL_4 = 4, OUTPOST_3 = 3 };

    enum class ArmorType { SMALL, LARGE, INVALID };
    std::string armorTypeToString(const ArmorType& type) noexcept;

    struct ArmorObject {
        ArmorColor color;
        ArmorNumber number;
        std::vector<cv::Point2f> pts;
        cv::Rect box;

        cv::Mat number_img;

        double confidence;

        cv::Mat whole_binary_img;
        cv::Mat whole_rgb_img;
        cv::Mat whole_gray_img;

        std::vector<Light> lights;
        cv::Point2f local_offset;
        cv::Point2f center;
        bool is_ok = false;
        ArmorType type;
        static constexpr const int N_LANDMARKS = 6;
        static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;

        template<typename PointType>
        static std::vector<PointType> buildObjectPoints(const double& w, const double& h) noexcept {
            if constexpr (N_LANDMARKS == 4) {
                return {
                    PointType(0, w / 2, -h / 2), // 右下
                    PointType(0, w / 2, h / 2), // 右上
                    PointType(0, -w / 2, h / 2), // 左上
                    PointType(0, -w / 2, -h / 2) // 左下
                };
            } else {
                return {
                    PointType(0, w / 2, -h / 2), // 右下
                    PointType(0, w / 2, 0.0), // 右中
                    PointType(0, w / 2, h / 2), // 右上

                    PointType(0, -w / 2, h / 2), // 左上
                    PointType(0, -w / 2, 0.0), // 左中
                    PointType(0, -w / 2, -h / 2) // 左下
                };
            }
        }
        template<typename IDType>
        static std::vector<std::pair<IDType, IDType>> buildSymPairs() noexcept {
            if constexpr (N_LANDMARKS == 4) {
                static const std::vector<std::pair<IDType, IDType>> pairs = {
                    { 0, 3 },
                    { 1, 2 },
                    // { 0, 2 },
                    //    { 1, 3 }
                };
                return pairs;
            } else {
                static const std::vector<std::pair<IDType, IDType>> pairs = {
                    { 0, 5 },
                    { 1, 4 },
                    { 2, 3 },
                    //    { 0, 3 },
                    //    { 2, 5 }

                };
                return pairs;
            }
        }
        std::vector<cv::Point2f> toPts() const noexcept;
        bool checkOkptsRight(double max_error) const noexcept;
        std::array<cv::Point2f, 4> sortCorners(const std::vector<cv::Point2f>& pts) const noexcept;

        // Landmarks start from bottom left in clockwise order
        std::vector<cv::Point2f> landmarks() const noexcept;
        void addOffset(const cv::Point2f& offset) noexcept {
            for (auto& pt: pts) {
                pt += offset;
            }
            center += offset;
            box.x += offset.x;
            box.y += offset.y;
            for (auto& l: lights) {
                l.addOffset(offset);
            }
        }
        void transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
            for (auto& l: lights) {
                l.transform(transform_matrix);
            }
            center = utils::transformPoint2D(transform_matrix, center);
            box = utils::transformRect(transform_matrix, box);
            for (auto& pt: pts) {
                pt = utils::transformPoint2D(transform_matrix, pt);
            }
        }
        ArmorObject(const Light& l1, const Light& l2);
        ArmorObject() = default;
    };

    struct Armor {
    public:
        ArmorNumber number;
        std::string type;
        ISO3 pose;

        float distance_to_image_center;
        float yaw;
        std::chrono::steady_clock::time_point timestamp;
        bool is_ok = false;
        bool is_none_purple = false;
        int id = -1;
        std::vector<cv::Point2f> toPtsDebug(
            const cv::Mat& camera_intrinsic,
            const cv::Mat& camera_distortion
        ) const noexcept;
    };
    struct Armors {
    public:
        std::vector<Armor> armors;
        std::chrono::steady_clock::time_point timestamp;
        int id;
        Frame frame_id;
    };

} // namespace auto_aim
} // namespace awakening