#include "type.hpp"
#include "common.hpp"
#include "config.hpp"
#include "wust_vl/common/utils/logger.hpp"
#include <numeric>
namespace awakening {
namespace auto_aim {

    Light::Light(const std::vector<cv::Point>& contour): cv::RotatedRect(cv::minAreaRect(contour)) {
        this->center = std::accumulate(
            contour.begin(),
            contour.end(),
            cv::Point2f(0, 0),
            [n = static_cast<float>(contour.size())](const cv::Point2f& a, const cv::Point& b) {
                return a + cv::Point2f(b.x, b.y) / n;
            }
        );

        cv::Point2f p[4];
        this->points(p);

        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        axis = (top - bottom) / cv::norm(top - bottom);

        tilt_angle =
            std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y)) / CV_PI * 180.0f;
    }

    void Light::addOffset(const cv::Point2f& offset) noexcept {
        this->center += offset;
        top += offset;
        bottom += offset;
    }
    void Light::transform(const Eigen::Matrix<float, 3, 3>& transform_matrix) noexcept {
        top = utils::transformPoint2D(transform_matrix, top);
        bottom = utils::transformPoint2D(transform_matrix, bottom);
        length = cv::norm(top - bottom);
        cv::Point2f p[4];
        this->points(p);

        width = cv::norm(
            utils::transformPoint2D(transform_matrix, p[0])
            - utils::transformPoint2D(transform_matrix, p[1])
        );
        const cv::Point2f p0 = center;
        const cv::Point2f p1 = center + axis;

        const cv::Point2f p0_t = utils::transformPoint2D(transform_matrix, p0);

        const cv::Point2f p1_t = utils::transformPoint2D(transform_matrix, p1);

        axis = p1_t - p0_t;
        axis /= cv::norm(axis);

        tilt_angle =
            std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y)) / CV_PI * 180.0f;
        center = utils::transformPoint2D(transform_matrix, center);
    }
    int formArmorColor(const ArmorColor& color) noexcept {
        switch (color) {
            case ArmorColor::RED:
                return 0;
            case ArmorColor::BLUE:
                return 1;
            case ArmorColor::NONE:
                return 2;
            case ArmorColor::PURPLE:
                return 3;
        }
        return -1;
    }

    std::ostream& operator<<(std::ostream& os, const ArmorNumber& number) noexcept {
        switch (number) {
            case ArmorNumber::SENTRY:
                return os << "SENTRY";
            case ArmorNumber::NO1:
                return os << "NO1";
            case ArmorNumber::NO2:
                return os << "NO2";
            case ArmorNumber::NO3:
                return os << "NO3";
            case ArmorNumber::NO4:
                return os << "NO4";
            case ArmorNumber::NO5:
                return os << "NO5";
            case ArmorNumber::OUTPOST:
                return os << "OUTPOST";
            case ArmorNumber::BASE:
                return os << "BASE";
            case ArmorNumber::UNKNOWN:
                return os << "UNKNOWN";
            default:
                return os << "InvalidArmorNumber(" << static_cast<int>(number) << ")";
        }
    }

    int formArmorNumber(const ArmorNumber& number) noexcept {
        switch (number) {
            case ArmorNumber::SENTRY:
                return 0;
            case ArmorNumber::NO1:
                return 1;
            case ArmorNumber::NO2:
                return 2;
            case ArmorNumber::NO3:
                return 3;
            case ArmorNumber::NO4:
                return 4;
            case ArmorNumber::NO5:
                return 5;
            case ArmorNumber::OUTPOST:
                return 6;
            case ArmorNumber::BASE:
                return 7;
            case ArmorNumber::UNKNOWN:
                return 8;
        }
        return -1;
    }
    ArmorNumber armorNumberFromString(const std::string& s) noexcept {
        if (s == "SENTRY")
            return ArmorNumber::SENTRY;
        if (s == "BASE")
            return ArmorNumber::BASE;
        if (s == "OUTPOST")
            return ArmorNumber::OUTPOST;
        if (s == "NO1")
            return ArmorNumber::NO1;
        if (s == "NO2")
            return ArmorNumber::NO2;
        if (s == "NO3")
            return ArmorNumber::NO3;
        if (s == "NO4")
            return ArmorNumber::NO4;
        if (s == "NO5")
            return ArmorNumber::NO5;
        return ArmorNumber::UNKNOWN;
    }

    std::string armorNumberToString(const ArmorNumber& num) noexcept {
        switch (num) {
            case ArmorNumber::SENTRY:
                return "SENTRY";
            case ArmorNumber::BASE:
                return "BASE";
            case ArmorNumber::OUTPOST:
                return "OUTPOST";
            case ArmorNumber::NO1:
                return "NO1";
            case ArmorNumber::NO2:
                return "NO2";
            case ArmorNumber::NO3:
                return "NO3";
            case ArmorNumber::NO4:
                return "NO4";
            case ArmorNumber::NO5:
                return "NO5";
            default:
                return "UNKNOWN";
        }
    }
    namespace {
        std::unordered_map<std::string, int> armor_map;
        std::unordered_map<int, std::vector<ArmorNumber>> tracker_to_armors;
        bool loaded = false;

        void loadArmorMapOnce() {
            if (loaded)
                return;

            try {
                YAML::Node config = YAML::LoadFile(AUTO_AIM_CONFIG)["armor_map"];

                for (auto it = config.begin(); it != config.end(); ++it) {
                    const std::string key = it->first.as<std::string>();
                    const int tracker_id = it->second.as<int>();

                    ArmorNumber armor_num = armorNumberFromString(key);

                    armor_map[key] = tracker_id;
                    tracker_to_armors[tracker_id].emplace_back(armor_num);
                }
                loaded = true;
            } catch (const std::exception& e) {
                std::cerr << "[ArmorMap] Failed to load armor_map.yaml: " << e.what() << std::endl;
            }
        }
    } // namespace

    int retypetotracker(const ArmorNumber& a) noexcept {
        loadArmorMapOnce();

        const std::string key = armorNumberToString(a);
        const auto it = armor_map.find(key);
        if (it != armor_map.end())
            return it->second;

        std::cerr << "[retypetotracker] Invalid ArmorNumber: " << static_cast<int>(a) << std::endl;
        return -1;
    }

    bool isSameTarget(const ArmorNumber& a, const ArmorNumber& b) noexcept {
        return retypetotracker(a) == retypetotracker(b);
    }

    std::string armorTypeToString(const ArmorType& type) noexcept {
        switch (type) {
            case ArmorType::SMALL:
                return "small";
            case ArmorType::LARGE:
                return "large";
            default:
                return "invalid";
        }
    }

    std::vector<cv::Point2f> ArmorObject::toPts() const noexcept {
        if (is_ok) {
            return { lights[0].top, lights[0].bottom, lights[1].bottom, lights[1].top };
        } else {
            return { pts[0], pts[1], pts[2], pts[3] };
        }
    }
    bool ArmorObject::checkOkptsRight(double max_error) const noexcept {
        double error = 0.0;
        for (int i = 0; i < 4; i++) {
            error += cv::norm(pts[i] - toPts()[i]);
        }
        return error < max_error;
    }
    std::array<cv::Point2f, 4> ArmorObject::sortCorners(const std::vector<cv::Point2f>& pts
    ) const noexcept {
        std::array<cv::Point2f, 4> ordered;

        // 先按 x 坐标分成左右两组
        std::vector<cv::Point2f> left, right;
        std::vector<cv::Point2f> sorted = pts;

        std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        left.push_back(sorted[0]);
        left.push_back(sorted[1]);
        right.push_back(sorted[2]);
        right.push_back(sorted[3]);

        // 左边两个点，按 y 分为上/下
        std::sort(left.begin(), left.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        ordered[1] = left[0]; // 左上
        ordered[0] = left[1]; // 左下

        // 右边两个点，按 y 分为上/下
        std::sort(right.begin(), right.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
        ordered[2] = right[0]; // 右上
        ordered[3] = right[1]; // 右下

        return ordered; // 顺序: 左下, 左上, 右上, 右下
    }
    std::vector<cv::Point2f> ArmorObject::landmarks() const noexcept {
        if constexpr (N_LANDMARKS == 4) {
            if (is_ok) {
                return { lights[0].bottom, lights[0].top, lights[1].top, lights[1].bottom };
            } else {
                const auto ordered = sortCorners(pts);
                return { ordered[0], ordered[1], ordered[2], ordered[3] };
            }

        } else {
            if (is_ok) {
                return { lights[0].bottom, lights[0].center, lights[0].top,
                         lights[1].top,    lights[1].center, lights[1].bottom };
            } else {
                const auto ordered = sortCorners(pts);
                return { ordered[0], (ordered[0] + ordered[1]) / 2.0, ordered[1],
                         ordered[2], (ordered[2] + ordered[3]) / 2.0, ordered[3] };
            }
        }
    }
    ArmorObject::ArmorObject(const Light& l1, const Light& l2) {
        pts.resize(4);
        if (l1.center.x < l2.center.x) {
            lights.push_back(l1);
            lights.push_back(l2);
            pts[0] = l1.top;
            pts[1] = l1.bottom;
            pts[2] = l2.bottom;
            pts[3] = l2.top;
        } else {
            lights.push_back(l2);
            lights.push_back(l1);
            pts[0] = l2.top;
            pts[1] = l2.bottom;
            pts[2] = l1.bottom;
            pts[3] = l1.top;
        }
        is_ok = true;
    }

    std::vector<cv::Point2f> Armor::toPtsDebug(
        const cv::Mat& camera_intrinsic,
        const cv::Mat& camera_distortion
    ) const noexcept {
        std::vector<cv::Point2f> image_points;
        const std::vector<cv::Point3f>* model_points;
        static std::vector<cv::Point3f> SMALL_ARMOR_3D_POINTS_BLOCK = {
            { 0, 0.025, -0.066 }, // 左上前
            { 0, -0.025, -0.066 }, // 左下前
            { 0, -0.025, 0.066 }, // 右下前
            { 0, 0.025, 0.066 }, // 右上前
            { 0.015, 0.025, -0.066 }, // 左上后
            { 0.015, -0.025, -0.066 }, // 左下后
            { 0.015, -0.025, 0.066 }, // 右下后
            { 0.015, 0.025, 0.066 }, // 右上后
        };

        static std::vector<cv::Point3f> BIG_ARMOR_3D_POINTS_BLOCK = {
            { 0, 0.025, -0.1125 },     { 0, -0.025, -0.1125 },    { 0, -0.025, 0.1125 },
            { 0, 0.025, 0.1125 },      { 0.015, 0.025, -0.1125 }, { 0.015, -0.025, -0.1125 },
            { 0.015, -0.025, 0.1125 }, { 0.015, 0.025, 0.1125 },
        };

        if (type == "large") {
            model_points = &BIG_ARMOR_3D_POINTS_BLOCK;
        } else if (type == "small") {
            model_points = &SMALL_ARMOR_3D_POINTS_BLOCK;
        }
        const Mat3 tf_rot = pose.rotation();
        const cv::Mat rot_mat =
            (cv::Mat_<double>(3, 3) << tf_rot(0, 0),
             tf_rot(0, 1),
             tf_rot(0, 2),
             tf_rot(1, 0),
             tf_rot(1, 1),
             tf_rot(1, 2),
             tf_rot(2, 0),
             tf_rot(2, 1),
             tf_rot(2, 2));

        cv::Mat rvec;
        cv::Rodrigues(rot_mat, rvec);
        auto pos = pose.translation();
        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << pos.x(), pos.y(), pos.z());

        cv::projectPoints(
            *model_points,
            rvec,
            tvec,
            camera_intrinsic,
            camera_distortion,
            image_points
        );
        return image_points;
    }

} // namespace auto_aim
} // namespace awakening