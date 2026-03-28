#pragma once
#include "tasks/auto_aim/type.hpp"

namespace awakening::auto_aim::armor_infer {

static constexpr float MERGE_CONF_ERROR = 0.15f;
static constexpr float MERGE_MIN_IOU = 0.9f;

enum class Mode { TUP, RP, AT, BOX416, BOX320, BOX };

inline Mode modeFromString(const std::string& m) {
    if (m == "tup" || m == "TUP")
        return Mode::TUP;
    if (m == "rp" || m == "RP")
        return Mode::RP;
    if (m == "at" || m == "AT")
        return Mode::AT;
    if (m == "box416" || m == "BOX416")
        return Mode::BOX416;
    if (m == "box320" || m == "BOX320")
        return Mode::BOX320;
    return Mode::TUP;
}

// ------------------------- model traits -------------------------
template<Mode M>
struct ModelTraits; // declare
// TUP
template<>
struct ModelTraits<Mode::TUP> {
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = false;
    static constexpr bool INPUT_RGB = true;
};

// RP
template<>
struct ModelTraits<Mode::RP> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_CLASSES = 9;
    static constexpr int NUM_COLORS = 4;
    static constexpr bool USE_NORM = true;
    static constexpr bool INPUT_RGB = false;
};

template<>
struct ModelTraits<Mode::AT> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int NUM_KPTS = 4;
    static constexpr bool USE_NORM = true;
    static constexpr bool INPUT_RGB = false;
    static constexpr std::array<std::pair<ArmorColor, ArmorNumber>, 64> CLASSES = { {
        { ArmorColor::BLUE, ArmorNumber::SENTRY },    { ArmorColor::BLUE, ArmorNumber::NO1 },
        { ArmorColor::BLUE, ArmorNumber::NO2 },       { ArmorColor::BLUE, ArmorNumber::NO3 },
        { ArmorColor::BLUE, ArmorNumber::NO4 },       { ArmorColor::BLUE, ArmorNumber::NO5 },
        { ArmorColor::BLUE, ArmorNumber::OUTPOST },   { ArmorColor::BLUE, ArmorNumber::BASE },
        { ArmorColor::BLUE, ArmorNumber::SENTRY },    { ArmorColor::BLUE, ArmorNumber::NO1 },
        { ArmorColor::BLUE, ArmorNumber::NO2 },       { ArmorColor::BLUE, ArmorNumber::NO3 },
        { ArmorColor::BLUE, ArmorNumber::NO4 },       { ArmorColor::BLUE, ArmorNumber::NO5 },
        { ArmorColor::BLUE, ArmorNumber::OUTPOST },   { ArmorColor::BLUE, ArmorNumber::BASE },
        { ArmorColor::RED, ArmorNumber::SENTRY },     { ArmorColor::RED, ArmorNumber::NO1 },
        { ArmorColor::RED, ArmorNumber::NO2 },        { ArmorColor::RED, ArmorNumber::NO3 },
        { ArmorColor::RED, ArmorNumber::NO4 },        { ArmorColor::RED, ArmorNumber::NO5 },
        { ArmorColor::RED, ArmorNumber::OUTPOST },    { ArmorColor::RED, ArmorNumber::BASE },
        { ArmorColor::RED, ArmorNumber::SENTRY },     { ArmorColor::RED, ArmorNumber::NO1 },
        { ArmorColor::RED, ArmorNumber::NO2 },        { ArmorColor::RED, ArmorNumber::NO3 },
        { ArmorColor::RED, ArmorNumber::NO4 },        { ArmorColor::RED, ArmorNumber::NO5 },
        { ArmorColor::RED, ArmorNumber::OUTPOST },    { ArmorColor::RED, ArmorNumber::BASE },
        { ArmorColor::NONE, ArmorNumber::SENTRY },    { ArmorColor::NONE, ArmorNumber::NO1 },
        { ArmorColor::NONE, ArmorNumber::NO2 },       { ArmorColor::NONE, ArmorNumber::NO3 },
        { ArmorColor::NONE, ArmorNumber::NO4 },       { ArmorColor::NONE, ArmorNumber::NO5 },
        { ArmorColor::NONE, ArmorNumber::OUTPOST },   { ArmorColor::NONE, ArmorNumber::BASE },
        { ArmorColor::NONE, ArmorNumber::SENTRY },    { ArmorColor::NONE, ArmorNumber::NO1 },
        { ArmorColor::NONE, ArmorNumber::NO2 },       { ArmorColor::NONE, ArmorNumber::NO3 },
        { ArmorColor::NONE, ArmorNumber::NO4 },       { ArmorColor::NONE, ArmorNumber::NO5 },
        { ArmorColor::NONE, ArmorNumber::OUTPOST },   { ArmorColor::NONE, ArmorNumber::BASE },
        { ArmorColor::PURPLE, ArmorNumber::SENTRY },  { ArmorColor::PURPLE, ArmorNumber::NO1 },
        { ArmorColor::PURPLE, ArmorNumber::NO2 },     { ArmorColor::PURPLE, ArmorNumber::NO3 },
        { ArmorColor::PURPLE, ArmorNumber::NO4 },     { ArmorColor::PURPLE, ArmorNumber::NO5 },
        { ArmorColor::PURPLE, ArmorNumber::OUTPOST }, { ArmorColor::PURPLE, ArmorNumber::BASE },
        { ArmorColor::PURPLE, ArmorNumber::SENTRY },  { ArmorColor::PURPLE, ArmorNumber::NO1 },
        { ArmorColor::PURPLE, ArmorNumber::NO2 },     { ArmorColor::PURPLE, ArmorNumber::NO3 },
        { ArmorColor::PURPLE, ArmorNumber::NO4 },     { ArmorColor::PURPLE, ArmorNumber::NO5 },
        { ArmorColor::PURPLE, ArmorNumber::OUTPOST }, { ArmorColor::PURPLE, ArmorNumber::BASE },
    } };
};
template<>
struct ModelTraits<Mode::BOX416> {
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
};
template<>
struct ModelTraits<Mode::BOX320> {
    static constexpr int INPUT_W = 320;
    static constexpr int INPUT_H = 320;
};
template<>
struct ModelTraits<Mode::BOX> {
    static constexpr int INPUT_W = 416;
    static constexpr int INPUT_H = 416;
    static constexpr bool INPUT_RGB = false;
    static constexpr bool USE_NORM = true;
    static constexpr std::array<std::pair<ArmorColor, ArmorNumber>, 12> CLASSES = { {
        { ArmorColor::BLUE, ArmorNumber::NO1 },
        { ArmorColor::BLUE, ArmorNumber::NO2 },
        { ArmorColor::BLUE, ArmorNumber::NO3 },
        { ArmorColor::BLUE, ArmorNumber::NO4 },
        { ArmorColor::BLUE, ArmorNumber::NO5 },
        { ArmorColor::BLUE, ArmorNumber::SENTRY },
        { ArmorColor::RED, ArmorNumber::NO1 },
        { ArmorColor::RED, ArmorNumber::NO2 },
        { ArmorColor::RED, ArmorNumber::NO3 },
        { ArmorColor::RED, ArmorNumber::NO4 },
        { ArmorColor::RED, ArmorNumber::NO5 },
        { ArmorColor::RED, ArmorNumber::SENTRY },

    } };
};

[[nodiscard]] inline double sigmoid(double x) noexcept {
    return x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

[[nodiscard]] inline float rectIoU(const cv::Rect2f& a, const cv::Rect2f& b) noexcept {
    const cv::Rect2f inter = a & b;
    const float inter_area = inter.area();
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.f || std::isnan(union_area))
        return 0.f;
    return inter_area / union_area;
}

// Merge / NMS helpers that mimic original intent but clearer
inline void nms_merge_sorted_bboxes(
    std::vector<ArmorObject>& objs,
    std::vector<int>& out_indices,
    float nms_threshold
) {
    out_indices.clear();
    const size_t n = objs.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i)
        areas[i] = objs[i].box.area();

    for (size_t i = 0; i < n; ++i) {
        ArmorObject& a = objs[i];
        bool keep = true;
        for (int idx: out_indices) {
            ArmorObject& b = objs[idx];
            const float iou = rectIoU(a.box, b.box);
            if (std::isnan(iou) || iou > nms_threshold) {
                keep = false;
                if (a.number == b.number && a.color == b.color && iou > MERGE_MIN_IOU
                    && std::abs(a.confidence - b.confidence) < MERGE_CONF_ERROR)
                {
                    // accumulate points for later averaging
                    for (const auto& pt: a.pts)
                        b.pts.push_back(pt);
                }
                break;
            }
        }
        if (keep)
            out_indices.push_back(static_cast<int>(i));
    }
}

inline std::vector<ArmorObject>
topKAndNms(std::vector<ArmorObject>& objs, int top_k, float nms_threshold) {
    std::sort(objs.begin(), objs.end(), [](const ArmorObject& a, const ArmorObject& b) {
        return a.confidence > b.confidence;
    });
    if (static_cast<int>(objs.size()) > top_k)
        objs.resize(static_cast<size_t>(top_k));

    std::vector<int> indices;
    nms_merge_sorted_bboxes(objs, indices, nms_threshold);

    std::vector<ArmorObject> result;
    result.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        result.push_back(std::move(objs[indices[i]]));
        // average merged extra points if any
        auto& ro = result.back();
        if (ro.pts.size() >= 8) {
            const size_t npts = ro.pts.size();
            std::array<cv::Point2f, 4> accum { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
            for (size_t j = 0; j < npts; ++j)
                accum[j % 4] += ro.pts[j];
            ro.pts.resize(4);
            for (int k = 0; k < 4; ++k) {
                float denom = static_cast<float>(npts) / 4.0f;
                ro.pts[k].x = accum[k].x / denom;
                ro.pts[k].y = accum[k].y / denom;
            }
        }
    }
    return result;
}

// However providing full modern unified class below that delegates using templated helpers.

class ArmorInfer {
public:
    ArmorInfer(
        Mode mode = Mode::TUP,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.45f,
        int top_k = 100
    ) noexcept:
        mode_(mode),
        conf_threshold_(conf_threshold),
        nms_threshold_(nms_threshold),
        top_k_(top_k) {
        setMode(mode_);
    }

    void setMode(Mode m) noexcept {
        mode_ = m;
        switch (mode_) {
            case Mode::TUP: {
                input_w_ = ModelTraits<Mode::TUP>::INPUT_W;
                input_h_ = ModelTraits<Mode::TUP>::INPUT_H;
                use_norm_ = ModelTraits<Mode::TUP>::USE_NORM;
                input_rgb_ = ModelTraits<Mode::TUP>::INPUT_RGB;
                break;
            }
            case Mode::RP: {
                input_w_ = ModelTraits<Mode::RP>::INPUT_W;
                input_h_ = ModelTraits<Mode::RP>::INPUT_H;
                use_norm_ = ModelTraits<Mode::RP>::USE_NORM;
                input_rgb_ = ModelTraits<Mode::RP>::INPUT_RGB;
                break;
            }
            case Mode::AT: {
                input_w_ = ModelTraits<Mode::AT>::INPUT_W;
                input_h_ = ModelTraits<Mode::AT>::INPUT_H;
                use_norm_ = ModelTraits<Mode::AT>::USE_NORM;
                input_rgb_ = ModelTraits<Mode::AT>::INPUT_RGB;
                break;
            }
            case Mode::BOX416: {
                input_w_ = ModelTraits<Mode::BOX416>::INPUT_W;
                input_h_ = ModelTraits<Mode::BOX416>::INPUT_H;
                use_norm_ = ModelTraits<Mode::BOX>::USE_NORM;
                input_rgb_ = ModelTraits<Mode::BOX>::INPUT_RGB;
                break;
            }
            case Mode::BOX320: {
                input_w_ = ModelTraits<Mode::BOX320>::INPUT_W;
                input_h_ = ModelTraits<Mode::BOX320>::INPUT_H;
                use_norm_ = ModelTraits<Mode::BOX>::USE_NORM;
                input_rgb_ = ModelTraits<Mode::BOX>::INPUT_RGB;
                break;
            } break;
        }
    }

    void setConfThreshold(float t) noexcept {
        conf_threshold_ = t;
    }
    void setNmsThreshold(float t) noexcept {
        nms_threshold_ = t;
    }
    void setTopK(int k) noexcept {
        top_k_ = k;
    }

    int inputW() const noexcept {
        return input_w_;
    }
    int inputH() const noexcept {
        return input_h_;
    }
    bool useNorm() const noexcept {
        return use_norm_;
    }
    bool inputRGB() const noexcept {
        return input_rgb_;
    }

    // main dispatching entry (keeps original signature)
    [[nodiscard]] std::vector<ArmorObject> postProcess(const cv::Mat& output_buffer) const {
        switch (mode_) {
            case Mode::TUP:
                return postProcessTUP_impl(output_buffer);
            case Mode::RP:
                return postProcessRP_impl(output_buffer);
            case Mode::AT:
                return postProcessAT_impl(output_buffer);
            case Mode::BOX416:
                return postProcessBOX_impl(output_buffer);
            case Mode::BOX320:
                return postProcessBOX_impl(output_buffer);
        }
        return {};
    }

private:
    std::vector<ArmorObject> postProcessTUP_impl(const cv::Mat& out) const;

    std::vector<ArmorObject> postProcessRP_impl(const cv::Mat& out) const;

    std::vector<ArmorObject> postProcessAT_impl(const cv::Mat& out) const;

    std::vector<ArmorObject> postProcessBOX_impl(const cv::Mat& out) const;

private:
    Mode mode_;
    int input_w_ { 0 };
    int input_h_ { 0 };
    float conf_threshold_ { 0.25f };
    float nms_threshold_ { 0.45f };
    int top_k_ { 100 };
    bool use_norm_ { false };
    bool input_rgb_ { false };
};

} // namespace awakening::auto_aim::armor_infer