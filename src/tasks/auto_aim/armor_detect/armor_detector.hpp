#pragma once
#include "tasks/auto_aim/type.hpp"
#include "utils/impl.hpp"
#include <vector>
namespace awakening::auto_aim {
class ArmorDetector {
public:
    using Ptr = std::unique_ptr<ArmorDetector>;
    ArmorDetector(const YAML::Node& config);
    [[nodiscard]] std::tuple<std::vector<Light>, std::vector<Armor>> detect(const CommonFrame& frame,bool need_light_detect
    );
    AWAKENING_IMPL_DEFINITION(ArmorDetector)
};
} // namespace awakening::auto_aim