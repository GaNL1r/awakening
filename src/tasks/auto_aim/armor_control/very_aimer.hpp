#pragma once
#include "tasks/auto_aim/armor_tracker/armor_target.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/base/common.hpp"
#include "utils/impl.hpp"
#include <memory>
#include <yaml-cpp/node/node.h>

namespace awakening::auto_aim {
class VeryAimer {
public:
    VeryAimer(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(VeryAimer)
    [[nodiscard]] GimbalCmd
    very_aim(ArmorTarget target, double bullet_speed, const AutoAimFsm& fsm);
};
} // namespace awakening::auto_aim