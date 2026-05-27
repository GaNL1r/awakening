#pragma once
#include "utils/impl.hpp"
#include <yaml-cpp/node/node.h>
namespace awakening::auto_buff {
class RuneTracker {
    RuneTracker(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(RuneTracker)
};
} // namespace awakening::auto_buff