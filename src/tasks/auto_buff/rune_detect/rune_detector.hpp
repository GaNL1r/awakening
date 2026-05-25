#pragma once
#include "utils/impl.hpp"
#include <yaml-cpp/node/node.h>
namespace awakening::auto_buff {
class RuneDetector {
    RuneDetector(const YAML::Node& config);
    AWAKENING_IMPL_DEFINITION(RuneDetector)
};
} // namespace awakening::auto_buff