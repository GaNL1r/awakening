#pragma once
#include "tasks/auto_aim/type.hpp"
namespace awakening {
namespace auto_aim {
    class NumberClassifierBase {
    public:
        virtual ~NumberClassifierBase() = default;
        virtual bool classifyNumber(ArmorObject& armor) = 0;
        virtual void initNumberClassifier() = 0;
    };
} // namespace auto_aim
} // namespace awakening