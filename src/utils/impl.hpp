#pragma once
#include <memory>
namespace awakening {}
#define RMCS_PIMPL_DEFINITION(CLASS) \
public: \
    explicit CLASS() noexcept; \
    ~CLASS() noexcept; \
    CLASS(const CLASS&) = delete; \
    CLASS& operator=(const CLASS&) = delete; \
\
private: \
    struct Impl; \
    std::unique_ptr<Impl> _impl; \
\
public:
