#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace awakening::io {

template <class Base>
class RegistryBase {
public:
    virtual ~RegistryBase() = default;
    virtual Base* create() = 0;
};

template <class Base>
class Factory final {
public:
    [[nodiscard]] static Factory& instance() {
        static Factory factory;
        return factory;
    }

    void register_type(std::string&& type_name, RegistryBase<Base>* registry) {
        registry_[std::move(type_name)] = registry;
    }

    [[nodiscard]] Base* create(const std::string& type_name) {
        return registry_.contains(type_name) ? registry_[type_name]->create() : nullptr;
    }

private:
    Factory() = default;
    ~Factory() = default;

    std::unordered_map<std::string, RegistryBase<Base>*> registry_;
};

template <class Base, class Derived>
class RegistrySub final : public RegistryBase<Base> {
public:
    explicit RegistrySub(std::string&& type_name) {
        static_assert(std::is_base_of<Base, Derived>());
        Factory<Base>::instance().register_type(std::move(type_name), this);
    }

    [[nodiscard]] Base* create() override { return new Derived(); }
};

#define ENABLE_FACTORY(_namespace, _type, _func)                                                                    \
    namespace _namespace {                                                                                          \
    class _type;                                                                                                   \
    inline _type* _func(const std::string& type_name) { return Factory<_type>::instance().create(type_name); } \
    }

} // namespace awakening::io
