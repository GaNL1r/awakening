#pragma once

#define ATTR_READER_VAL(_var, _func) \
    [[nodiscard]] auto _func() const { return _var; }

#define ATTR_WRITER_VAL(_var, _func) \
    template <typename T> \
    void _func(const T& value) { \
        _var = value; \
    }

#define ATTR_READER_REF(_var, _func) \
    [[nodiscard]] auto&& _func() const { return _var; }

#ifndef REF_IN
#define REF_IN const&
#endif

#ifndef PTR_IN
#define PTR_IN const*
#endif

#ifndef FWD_IN
#define FWD_IN &&
#endif

#ifndef REF_OUT
#define REF_OUT &
#endif

#ifndef PTR_OUT
#define PTR_OUT *
#endif
