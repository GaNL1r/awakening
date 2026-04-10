#pragma once

#include "tags.hpp"
#include <array>
#include <mutex>

namespace awakening::io {

template <typename T, size_t N>
class Buffer final {
public:
    Buffer() = default;
    ~Buffer() = default;

    void push(T FWD_IN obj) {
        std::lock_guard lock{lock_};
        data_[tail_] = std::forward<T>(obj);
        ++tail_ %= N;
        if (full_) {
            ++head_ %= N;
        }
        full_ = head_ == tail_;
    }

    bool pop(T REF_OUT obj) {
        std::lock_guard lock{lock_};
        if (empty()) {
            return false;
        }
        obj = std::move(data_[head_]);
        ++head_ %= N;
        full_ = false;
        return true;
    }

    [[nodiscard]] bool empty() const { return head_ == tail_ && !full_; }

    ATTR_READER_VAL(full_, full)

private:
    std::array<T, N> data_;
    size_t head_{};
    size_t tail_{};
    bool full_{};
    std::mutex lock_;
};

} // namespace awakening::io
