#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>

namespace awakening::io {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size, std::function<void()> full_handler = [] {})
        : max_size_(max_size)
        , full_handler_(std::move(full_handler)) {
    }

    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.size() >= max_size_) {
            full_handler_();
            return;
        }

        queue_.push(value);
        not_empty_condition_.notify_all();
    }

    void push(T&& value) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.size() >= max_size_) {
            full_handler_();
            return;
        }

        queue_.push(std::move(value));
        not_empty_condition_.notify_all();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    std::optional<T> try_pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    T front() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });

        return queue_.front();
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        not_empty_condition_.notify_all();
    }

private:
    std::queue<T> queue_;
    size_t max_size_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_condition_;
    std::function<void()> full_handler_;
};

} // namespace awakening::io
