#pragma once
#include <array>
#include <atomic>
#include <cstddef>

namespace awakening {

template<typename T>
struct OneBuffer {
public:
    OneBuffer() {
        write_index.store(0, std::memory_order_relaxed);
        read_index.store(1, std::memory_order_relaxed);
        back_index.store(2, std::memory_order_relaxed);
    }
    template<typename Fn>
    void write(Fn&& fn) {
        T& buf = buffers[write_index.load(std::memory_order_relaxed)];
        fn(buf);
        size_t prev_write = write_index.load(std::memory_order_relaxed);
        write_index.store(back_index.load(std::memory_order_relaxed), std::memory_order_release);
        back_index.store(prev_write, std::memory_order_relaxed);
    }
    void write(const T& value) {
        buffers[write_index.load(std::memory_order_relaxed)] = value;
        rotate_write();
    }

    void write(T&& value) {
        buffers[write_index.load(std::memory_order_relaxed)] = std::move(value);
        rotate_write();
    }
    T read() const {
        size_t idx = read_index.load(std::memory_order_acquire);
        T val = buffers[idx];
        read_index.store(write_index.load(std::memory_order_acquire), std::memory_order_release);
        return val;
    }

private:
    std::array<T, 3> buffers;
    std::atomic<size_t> write_index;
    mutable std::atomic<size_t> read_index;
    std::atomic<size_t> back_index;

    void rotate_write() {
        size_t prevWrite = write_index.load(std::memory_order_relaxed);
        write_index.store(back_index.load(std::memory_order_relaxed), std::memory_order_release);
        back_index.store(prevWrite, std::memory_order_relaxed);
    }
};

} // namespace awakening