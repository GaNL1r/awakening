#pragma once

#include "io/core/packet.hpp"
#include "utils/logger.hpp"
#include <boost/asio.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <yaml-cpp/yaml.h>

namespace awakening::io {

class AsyncSerial {
public:
    AsyncSerial() : port_(io_) {}
    ~AsyncSerial();

    bool initialize(const std::string& config_path);
    bool read(Packet& packet, int16_t size);
    bool write(const Packet& packet);
    void close();

private:
    struct Params {
        unsigned int baud_rate = 115200;
        unsigned int char_size = 8;
        boost::asio::serial_port_base::parity::type parity = boost::asio::serial_port_base::parity::none;
        boost::asio::serial_port_base::stop_bits::type stop_bits = boost::asio::serial_port_base::stop_bits::one;
        boost::asio::serial_port_base::flow_control::type flow_control =
            boost::asio::serial_port_base::flow_control::none;
        std::string device_name;
    };

    bool open_port();
    void start_read();
    void handle_read(const boost::system::error_code& ec, size_t bytes_transferred);
    void do_write();
    void handle_write(const boost::system::error_code& ec, size_t bytes_transferred);
    bool find_device_name();

private:
    boost::asio::io_context io_;
    boost::asio::serial_port port_;
    std::vector<uint8_t> read_buffer_;
    std::deque<Packet> write_queue_;
    std::thread io_thread_;
    std::atomic<bool> running_{false};
    Params params_;

    std::mutex read_mutex_;
    std::condition_variable read_cv_;
    std::deque<Packet> read_queue_;
    int16_t expected_size_{0};
    std::vector<uint8_t> accumulate_buffer_;
};

} // namespace awakening::io
