#include "async_serial.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace awakening::io {

AsyncSerial::~AsyncSerial() {
    close();
}

bool AsyncSerial::find_device_name() {
#if defined(__linux__)
    FILE* ls = popen("ls /dev/ttyACM*", "r");
#else
    return false;
#endif

    if (!ls) {
        AWAKENING_ERROR("Failed to list serial devices");
        return false;
    }

    char name[256];
    if (fscanf(ls, "%s", name) == -1) {
        pclose(ls);
        AWAKENING_ERROR("No UART device found");
        return false;
    }
    pclose(ls);

    params_.device_name = name;

#if defined(__linux__)
    if (chmod(params_.device_name.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
        AWAKENING_ERROR("Running in user mode, manually setting permission is required.");
        AWAKENING_ERROR("Run: sudo chmod 777 {}", params_.device_name);
    }
#endif

    return true;
}

bool AsyncSerial::initialize(const std::string& config_path) {
    auto yaml = YAML::LoadFile(config_path);

    if (yaml["serial"]["baud_rate"]) {
        params_.baud_rate = yaml["serial"]["baud_rate"].as<unsigned int>();
    }

    if (yaml["serial"]["device_name"]) {
        params_.device_name = yaml["serial"]["device_name"].as<std::string>();
    } else {
        if (!find_device_name()) {
            return false;
        }
    }

    read_buffer_.resize(4096);
    accumulate_buffer_.reserve(4096);

    if (!open_port()) {
        return false;
    }

    running_ = true;
    io_thread_ = std::thread([this]() {
        start_read();
        io_.run();
    });

    AWAKENING_INFO("AsyncSerial initialized successfully: {}", params_.device_name);
    return true;
}

bool AsyncSerial::open_port() {
    boost::system::error_code ec;

    port_.open(params_.device_name, ec);
    if (ec) {
        AWAKENING_ERROR("Failed to open serial port {}: {}", params_.device_name, ec.message());
        return false;
    }

    port_.set_option(boost::asio::serial_port_base::baud_rate(params_.baud_rate), ec);
    port_.set_option(boost::asio::serial_port_base::character_size(params_.char_size), ec);
    port_.set_option(boost::asio::serial_port_base::parity(params_.parity), ec);
    port_.set_option(boost::asio::serial_port_base::stop_bits(params_.stop_bits), ec);
    port_.set_option(boost::asio::serial_port_base::flow_control(params_.flow_control), ec);

    if (ec) {
        AWAKENING_ERROR("Failed to configure serial port: {}", ec.message());
        return false;
    }

    AWAKENING_INFO("Serial port {} is open", params_.device_name);
    return true;
}

void AsyncSerial::start_read() {
    if (!running_) return;

    port_.async_read_some(
        boost::asio::buffer(read_buffer_),
        [this](const boost::system::error_code& ec, size_t bytes_transferred) {
            handle_read(ec, bytes_transferred);
        }
    );
}

void AsyncSerial::handle_read(const boost::system::error_code& ec, size_t bytes_transferred) {
    if (ec) {
        if (ec != boost::asio::error::operation_aborted) {
            AWAKENING_ERROR("Serial read error: {}", ec.message());
        }
        return;
    }

    if (bytes_transferred > 0) {
        accumulate_buffer_.insert(
            accumulate_buffer_.end(),
            read_buffer_.begin(),
            read_buffer_.begin() + bytes_transferred
        );

        while (accumulate_buffer_.size() >= 2) {
            if (expected_size_ == 0) {
                expected_size_ = *reinterpret_cast<const int16_t*>(accumulate_buffer_.data());
                if (expected_size_ <= 0 || expected_size_ > 4096) {
                    AWAKENING_ERROR("Invalid packet size: {}", expected_size_);
                    accumulate_buffer_.clear();
                    expected_size_ = 0;
                    break;
                }
            }

            if (accumulate_buffer_.size() >= static_cast<size_t>(2 + expected_size_)) {
                Packet packet(expected_size_);
                std::copy(
                    accumulate_buffer_.begin() + 2,
                    accumulate_buffer_.begin() + 2 + expected_size_,
                    packet.begin()
                );

                {
                    std::lock_guard<std::mutex> lock(read_mutex_);
                    read_queue_.push_back(std::move(packet));
                    if (read_queue_.size() > 100) {
                        read_queue_.pop_front();
                    }
                }
                read_cv_.notify_one();

                accumulate_buffer_.erase(
                    accumulate_buffer_.begin(),
                    accumulate_buffer_.begin() + 2 + expected_size_
                );
                expected_size_ = 0;
            } else {
                break;
            }
        }
    }

    start_read();
}

bool AsyncSerial::read(Packet& packet, int16_t size) {
    std::unique_lock<std::mutex> lock(read_mutex_);

    if (!read_cv_.wait_for(lock, std::chrono::seconds(3), [this] { return !read_queue_.empty(); })) {
        return false;
    }

    if (read_queue_.empty()) {
        return false;
    }

    packet = std::move(read_queue_.front());
    read_queue_.pop_front();

    return static_cast<int16_t>(packet.size_packet()) == size;
}

bool AsyncSerial::write(const Packet& packet) {
    if (!running_) {
        return false;
    }

    Packet send_packet;
    int16_t size = static_cast<int16_t>(packet.size_packet());
    send_packet.write(size);
    send_packet.insert(send_packet.end(), packet.begin(), packet.end());

    boost::asio::post(io_, [this, send_packet]() {
        write_queue_.push_back(send_packet);
        if (write_queue_.size() == 1) {
            do_write();
        }
    });

    return true;
}

void AsyncSerial::do_write() {
    if (write_queue_.empty() || !running_) {
        return;
    }

    boost::asio::async_write(
        port_,
        boost::asio::buffer(write_queue_.front().data_ptr(), write_queue_.front().size_packet()),
        [this](const boost::system::error_code& ec, size_t /*bytes_transferred*/) {
            handle_write(ec, 0);
        }
    );
}

void AsyncSerial::handle_write(const boost::system::error_code& ec, size_t /*bytes_transferred*/) {
    if (ec) {
        AWAKENING_ERROR("Serial write error: {}", ec.message());
        return;
    }

    write_queue_.pop_front();

    if (!write_queue_.empty()) {
        do_write();
    }
}

void AsyncSerial::close() {
    if (!running_) return;

    running_ = false;

    boost::system::error_code ec;
    port_.cancel(ec);
    port_.close(ec);

    io_.stop();

    if (io_thread_.joinable()) {
        io_thread_.join();
    }

    AWAKENING_INFO("AsyncSerial closed");
}

} // namespace awakening::io
