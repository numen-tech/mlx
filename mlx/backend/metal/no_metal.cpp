// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core {

namespace metal {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

void set_metallib_path(const std::string& path) {}

const std::string& get_metallib_path() {
  throw std::runtime_error(
      "[metal::get_metallib_path] Cannot get metallib path without metal backend");
}

Counters counters() {
  return {0, 0, 0, 0};
}
void reset() {}

void kernel_trace_start() {}
std::vector<std::string> kernel_trace_stop() {
  return {};
}

void set_qmv_mma(bool) {}
void set_quantized_row_stride(bool) {}
bool qmv_mma_device() {
  return false;
}

array nonresident_array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter) {
  return array(data, std::move(shape), dtype, deleter);
}

} // namespace metal

} // namespace mlx::core
