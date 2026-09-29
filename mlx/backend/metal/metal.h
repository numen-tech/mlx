// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

// Process-wide Metal work counters. syncs counts mx::synchronize() calls,
// waits counts host waits on GPU work. Read or reset only while no eval runs.
struct Counters {
  uint64_t dispatches;
  uint64_t commits;
  uint64_t syncs;
  uint64_t waits;
};
MLX_API Counters counters();
MLX_API void reset();

} // namespace mlx::core::metal
