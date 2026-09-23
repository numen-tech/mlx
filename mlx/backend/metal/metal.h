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

/** Process-wide Metal work counters, summed over every stream.
 *
 * `dispatches` counts compute kernel dispatches (dispatchThreadgroups /
 * dispatchThreads), `commits` counts command-buffer commits, and `syncs`
 * counts blocking CommandEncoder::synchronize() calls (waitUntilCompleted).
 * Host waits on an array's completion event (eval(), item()) are not syncs.
 *
 * Increments are relaxed atomics with no happens-before relation to the GPU
 * or to other threads encoding work: read them only after the eval whose work
 * you are measuring has completed and no other thread is encoding. */
struct Counters {
  uint64_t dispatches;
  uint64_t commits;
  uint64_t syncs;
};
MLX_API Counters counters();
MLX_API void reset();

} // namespace mlx::core::metal
