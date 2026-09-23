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
 * - `dispatches`: compute kernel dispatches (dispatchThreadgroups /
 *   dispatchThreads).
 * - `commits`: command-buffer commits, including the automatic splits made
 *   when a buffer exceeds MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER.
 *   A commit is not a host wait.
 * - `syncs`: explicit stream synchronizations (CommandEncoder::synchronize(),
 *   i.e. mx::synchronize()).
 * - `waits`: host blocking waits on GPU completion. Counted once per wait
 *   *call*, whether or not the GPU had already finished, never as blocking
 *   time: Metal shared-event waits (EventImpl::wait: an eval()/item() on an
 *   array still in flight, or a CPU stream waiting on a GPU event), the
 *   waitUntilCompleted inside synchronize() (so every sync is also a wait),
 *   and the CPU-stream spin on a fast fence. Note array::wait() skips the
 *   event wait when the event is already signaled, so an eval() whose GPU
 *   work finished before the host checked records no wait.
 *
 * Every field is an independent relaxed atomic with no happens-before
 * relation to the GPU or to other threads encoding work. counters() is four
 * independent loads and reset() four independent stores, so both are only
 * meaningful while no eval is in flight on any thread: call them after the
 * measured eval has completed and before the next one starts. */
struct Counters {
  uint64_t dispatches;
  uint64_t commits;
  uint64_t syncs;
  uint64_t waits;
};
MLX_API Counters counters();
MLX_API void reset();

} // namespace mlx::core::metal
