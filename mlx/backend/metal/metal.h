// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mlx/api.h"
#include "mlx/array.h"

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

// A no-copy array over `data`, like array(data, shape, dtype, deleter), whose
// Metal buffer never joins the residency sets: no wired limit
// (set_wired_limit) ever wires it, so its pages stay pageable for as long as
// no GPU command binds the buffer. Use it for a mapping only the CPU stream
// reads (numen-tech/gemma4-qat#1228). If Metal refuses the pointer, or there
// is no Metal backend, this is array(data, shape, dtype, deleter).
// Version: kNonresidentArrayVersion (device.h).
MLX_API array nonresident_array(
    void* data,
    Shape shape,
    Dtype dtype,
    const std::function<void(void*)>& deleter);

// Records the name of every kernel looked up, cached or not, from
// kernel_trace_start() to kernel_trace_stop(), in order with repeats. A harness
// that compares two routes uses it to prove which kernel served an op, since a
// fallback gives a plausible answer too. Process-wide: only one trace at a
// time, and eval() the op inside it. Off, it costs one relaxed atomic load per
// kernel lookup. Without a Metal backend the trace is always empty.
MLX_API void kernel_trace_start();
MLX_API std::vector<std::string> kernel_trace_stop();

// The affine qmv_mma route (ml-explore/mlx#4787, numen-tech/gemma4-qat#1701),
// off by default. set_qmv_mma(true) turns it on for the process as
// MLX_QMV_MMA=1 does; the route is on while either is. A host that switches it
// per model load calls this instead of setenv/unsetenv, which may race another
// thread's dispatch reading the variable. qmv_mma_device() is whether this GPU
// can take the route at all (no NAX, not Apple6, not an Ultra); false without a
// Metal backend. Version: kQmvMmaVersion (device.h).
MLX_API void set_qmv_mma(bool enabled);
MLX_API bool qmv_mma_device();

// The head-dim-512 prefill route (numen-tech/gemma4-qat#1972), off by default.
// set_sdpa_d512_prefill(true) lets a causal head-dim-512 attention of 9 or more
// query rows (no array mask, sinks or logsumexp) take the fused d512 paths,
// which otherwise need 1,024 rows: the blocked path on a GPU without NAX, the
// NAX d512 kernel on a NAX GPU. Neither materializes the full
// [heads x rows x keys] score array the unfused path builds. A host that
// switches it per model load calls this; the decision is taken when the op is
// built, and a call built with it on still runs fused if it is switched off
// before the call is evaluated. Version: kSdpaD512PrefillVersion (device.h).
MLX_API void set_sdpa_d512_prefill(bool enabled);
MLX_API bool sdpa_d512_prefill();

} // namespace mlx::core::metal
