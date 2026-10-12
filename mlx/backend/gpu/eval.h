// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::gpu {

void init();
void new_stream(Stream s);
void new_thread_unsafe_stream(Stream s);
void eval(array& arr);
void finalize(Stream s);
// explicit_sync is false for internal flushes (e.g. eval error recovery).
void synchronize(Stream s, bool explicit_sync = true);
void clear_streams();

} // namespace mlx::core::gpu
