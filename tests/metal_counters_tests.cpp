// Copyright © 2026 Apple Inc.

#include <thread>

#include "doctest/doctest.h"

#include "mlx/backend/metal/metal.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test metal counters count dispatches, commits, syncs and waits") {
  // Use the GPU explicitly: DEVICE=cpu makes the CPU the default device.
  Device gpu = Device::gpu;
  auto a = random::normal({64, 64}, float32, 0.0, 1.0, std::nullopt, gpu);
  auto b = random::normal({64, 64}, float32, 0.0, 1.0, std::nullopt, gpu);
  eval(a, b);

  metal::reset();
  auto z = metal::counters();
  CHECK_EQ(z.dispatches, 0);
  CHECK_EQ(z.commits, 0);
  CHECK_EQ(z.syncs, 0);
  CHECK_EQ(z.waits, 0);

  auto c = matmul(a, b, gpu);
  eval(c);
  auto after_eval = metal::counters();
  CHECK_GE(after_eval.dispatches, 1);
  CHECK_GE(after_eval.commits, 1);
  CHECK_EQ(after_eval.syncs, 0);
  // eval() blocks on c's completion event unless the GPU already signaled it
  // before the host checked, so at most one wait.
  CHECK_LE(after_eval.waits, 1);

  // Re-evaluating (or reading) an evaluated array does not wait again.
  eval(c);
  CHECK_EQ(metal::counters().waits, after_eval.waits);

  synchronize(default_stream(gpu));
  auto after_sync = metal::counters();
  CHECK_EQ(after_sync.syncs, 1);
  CHECK_EQ(after_sync.waits, after_eval.waits + 1);
  CHECK_GE(after_sync.commits, after_eval.commits + 1);
  CHECK_EQ(after_sync.dispatches, after_eval.dispatches);

  metal::reset();
  auto r = metal::counters();
  CHECK_EQ(r.dispatches, 0);
  CHECK_EQ(r.commits, 0);
  CHECK_EQ(r.syncs, 0);
  CHECK_EQ(r.waits, 0);
}

TEST_CASE("test metal counters skip CPU-only work") {
  Device cpu = Device::cpu;
  auto a = random::normal({64, 64}, float32, 0.0, 1.0, std::nullopt, cpu);
  auto b = random::normal({64, 64}, float32, 0.0, 1.0, std::nullopt, cpu);
  eval(a, b);

  metal::reset();
  auto c = matmul(a, b, cpu);
  eval(c);
  synchronize(default_stream(cpu));
  auto r = metal::counters();
  CHECK_EQ(r.dispatches, 0);
  CHECK_EQ(r.commits, 0);
  CHECK_EQ(r.syncs, 0);
  CHECK_EQ(r.waits, 0);
}

TEST_CASE("test metal counters skip encoder teardown") {
  // A new thread gets its own GPU stream, so clear_streams() there does not
  // touch the streams of the main thread.
  std::thread t([]() {
    Device gpu = Device::gpu;
    auto a = random::normal({64, 64}, float32, 0.0, 1.0, std::nullopt, gpu);
    eval(a);
    metal::reset();
    clear_streams();
  });
  t.join();

  auto r = metal::counters();
  CHECK_EQ(r.syncs, 0);
  CHECK_EQ(r.waits, 0);
}
