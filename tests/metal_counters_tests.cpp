// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/backend/metal/metal.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test metal counters count dispatches, commits and syncs") {
  auto a = random::normal({64, 64});
  auto b = random::normal({64, 64});
  eval(a, b);

  metal::reset();
  auto z = metal::counters();
  CHECK_EQ(z.dispatches, 0);
  CHECK_EQ(z.commits, 0);
  CHECK_EQ(z.syncs, 0);

  auto c = matmul(a, b);
  eval(c);
  auto after_eval = metal::counters();
  CHECK_GE(after_eval.dispatches, 1);
  CHECK_GE(after_eval.commits, 1);
  CHECK_EQ(after_eval.syncs, 0);

  synchronize();
  auto after_sync = metal::counters();
  CHECK_EQ(after_sync.syncs, 1);
  CHECK_GE(after_sync.commits, after_eval.commits + 1);
  CHECK_EQ(after_sync.dispatches, after_eval.dispatches);

  metal::reset();
  auto r = metal::counters();
  CHECK_EQ(r.dispatches, 0);
  CHECK_EQ(r.commits, 0);
  CHECK_EQ(r.syncs, 0);
}
