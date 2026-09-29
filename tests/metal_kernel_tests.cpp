// Copyright © 2026 Apple Inc.

#include <thread>

#include "doctest/doctest.h"

#include "mlx/fast_primitives.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

using TemplateArgs = std::vector<std::pair<std::string, fast::TemplateArg>>;

fast::CustomKernelFunction sizeof_kernel() {
  return fast::metal_kernel(
      "sizeof_kernel",
      {"inp"},
      {"out"},
      "out[0] = static_cast<int>(sizeof(V));");
}

array call_sizeof(
    const fast::CustomKernelFunction& kernel,
    const array& inp,
    const TemplateArgs& template_args,
    StreamOrDevice s = Device::gpu) {
  return kernel(
      {inp},
      {{1}},
      {int32},
      {1, 1, 1},
      {1, 1, 1},
      template_args,
      {},
      false,
      s)[0];
}

// Read before evaluating `out`: evaluation detaches the primitive.
std::pair<std::string, std::string> name_and_source(const array& out) {
  auto state = static_cast<const fast::CustomKernel&>(out.primitive()).state();
  return {std::get<0>(state), std::get<1>(state)};
}

} // namespace

TEST_CASE("test metal kernel template argument kinds do not collide") {
  auto kernel = sizeof_kernel();
  auto inp = zeros({1}, int32);

  // `int` 1 and `bool` true print the same template value.
  auto as_int = call_sizeof(kernel, inp, {{"V", 1}});
  auto as_bool = call_sizeof(kernel, inp, {{"V", true}});
  CHECK_NE(name_and_source(as_int).second, name_and_source(as_bool).second);
  eval(as_int, as_bool);
  CHECK_EQ(as_int.item<int>(), sizeof(int));
  CHECK_EQ(as_bool.item<int>(), sizeof(bool));

  // Memoized variants give the same results again.
  auto as_bool_again = call_sizeof(kernel, inp, {{"V", true}});
  auto as_int_again = call_sizeof(kernel, inp, {{"V", 1}});
  CHECK_EQ(as_bool_again.item<int>(), sizeof(bool));
  CHECK_EQ(as_int_again.item<int>(), sizeof(int));
}

TEST_CASE("test metal kernel memoized source matches a fresh build") {
  auto kernel = sizeof_kernel();
  std::vector<std::pair<array, TemplateArgs>> variants = {
      {array(1, int32), {{"V", 1}}},
      {array(1, float32), {{"V", 1}}},
      {zeros({2}, int32), {{"V", 1}}},
      {zeros({16}, int32), {{"V", 1}}},
      {zeros({16}, int32), {{"V", 2}}},
      {zeros({16}, int32), {{"V", false}}},
      {zeros({16}, float16), {{"V", float16}}},
      {zeros({16}, float16), {{"V", bfloat16}}},
  };
  for (int round = 0; round < 2; ++round) {
    for (const auto& [inp, template_args] : variants) {
      auto memoized = call_sizeof(kernel, inp, template_args);
      auto fresh = call_sizeof(sizeof_kernel(), inp, template_args);
      CHECK_EQ(name_and_source(memoized), name_and_source(fresh));
      CHECK_EQ(memoized.item<int>(), fresh.item<int>());
    }
  }
}

TEST_CASE("test metal kernel copies share the memo across threads") {
  auto kernel = sizeof_kernel();
  auto s = default_stream(Device::gpu);
  auto inp = zeros({1}, int32);
  constexpr int n_threads = 8;
  constexpr int n_calls = 64;
  std::vector<std::vector<array>> outs(n_threads);
  std::vector<std::thread> threads;
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t, kernel]() {
      for (int i = 0; i < n_calls; ++i) {
        TemplateArgs template_args;
        if ((t + i) % 2 == 0) {
          template_args = {{"V", 1}};
        } else {
          template_args = {{"V", true}};
        }
        outs[t].push_back(call_sizeof(kernel, inp, template_args, s));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (int t = 0; t < n_threads; ++t) {
    eval(outs[t]);
    for (int i = 0; i < n_calls; ++i) {
      auto expected = (t + i) % 2 == 0 ? sizeof(int) : sizeof(bool);
      CHECK_EQ(outs[t][i].item<int>(), expected);
    }
  }
}

TEST_CASE("test metal kernel variant memo is bounded") {
  auto kernel = fast::metal_kernel(
      "value_kernel", {"inp"}, {"out"}, "out[0] = static_cast<int>(V);");
  auto inp = zeros({1}, int32);
  auto call = [&](int v) { return call_sizeof(kernel, inp, {{"V", v}}); };
  auto source_of = [](const array& out) {
    return static_cast<const fast::CustomKernel&>(out.primitive())
        .shared_source();
  };

  // A repeat call shares the memoized source.
  auto pending = call(0);
  auto pending_source = source_of(pending);
  CHECK_EQ(source_of(call(0)), pending_source);

  constexpr int n_values = 3 * fast::metal_kernel_max_cached_variants + 5;
  std::vector<std::weak_ptr<const std::string>> sources;
  for (int v = 1; v < n_values; ++v) {
    auto out = call(v);
    sources.push_back(source_of(out));
    CHECK_EQ(out.item<int>(), v);
  }

  // Only the memo holds the sources of evaluated arrays.
  size_t live = 0;
  for (auto& source : sources) {
    live += !source.expired();
  }
  CHECK_GT(live, 0);
  CHECK_LE(live, fast::metal_kernel_max_cached_variants);

  // The entry for 0 was evicted: a new call builds an equal, new source.
  auto rebuilt = call(0);
  CHECK_NE(source_of(rebuilt), pending_source);
  CHECK_EQ(*source_of(rebuilt), *pending_source);

  // The pending array still owns its source and evaluates correctly.
  pending_source.reset();
  CHECK_EQ(pending.item<int>(), 0);
  CHECK_EQ(rebuilt.item<int>(), 0);
}
