// Copyright © 2026 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void SpecDecodeVerify::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& draft = inputs[0]; // [B, K] int32
  auto& target = inputs[1]; // [B, K+1] int32
  auto& n_accepted = outputs[0]; // [B] int32
  auto& committed = outputs[1]; // [B, K+1] int32
  n_accepted.set_data(allocator::malloc(n_accepted.nbytes()));
  committed.set_data(allocator::malloc(committed.nbytes()));
  if (n_accepted.size() == 0) {
    return;
  }

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(draft);
  encoder.set_input_array(target);
  encoder.set_output_array(n_accepted);
  encoder.set_output_array(committed);
  encoder.dispatch([draft = array::unsafe_weak_copy(draft),
                    target = array::unsafe_weak_copy(target),
                    n_accepted = array::unsafe_weak_copy(n_accepted),
                    committed = array::unsafe_weak_copy(committed)]() mutable {
    int B = draft.shape(0);
    int K = draft.shape(1);
    auto ds = draft.strides();
    auto ts = target.strides();
    auto* d = draft.data<int32_t>();
    auto* t = target.data<int32_t>();
    auto* n_out = n_accepted.data<int32_t>();
    auto* c_out = committed.data<int32_t>();
    for (int b = 0; b < B; ++b) {
      auto dt = [&](int j) { return d[b * ds[0] + j * ds[1]]; };
      auto tt = [&](int j) { return t[b * ts[0] + j * ts[1]]; };
      int n = K;
      for (int j = 0; j < K; ++j) {
        if (dt(j) != tt(j)) {
          n = j;
          break;
        }
      }
      n_out[b] = n;
      int32_t* c = c_out + b * (K + 1);
      for (int j = 0; j < K + 1; ++j) {
        c[j] = (j < n) ? dt(j) : ((j == n) ? tt(n) : 0);
      }
    }
  });
}

} // namespace mlx::core::fast
