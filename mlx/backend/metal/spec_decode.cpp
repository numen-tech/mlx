// Copyright © 2024 Apple Inc.

#include <algorithm>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

// The kernel indexes dense rows (row * K, row * (K + 1)). A row-contiguous
// input is bound as is; a view (transposed, sliced, broadcast) is copied
// first and the copy kept alive until the command buffer completes.
array ensure_row_contiguous(const array& x, const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

} // namespace

bool SpecDecodeVerify::use_fallback(Stream s) {
  return false;
}

void SpecDecodeVerify::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  // Copies (if any) are encoded before this kernel's pipeline is set.
  array draft = ensure_row_contiguous(inputs[0], s); // [B, K] int32
  array target = ensure_row_contiguous(inputs[1], s); // [B, K+1] int32
  auto& n_accepted = outputs[0]; // [B] int32
  auto& committed = outputs[1]; // [B, K+1] int32

  n_accepted.set_data(allocator::malloc(n_accepted.nbytes()));
  committed.set_data(allocator::malloc(committed.nbytes()));

  int B = draft.shape(0);
  int K = draft.shape(1);

  auto& d = metal::device(s.device);
  auto kernel = d.get_kernel("spec_decode_verify");
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  // K = 0 has no draft buffer, and the kernel reads none: bind target there.
  compute_encoder.set_input_array(K > 0 ? draft : target, 0);
  compute_encoder.set_input_array(target, 1);
  compute_encoder.set_output_array(n_accepted, 2);
  compute_encoder.set_output_array(committed, 3);
  compute_encoder.set_bytes(K, 4);
  compute_encoder.set_bytes(B, 5);

  // One thread per batch row (K is tiny; no reduction).
  int tgroup = std::min(B, 256);
  MTL::Size grid_dims = MTL::Size(B, 1, 1);
  MTL::Size group_dims = MTL::Size(tgroup, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
