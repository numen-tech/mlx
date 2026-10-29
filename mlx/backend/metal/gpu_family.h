// Copyright © 2026 numen-tech

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

// GPU-family route for Apple6 (A13 / G12) GPUs (numen-tech/gemma4-qat#703).
//
// The G12 backend compiler cannot lower `simdgroup_multiply_accumulate`
// (SIMD-group matrix MMA is an Apple7+ feature): building a pipeline for any
// kernel that contains one aborts MTLCompilerService (`Emitted non-G12
// instruction for G12`). Every steel GEMM, qmm, steel attention and implicit
// GEMM convolution kernel holds one, so on an Apple6 GPU MLX
//
//   * routes affine 2/4/8-bit quantized matmuls of 5 or more rows to the SIMT
//     GEMM (affine_qgemm_simt, numen-tech/gemma4-qat#991) and the rest of at
//     most 8 rows to the MMA-free qmv / qmv_wide kernels, dense matmuls of at
//     most 64 rows to gemv / gemv_wide, and full attention of any query length
//     to sdpa_vector (see kApple6RouteVersion for the shapes per version), and
//   * throws UnsupportedGPUFamilyError, before the compiler is reached, from
//     every dispatch that would still need an MMA kernel (other quantization
//     modes and widths past 8 rows, dense matmuls past 64 rows, attention
//     without a vector kernel for its head dim).
//
// The route is selected at runtime by the GPU family only
// (Device::is_apple6()); every other GPU takes the code paths it always took.
// MLX_FORCE_GPU_FAMILY=apple6 (read once, at device init) selects the route on
// any GPU, for tests.

namespace mlx::core::metal {

// The version of the Apple6 route this build carries. A downstream package
// that reads it (mlx-swift's `mlx_apple6_route_version()` shim, numen-tech/
// gemma4-qat) only opens its A13 allowlist when it is >= 1, so a build whose
// mlx core predates the route keeps refusing on Apple6 instead of reaching the
// GPU compiler. Bump it when the set of shapes the route accepts changes.
//
//   1: quantized matmul, dense matmul and full attention of at most 8 rows
//      (numen-tech/gemma4-qat#703).
//   2: affine 2/4/8-bit quantized matmul of any row count through the SIMT
//      GEMM (affine_qgemm_simt), full attention of any query length through
//      sdpa_vector (never the 2-pass kernel), dense matmul of up to 64 rows
//      (numen-tech/gemma4-qat#991).
inline constexpr int kApple6RouteVersion = 2;

// The exception message starts with this prefix, so a caller that only sees the
// message (mlx-c hands the Swift/Python side a string) can still recognize it.
inline constexpr std::string_view kUnsupportedGPUFamilyPrefix =
    "[mlx:unsupported-gpu-family] ";

// Thrown when an Apple6 GPU would need a kernel it cannot compile.
class UnsupportedGPUFamilyError : public std::runtime_error {
 public:
  explicit UnsupportedGPUFamilyError(const std::string& what)
      : std::runtime_error(std::string(kUnsupportedGPUFamilyPrefix) + what) {}
};

// Whether a Metal kernel (the base name MLX passes to Device::get_kernel)
// contains a SIMD-group matrix multiply-accumulate, and so cannot be built for
// an Apple6 GPU. A name-based backstop behind the dispatch-level routing: the
// list is checked against the AIR of every kernel in mlx.metallib by
// tests/metal_gpu_family_tests.cpp's fixtures and by the classification in
// numen-tech/gemma4-qat#703.
inline bool is_mma_kernel_name(std::string_view name) {
  auto has_prefix = [&](std::string_view p) {
    return name.substr(0, p.size()) == p;
  };
  auto contains = [&](std::string_view p) {
    return name.find(p) != std::string_view::npos;
  };
  // Steel GEMM (fused, split-K, masked, gather, segmented), steel attention
  // and the implicit-GEMM / Winograd / general convolutions.
  if (has_prefix("steel_") || has_prefix("implicit_gemm_conv") ||
      has_prefix("winograd_conv")) {
    return true;
  }
  // Quantized matrix-matrix products (qmm, split-K qmm, gather qmm) of every
  // mode; the vector products (qmv, qvm, qmv_quad, qmv_wide, ...) are MMA-free.
  if (contains("_qmm_") || contains("_gather_qmm_")) {
    return true;
  }
  // The fork's fused chunked gated-delta kernels, and every NAX kernel.
  if (has_prefix("gated_delta_fused") || contains("_nax")) {
    return true;
  }
  return false;
}

} // namespace mlx::core::metal
