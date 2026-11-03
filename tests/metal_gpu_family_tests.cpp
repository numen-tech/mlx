// Copyright © 2026 numen-tech

#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/gpu_family.h"
#include "mlx/mlx.h"
#include "mlx/utils.h"

using namespace mlx::core::metal;

// The names below come from classifying every kernel function of mlx.metallib
// by its AIR: a kernel is "MMA" when it calls
// air.simdgroup_matrix_8x8_multiply_accumulate, which the G12 (A13) backend
// compiler cannot lower. is_mma_kernel_name must match all of those (a miss
// lets the kernel reach the compiler, which aborts MTLCompilerService) and may
// match more (NAX kernels, the split-K accumulate).

TEST_CASE("test apple6 MMA kernel names are denied") {
  const std::vector<std::string> mma = {
      "affine_qmm_t_float16_t_gs_128_b_2_alN_true_batch_0",
      "affine_qmm_t_float16_t_gs_128_b_2_alN_true_batch_0_ib",
      "affine_qmm_n_float16_t_gs_128_b_2_batch_0",
      "affine_qmm_t_splitk_float16_t_gs_128_b_2_alN_true",
      "affine_gather_qmm_t_float16_t_gs_128_b_4_alN_true",
      "affine_gather_qmm_rhs_nt_float16_t_gs_128_b_4_bm_16_bn_32_bk_32",
      "affine_gather_qmm_n_float16_t_gs_64_b_8",
      "mxfp4_qmm_t_float16_t_gs_32_b_4_alN_true_batch_0",
      "nvfp4_gather_qmm_t_float16_t_gs_16_b_4_alN_true",
      "steel_gemm_fused_nt_float16_float16_bm64_bn64_bk16_wm2_wn2_has_batch_t_"
      "use_out_source_n_do_axpby_n_align_M_t_align_N_t_align_K_t",
      "steel_gemm_splitk_nt_float16_float32_bm32_bn16_bk16_wm2_wn2_MN_taligned_"
      "K_taligned",
      "steel_gemm_block_outmask_bool__opmask_bool__nn_float16_float16_bm32_bn32_"
      "bk16_wm2_wn2_MN_naligned_K_naligned",
      "steel_gather_mm_tn_float16_float16_bm16_bn32_bk16_wm1_wn2",
      "steel_segmented_mm_nn_float16_float16_bm32_bn32_bk16_wm2_wn2",
      "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
      "implicit_gemm_conv_2d_float16_bm32_bn32_bk16_wm2_wn2_channel_l_filter_l",
      "implicit_gemm_conv_3d_float16_bm32_bn32_bk16_wm2_wn2",
      "winograd_conv_2d_input_transform_float16_bc32",
      "gated_delta_fused_chunk_float16_t_128_128_16_16_8",
      "gated_delta_fused_nax_float16_t_128_128_16_16_16_16",
      "affine_qmm_t_nax_float16_t_gs_128_b_4_bm64_bn64_bk64",
      "steel_gemm_fused_nax_nn_float16_float16",
  };
  for (const auto& name : mma) {
    CAPTURE(name);
    CHECK(is_mma_kernel_name(name));
  }
}

TEST_CASE("test apple6 MMA-free kernel names are allowed") {
  const std::vector<std::string> free = {
      // The quantized vector products the Apple6 route relies on.
      "affine_qmv_fast_float16_t_gs_128_b_2_batch_0_pr_0_hgs_0_r_4",
      "affine_qmv_fast_float16_t_gs_128_b_2_batch_0_pr_0_hgs_0_r_4_ib",
      "affine_qmv_float16_t_gs_128_b_2_batch_0",
      "affine_qmv_wide_float16_t_gs_128_b_2_nv_5_kl_8_batch_0",
      "affine_qmv_wide_float16_t_gs_128_b_2_nv_5_kl_8_batch_0_ib",
      "affine_qmv_wide_rr_float16_t_gs_128_b_2_nv_5_batch_0",
      "affine_qgemm_simt_float16_t_gs_128_b_2_bm_32",
      "affine_qgemm_simt_bfloat16_t_gs_64_b_4_bm_8_ib",
      "affine_qmv_quad_float16_t_gs_128_b_2_d_128_batch_0",
      "affine_sym_qmv_fast_float16_t_gs_128_b_2_batch_0_pr_0_hgs_0_r_4",
      "affine_qvm_float16_t_gs_128_b_4",
      "affine_qvm_split_k_float16_t_gs_128_b_4_spk_8",
      "mxfp4_qmv_float16_t_gs_32_b_4_batch_0",
      "affine_dequantize_float16_t_gs_128_b_2",
      "affine_quantize_float16_t_gs_128_b_2",
      // Attention, dense vector products, conv, recurrence, the rest.
      "sdpa_vector_float16_t_256_256_boolmask_qnt_nc_nosinks",
      "sdpa_vector_2pass_1_float16_t_256_256",
      "sdpa_blocked_causal_mask_float16",
      "gemv_float16_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
      "gemv_wide_float16_nv5_kl32_nc0_axpby0",
      "depthwise_conv_1d_float16",
      "depthwise_conv_2d_float16",
      "seq_gated_delta_float16_t_128_128_16_16_1",
      "n2hadamard_512_float32",
      "rmsfloat16",
      "rope_float16_",
      "block_softmax_float32",
      "gather_frontfloat16_int32_int_2",
      "vn_copyfloat32float16",
      "g2_Selectfloat16",
      "custom_kernel_gated_delta_step",
  };
  for (const auto& name : free) {
    CAPTURE(name);
    CHECK_FALSE(is_mma_kernel_name(name));
  }
}

TEST_CASE("test apple6 typed error carries the prefix") {
  try {
    throw UnsupportedGPUFamilyError("quantized_matmul with 9 rows");
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    CHECK(msg.rfind(std::string(kUnsupportedGPUFamilyPrefix), 0) == 0);
    CHECK(msg.find("quantized_matmul with 9 rows") != std::string::npos);
  }
}

TEST_CASE("test apple6 route version is published") {
  // Downstream packages gate their A13 allowlist on this being >= 1, and the
  // wider prefill window (SIMT GEMM, any-length sdpa_vector) on >= 2.
  CHECK(kApple6RouteVersion >= 2);
}

// MLX_DISABLE_NAX (numen-tech/gemma4-qat#1085): the kill switch for a GPU
// generation the NAX kernels are not verified on.
TEST_CASE("test nax gate honours the OS, the generation and MLX_DISABLE_NAX") {
  CHECK(nax_gate(true, 18, false));
  CHECK(nax_gate(true, 19, false));
  CHECK_FALSE(nax_gate(true, 19, true));
  CHECK_FALSE(nax_gate(true, 18, true));
  CHECK_FALSE(nax_gate(true, 17, false));
  CHECK_FALSE(nax_gate(false, 19, false));
}

// MLX_METAL_NAX (numen-tech/gemma4-qat#312): a gen 17 part needs the force, and
// no value reaches an unsupported OS, a GPU below gen 17, or past the kill
// switch.
TEST_CASE("test nax gate honours MLX_METAL_NAX") {
  // Forced on.
  CHECK(nax_gate(true, 17, false, 1));
  CHECK(nax_gate(true, 18, false, 1));
  CHECK(nax_gate(true, 19, false, 1));
  CHECK_FALSE(nax_gate(true, 16, false, 1));
  CHECK_FALSE(nax_gate(true, 15, false, 1));
  CHECK_FALSE(nax_gate(false, 17, false, 1));
  CHECK_FALSE(nax_gate(false, 19, false, 1));
  // The kill switch wins.
  CHECK_FALSE(nax_gate(true, 17, true, 1));
  CHECK_FALSE(nax_gate(true, 19, true, 1));
  // Forced off.
  CHECK_FALSE(nax_gate(true, 19, false, 0));
  CHECK_FALSE(nax_gate(true, 18, false, 0));
  CHECK_FALSE(nax_gate(true, 17, false, 0));
  // Unset keeps the compiled rule.
  CHECK(nax_gate(true, 18, false, -1));
  CHECK_FALSE(nax_gate(true, 17, false, -1));
}

TEST_CASE("test MLX_METAL_NAX reads the whole string") {
  CHECK(env::parse_metal_nax("1") == 1);
  CHECK(env::parse_metal_nax("0") == 0);
  CHECK(env::parse_metal_nax(nullptr) == -1);
  for (auto value : {"", "1garbage", "garbage", "2", "-1", "00", "01", "true"}) {
    CAPTURE(value);
    CHECK(env::parse_metal_nax(value) == -1);
  }
}

TEST_CASE("test the switch version says MLX_METAL_NAX is there") {
  // Downstream packages gate their force-on on this being >= 2.
  CHECK(kNaxDisableEnvVersion >= 2);
}

TEST_CASE("test kernel trace records every lookup while on, and only then") {
  using namespace mlx::core;
  auto dev = mlx::core::Device::gpu;
  auto a = ones({8, 8}, float32, dev);
  auto b = ones({8, 8}, float32, dev);
  eval(a, b);

  // A first lookup and a cached one must both show.
  for (int run = 0; run < 2; ++run) {
    kernel_trace_start();
    auto c = add(a, b, dev);
    eval(c);
    auto names = kernel_trace_stop();
    CAPTURE(run);
    CHECK_FALSE(names.empty());
  }

  // Off: nothing accumulates into the next trace.
  auto d = subtract(a, b, dev);
  eval(d);
  kernel_trace_start();
  auto names = kernel_trace_stop();
  CHECK(names.empty());
}

TEST_CASE("test is_nax_available is off under MLX_DISABLE_NAX") {
  // On a gen < 18 GPU NAX is off either way; on gen >= 18 this run checks that
  // the environment variable reaches the cached decision.
  if (mlx::core::env::disable_nax()) {
    CHECK_FALSE(is_nax_available());
  } else {
    MESSAGE("MLX_DISABLE_NAX not set; run with MLX_DISABLE_NAX=1 to check it");
  }
}
