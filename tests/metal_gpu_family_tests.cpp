// Copyright © 2026 numen-tech

#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/backend/metal/gpu_family.h"

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
