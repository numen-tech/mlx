// Copyright © 2026 Apple Inc.

// JIT-only: these tests call the jit_kernels.cpp entry points directly, so they
// build each NAX kernel from source even on a GPU where is_nax_available() is
// false. They are compiled only when MLX_METAL_JIT is on.

#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

struct AttnNaxConfig {
  int bd;
  int bv;
};

// The head dims sdpa_full_self_attention_nax accepts (72 and 80 are padded to
// 96 first; (96, 64) is the one asymmetric pair). Tiles follow that function:
// bq = 32 for D 512 else 64, bk = 32, wm = 2 for D 512 else 4, and d-split
// (wn = D / 128) for D 256 and 512.
const std::vector<AttnNaxConfig> kAttnNaxConfigs = {
    {64, 64},
    {96, 96},
    {96, 64},
    {128, 128},
    {256, 256},
    {512, 512},
};

} // namespace

TEST_CASE("test steel attention nax JIT builds every SDPA configuration") {
  if (std::string(metal::steel_attention_nax()).empty()) {
    MESSAGE("NAX JIT sources not built (MLX_METAL_NO_NAX); skipping");
    return;
  }
  auto& d = metal::device(Device::gpu);

  for (auto dtype : {float16, bfloat16, float32}) {
    for (auto [bd, bv] : kAttnNaxConfigs) {
      for (bool bool_mask : {false, true}) {
        int bq = bd == 512 ? 32 : 64;
        int bk = 32;
        bool split_d = (bd == 256 || bd == 512);
        int wm = bd == 512 ? 2 : 4;
        int wn = split_d ? bd / 128 : 1;

        array q = zeros({1, 1, 1, bd}, dtype);
        array m = zeros({1}, bool_mask ? bool_ : dtype);

        std::string base_name;
        concatenate(
            base_name,
            split_d ? "steel_attention_dsplit_" : "steel_attention_",
            type_to_name(q),
            "_bq",
            bq,
            "_bk",
            bk,
            "_bd",
            bd,
            "_bv",
            bv,
            "_wm",
            wm,
            "_wn",
            wn,
            "_mask",
            type_to_name(m));
        CAPTURE(base_name);

        bool align_Q = false;
        bool align_K = false;
        bool has_mask = bool_mask;
        bool do_causal = !bool_mask;
        bool has_sinks = false;
        bool save_lse = false;
        metal::MTLFCList func_consts = {
            {&align_Q, MTL::DataType::DataTypeBool, 200},
            {&align_K, MTL::DataType::DataTypeBool, 201},
            {&has_mask, MTL::DataType::DataTypeBool, 300},
            {&do_causal, MTL::DataType::DataTypeBool, 301},
            {&has_sinks, MTL::DataType::DataTypeBool, 302},
            {&save_lse, MTL::DataType::DataTypeBool, 303}};

        MTL::ComputePipelineState* kernel = nullptr;
        CHECK_NOTHROW(
            kernel = get_steel_attention_nax_kernel(
                d,
                base_name,
                base_name + "_jit_test",
                func_consts,
                q,
                bq,
                bk,
                bd,
                bv,
                wm,
                wn,
                m,
                split_d));
        CHECK(kernel != nullptr);
      }
    }
  }
}
