# Copyright © 2026 numen-tech

"""The Apple6 (A13 / G12) route, forced on any Apple GPU.

An A13 cannot build a Metal kernel that holds a SIMD-group matrix
multiply-accumulate (mlx/backend/metal/gpu_family.h), so on that GPU MLX routes
small matmuls, attention and the gated-delta update around the MMA kernels and
raises a typed error for anything that would still need one. MLX_FORCE_GPU_FAMILY=apple6,
read once at device init, selects the same route on any GPU. These tests run
each case twice, once in this process (the GPU's normal route) and once in a
child process with the route forced, and compare:

* results agree within fp16 / 2-bit tolerance for the Ternary Bonsai 2 2B
  shapes (hidden 2048, MLP 6144, 8 query / 2 KV heads of 256, depthwise
  conv1d of width 4 over 6144 channels, 128-wide gated-delta heads);
* inputs that would still need an MMA kernel (route version 2, gemma4-qat#991:
  quantized matmuls past 8 rows that the SIMT GEMM does not take, dense
  matmuls past 64 rows, attention without a vector kernel for its head dim)
  raise RuntimeError starting with "[mlx:unsupported-gpu-family]";
* with MLX_LOG_KERNEL_LOAD=1 every kernel the forced route builds is reported
  with mma=0.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np

PREFIX = "[mlx:unsupported-gpu-family]"

# Ternary Bonsai 2 2B linear shapes (K, N): GDN in_proj_qkv / in_proj_z / out_proj,
# attention q / k / v / o, MLP gate-up / down, and the tied lm_head.
BONSAI_2B_SHAPES = [
    (2048, 6144),
    (2048, 2048),
    (6144, 2048),
    (2048, 4096),
    (2048, 512),
    (2048, 248320),
]


def _rng(seed):
    return np.random.default_rng(seed)


def _quantized_cases():
    """(name, thunk) pairs computing one result array each, deterministically."""
    cases = []

    def qmm_case(K, N, bits, M, sym):
        def run():
            r = _rng(K * 31 + N + bits)
            w = mx.array(r.standard_normal((N, K)).astype(np.float32) * 0.05).astype(
                mx.float16
            )
            if sym:
                # Ternary weights with the symmetric (bias-free) affine path.
                w = mx.array(r.integers(-1, 2, size=(N, K)).astype(np.float32) * 0.03).astype(
                    mx.float16
                )
            wq, scales, biases = mx.quantize(w, group_size=128, bits=bits)
            x = mx.array(_rng(M + 7).standard_normal((M, K)).astype(np.float32)).astype(
                mx.float16
            )
            if sym:
                out = mx.quantized_matmul(
                    x, wq, scales, None, transpose=True, group_size=128, bits=bits
                )
            else:
                out = mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=128, bits=bits
                )
            return out

        cases.append((f"qmm_b{bits}_K{K}_N{N}_M{M}{'_sym' if sym else ''}", run))

    for K, N in BONSAI_2B_SHAPES:
        for M in (1, 2, 3, 5, 8):
            qmm_case(K, N, 2, M, False)
    # Route version 2 (gemma4-qat#991): more than 8 rows go through the SIMT GEMM
    # (affine_qgemm_simt: BM 8 / 16 / 32 tiles, M tiled in the grid, ragged M and N).
    for K, N in BONSAI_2B_SHAPES[:5]:
        for M in (9, 16, 17, 32, 33, 64):
            qmm_case(K, N, 2, M, False)
    qmm_case(2048, 6144, 2, 100, False)
    # Other widths on one shape, and the bias-free 2-bit path the Hadamard packs use.
    for bits in (4, 8):
        for M in (2, 8, 9, 32, 33):
            qmm_case(2048, 6144, bits, M, False)
    for M in (1, 4, 8):
        qmm_case(2048, 2048, 2, M, True)
    return cases


def _sdpa_cases():
    cases = []

    def sdpa_case(qL, kL, mask_kind):
        def run():
            r = _rng(qL * 1000 + kL)
            q = mx.array(r.standard_normal((1, 8, qL, 256)).astype(np.float32)).astype(
                mx.float16
            )
            k = mx.array(r.standard_normal((1, 2, kL, 256)).astype(np.float32)).astype(
                mx.float16
            )
            v = mx.array(r.standard_normal((1, 2, kL, 256)).astype(np.float32)).astype(
                mx.float16
            )
            if mask_kind == "causal":
                mask = "causal"
            elif mask_kind == "array":
                # Bottom-right aligned causal mask as a boolean array.
                i = np.arange(qL)[:, None] + (kL - qL)
                j = np.arange(kL)[None, :]
                mask = mx.array(j <= i)
            else:
                mask = None
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=256**-0.5, mask=mask
            )

        cases.append((f"sdpa_q{qL}_k{kL}_{mask_kind}", run))

    for qL in (1, 2, 5, 8, 9, 16, 32, 33):
        for kL, mask_kind in ((64, "causal"), (300, "array"), (8, "causal")):
            if qL <= kL:
                sdpa_case(qL, kL, mask_kind)
    # Keys long enough for the 2-pass kernel on the normal route (GQA 4, >= 4096).
    sdpa_case(1, 4200, "none")
    sdpa_case(32, 4200, "causal")
    return cases


def _other_cases():
    cases = []

    for M in (2, 5, 8, 9, 16, 32, 33, 64):
        # GDN in_proj_a / in_proj_b: a dense fp16 [16, 2048] Linear over the chunk.
        def dense(M=M):
            r = _rng(M)
            x = mx.array(r.standard_normal((1, M, 2048)).astype(np.float32)).astype(
                mx.float16
            )
            w = mx.array(r.standard_normal((16, 2048)).astype(np.float32) * 0.05).astype(
                mx.float16
            )
            return x @ w.T

        cases.append((f"dense_matmul_M{M}_N16", dense))

    for L in (1, 4, 8, 32, 33):
        # GDN depthwise causal conv1d: state(3) + chunk, width 4, 6144 channels.
        def conv(L=L):
            r = _rng(L + 99)
            x = mx.array(r.standard_normal((1, L + 3, 6144)).astype(np.float32)).astype(
                mx.float16
            )
            w = mx.array(r.standard_normal((6144, 4, 1)).astype(np.float32) * 0.2).astype(
                mx.float16
            )
            return mx.conv1d(x, w, groups=6144)

        cases.append((f"depthwise_conv1d_L{L}", conv))

    for T in (1, 8, 24, 32, 33):
        # GDN recurrence through the fork's fused op (forced route is always sequential).
        def gdn(T=T):
            r = _rng(T + 5)
            B, Hk, Hv, Dk, Dv = 1, 16, 16, 128, 128
            q = mx.array(r.standard_normal((B, T, Hk, Dk)).astype(np.float32) * 0.1).astype(
                mx.float16
            )
            k = mx.array(r.standard_normal((B, T, Hk, Dk)).astype(np.float32) * 0.1).astype(
                mx.float16
            )
            v = mx.array(r.standard_normal((B, T, Hv, Dv)).astype(np.float32) * 0.1).astype(
                mx.float16
            )
            g = mx.array(r.uniform(0.8, 1.0, (B, T, Hv)).astype(np.float32))
            beta = mx.array(r.uniform(0.1, 0.9, (B, T, Hv)).astype(np.float32))
            y, state = mx.fast.gated_delta_update(q, k, v, g, beta)
            return mx.concatenate([y.reshape(-1).astype(mx.float32), state.reshape(-1)])

        cases.append((f"gated_delta_update_T{T}", gdn))

    for n in (512, 2048):
        def had(n=n):
            r = _rng(n)
            x = mx.array(r.standard_normal((8, n)).astype(np.float32))
            return mx.hadamard_transform(x, scale=1.0 / np.sqrt(512))

        cases.append((f"hadamard_n{n}", had))

    def gather_dequant():
        r = _rng(3)
        w = mx.array(r.standard_normal((4096, 2048)).astype(np.float32) * 0.05).astype(
            mx.float16
        )
        wq, scales, biases = mx.quantize(w, group_size=128, bits=2)
        ids = mx.array([5, 17, 4000, 3, 99, 1024, 7, 2])
        return mx.dequantize(
            wq[ids], scales[ids], biases[ids], group_size=128, bits=2
        )

    cases.append(("embedding_gather_dequantize", gather_dequant))

    def norm_rope_softmax():
        r = _rng(11)
        x = mx.array(r.standard_normal((1, 8, 2048)).astype(np.float32)).astype(mx.float16)
        w = mx.array(r.standard_normal((2048,)).astype(np.float32)).astype(mx.float16)
        y = mx.fast.rms_norm(x, w, 1e-6)
        z = mx.fast.rope(
            y.reshape(1, 8, 8, 256).transpose(0, 2, 1, 3),
            64,
            traditional=False,
            base=1e7,
            scale=1.0,
            offset=3,
        )
        return mx.softmax(z.astype(mx.float32), axis=-1)

    cases.append(("rms_norm_rope_softmax", norm_rope_softmax))
    return cases


def _minicpm_cases():
    """MiniCPM5 1B (llama, bfloat16, 4-bit group 64, hidden 1536, MLP 4608, 16 query / 2 KV
    heads of 128, so GQA 8): sdpa_vector takes at most 4 query rows (4 * 8 = 32)."""
    cases = []

    def qmm(K, N, M):
        def run():
            r = _rng(K + N + M)
            w = mx.array(r.standard_normal((N, K)).astype(np.float32) * 0.05).astype(mx.bfloat16)
            wq, scales, biases = mx.quantize(w, group_size=64, bits=4)
            x = mx.array(r.standard_normal((1, M, K)).astype(np.float32)).astype(mx.bfloat16)
            return mx.quantized_matmul(
                x, wq, scales, biases, transpose=True, group_size=64, bits=4
            )

        cases.append((f"minicpm_qmm_K{K}_N{N}_M{M}", run))

    for K, N in ((1536, 2048), (1536, 256), (2048, 1536), (1536, 4608), (4608, 1536), (1536, 130560)):
        for M in (1, 4, 8, 9, 16, 32, 33):
            qmm(K, N, M)

    def sdpa(qL, kL, mask_kind):
        def run():
            r = _rng(qL * 77 + kL)
            q = mx.array(r.standard_normal((1, 16, qL, 128)).astype(np.float32)).astype(mx.bfloat16)
            k = mx.array(r.standard_normal((1, 2, kL, 128)).astype(np.float32)).astype(mx.bfloat16)
            v = mx.array(r.standard_normal((1, 2, kL, 128)).astype(np.float32)).astype(mx.bfloat16)
            if mask_kind == "array":
                i = np.arange(qL)[:, None] + (kL - qL)
                mask = mx.array(np.arange(kL)[None, :] <= i)
            else:
                mask = "causal"
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5, mask=mask)

        cases.append((f"minicpm_sdpa_q{qL}_k{kL}_{mask_kind}", run))

    for qL in (1, 2, 4, 5, 16, 32):
        sdpa(qL, 64, "causal")
        sdpa(qL, 300, "array")
    return cases


def all_cases():
    return _quantized_cases() + _sdpa_cases() + _other_cases() + _minicpm_cases()


def _error_cases():
    """(name, thunk): each must raise the typed error on the forced route."""
    cases = []

    def qmm_rows(M, bits=2, K=2048, N=6144):
        def run():
            r = _rng(M)
            w = mx.array(r.standard_normal((N, K)).astype(np.float32) * 0.05).astype(
                mx.float16
            )
            wq, scales, biases = mx.quantize(w, group_size=128, bits=bits)
            x = mx.array(r.standard_normal((M, K)).astype(np.float32)).astype(mx.float16)
            mx.eval(
                mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=128, bits=bits
                )
            )

        return run

    # Not eligible for the SIMT GEMM (affine 2/4/8-bit, transposed, non-batched,
    # K and group_size multiples of 32): 3-bit weights past 8 rows are still refused.
    cases.append(("qmm_3bit_9_rows", qmm_rows(9, bits=3)))
    cases.append(("qmm_3bit_64_rows", qmm_rows(64, bits=3)))

    def qmm_not_transposed():
        r = _rng(5)
        w = mx.array(r.standard_normal((2048, 512)).astype(np.float32) * 0.05).astype(mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=128, bits=4)
        x = mx.array(r.standard_normal((16, 2048)).astype(np.float32)).astype(mx.float16)
        mx.eval(mx.quantized_matmul(x, wq, scales, biases, transpose=False, group_size=128, bits=4))

    cases.append(("qmm_not_transposed_16_rows", qmm_not_transposed))

    def minicpm_quantized_kv_scores():
        # Quantized-KV attention is a batched quantized matmul (gqa * queryLength rows): batched
        # weights are not eligible for the SIMT GEMM.
        r = _rng(3)
        kq, ks, kb = mx.quantize(
            mx.array(r.standard_normal((1, 2, 300, 128)).astype(np.float32)).astype(mx.bfloat16),
            group_size=64,
            bits=8,
        )
        q = mx.array(r.standard_normal((1, 2, 16, 128)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(mx.quantized_matmul(q, kq, ks, kb, transpose=True, group_size=64, bits=8))

    cases.append(("minicpm_quantized_kv_two_queries", minicpm_quantized_kv_scores))

    def dense_big():
        # Past kApple6GemvWideMaxRows (64).
        a = mx.ones((65, 256), dtype=mx.float16)
        b = mx.ones((256, 256), dtype=mx.float16)
        mx.eval(a @ b)

    cases.append(("dense_matmul_65_rows", dense_big))

    def dense_vocab_wide():
        a = mx.ones((9, 2048), dtype=mx.float16)
        b = mx.ones((65536, 2048), dtype=mx.float16)
        mx.eval(a @ b.T)

    cases.append(("dense_matmul_9_rows_vocab_wide", dense_vocab_wide))

    def sdpa_head_dim_without_a_vector_kernel():
        # Head dim 80 has a steel kernel but no vector kernel: the unfused path would need steel GEMMs.
        r = _rng(7)
        q = mx.array(r.standard_normal((1, 8, 16, 80)).astype(np.float32)).astype(mx.float16)
        k = mx.array(r.standard_normal((1, 2, 64, 80)).astype(np.float32)).astype(mx.float16)
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, k, scale=1.0, mask="causal"))

    cases.append(("sdpa_head_dim_80", sdpa_head_dim_without_a_vector_kernel))

    def conv2d():
        x = mx.ones((1, 32, 32, 16), dtype=mx.float16)
        w = mx.ones((32, 3, 3, 16), dtype=mx.float16)
        mx.eval(mx.conv2d(x, w))

    cases.append(("conv2d_implicit_gemm", conv2d))
    return cases


def _child(mode, out_dir):
    """Runs in a child with MLX_FORCE_GPU_FAMILY=apple6."""
    if mode == "results":
        for name, thunk in all_cases():
            out = thunk()
            mx.eval(out)
            np.save(os.path.join(out_dir, name + ".npy"), np.array(out.astype(mx.float32)))
    elif mode == "errors":
        report = {}
        for name, thunk in _error_cases():
            try:
                thunk()
                report[name] = None
            except RuntimeError as e:
                report[name] = str(e)
        with open(os.path.join(out_dir, "errors.json"), "w") as f:
            json.dump(report, f)
    else:
        raise SystemExit(f"unknown child mode {mode}")


def _run_child(mode, out_dir, **extra_env):
    env = dict(os.environ)
    env["MLX_FORCE_GPU_FAMILY"] = "apple6"
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", mode, out_dir],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@unittest.skipIf(not mx.metal.is_available(), "requires Metal")
class TestApple6Route(mlx_tests.MLXTestCase):
    def test_forced_route_matches_normal_route(self):
        # The forced route runs different kernels for the same math (qmv / qmv_wide
        # for qmm, gemv_wide for the dense products); fp16 outputs agree to a few
        # parts in a thousand of their scale, the tier the Bonsai 2 oracle uses.
        with tempfile.TemporaryDirectory() as d:
            done = _run_child("results", d)
            self.assertEqual(done.returncode, 0, done.stderr[-4000:])
            for name, thunk in all_cases():
                with self.subTest(case=name):
                    expect = np.array(thunk().astype(mx.float32))
                    got = np.load(os.path.join(d, name + ".npy"))
                    self.assertEqual(got.shape, expect.shape)
                    scale = max(float(np.abs(expect).max()), 1e-6)
                    err = float(np.abs(got - expect).max()) / scale
                    # bfloat16 (MiniCPM5) carries 8 mantissa bits: its two kernels differ by a few ulps.
                    tol = 2e-2 if name.startswith("minicpm_") else 4e-3
                    self.assertLess(err, tol, f"{name}: max abs error / max |expect| = {err}")

    def test_forced_route_raises_typed_error_for_mma_paths(self):
        with tempfile.TemporaryDirectory() as d:
            done = _run_child("errors", d)
            self.assertEqual(done.returncode, 0, done.stderr[-4000:])
            with open(os.path.join(d, "errors.json")) as f:
                report = json.load(f)
        self.assertEqual(set(report), {n for n, _ in _error_cases()})
        for name, message in report.items():
            with self.subTest(case=name):
                self.assertIsNotNone(message, f"{name} did not raise")
                self.assertTrue(
                    message.startswith(PREFIX), f"{name}: unexpected error {message[:300]}"
                )

    def test_forced_route_builds_no_mma_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            done = _run_child("results", d, MLX_LOG_KERNEL_LOAD="1")
            self.assertEqual(done.returncode, 0, done.stderr[-4000:])
            loaded = [
                line for line in done.stderr.splitlines() if line.startswith("[mlx:kernel-load] ")
            ]
            self.assertGreater(len(loaded), 20)
            mma = [line for line in loaded if line.endswith(" mma=1")]
            self.assertEqual(mma, [])

    def test_normal_route_is_unchanged_on_this_gpu(self):
        # Without the override a GPU of family Apple7+ keeps its kernels: a 33-row
        # product still goes through qmm.
        r = _rng(0)
        w = mx.array(r.standard_normal((512, 2048)).astype(np.float32) * 0.05).astype(
            mx.float16
        )
        wq, scales, biases = mx.quantize(w, group_size=128, bits=4)
        x = mx.array(r.standard_normal((33, 2048)).astype(np.float32)).astype(mx.float16)
        out = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=128, bits=4)
        ref = x.astype(mx.float32) @ mx.dequantize(
            wq, scales, biases, group_size=128, bits=4
        ).astype(mx.float32).T
        mx.eval(out, ref)
        self.assertLess(
            float(mx.abs(out.astype(mx.float32) - ref).max()) / float(mx.abs(ref).max()), 5e-3
        )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3])
    else:
        mlx_tests.MLXTestRunner()
