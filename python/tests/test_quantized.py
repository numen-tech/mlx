# Copyright © 2023-2026 Apple Inc.

import math
import os
import platform
import re
import subprocess
import unittest
from itertools import product

import mlx.core as mx
import numpy as np
import mlx_tests


def is_m1_mac():
    if platform.system() != "Darwin":
        return False
    cmd = "sysctl -n machdep.cpu.brand_string"
    cpu = subprocess.check_output(cmd, shell=True).decode().strip()
    return cpu.startswith("Apple M1")


class TestQuantized(mlx_tests.MLXTestCase):
    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 512))
        for gs in [32, 64, 128]:
            for b in [1, 2, 3, 5, 6, 4, 8]:
                if b == 1 and mx.cuda.is_available():
                    continue
                with self.subTest(gs=gs, b=b):
                    w_q, scales, biases = mx.quantize(w, group_size=gs, bits=b)
                    w_hat = mx.dequantize(w_q, scales, biases, gs, b)
                    errors = (w - w_hat).abs().reshape(*scales.shape, -1)
                    eps = 1e-6
                    self.assertTrue((errors <= (scales[..., None] + eps).abs()).all())

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        for gs in [32, 64, 128]:
            for b in [1, 2, 3, 4, 5, 6, 8]:
                if b == 1 and mx.cuda.is_available():
                    continue
                w_q, scales, biases = mx.quantize(a, gs, b)
                a_hat = mx.dequantize(w_q, scales, biases, gs, b)
                self.assertTrue(mx.all(a_hat == 0))

        # slices
        if mx.default_device() == mx.gpu:
            w = mx.random.normal(shape=(2, 256, 32))
            quant = {"group_size": 32, "bits": 4}
            wq, scales, biases = mx.quantize(w, **quant)
            wq_s = wq[:, :16, :]
            scales_s = scales[:, :16, :]
            biases_s = biases[:, :16, :]
            dq_cpu = mx.dequantize(wq_s, scales_s, biases_s, **quant, stream=mx.cpu)
            dq_gpu = mx.dequantize(wq_s, scales_s, biases_s, **quant, stream=mx.gpu)
            self.assertTrue(mx.abs(dq_cpu - dq_gpu).max().item() < 1e-6)

    def test_affine_quantize_signed_groups(self):
        for gs, bits in product([32, 64, 128], [1, 2, 3, 4, 5, 6, 8]):
            if bits == 1 and mx.default_device() == mx.gpu and mx.cuda.is_available():
                continue
            n_bins = (1 << bits) - 1
            grid = [i % (n_bins + 1) for i in range(gs)]
            grid[-1] = n_bins
            groups = [
                [v - 2 * n_bins for v in grid],
                [2 * n_bins - v for v in grid],
                [v - (1 << (bits - 1)) for v in grid],
                [-2] * gs,
                [0] * gs,
            ]
            with self.subTest(gs=gs, bits=bits):
                w = mx.array(groups, dtype=mx.float32)
                wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
                restored = mx.dequantize(wq, scales, biases, gs, bits)
                # These integer grids have unit spacing in the quantized range.
                self.assertLess(mx.max(mx.abs(w - restored)).item(), 1e-5)

    def test_affine_quantize_all_negative_kv_group_8bit(self):
        # ml-explore/mlx#4775: an all-negative group used to scale over [min, 0].
        if not mx.metal.is_available():
            self.skipTest("Compares the GPU's quantization with the CPU's")
        shape = (1, 2, 8, 256)
        k = mx.random.uniform(shape=shape, key=mx.random.key(4775)) * 0.1 - 3.0
        k = k.astype(mx.float16)
        groups = k.astype(mx.float32).reshape(-1, 64)
        step = (groups.max(axis=-1) - groups.min(axis=-1)) / 255
        for device in [mx.cpu, mx.gpu]:
            with self.subTest(device=device):
                wq, scales, biases = mx.quantize(
                    k, group_size=64, bits=8, stream=device
                )
                restored = mx.dequantize(
                    wq, scales, biases, group_size=64, bits=8, stream=device
                )
                error = mx.abs(restored.astype(mx.float32) - k.astype(mx.float32))
                error = error.reshape(-1, 64).max(axis=-1)
                # Half a step plus fp16 rounding; the old scale erred by up to 6e-3.
                self.assertTrue(mx.all(error <= 0.5 * step + 3e-3).item())
        wq_c, s_c, b_c = mx.quantize(k, group_size=64, bits=8, stream=mx.cpu)
        wq_g, s_g, b_g = mx.quantize(k, group_size=64, bits=8, stream=mx.gpu)
        dq_c = mx.dequantize(wq_c, s_c, b_c, group_size=64, bits=8, stream=mx.cpu)
        dq_g = mx.dequantize(wq_g, s_g, b_g, group_size=64, bits=8, stream=mx.gpu)
        self.assertLess(mx.abs(dq_c - dq_g).max().item(), 3e-3)

    def test_mxfp4_quantize_dequantize(self):
        lut = mx.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[mx.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 32)
        w[:, 0] = 6
        w = (w + 3e-6).astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=64, mode="mxfp4")

        w_q, scales = mx.quantize(w, mode="mxfp4")
        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=64, mode="mxfp4")

        # Invalid output type
        with self.assertRaises(ValueError):
            mx.dequantize(
                w_q, scales, group_size=32, bits=4, mode="mxfp4", dtype=mx.int32
            )

        w_hat = mx.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="mxfp4")
        w_hat = mx.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(mx.all(w_hat == 0))

    def test_mxfp8_quantize_dequantize(self):
        w = 2 * mx.random.uniform(shape=(512, 32)) - 1
        w = w.astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=32, bits=7, mode="mxfp8")
        w_q, scales = mx.quantize(w, group_size=32, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=16, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=4, mode="mxfp8")

        w_hat = mx.dequantize(w_q, scales, mode="mxfp8")

        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-1, atol=1e-1))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="mxfp8")
        w_hat = mx.dequantize(w_q, scales, mode="mxfp8")
        self.assertTrue(mx.all(w_hat == 0))

    def test_mxfp8_block_scale_does_not_saturate(self):
        # E4M3 has three mantissa bits, so an in-range element loses at most
        # half a step, 6.25%. More than that means the block scale rounded
        # below amax/448 and the block maximum saturated.
        mx.random.seed(0)
        group_size = 32
        n_blocks = 512

        # Sweep the block magnitude across one binade so both scale rounding
        # directions are covered.
        w = mx.random.normal(shape=(n_blocks, group_size))
        w = w * mx.exp(mx.arange(n_blocks) / n_blocks * math.log(2.0)).reshape(-1, 1)

        w_q, scales = mx.quantize(w, group_size=group_size, mode="mxfp8")
        w_hat = mx.dequantize(w_q, scales, group_size=group_size, mode="mxfp8")

        # Quantization is monotone in |w|, so a block's largest output is the
        # reconstruction of its largest input.
        amax = mx.max(mx.abs(w), axis=1)
        rel = mx.abs(amax - mx.max(mx.abs(w_hat), axis=1)) / amax
        self.assertLess(mx.max(rel).item(), 0.0626)

    def test_nvfp4_quantize_dequantize(self):
        lut = mx.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[mx.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 16)
        w[:, 0] = 6
        w = (w + 3e-6).astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=64, mode="nvfp4")

        w_q, scales = mx.quantize(w, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=32, mode="nvfp4")

        w_hat = mx.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # A scale shared across a 32-value SIMD group instead of computed
        # per 16-value group cannot represent the low-magnitude groups.
        alternating = mx.zeros((64, 16), dtype=mx.bfloat16)
        alternating[::2] = 6 * 2**-9
        alternating[1::2] = 6.0
        w_q, scales = mx.quantize(alternating, mode="nvfp4")
        w_hat = mx.dequantize(w_q, scales, mode="nvfp4", dtype=mx.bfloat16)
        self.assertTrue(mx.allclose(alternating, w_hat, rtol=1e-5, atol=1e-6))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="nvfp4")
        w_hat = mx.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(mx.all(w_hat == 0))

        # Test nvfp4 quantize/dequantize with tensor-scale global_scale
        global_scale = w.abs().max().astype(mx.float32)

        w_q, scales = mx.quantize(w, mode="nvfp4", global_scale=global_scale)
        w_hat = mx.dequantize(
            w_q, scales, group_size=16, bits=4, mode="nvfp4", global_scale=global_scale
        )
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

    @unittest.skipIf(mx.cuda.is_available(), "1-bit quantization NYI on CUDA")
    def test_1bit_quantize_dequantize(self):
        """Test 1-bit affine quantization."""

        # Symmetric binary weights {-0.5, +0.5} should round-trip perfectly
        # (affine formula gives scale=1.0, bias=-0.5)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="pack_symmetric_weights"):
                signs = (mx.random.uniform(shape=(128, 512)) > 0.5).astype(mx.float32)
                w = signs * 1.0 - (1 - signs) * 1.0  # {-1.0, +1.0}
                w = w * 0.5  # {-0.5, +0.5}

                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                self.assertLess((w - w_hat).abs().max().item(), 1e-5)

        # Asymmetric binary weights {0.1, 0.9} should round-trip perfectly
        # (affine formula gives scale=0.8, bias=0.1)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="pack_asymmetric_weights"):
                bits = (mx.random.uniform(shape=(128, 512)) > 0.5).astype(mx.float32)
                w = bits * 0.9 + (1 - bits) * 0.1  # {0.1, 0.9}

                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                self.assertLess((w - w_hat).abs().max().item(), 1e-5)

        # Verify dequantized values are exactly {bias, bias + scale}
        w = mx.random.normal(shape=(64, 256))
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="dequant_values"):
                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                for i in range(scales.shape[0]):
                    for j in range(scales.shape[1]):
                        s = scales[i, j].item()
                        b = biases[i, j].item()
                        row_start = j * gs
                        row_end = row_start + gs
                        vals = w_hat[i, row_start:row_end]
                        mx.eval(vals)
                        for v in vals.tolist():
                            self.assertTrue(
                                abs(v - b) < 1e-5 or abs(v - (b + s)) < 1e-5,
                                f"Value {v} not in {{bias={b}, bias+scale={b + s}}}",
                            )

        # 1-bit quantize/dequantize zeros — scale floors to eps, bias=0
        a = mx.zeros((256, 512))
        for gs in [32, 64, 128]:
            w_q, scales, biases = mx.quantize(a, gs, 1)
            a_hat = mx.dequantize(w_q, scales, biases, gs, 1)
            self.assertLess(a_hat.abs().max().item(), 1e-5)

        # Quantized matmul with symmetric binary weights
        key = mx.random.key(42)
        k1, k2 = mx.random.split(key)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="quantized_matmul_symmetric"):
                x = mx.random.normal(shape=(4, 256), key=k1)
                signs = (mx.random.uniform(shape=(128, 256), key=k2) > 0.5).astype(
                    mx.float32
                )
                w = signs * 0.3 - (1 - signs) * 0.3  # {-0.3, +0.3}

                w_q, scales, biases = mx.quantize(w, gs, 1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)
                y_q = mx.quantized_matmul(x, w_q, scales, biases, True, gs, 1)
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max().item(), 1e-5)

        # Quantized matmul with asymmetric binary weights
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="quantized_matmul_asymmetric"):
                x = mx.random.normal(shape=(4, 256), key=k1)
                bits = (mx.random.uniform(shape=(128, 256), key=k2) > 0.5).astype(
                    mx.float32
                )
                w = bits * 0.7 + (1 - bits) * 0.1  # {0.1, 0.7}

                w_q, scales, biases = mx.quantize(w, gs, 1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)
                y_q = mx.quantized_matmul(x, w_q, scales, biases, True, gs, 1)
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                tol = 2e-5 if platform.system() == "Windows" else 1e-5
                self.assertLess((y_q - y_hat).abs().max().item(), tol)

    def test_qqmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [256, 512, 67],  # M
            [64, 256, 512],  # N
            ["nvfp4", "mxfp8"],  # mode
        )
        for M, N, mode in tests:
            with self.subTest(shape=(M, N), mode=mode):
                x_shape = (1, N)
                w_shape = (M, N)

                # TODO: Fix qmv with global scale in CPU backend.
                has_global_scale = mode == "nvfp4" and mx.default_device() == mx.gpu

                x = mx.random.normal(shape=x_shape, key=k1)
                global_scale_x = mx.max(mx.abs(x)) if has_global_scale else None
                x_hat = mx.dequantize(
                    *mx.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=mx.float32,
                    global_scale=global_scale_x,
                )

                w = mx.random.normal(shape=w_shape, key=k2)
                global_scale_w = mx.max(mx.abs(w)) if has_global_scale else None
                w_q, scales = mx.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = mx.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=mx.float32,
                )
                y_q = mx.qqmm(
                    x,
                    w_q,
                    scales,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = x_hat @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qqmm(self):
        if mx.default_device() == mx.cpu:
            self.skipTest("Not implemented for CPU")
            return

        tol = 5e-2 if mx.cuda.is_available() else 1e-3
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [8, 32, 33, 64],  # M
            [128, 256],  # N
            [128, 256],  # K
            ["nvfp4", "mxfp8"],  # mode
        )
        for M, N, K, mode in tests:
            with self.subTest(shape=(M, N, K), mode=mode):
                x_shape = (M, K)
                w_shape = (N, K)

                x = mx.random.normal(shape=x_shape, key=k1)
                global_scale_x = mx.max(mx.abs(x)) if mode == "nvfp4" else None
                x_hat = mx.dequantize(
                    *mx.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=mx.float32,
                    global_scale=global_scale_x,
                )

                w = mx.random.normal(shape=w_shape, key=k2)
                global_scale_w = mx.max(mx.abs(w)) if mode == "nvfp4" else None
                w_q, scales = mx.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = mx.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=mx.float32,
                )
                y_q = mx.qqmm(
                    x,
                    w_q,
                    scales,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = x_hat @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), tol)

    @unittest.skipIf(
        not mx.metal.is_available(), "Global scale is only supported on Metal backend"
    )
    def test_qqmm_global_scale_matrix_kernels(self):
        mx.random.seed(0)
        dtype = mx.bfloat16
        for M, N, K in product(
            [32, 64],
            [128, 130],
            [96, 128],
        ):
            with self.subTest(shape=(M, N, K)):
                x = mx.random.normal((M, K)).astype(dtype)
                w = mx.random.normal((N, K)).astype(dtype)
                global_scale_x = mx.max(mx.abs(x)).astype(mx.float32)
                global_scale_w = mx.max(mx.abs(w)).astype(mx.float32)
                x_hat = mx.dequantize(
                    *mx.quantize(x, mode="nvfp4", global_scale=global_scale_x),
                    mode="nvfp4",
                    dtype=dtype,
                    global_scale=global_scale_x,
                )
                w_q, scales = mx.quantize(w, mode="nvfp4", global_scale=global_scale_w)
                w_hat = mx.dequantize(
                    w_q,
                    scales,
                    mode="nvfp4",
                    dtype=dtype,
                    global_scale=global_scale_w,
                )

                expected = x_hat @ mx.swapaxes(w_hat, -1, -2)
                actual = mx.qqmm(
                    x,
                    w_q,
                    scales,
                    mode="nvfp4",
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                delta = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
                relative_error = delta.max() / mx.maximum(
                    mx.abs(expected.astype(mx.float32)).max(), 1e-20
                )
                self.assertLess(relative_error, 3e-2)

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
        tests = product(
            [128, 64, 32],  # group_size
            [1, 2, 4, 8],  # bits
            [8, 32, 33, 64],  # M
            [128, 256],  # N
            [128, 256],  # K
            [True, False],  # transposed
        )
        for group_size, bits, M, N, K, transposed in tests:
            if bits == 1 and mx.cuda.is_available():
                continue
            with self.subTest(
                shape=(M, N, K),
                group_size=group_size,
                bits=bits,
                transposed=transposed,
            ):
                x = mx.random.normal(shape=(M, K), key=k1) / K**0.5
                w = (
                    mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
                    / K**0.5
                )
                x = x.astype(dtype)
                w = w.astype(dtype)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )
                y_hat = (x @ w_hat.T) if transposed else (x @ w_hat)
                self.assertEqual(y_q.shape, y_hat.shape)

                tol = 1e-3 if dtype == mx.float32 else 1.5e-3
                self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_qmm_large_dims(self):
        # Regression test for an int16 overflow in the NAX qmm kernels:
        # the per-simdgroup edge sizes were computed as
        # min(SN, short(N - (y_col + tn))), which wraps for distances
        # over 32767 and made store_safe skip a contiguous band of output
        # columns [N - 65536, N - 32768) whenever the M-tile was partial.
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
        group_size, bits = 64, 4
        K = 128
        tests = [
            (16, 32840),  # unaligned N > 2**15, M < 32: partial M-tile
            (32, 32840),  # M at the small-block dispatch boundary
            (33, 32840),  # unaligned N > 2**15, M % 32 != 0
            (33000, 64),  # M > 2**15: row distance overflows (aligned N)
        ]
        for M, N in tests:
            with self.subTest(shape=(M, N, K)):
                x = mx.random.normal(shape=(M, K), key=k1) / K**0.5
                w = mx.random.normal(shape=(N, K), key=k2) / K**0.5
                x = x.astype(dtype)
                w = w.astype(dtype)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                tol = 1e-3 if dtype == mx.float32 else 1.5e-3
                self.assertLess((y_q - y_hat).abs().max(), tol)

    @unittest.skipIf("CI" in os.environ, "too slow in CI")
    def test_qmm_non_transposed(self):
        # The non-transposed matmul (w is [K, N]) is reachable mainly from the
        # vjp of a quantized linear layer, so it gets much less coverage than
        # the transposed one. Sweep it over transformer-sized K/N and over M
        # values that leave a partial M-tile.
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        modes = ["mxfp4", "nvfp4", "mxfp8"]
        if mx.default_device() == mx.gpu:
            dtypes = [mx.float16, mx.bfloat16]
        else:
            dtypes = [mx.float32]

        def check_affine(M, K, N, group_size, bits, dtype, batch=()):
            x = mx.random.normal(shape=(*batch, M, K), key=k1, dtype=dtype) / K**0.5
            w = mx.random.normal(shape=(K, N), key=k2, dtype=dtype) / K**0.5
            w_q, scales, biases = mx.quantize(w, group_size, bits)
            w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            tol = 1e-3 if dtype == mx.float32 else 1.5e-3
            self.assertLess((y_q - y_hat).abs().max(), tol)

        def check_fp(M, K, N, mode, dtype, batch=()):
            x = mx.random.normal(shape=(*batch, M, K), key=k1, dtype=dtype) / K**0.5
            w = mx.random.normal(shape=(K, N), key=k2, dtype=dtype) / K**0.5
            w_q, scales = mx.quantize(w, mode=mode)
            w_hat = mx.dequantize(w_q, scales, mode=mode)
            y_q = mx.quantized_matmul(x, w_q, scales, None, False, mode=mode)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            tol = 1e-3 if dtype == mx.float32 else 1.5e-3
            self.assertLess((y_q - y_hat).abs().max(), tol)

        for dtype in dtypes:
            # M sweep. 33..63 is the interesting range: a whole simdgroup of the
            # threadgroup's M-tile falls past the end of the matrix.
            for M in [1, 2, 31, 32, 33, 63, 64, 65, 96, 97, 100, 127, 128, 129]:
                for group_size, bits in [(64, 4), (128, 4), (64, 8)]:
                    with self.subTest(
                        M=M, group_size=group_size, bits=bits, dtype=dtype
                    ):
                        check_affine(M, 512, 1024, group_size, bits, dtype)
                for mode in modes:
                    with self.subTest(M=M, mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype)

            # Transformer-sized K/N, aligned and unaligned M.
            for K, N in [(2048, 2048), (512, 2048), (2048, 512), (11008, 2048)]:
                for M in [100, 256]:
                    with self.subTest(shape=(M, K, N), dtype=dtype):
                        check_affine(M, K, N, 64, 4, dtype)
                for mode in modes:
                    with self.subTest(shape=(M, K, N), mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype)

            # Batched x, unaligned M.
            for batch in [(2,), (2, 3)]:
                for M in [33, 250]:
                    with self.subTest(batch=batch, M=M, dtype=dtype):
                        check_affine(M, 512, 1024, 64, 4, dtype, batch=batch)
                for mode in modes:
                    with self.subTest(batch=batch, mode=mode, dtype=dtype):
                        check_fp(M, 512, 1024, mode, dtype, batch=batch)

            # M > 2**15 with a partial M-tile, so the per-simdgroup row count is a
            # distance that does not fit in an int16. Same failure mode as the one
            # test_qmm_large_dims covers for the transposed kernel.
            with self.subTest(shape=(33000, 128, 64), dtype=dtype):
                check_affine(33000, 128, 64, 64, 4, dtype)
                check_fp(33000, 128, 64, mode, dtype)

            # K=64 is the single reduction-tile control; K > 64 spans two or more
            # tiles, which exposed the over-advanced scale pointer.
            for M in [8, 33, 65]:
                for K in [64, 128, 256]:
                    for bits in [2, 4, 8]:
                        with self.subTest(M=M, K=K, bits=bits, dtype=dtype):
                            check_affine(M, K, 128, 32, bits, dtype)
                    for mode in modes:
                        with self.subTest(M=M, K=K, mode=mode, dtype=dtype):
                            check_fp(M, K, 128, mode, dtype)

    def test_qmm_small_m_block(self):
        # The batched and fp-mode variants of the small-M block, which the
        # test_qmm_large_dims shapes cannot reach.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        K = 1024
        tests = [
            # mode, group_size, bits, M, N, batch
            ("affine", 64, 4, 14, 8256, (2,)),  # batched w
            ("mxfp4", None, None, 14, 8256, ()),
        ]
        for mode, group_size, bits, M, N, batch in tests:
            dtype = mx.float16 if mode == "affine" else mx.bfloat16
            with self.subTest(
                mode=mode, group_size=group_size, bits=bits, M=M, N=N, batch=batch
            ):
                x = (mx.random.normal(batch + (M, K), key=k1) / K**0.5).astype(dtype)
                w = (mx.random.normal(batch + (N, K), key=k2) / K**0.5).astype(dtype)
                if mode == "affine":
                    wq = mx.quantize(w, group_size=group_size, bits=bits)
                else:
                    wq = mx.quantize(w, mode=mode)
                w_hat = mx.dequantize(*wq, group_size=group_size, bits=bits, mode=mode)
                y_ref = x @ w_hat.swapaxes(-1, -2)
                y = mx.quantized_matmul(
                    x,
                    *wq,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                self.assertLess((y_ref - y).abs().max(), 1e-3)

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmm_splitk_precision(self):
        # Split-K sums the partitions in float32 and rounds the result to the
        # input type once. Rounding each partition first gives two to three
        # times the error of a single rounding at these shapes.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        K, N = 2560, 1024
        # Keep M at or above get_qmv_batch_limit on all devices. Below it the
        # qmv kernel runs, which dequantizes in float32 and not in the input type.
        for mode, dtype, M in product(
            ["affine", "mxfp4"], [mx.bfloat16, mx.float16], [25, 33, 65]
        ):
            with self.subTest(mode=mode, dtype=dtype, M=M):
                k1, k2 = mx.random.split(mx.random.key(0))
                x = mx.random.normal((M, K), key=k1).astype(dtype)
                w = (0.02 * mx.random.normal((N, K), key=k2)).astype(dtype)
                group_size, bits = (64, 4) if mode == "affine" else (None, None)
                wq = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
                w_hat = mx.dequantize(*wq, group_size=group_size, bits=bits, mode=mode)
                y_ref = x.astype(mx.float32) @ w_hat.astype(mx.float32).T
                y = mx.quantized_matmul(
                    x, *wq, transpose=True, group_size=group_size, bits=bits, mode=mode
                )
                error = (y.astype(mx.float32) - y_ref).abs().mean()
                rounding = (y_ref.astype(dtype).astype(mx.float32) - y_ref).abs().mean()
                self.assertLess(error.item(), 1.1 * rounding.item())

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmm_splitk_nax_widths(self):
        # Split-K NAX (numen-tech/gemma4-qat#315): on a NAX GPU an affine
        # 2/4/8-bit split-K qmm of group size 32/64/128 takes the tensor-unit
        # body for FP16 input; MLX_DISABLE_QMM_SPLITK_NAX=1 (read on every
        # call) takes the SIMD body. Both must match the dequantized reference,
        # with an explicit bias and with the implied one (a 0-d factor), at M
        # 13..16 (one 32-row tile, at most 16 rows: the NAX body) and 40 (a
        # 32-row SIMD tile and an 8-row NAX one). K > 4096 keeps every M here
        # at or above get_qmv_batch_limit on every GPU, and N = 1536 leaves
        # few enough tiles for split-K. On a GPU without NAX both runs take
        # the SIMD body.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        K, N = 5120, 1536
        k1, k2 = mx.random.split(mx.random.key(7))
        w = (0.02 * mx.random.normal((N, K), key=k2)).astype(mx.float16)
        x_all = mx.random.normal((40, K), key=k1).astype(mx.float16)
        key = "MLX_DISABLE_QMM_SPLITK_NAX"
        saved = os.environ.get(key)

        def run(x, wq, scales, biases, gs, bits, disabled):
            if disabled:
                os.environ[key] = "1"
            else:
                os.environ.pop(key, None)
            try:
                y = mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=gs, bits=bits
                )
                mx.eval(y)
                return y
            finally:
                if saved is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = saved

        for bits, gs in product([2, 4, 8], [32, 64, 128]):
            wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
            w_hat = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
            factor = -(2 ** (bits - 1))
            implied = mx.array(factor, mx.float16)
            implied_full = scales * implied
            for M in [13, 14, 15, 16, 40]:
                x = x_all[:M]
                y_ref = x.astype(mx.float32) @ w_hat.astype(mx.float32).T
                scale = y_ref.abs().max().item()
                for disabled in [False, True]:
                    with self.subTest(bits=bits, gs=gs, M=M, disabled=disabled):
                        y = run(x, wq, scales, biases, gs, bits, disabled)
                        err = (y.astype(mx.float32) - y_ref).abs().max().item()
                        self.assertLess(err / scale, 2e-3)
                        # The implied bias must reproduce the materialized
                        # one bit for bit on the same route.
                        self.assert_bitwise_equal(
                            run(x, wq, scales, implied, gs, bits, disabled),
                            run(x, wq, scales, implied_full, gs, bits, disabled),
                        )

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmm_tile64(self):
        # N 1312 is 32- but not 64-aligned, so the aligned weight loader
        # must not be chosen for the 64-wide tile.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        key = "MLX_QMM_TILE64"
        saved = os.environ.get(key)

        def run(x, wq, scales, biases, gs, bits, on):
            if on:
                os.environ[key] = "1"
            else:
                os.environ.pop(key, None)
            try:
                y = mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=gs, bits=bits
                )
                mx.eval(y)
                return y
            finally:
                if saved is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = saved

        k1, k2 = mx.random.split(mx.random.key(7))
        K = 256
        shapes = [
            (511, 1280, 4),
            (512, 1280, 4),
            (512, 1312, 4),
            (639, 1281, 4),
            (640, 1280, 4),
            (2048, 2048, 4),
            (2047, 1280, 8),
            (2049, 1312, 8),
        ]
        for (M, N, bits), gs, dtype in product(
            shapes, [32, 64, 128], [mx.float32, mx.float16, mx.bfloat16]
        ):
            x = (mx.random.normal((M, K), key=k1) / K**0.5).astype(dtype)
            w = (mx.random.normal((N, K), key=k2) / K**0.5).astype(dtype)
            wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
            w_hat = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
            y_ref = x.astype(mx.float32) @ w_hat.astype(mx.float32).T
            implied = mx.array(-(2 ** (bits - 1)), dtype)
            implied_full = scales * implied
            tol = 1e-3 if dtype == mx.float32 else 1.5e-2
            with self.subTest(M=M, N=N, bits=bits, gs=gs, dtype=dtype):
                ys = {}
                for on in [False, True]:
                    y = run(x, wq, scales, biases, gs, bits, on)
                    self.assertEqual(y.shape, (M, N))
                    err = (y.astype(mx.float32) - y_ref).abs().max().item()
                    self.assertLess(err, tol)
                    self.assert_bitwise_equal(
                        run(x, wq, scales, implied, gs, bits, on),
                        run(x, wq, scales, implied_full, gs, bits, on),
                    )
                    ys[on] = y
                if (bits == 4 and M < 512) or (bits == 8 and M < 2048):
                    self.assert_bitwise_equal(ys[False], ys[True])

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmv_mma_switch(self):
        # numen-tech/gemma4-qat#1701: MLX_QMV_MMA=1, or mx.metal._set_qmv_mma,
        # routes 4/8-bit affine products of 4-16 rows (3-16 below gen 15) to
        # qmv_mma on a GPU that can take it (no NAX, not Apple6, not an Ultra);
        # with neither, no product takes it.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        key = "MLX_QMV_MMA"
        saved = os.environ.get(key)

        def run(x, wq, scales, biases, gs, bits, on):
            if on:
                os.environ[key] = "1"
            else:
                os.environ.pop(key, None)
            try:
                mx.metal._kernel_trace_start()
                y = mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=gs, bits=bits
                )
                mx.eval(y)
                return y, mx.metal._kernel_trace_stop()
            finally:
                if saved is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = saved

        def took(trace):
            return any("qmv_mma" in n for n in trace)

        eligible = mx.metal._qmv_mma_device()
        arch = mx.metal.device_info()["architecture"]
        min_rows = (
            4
            if int("".join(c for c in arch.split("_g")[-1] if c.isdigit())) >= 15
            else 3
        )
        K, N = 1024, 2048
        k1, k2 = mx.random.split(mx.random.key(17))
        w = (0.02 * mx.random.normal((N, K), key=k2)).astype(mx.float16)
        x_all = mx.random.normal((16, K), key=k1).astype(mx.float16)
        for bits, gs in product([2, 4, 8], [32, 64, 128]):
            wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
            w_hat = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
            implied = mx.array(-(2 ** (bits - 1)), mx.float16)
            implied_full = scales * implied
            w_hat_ib = mx.dequantize(wq, scales, implied_full, group_size=gs, bits=bits)
            for M in range(1, 17):
                x = x_all[:M]
                with self.subTest(bits=bits, gs=gs, M=M):
                    for b, wh in ((biases, w_hat), (implied, w_hat_ib)):
                        y_ref = x.astype(mx.float32) @ wh.astype(mx.float32).T
                        scale = y_ref.abs().max().item()
                        _, t0 = run(x, wq, scales, b, gs, bits, False)
                        y1, t1 = run(x, wq, scales, b, gs, bits, True)
                        self.assertFalse(took(t0), t0)
                        err = (y1.astype(mx.float32) - y_ref).abs().max().item()
                        self.assertLess(err / scale, 2e-3)
                        self.assertEqual(
                            took(t1), eligible and bits != 2 and M >= min_rows, t1
                        )
                    # On the qmv_mma route the implied factor and the
                    # materialized bias agree bit for bit.
                    y_ib, t_ib = run(x, wq, scales, implied, gs, bits, True)
                    y_full, t_full = run(x, wq, scales, implied_full, gs, bits, True)
                    if took(t_ib) and took(t_full):
                        self.assert_bitwise_equal(y_ib, y_full)
            if bits in (4, 8):
                # The process flag takes the route as the variable does, and
                # off again leaves it to the (unset) variable.
                os.environ.pop(key, None)
                try:
                    for on in (True, False):
                        mx.metal._set_qmv_mma(on)
                        mx.metal._kernel_trace_start()
                        mx.eval(
                            mx.quantized_matmul(
                                x_all[:4],
                                wq,
                                scales,
                                biases,
                                transpose=True,
                                group_size=gs,
                                bits=bits,
                            )
                        )
                        t = mx.metal._kernel_trace_stop()
                        self.assertEqual(took(t), on and eligible, t)
                finally:
                    mx.metal._set_qmv_mma(False)
                    if saved is not None:
                        os.environ[key] = saved

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmm_splitk_fallthrough(self):
        # On a NAX GPU an FP16 split-K qmm of more than 16 rows goes to the NAX
        # qmm. Both routes must match the reference; at M 16 they are equal.
        # K > 4096 keeps every M at or above the qmv batch limit.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        K, N = 5120, 512
        k1, k2 = mx.random.split(mx.random.key(11))
        w = (0.02 * mx.random.normal((N, K), key=k2)).astype(mx.float16)
        x_all = mx.random.normal((128, K), key=k1).astype(mx.float16)
        key = "MLX_DISABLE_QMM_SPLITK_FALLTHROUGH"
        saved = os.environ.get(key)

        def run(x, wq, scales, biases, gs, bits, disabled):
            if disabled:
                os.environ[key] = "1"
            else:
                os.environ.pop(key, None)
            try:
                y = mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=gs, bits=bits
                )
                mx.eval(y)
                return y
            finally:
                if saved is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = saved

        for bits, gs in product([2, 4, 8], [32, 64, 128]):
            wq, scales, biases = mx.quantize(w, group_size=gs, bits=bits)
            w_hat = mx.dequantize(wq, scales, biases, group_size=gs, bits=bits)
            implied = mx.array(-(2 ** (bits - 1)), mx.float16)
            implied_full = scales * implied
            for M in [16, 17, 24, 32, 33, 64, 128]:
                x = x_all[:M]
                y_ref = x.astype(mx.float32) @ w_hat.astype(mx.float32).T
                scale = y_ref.abs().max().item()
                with self.subTest(bits=bits, gs=gs, M=M):
                    ys = {}
                    for disabled in [False, True]:
                        y = run(x, wq, scales, biases, gs, bits, disabled)
                        err = (y.astype(mx.float32) - y_ref).abs().max().item()
                        self.assertLess(err / scale, 2e-3)
                        self.assert_bitwise_equal(
                            run(x, wq, scales, implied, gs, bits, disabled),
                            run(x, wq, scales, implied_full, gs, bits, disabled),
                        )
                        ys[disabled] = y
                    if M <= 16:
                        self.assert_bitwise_equal(ys[False], ys[True])

    def test_qmm_vjp(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 1024
        K = 512

        x = mx.random.normal(shape=(2, M, K), key=k1)
        c = mx.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = mx.quantize(w, group_size, bits)

            def fn(x):
                return mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, vjp_out = mx.vjp(fn, primals=(x,), cotangents=(c,))

            expected_out = mx.quantized_matmul(
                c, w_q, scales, biases, not transposed, group_size, bits
            )
            self.assertTrue(mx.allclose(vjp_out[0], expected_out))

    def test_qmm_jvp(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 128
        K = 128

        x = mx.random.normal(shape=(2, M, K), key=k1)
        x_tan = mx.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = mx.quantize(w, group_size, bits)

            def fn(x):
                return mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, jvp_out = mx.jvp(fn, primals=(x,), tangents=(x_tan,))

            expected_out = mx.quantized_matmul(
                x_tan, w_q, scales, biases, transposed, group_size, bits
            )
            self.assertTrue(mx.allclose(jvp_out[0], expected_out))

    def test_qmm_shapes(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        group_size = 64
        bits = 4
        w = mx.random.normal(shape=(32, 256), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        tol = 2e-2 if mx.cuda.is_available() else 1e-3
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = mx.random.normal(shape=s, key=k1)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, True, group_size, bits)
            y_hat = x @ w_hat.T
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), tol)

        w = mx.random.normal(shape=(256, 256), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = mx.random.normal(shape=s, key=k1)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 5, 6, 8],  # bits
            [256, 512, 67],  # M
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            if group_size > N:
                continue
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                w_shape = (M, N) if B == 0 else (B, M, N)
                x = mx.random.normal(shape=x_shape, key=k1) / N**0.5
                w = mx.random.normal(shape=w_shape, key=k2) / N**0.5
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmv_affine_bias_sum_precision(self):
        # The bias contributes bias * sum(x), and a sum accumulated in the input
        # type loses the small values: the sum of [1, 1 / denom, -1, 0] cancels
        # to zero.
        for n, k, gs, dtype, sign, bits in product(
            [1, 4, 8, 12],
            [64, 96, 512],
            [32, 64],
            [mx.bfloat16, mx.float16],
            [-1, 1],
            [2, 3, 4, 5, 6, 8],
        ):
            if k % gs:
                continue
            with self.subTest(n=n, k=k, gs=gs, dtype=dtype, sign=sign, bits=bits):
                # 1 / denom is exactly half an ulp at 1.0, so 1 + 1 / denom
                # rounds back to 1 (ties-to-even) in the input type.
                denom = 2048 if dtype == mx.float16 else 256
                x = mx.tile(mx.array([1, 1 / denom, -1, 0], dtype), k // 4)
                x = (sign * x).reshape(1, k)
                # Zero scales and unit biases dequantize to all ones, so the
                # product is exactly sum(x) = sign * k / (4 * denom).
                q = mx.zeros((n, k * bits // 32), mx.uint32)
                scales = mx.zeros((n, k // gs), dtype)
                biases = mx.ones((n, k // gs), dtype)
                y = mx.quantized_matmul(x, q, scales, biases, group_size=gs, bits=bits)
                expected = mx.full((1, n), sign * k / (4 * denom), dtype)
                self.assertTrue(mx.array_equal(y, expected).item())

    def test_qmm_accumulation_precision(self):
        # x repeats [1, eps, -1, 0] and each block of four weights along K is
        # equal, so each block adds eps * w. A sum in the input type loses
        # eps * w next to w. In float32 every partial sum is exact.
        N, K = 128, 256
        for dtype, transpose, gs, bits in product(
            [mx.bfloat16, mx.float16],
            [True, False],
            [32, 64, 128],
            [2, 4, 8],
        ):
            with self.subTest(dtype=dtype, transpose=transpose, gs=gs, bits=bits):
                eps = 1 / 256 if dtype == mx.bfloat16 else 1 / 2048
                x = mx.tile(mx.array([1, eps, -1, 0], dtype), K // 4)
                x = mx.stack([x, -x])
                # All bits set give the code 2**bits - 1, and the bias moves w to v.
                if transpose:
                    rows, cols = N, K
                    v = mx.arange(rows)[:, None] + 3 * mx.arange(cols // gs)
                else:
                    rows, cols = K, N
                    v = mx.arange(rows)[:, None] // 4 + 3 * mx.arange(cols // gs)
                v = v % 8 + 1
                q = mx.full((rows, cols * bits // 32), 0xFFFFFFFF, mx.uint32)
                scales = mx.ones(v.shape, dtype)
                biases = (v - (2**bits - 1)).astype(dtype)
                w = mx.dequantize(q, scales, biases, group_size=gs, bits=bits)
                w = w.astype(mx.float32)
                y_ref = x.astype(mx.float32) @ (w.T if transpose else w)
                y = mx.quantized_matmul(
                    x, q, scales, biases, transpose=transpose, group_size=gs, bits=bits
                )
                self.assertTrue(mx.array_equal(y, y_ref.astype(dtype)).item())

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmv_fast_unaligned_n(self):
        # Output sizes that are not a multiple of 8 with an aligned input size.
        # "vector" is one input row, "batched" two weight batches, and "rows"
        # is M = 3, which reaches this kernel only on GPUs before gen 15, since
        # later ones route M >= 2 to qmv_wide.
        for n, k, bits, gs, dtype, layout in product(
            [1, 3, 5, 12, 17],
            [512, 1024],
            [2, 3, 4, 5, 6, 8],
            [64],
            [mx.float32, mx.bfloat16],
            ["vector", "batched", "rows"],
        ):
            with self.subTest(n=n, k=k, bits=bits, gs=gs, dtype=dtype, layout=layout):
                key = mx.random.key(n * 7 + k + bits)
                if layout == "batched":
                    w_shape, x_shape = (2, n, k), (2, 1, k)
                elif layout == "rows":
                    w_shape, x_shape = (n, k), (3, k)
                else:
                    w_shape, x_shape = (n, k), (1, k)
                w = mx.random.normal(w_shape, key=key) / k**0.5
                x = mx.random.normal(x_shape, key=mx.random.split(key)[0])
                x = (x / k**0.5).astype(dtype)
                q, s, b = mx.quantize(w.astype(dtype), group_size=gs, bits=bits)
                w_hat = mx.dequantize(q, s, b, group_size=gs, bits=bits)
                y_hat = x.astype(mx.float32) @ mx.swapaxes(
                    w_hat.astype(mx.float32), -1, -2
                )
                y = mx.quantized_matmul(x, q, s, b, group_size=gs, bits=bits)
                tol = 1e-3 if dtype == mx.float32 else 1.5e-3
                self.assertEqual(y.shape, y_hat.shape)
                self.assertLess((y - y_hat).abs().max().item(), tol)

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmv_fast_tail_non_aligned_k(self):
        # GPU vs CPU for K that is a multiple of the group size but not of
        # qmv_fast's block, with N = 67 not a multiple of 8.
        #
        # The gs * 17 / gs * 33 shapes are on no qmv_fast gate and cover plain
        # qmv at non-aligned K. The host gate takes 1-bit K % 512 and affine
        # 4-bit K % 256 (half a kernel block; numen-tech/gemma4-qat#192), so
        # the 1-bit K = 1536/2560 and 4-bit K = 768/1280 shapes run the
        # qmv_fast partial tail and, at N = 67, the last-row clamp (M = 3 where
        # it stays on qmv: 1-bit, and 4-bit before gen 15): the GPU cannot
        # match the CPU if a tail lane reads past K or past the last row.
        cases = [
            (bits, gs, K, N, 1)
            for bits, gs in ((1, 32), (1, 64), (4, 32), (4, 64))
            for K in (gs * 17, gs * 33)
            for N in (67, 256)
        ]
        cases += [
            (bits, gs, K, N, M)
            for bits, gss, Ks in (
                (1, (32, 64, 128), (1536, 2560)),
                (4, (32, 64), (768, 1280)),
            )
            for gs in gss
            for K in Ks
            for N in (67, 256)
            for M in (1, 3)
        ]
        key = mx.random.key(0)
        for bits, gs, K, N, M in cases:
            with self.subTest(bits=bits, gs=gs, K=K, N=N, M=M):
                k1, k2, key = mx.random.split(key, 3)
                w = mx.random.normal((N, K), key=k1)
                x = mx.random.normal((M, K), key=k2).astype(mx.float16)
                wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
                y = mx.quantized_matmul(x, wq, s, b, True, gs, bits, stream=mx.gpu)
                y_cpu = mx.quantized_matmul(x, wq, s, b, True, gs, bits, stream=mx.cpu)
                self.assertTrue(mx.allclose(y, y_cpu, atol=1e-2, rtol=1e-2))

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_qmv_quad_ragged_n(self):
        # qmv_quad (K = 128, or K = 64 at bits >= 2; power-of-two bits) runs
        # ceil(N / 64) tiles with no N % 64 restriction. On a ragged last tile,
        # quad rows past N must read no scale or bias (gemma4-qat#302). N = 65
        # leaves one valid row in tile 2, N = 72 leaves eight. Explicit and
        # implied (0-d factor) biases are both covered; affine_sym never reaches
        # qmv_quad. Group size 128 runs at K = 128 only.
        key = mx.random.key(7)
        cases = [
            (bits, gs, K)
            for bits in (1, 2, 4, 8)
            for gs in (32, 64, 128)
            for K in (64, 128)
            if gs <= K
        ]
        for bits, gs, K in cases:
            if K == 64 and bits == 1:
                continue  # 1-bit qmv_quad needs K == 128
            for N in (65, 72):
                for implied in (False, True):
                    with self.subTest(bits=bits, gs=gs, K=K, N=N, implied=implied):
                        k1, k2, key = mx.random.split(key, 3)
                        w = mx.random.normal((N, K), key=k1).astype(mx.float16)
                        x = mx.random.normal((1, K), key=k2).astype(mx.float16)
                        wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
                        if implied:
                            f = -(2.0 ** (bits - 1)) if bits > 1 else -0.5
                            b = mx.array(f, mx.float16)
                            full = s * b
                        else:
                            full = b
                        y = mx.quantized_matmul(
                            x, wq, s, b, True, gs, bits, stream=mx.gpu
                        )
                        w_ref = mx.dequantize(wq, s, full, gs, bits, stream=mx.cpu)
                        y_ref = x.astype(mx.float32) @ w_ref.astype(mx.float32).T
                        self.assertEqual(y.shape, (1, N))
                        self.assertTrue(
                            mx.allclose(
                                y.astype(mx.float32), y_ref, atol=2e-2, rtol=2e-2
                            )
                        )

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_fp_qmv_fast_unaligned_n(self):
        for n, k, mode, dtype, batched in product(
            [1, 3, 5, 7, 8, 9, 12, 17],
            [256, 512, 544, 1024],
            ["mxfp4", "mxfp8", "nvfp4"],
            [mx.float32, mx.float16, mx.bfloat16],
            [False, True],
        ):
            with self.subTest(n=n, k=k, mode=mode, dtype=dtype, batched=batched):
                key = mx.random.key(n * 7 + k)
                w_shape = (2, n, k) if batched else (n, k)
                x_shape = (2, 1, k) if batched else (1, k)
                w = mx.random.normal(w_shape, key=key) / k**0.5
                x = mx.random.normal(x_shape, key=mx.random.split(key)[0])
                x = (x / k**0.5).astype(dtype)
                q, s = mx.quantize(w, mode=mode)
                w_hat = mx.dequantize(q, s, mode=mode, dtype=mx.float32)
                expected = x.astype(mx.float32) @ mx.swapaxes(w_hat, -1, -2)
                actual = mx.quantized_matmul(x, q, s, mode=mode)
                self.assertEqual(actual.shape, expected.shape)
                self.assertLess((actual - expected).abs().max().item(), 1e-3)

                # Compare aligned inputs with the existing full-tile kernel.
                alignment = 256 if mode == "mxfp8" else 512
                if k % alignment == 0:
                    padding = [(0, 0)] * q.ndim
                    padding[-2] = (0, (-n) % 8)
                    padded = mx.quantized_matmul(
                        x, mx.pad(q, padding), mx.pad(s, padding), mode=mode
                    )
                    self.assertTrue(mx.array_equal(actual, padded[..., :n]).item())

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_fp_qmv_fast_unaligned_n_global_scale(self):
        for n, k, m, dtype in product(
            [1, 3, 5, 7, 8, 9, 12, 17],
            [512, 544, 1024],
            [1, 3],
            [mx.float32, mx.float16, mx.bfloat16],
        ):
            with self.subTest(n=n, k=k, m=m, dtype=dtype):
                key = mx.random.key(n * 7 + k)
                w = mx.random.normal((n, k), key=key) / k**0.5
                x = mx.random.normal((m, k), key=mx.random.split(key)[0])
                x = (x / k**0.5).astype(dtype)
                global_scale_x = mx.max(mx.abs(x)).astype(mx.float32)
                global_scale_w = mx.max(mx.abs(w)).astype(mx.float32)
                q, s = mx.quantize(w, mode="nvfp4", global_scale=global_scale_w)
                w_hat = mx.dequantize(
                    q, s, mode="nvfp4", global_scale=global_scale_w, dtype=mx.float32
                )
                x_hat = mx.dequantize(
                    *mx.quantize(x, mode="nvfp4", global_scale=global_scale_x),
                    mode="nvfp4",
                    global_scale=global_scale_x,
                    dtype=dtype,
                )
                expected = x_hat.astype(mx.float32) @ w_hat.T
                actual = mx.qqmm(
                    x,
                    q,
                    s,
                    mode="nvfp4",
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertLess((actual - expected).abs().max().item(), 1e-3)

                if k % 512 == 0:
                    padded = mx.qqmm(
                        x,
                        mx.pad(q, [(0, (-n) % 8), (0, 0)]),
                        mx.pad(s, [(0, (-n) % 8), (0, 0)]),
                        mode="nvfp4",
                        global_scale_x=global_scale_x,
                        global_scale_w=global_scale_w,
                    )
                    self.assertTrue(mx.array_equal(actual, padded[..., :n]).item())

    def test_fp_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tol = 2e-2 if mx.cuda.is_available() else 1e-3
        tests = product(
            [256, 512, 67, 5, 7],  # M -- 5, 7 exercise out_vec_size < 8 (#3762)
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        modes = ["mxfp4", "nvfp4", "mxfp8"]
        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                    w_shape = (M, N) if B == 0 else (B, M, N)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode)
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=True,
                        mode=mode,
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test multiple of 16 but not 32
        M = 128
        N = 48
        mode = "nvfp4"
        with self.subTest(shape=(B, M, N), mode=mode):
            x_shape = (1, N)
            w_shape = (M, N)
            x = mx.random.normal(shape=x_shape, key=k1)
            w = mx.random.normal(shape=w_shape, key=k2)
            w_q, scales = mx.quantize(w, mode=mode)
            w_hat = mx.dequantize(w_q, scales, mode=mode)
            y_q = mx.quantized_matmul(
                x,
                w_q,
                scales,
                transpose=True,
                mode=mode,
            )
            y_hat = x @ mx.swapaxes(w_hat, -1, -2)
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_fp_qmv_large_output(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        K = 512
        N = 4096

        for B in [1, 2]:
            with self.subTest(B=B, N=N, K=K):
                x_shape = (1, K) if B == 1 else (B, 1, K)
                w_shape = (N, K) if B == 1 else (B, N, K)
                x = mx.random.normal(shape=x_shape, key=k1) / K**0.5
                w = mx.random.normal(shape=w_shape, key=k2)
                w_q, scales = mx.quantize(w, mode="nvfp4")

                dtypes = (
                    [mx.float16, mx.bfloat16, mx.float32]
                    if mx.default_device() == mx.gpu
                    else [mx.float32]
                )
                for dtype in dtypes:
                    with self.subTest(dtype=dtype):
                        x_t = x.astype(dtype)
                        w_hat = mx.dequantize(w_q, scales, mode="nvfp4", dtype=dtype)
                        y_q = mx.quantized_matmul(
                            x_t,
                            w_q,
                            scales,
                            transpose=True,
                            mode="nvfp4",
                        )
                        y_hat = x_t @ mx.swapaxes(w_hat, -1, -2)
                        self.assertEqual(y_q.shape, y_hat.shape)
                        tol = 1e-2 if dtype == mx.bfloat16 else 1e-3
                        self.assertTrue(mx.allclose(y_q, y_hat, rtol=tol, atol=tol))

    def test_qmv_wide(self):
        tol = 5e-2 if mx.cuda.is_available() else 1e-3
        # M in [2, vector_limit) routes to qmv_wide -- except K in {64, 128}
        # with power-of-2 bits, which stays on qmv_quad, and affine 1-bit (and
        # 2-bit at M = 2) or affine before gen 15, which stay on qmv. Check all
        # paths against a dequantize-then-matmul reference, with ragged M (token
        # tail) and ragged N (output-tile remainder). B > 1 stacks a distinct
        # weight matrix per slab and exercises the batched variant.
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        # M <= 9 < vector_limit for these shapes (K, N <= 2048), so all stay on
        # the mat-vec path; 7 and 9 also exercise the token-tail guard.
        Ms = [2, 3, 4, 5, 6, 7, 9]
        Ns = [256, 67]  # 67 is a non-multiple of the 4-row output tile
        Bs = [1, 3]

        # Affine: every bit-width and group size.
        for group_size, bits, K in product(
            [32, 64, 128], [1, 2, 3, 4, 5, 6, 8], [128, 512]
        ):
            for M, N, B in product(Ms, Ns, Bs):
                with self.subTest(M=M, N=N, K=K, B=B, group_size=group_size, bits=bits):
                    x_shape = (M, K) if B == 1 else (B, M, K)
                    w_shape = (N, K) if B == 1 else (B, N, K)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales, biases = mx.quantize(w, group_size, bits)
                    w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                    y_q = mx.quantized_matmul(
                        x, w_q, scales, biases, True, group_size, bits
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), tol)

        # FP modes (group_size and bits implied by the mode).
        for mode, K in product(["mxfp4", "nvfp4", "mxfp8"], [128, 512]):
            for M, N, B in product(Ms, Ns, Bs):
                with self.subTest(M=M, N=N, K=K, B=B, mode=mode):
                    x_shape = (M, K) if B == 1 else (B, M, K)
                    w_shape = (N, K) if B == 1 else (B, N, K)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode)
                    y_q = mx.quantized_matmul(x, w_q, scales, transpose=True, mode=mode)
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), tol)

        # Tiny shapes (M, K, N): small K and non-multiple output rows.
        tiny = [(2, 32, 10), (4, 32, 7), (3, 64, 5), (5, 64, 3)]
        settings = [
            (1, 32, "affine"),
            (4, 32, "affine"),
            (6, 32, "affine"),
            (4, 16, "nvfp4"),
        ]
        for M, K, N in tiny:
            for bits, group_size, mode in settings:
                with self.subTest(
                    M=M, K=K, N=N, bits=bits, group_size=group_size, mode=mode
                ):
                    x = mx.random.normal(shape=(M, K), key=k1)
                    w = mx.random.normal(shape=(N, K), key=k2)
                    w_q, *sb = mx.quantize(
                        w, group_size=group_size, bits=bits, mode=mode
                    )
                    w_hat = mx.dequantize(
                        w_q, *sb, group_size=group_size, bits=bits, mode=mode
                    )
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        *sb,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), tol)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide is a Metal kernel")
    def test_qmv_wide_tile_matches_split(self):
        # One call over M rows must match the split calls bit for bit.
        key = mx.random.key(6)
        k1, k2 = mx.random.split(key)
        cases = [(bits, gs, None) for bits in [2, 3, 4, 5, 6, 8] for gs in [32, 64]]
        cases += [(b, gs, f) for b, gs, f in self.IMPLIED_BIAS_CASES if b > 1]
        splits = {6: [3, 3], 7: [4, 3], 8: [4, 4], 11: [4, 4, 3], 12: [4, 4, 4]}
        for (bits, gs, factor), dtype, K, B in product(
            cases, self.IMPLIED_BIAS_DTYPES, [512, 1536], [1, 2]
        ):
            lead = (B,) if B > 1 else ()
            N = 72
            w = mx.random.normal(shape=lead + (N, K), key=k2).astype(dtype)
            w_q, scales, biases = mx.quantize(w, gs, bits)
            if factor is not None:
                biases = mx.array(factor, dtype)

            def qmm(x):
                return mx.quantized_matmul(x, w_q, scales, biases, True, gs, bits)

            for M, chunks in splits.items():
                with self.subTest(
                    bits=bits, gs=gs, factor=factor, dtype=dtype, K=K, B=B, M=M
                ):
                    x = mx.random.normal(shape=lead + (M, K), key=k1).astype(dtype)
                    parts, r = [], 0
                    for c in chunks:
                        parts.append(qmm(x[..., r : r + c, :]))
                        r += c
                    self.assert_bitwise_equal(qmm(x), mx.concatenate(parts, axis=-2))

    # Row-reuse qmv_wide (gemma4-qat#313). MLX_DISABLE_QMV_WIDE_RR=1 forces
    # affine_qmv_wide; the host reads it on every qmv_wide() call, so a test pins
    # both arms with scoped_env (which restores any prior value).
    RR_ENV = "MLX_DISABLE_QMV_WIDE_RR"

    @staticmethod
    def _gpu_gen():
        # "applegpu_g15s" -> 15
        m = re.search(r"_g(\d+)", mx.device_info()["architecture"])
        return int(m.group(1)) if m else 0

    def _rr_expected(self, bits, group_size, K, M):
        # Mirrors use_qmv_wide + use_qmv_wide_rr in mlx/backend/metal/quantized.cpp:
        # gen >= 15, affine 2/8-bit (4-bit is off), K % 256 == 0, and qmv_wide at
        # all (2-bit M < 3 stays on qmv).
        return (
            self._gpu_gen() >= 15
            and bits in (2, 8)
            and not (bits == 2 and M < 3)
            and group_size % 8 == 0
            and K % 256 == 0
        )

    def _wide_arm(self, fn):
        with mlx_tests.scoped_env(**{self.RR_ENV: "1"}):
            y = fn()
            mx.eval(y)
        return y

    def _assert_rr_route(self, bits, group_size, M, B=1, implied=False):
        # Route sentinel (from the #313 fork review): zero codes with scales =
        # biases = 1 make every weight 1, and each x row is 2^24 at k = 0, 1 at
        # k = 8 and -2^24 at k = 256, so y = 1 exactly. affine_qmv_wide adds the 1
        # to 2^24 first and loses it (y = 0); affine_qmv_wide_rr adds the two
        # 2^24 terms first (y = 1). Where rr is not expected both arms run the
        # same kernel and must match bitwise.
        K, N = 512, 8
        lead = (B,) if B > 1 else ()
        x = mx.zeros(lead + (M, K), mx.float32)
        x[..., 0] = 2.0**24
        x[..., 8] = 1.0
        x[..., 256] = -(2.0**24)
        w_q = mx.zeros(lead + (N, K * bits // 32), mx.uint32)
        s = mx.ones(lead + (N, K // group_size), mx.float32)
        b = mx.array(1.0, mx.float32) if implied else mx.ones_like(s)

        def qmm():
            return mx.quantized_matmul(
                x, w_q, s, b, True, group_size, bits, stream=mx.gpu
            )

        y_wide = self._wide_arm(qmm)
        with mlx_tests.scoped_env(**{self.RR_ENV: None}):
            y = qmm()
            mx.eval(y)
        if self._rr_expected(bits, group_size, K, M):
            self.assertTrue(mx.all(y == 1).item(), "expected affine_qmv_wide_rr")
            self.assertTrue(mx.all(y_wide == 0).item(), "expected affine_qmv_wide")
        else:
            self.assert_bitwise_equal(y, y_wide)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr(self):
        # Row-reuse qmv_wide (gemma4-qat#313): affine 2/8-bit, K % 256 == 0, M in
        # [2, vector_limit). These (M, N, B) route to affine_qmv_wide_rr on gen >= 15
        # (2-bit M = 2 stays on qmv, 4-bit on affine_qmv_wide); the reference is
        # dequantize-then-matmul, tol as test_qmv_wide. N = 67 is a ragged 8-row
        # tile, M = 7 two tiles (4 + 3), M = 9 a token tail (5 + 4), B = 3 the
        # batched variant.
        mx.set_default_device(mx.gpu)
        key = mx.random.key(313)
        k1, k2 = mx.random.split(key)
        with mlx_tests.scoped_env(**{self.RR_ENV: None}):
            for group_size, bits, K in product([32, 64, 128], [2, 4, 8], [256, 1536]):
                for M, N, B, dtype in product(
                    [2, 3, 5, 7, 8, 9],
                    [256, 67],
                    [1, 3],
                    [mx.float16, mx.bfloat16, mx.float32],
                ):
                    with self.subTest(
                        M=M,
                        N=N,
                        K=K,
                        B=B,
                        group_size=group_size,
                        bits=bits,
                        dtype=dtype,
                    ):
                        x_shape = (M, K) if B == 1 else (B, M, K)
                        w_shape = (N, K) if B == 1 else (B, N, K)
                        x = mx.random.normal(shape=x_shape, key=k1).astype(dtype)
                        w = mx.random.normal(shape=w_shape, key=k2)
                        w_q, scales, biases = mx.quantize(w, group_size, bits)
                        scales, biases = scales.astype(dtype), biases.astype(dtype)
                        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                        w_hat = w_hat.astype(mx.float32)
                        y_q = mx.quantized_matmul(
                            x,
                            w_q,
                            scales,
                            biases,
                            True,
                            group_size,
                            bits,
                            stream=mx.gpu,
                        )
                        y_hat = x.astype(mx.float32) @ mx.swapaxes(w_hat, -1, -2)
                        rel = {mx.float16: 2e-2, mx.bfloat16: 5e-2, mx.float32: 1e-3}
                        tol = rel[dtype] * mx.abs(y_hat).max().item() + 1e-3
                        self.assertEqual(y_q.shape, y_hat.shape)
                        err = (y_q.astype(mx.float32) - y_hat).abs().max().item()
                        self.assertLess(err, tol)
                        # The route depends on K only through K % 256 == 0,
                        # which the probe's K = 512 shares with these K.
                        self._assert_rr_route(bits, group_size, M, B)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr_no_worse_than_wide(self):
        # #178 no-worse rule: on the same weights, rr's error against the fp32
        # reference is at most qmv_wide's plus one ulp of the largest reference
        # value. Same process, two routes; the 4-bit and 2-bit M = 2 cases run one
        # kernel in both arms and must match bitwise.
        mx.set_default_device(mx.gpu)
        key = mx.random.key(3131)
        k1, k2 = mx.random.split(key)
        for bits, group_size, K, N, M, implied in product(
            [2, 4, 8],
            [32, 64, 128],
            [256, 1536],
            [256, 67],
            [2, 4, 7, 8],
            [False, True],
        ):
            with self.subTest(
                bits=bits, group_size=group_size, K=K, N=N, M=M, implied=implied
            ):
                x = mx.random.normal(shape=(M, K), key=k1).astype(mx.float16)
                w_q, s, b = mx.quantize(
                    mx.random.normal(shape=(N, K), key=k2), group_size, bits
                )
                s = s.astype(mx.float16)
                if implied:
                    f = mx.array(-(2.0 ** (bits - 1)), mx.float16)
                    b_ref = s * f
                    b = f
                else:
                    b = b.astype(mx.float16)
                    b_ref = b
                w_ref = mx.dequantize(
                    w_q,
                    s.astype(mx.float32),
                    b_ref.astype(mx.float32),
                    group_size,
                    bits,
                )
                ref = x.astype(mx.float32) @ w_ref.T
                ulp = mx.abs(ref).max().item() * 2.0**-10

                def qmm():
                    return mx.quantized_matmul(
                        x, w_q, s, b, True, group_size, bits, stream=mx.gpu
                    )

                y_wide = self._wide_arm(qmm)
                with mlx_tests.scoped_env(**{self.RR_ENV: None}):
                    y_rr = qmm()
                    mx.eval(y_rr)
                err_wide = (y_wide.astype(mx.float32) - ref).abs().max().item()
                err_rr = (y_rr.astype(mx.float32) - ref).abs().max().item()
                self.assertLessEqual(err_rr, err_wide + ulp)
                if not self._rr_expected(bits, group_size, K, M):
                    self.assert_bitwise_equal(y_rr, y_wide)
                self._assert_rr_route(bits, group_size, M, implied=implied)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr_implied_bias_matches_materialized(self):
        # The 0-d factor must be applied as affine_bias does (f * scale in T), so
        # rr on the implied form is bitwise equal to rr on the materialized biases,
        # including the subnormal-scale fp16 case from
        # test_implied_bias_dtype_promotion.
        mx.set_default_device(mx.gpu)
        with mlx_tests.scoped_env(**{self.RR_ENV: None}):
            for bits, group_size in product([2, 4, 8], [32, 64, 128]):
                for scale_value in (0.05, 2.0**-24):
                    with self.subTest(
                        bits=bits, group_size=group_size, scale=scale_value
                    ):
                        K, N, M = 512, 72, 5
                        x = mx.random.normal(shape=(M, K), key=mx.random.key(1))
                        x = x.astype(mx.float16)
                        w_q = mx.random.randint(
                            0, 2**bits, (N, K * bits // 32), dtype=mx.uint32
                        )
                        s = mx.full((N, K // group_size), scale_value, mx.float16)
                        f = mx.array(-(2.0 ** (bits - 1)), mx.float16)

                        def qmm(biases):
                            return mx.quantized_matmul(
                                x, w_q, s, biases, True, group_size, bits, stream=mx.gpu
                            )

                        y_implied = qmm(f)
                        y_full = qmm(s * f)
                        self.assert_bitwise_equal(y_implied, y_full)
                        self._assert_rr_route(bits, group_size, M, implied=True)

    @unittest.skipUnless(mx.metal.is_available(), "Bias-free affine is Metal-only")
    def test_qmv_affine_sym(self):
        def assert_sym_matches_biased(y_sym, y_biased):
            # The biased reference may run a different kernel, so allow a few
            # ulps of the output dtype.
            tol = 4 * mx.finfo(y_biased.dtype).eps * y_biased.abs().max()
            diff = (y_sym.astype(mx.float32) - y_biased.astype(mx.float32)).abs()
            self.assertLessEqual(diff.max().item(), tol.item())

        # Bias-free affine (biases=None) must match the biased kernels fed the
        # derived biases and a dequantized reference, on both qmv variants.
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        Ms = [1, 2, 3, 8]
        Ns = [256, 67]  # 67 is a non-multiple of the 8-row output tile
        for bits, group_size, K in product(
            [1, 2], [32, 64, 128], [64, 128, 512, 1024, 2048]
        ):
            if K < group_size:
                continue
            for M, N in product(Ms, Ns):
                with self.subTest(M=M, N=N, K=K, group_size=group_size, bits=bits):
                    x = mx.random.normal(shape=(M, K), key=k1)
                    w = mx.random.normal(shape=(N, K), key=k2)
                    w_q, scales, _ = mx.quantize(w, group_size, bits)
                    sym_biases = -0.5 * scales if bits == 1 else -scales
                    w_hat = mx.dequantize(w_q, scales, sym_biases, group_size, bits)
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    y_biased = mx.quantized_matmul(
                        x, w_q, scales, sym_biases, True, group_size, bits
                    )
                    y_sym = mx.quantized_matmul(
                        x, w_q, scales, None, True, group_size, bits
                    )
                    mx.eval(y_sym)
                    self.assertEqual(y_sym.shape, y_hat.shape)
                    self.assertEqual(y_sym.dtype, x.dtype)
                    assert_sym_matches_biased(y_sym, y_biased)
                    self.assertLess((y_sym - y_hat).abs().max(), 1e-3)

        # Half-precision activations: the kernels are templated on the
        # activation type and derive the bias in fp32 from the fp16/bf16 scale.
        for dtype, bits, K in product([mx.float16, mx.bfloat16], [1, 2], [512, 2048]):
            for M in [1, 3]:
                with self.subTest(dtype=dtype, M=M, K=K, bits=bits):
                    x = mx.random.normal(shape=(M, K), key=k1).astype(dtype)
                    w = mx.random.normal(shape=(256, K), key=k2).astype(dtype)
                    w_q, scales, _ = mx.quantize(w, 64, bits)
                    sym_biases = -0.5 * scales if bits == 1 else -scales
                    y_biased = mx.quantized_matmul(
                        x, w_q, scales, sym_biases, True, 64, bits
                    )
                    y_sym = mx.quantized_matmul(x, w_q, scales, None, True, 64, bits)
                    self.assertEqual(y_sym.dtype, dtype)
                    assert_sym_matches_biased(y_sym, y_biased)

    @unittest.skipUnless(mx.metal.is_available(), "Bias-free affine is Metal-only")
    def test_qmv_affine_sym_throws(self):
        x = mx.random.normal(shape=(1, 512))
        w = mx.random.normal(shape=(64, 512))

        # Bias-free affine is defined for 1- and 2-bit only.
        for bits in [3, 4, 8]:
            w_q, scales, _ = mx.quantize(w, 64, bits)
            with self.assertRaises(ValueError):
                mx.quantized_matmul(x, w_q, scales, None, True, 64, bits)
        w_q, scales, _ = mx.quantize(w, 64, 2)
        for bits in [0, -1]:
            with self.assertRaises(ValueError):
                mx.quantized_matmul(x, w_q, scales, None, True, 64, bits)

        w_q, scales, _ = mx.quantize(w, 64, 2)
        x_qvm = mx.random.normal(shape=(1, 64))
        # M = 33 is the smallest M above the qmv batch limit on every GPU.
        x_qmm_edge = mx.random.normal(shape=(33, 512))
        x_qmm = mx.random.normal(shape=(512, 512))
        # Evaluate the inputs up front: the checks below throw inside eval, and
        # an input left inside a failed eval would block the CPU stream.
        mx.eval(x, w_q, scales, x_qvm, x_qmm_edge, x_qmm)

        # Non-transposed (qvm) path has no bias-free kernel.
        with self.assertRaises(RuntimeError):
            mx.eval(mx.quantized_matmul(x_qvm, w_q, scales, None, False, 64, 2))
        # Matrix regime (M >= the device's qmv batch limit) is qmm-only: the
        # bias-free path says so instead of silently taking the qmm route.
        for x_big in [x_qmm_edge, x_qmm]:
            with self.subTest(M=x_big.shape[0]):
                with self.assertRaisesRegex(RuntimeError, "Bias-free affine"):
                    mx.eval(mx.quantized_matmul(x_big, w_q, scales, None, True, 64, 2))
        # CPU has no bias-free kernel either.
        with self.assertRaises(RuntimeError):
            mx.eval(
                mx.quantized_matmul(x, w_q, scales, None, True, 64, 2, stream=mx.cpu)
            )
        # Sanity: the very same call on the GPU mat-vec path works.
        mx.eval(mx.quantized_matmul(x, w_q, scales, None, True, 64, 2))

    @unittest.skipUnless(mx.metal.is_available(), "Bias-free affine is Metal-only")
    def test_qmv_affine_sym_grad(self):
        # The reverse products use the biased kernels with the derived bias, so
        # the gradients match the explicitly biased call.
        key = mx.random.key(1)
        k1, k2, k3 = mx.random.split(key, 3)
        for bits, K in product([1, 2], [512, 1024]):
            with self.subTest(bits=bits, K=K):
                x = mx.random.normal(shape=(2, K), key=k1)
                w = mx.random.normal(shape=(64, K), key=k2)
                w_q, scales, _ = mx.quantize(w, 64, bits)
                cotan = mx.random.normal(shape=(2, 64), key=k3)

                def f_sym(x, scales):
                    return mx.quantized_matmul(x, w_q, scales, None, True, 64, bits)

                def f_biased(x, scales):
                    biases = -0.5 * scales if bits == 1 else -scales
                    return mx.quantized_matmul(x, w_q, scales, biases, True, 64, bits)

                _, vjp_sym = mx.vjp(f_sym, [x, scales], [cotan])
                _, vjp_biased = mx.vjp(f_biased, [x, scales], [cotan])
                for a, b in zip(vjp_sym, vjp_biased):
                    self.assertEqual(a.shape, b.shape)
                    self.assertLess((a - b).abs().max().item(), 1e-3)

                _, jvp_sym = mx.jvp(lambda x: f_sym(x, scales), [x], [x])
                _, jvp_biased = mx.jvp(lambda x: f_biased(x, scales), [x], [x])
                self.assertLess((jvp_sym[0] - jvp_biased[0]).abs().max().item(), 1e-3)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_fast is a Metal kernel")
    def test_qmv_fast_half_block(self):
        # Affine 4-bit K % 256 == 0 and 1-bit K % 512 == 0 take qmv_fast with a
        # partial last block. The generic qmv baseline runs the same product
        # with K padded by one group that x zeroes out: K + gs is on no
        # qmv_fast gate. (An extra output row no longer leaves qmv_fast: since
        # ml-explore/mlx#4516 N % 8 != 0 takes its partial_rows variant.) At
        # B = 0 the fast call's x is a view whose memory past K is that group
        # drawn non-zero, so a tail lane that read past K would show. At B = 2
        # ensure_row_contiguous_matrix copies the slice: past K, batch 0 reads
        # batch 1's x (still non-zero) and batch 1 reads past the buffer.
        key = mx.random.key(5)
        k1, k2 = mx.random.split(key)
        f32 = mx.float32
        cases = [(4, 256), (4, 768), (4, 3840), (1, 512), (1, 1536), (1, 2560)]
        dtypes = [f32, mx.float16, mx.bfloat16]
        for (bits, K), gs, dtype in product(cases, [32, 64, 128], dtypes):
            for sym, B in product([False, True] if bits == 1 else [False], [0, 2]):
                with self.subTest(bits=bits, K=K, gs=gs, dtype=dtype, sym=sym, B=B):
                    N = 64
                    Kp = K + gs
                    lead = (B,) if B else ()
                    w = mx.random.normal(lead + (N, Kp), key=k2).astype(dtype)
                    w_q, scales, biases = mx.quantize(w, gs, bits)
                    ref_b = -0.5 * scales if sym else biases
                    if sym:
                        biases = None
                    x = mx.random.normal(lead + (1, Kp), key=k1).astype(dtype)
                    pad = mx.zeros(lead + (1, gs), dtype)
                    x0 = mx.concatenate([x[..., :K], pad], axis=-1)
                    w_hat = mx.dequantize(
                        w_q, scales.astype(f32), ref_b.astype(f32), gs, bits
                    )
                    ref = x0.astype(f32) @ mx.swapaxes(w_hat, -1, -2)

                    def qmm(x, k):
                        b = None if biases is None else biases[..., : k // gs]
                        y = mx.quantized_matmul(
                            x[..., :k],
                            w_q[..., : k * bits // 32],
                            scales[..., : k // gs],
                            b,
                            True,
                            gs,
                            bits,
                        )
                        return y.astype(f32)

                    fast, generic = qmm(x, K), qmm(x0, Kp)
                    err_fast = (fast - ref).abs().max().item()
                    err_generic = (generic - ref).abs().max().item()
                    # One ulp of the output dtype covers the rounding of the
                    # different accumulation order in fp16 / bf16; in fp32 the
                    # order itself shows (up to ~1.1 ulp here), so allow 4.
                    n_ulp = 4 if dtype == f32 else 1
                    ulp = n_ulp * mx.finfo(dtype).eps * ref.abs().max().item()
                    self.assertLessEqual(err_fast, err_generic + ulp)

        # gather_qmv shares the kernel.
        for dtype in dtypes:
            with self.subTest(gather=True, dtype=dtype):
                w = mx.random.normal((3, 64, 3840), key=k2).astype(dtype)
                w_q, scales, biases = mx.quantize(w, 32, 4)
                x = mx.random.normal((2, 1, 3840), key=k1).astype(dtype)
                idx = mx.array([2, 0])
                y = mx.gather_qmm(
                    x,
                    w_q,
                    scales,
                    biases,
                    rhs_indices=idx,
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
                w_hat = mx.dequantize(
                    w_q, scales.astype(f32), biases.astype(f32), 32, 4
                )
                ref = x.astype(f32) @ mx.swapaxes(w_hat[idx], -1, -2)
                tol = 2 * mx.finfo(dtype).eps * ref.abs().max().item() + 1e-4
                self.assertLessEqual((y.astype(f32) - ref).abs().max().item(), tol)

    # Implied bias: a 0-d `biases` array is a factor f and the kernels form
    # bias = f * scale per group. The factors of the shipped checkpoints are
    # powers of two, so f * scale is exact in every dtype and the implied path
    # must reproduce the rebuilt-bias path bit for bit.
    IMPLIED_BIAS_CASES = [
        (2, 64, -2.0),  # QAT 2-bit
        (4, 64, -8.0),  # QAT 4-bit
        (8, 64, -128.0),  # QAT 8-bit
        (1, 128, -0.5),  # Bonsai 1-bit
        (2, 128, -1.0),  # Ternary Bonsai 2 2-bit
    ]
    IMPLIED_BIAS_DTYPES = [mx.float32, mx.float16, mx.bfloat16]

    @staticmethod
    def implied_bias_devices():
        return [mx.cpu, mx.gpu] if mx.metal.is_available() else [mx.cpu]

    @staticmethod
    def reference_bias(scales, factor):
        # The implied-bias contract: a 0-d factor f stands for the materialized
        # bias `scales * T(f)` in the scales' storage dtype T (f cast to T first).
        return scales * mx.array(factor).astype(scales.dtype)

    def assert_bitwise_equal(self, a, b):
        # Unlike assertEqualArray(atol=0), equal inf/nan compare equal.
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        mx.eval(a, b)
        self.assertTrue(
            mx.array_equal(a, b, equal_nan=True).item(),
            f"max |diff| = {(a.astype(mx.float32) - b.astype(mx.float32)).abs().max().item()}",
        )

    def assert_implied_bias_matches(self, op, pre, full, factor, *post, **kwargs):
        # op(*pre, biases, *post): the 0-d factor must reproduce `full` bitwise.
        self.assert_bitwise_equal(
            op(*pre, factor, *post, **kwargs), op(*pre, full, *post, **kwargs)
        )

    def test_implied_bias_matmul(self):
        key = mx.random.key(3)
        k1, k2 = mx.random.split(key)
        for dev in self.implied_bias_devices():
            # M = 1..8 (qmv / qmv_quad / qmv_wide), 33 (above every GPU's qmv
            # batch limit: qmm / qmm_splitk) and a prefill-sized 512.
            Ms = [1, 2, 3, 4, 5, 6, 7, 8, 33, 512]
            for dtype, (bits, gs, factor) in product(
                self.IMPLIED_BIAS_DTYPES, self.IMPLIED_BIAS_CASES
            ):
                # K = 64 / 128 route to qmv_quad; 512 / 1536 to qmv_fast (and,
                # on gen 15+, affine qmv_wide for 2 <= M <= 8 at bits > 1, from
                # M = 3 at 2-bit);
                # K = 768 runs the 4-bit qmv_fast tail; N = 72 is not a
                # multiple of the 32-wide qmm tile.
                for K, N in product([64, 128, 512, 768, 1536], [256, 72]):
                    if K < gs:
                        continue
                    w = mx.random.normal(shape=(N, K), key=k2).astype(dtype)
                    w_q, scales, _ = mx.quantize(w, gs, bits, stream=dev)
                    full = self.reference_bias(scales, factor)
                    f = mx.array(factor, dtype)
                    for M in Ms:
                        with self.subTest(
                            dev=dev, dtype=dtype, bits=bits, gs=gs, K=K, N=N, M=M
                        ):
                            x = mx.random.normal(shape=(M, K), key=k1).astype(dtype)
                            self.assert_implied_bias_matches(
                                mx.quantized_matmul,
                                (x, w_q, scales),
                                full,
                                f,
                                True,
                                gs,
                                bits,
                                stream=dev,
                            )
                    # Non-transposed (qvm): inner dim N < 1024.
                    for M in [1, 3]:
                        with self.subTest(
                            dev=dev,
                            dtype=dtype,
                            bits=bits,
                            gs=gs,
                            K=K,
                            N=N,
                            M=M,
                            qvm=True,
                        ):
                            xt = mx.random.normal(shape=(M, N), key=k1).astype(dtype)
                            self.assert_implied_bias_matches(
                                mx.quantized_matmul,
                                (xt, w_q, scales),
                                full,
                                f,
                                False,
                                gs,
                                bits,
                                stream=dev,
                            )

    @unittest.skipUnless(mx.metal.is_available(), "GPU dispatch paths")
    def test_implied_bias_matmul_gpu_paths(self):
        # The remaining GPU dispatch routes: qvm_split_k (non-transposed with a
        # large inner dim), batched qmv / qmm (3-d x; the 0-d factor is never
        # broadcast and its zero batch strides keep it in place), the
        # non-aligned qmv (N % 8 != 0) and a factor given in another dtype.
        key = mx.random.key(4)
        k1, k2 = mx.random.split(key)
        for dtype, (bits, gs, factor) in product(
            self.IMPLIED_BIAS_DTYPES, self.IMPLIED_BIAS_CASES
        ):
            f = mx.array(factor, dtype)

            def check(x, w_q, scales, full, transpose=True, factor=f, **sub):
                with self.subTest(dtype=dtype, bits=bits, gs=gs, **sub):
                    self.assert_implied_bias_matches(
                        mx.quantized_matmul,
                        (x, w_q, scales),
                        full,
                        factor,
                        transpose,
                        gs,
                        bits,
                    )

            # qvm_split_k: w is (1536, 256), x @ w_hat with inner dim 1536.
            w = mx.random.normal(shape=(1536, 256), key=k2).astype(dtype)
            w_q, scales, _ = mx.quantize(w, gs, bits)
            full = self.reference_bias(scales, factor)
            for M in [1, 3]:
                xt = mx.random.normal(shape=(M, 1536), key=k1).astype(dtype)
                check(xt, w_q, scales, full, False, M=M, path="qvm_split_k")

            # Batched: x is (B, M, K) over a 2-d w (broadcast to B); qmv for
            # M = 1, qmv_quad for K = 128, qmm for M = 64.
            for K in [128, 512]:
                if K < gs:
                    continue
                w = mx.random.normal(shape=(256, K), key=k2).astype(dtype)
                w_q, scales, _ = mx.quantize(w, gs, bits)
                full = self.reference_bias(scales, factor)
                for M in [1, 3, 64]:
                    x = mx.random.normal(shape=(3, M, K), key=k1).astype(dtype)
                    check(x, w_q, scales, full, K=K, M=M, path="batched")
                # Batched w as well (B, N, K) with per-batch scales.
                wb = mx.random.normal(shape=(2, 256, K), key=k2).astype(dtype)
                wb_q, scales_b, _ = mx.quantize(wb, gs, bits)
                full_b = self.reference_bias(scales_b, factor)
                for M in [1, 64]:
                    x = mx.random.normal(shape=(2, M, K), key=k1).astype(dtype)
                    check(x, wb_q, scales_b, full_b, K=K, M=M, path="batched_w")

            # N % 8 != 0: qmv_fast with partial rows (K = 512 is fast-aligned
            # for every width here); qmv_wide for M = 4 where the device routes
            # affine there.
            w = mx.random.normal(shape=(67, 512), key=k2).astype(dtype)
            w_q, scales, _ = mx.quantize(w, gs, bits)
            full = self.reference_bias(scales, factor)
            for M in [1, 4]:
                x = mx.random.normal(shape=(M, 512), key=k1).astype(dtype)
                check(x, w_q, scales, full, M=M, path="n_not_mult_8")

            # A float32 factor with half-precision scales adopts the scales'
            # dtype (no promotion of the whole matmul to float32).
            if dtype != mx.float32:
                x = mx.random.normal(shape=(2, 512), key=k1).astype(dtype)
                check(
                    x, w_q, scales, full, factor=mx.array(factor), path="factor_dtype"
                )

    def test_implied_bias_subnormal_scales(self):
        # Subnormal fp16 scales (Bonsai-27B-1bit / Ternary Bonsai 2 carry a few):
        # f * scale with f = -0.5 and an odd subnormal mantissa is not
        # representable in fp16, so the kernels must round the product through
        # the scales dtype to stay bit-identical to a materialized bias.
        for bits, gs, factor in [(1, 128, -0.5), (2, 128, -1.0), (2, 64, -2.0)]:
            K, N = 512, 72
            w = mx.random.randint(0, 2**31 - 1, (N, K * bits // 32)).astype(mx.uint32)
            rng = np.random.default_rng(0)
            s = rng.uniform(0.01, 1.0, (N, K // gs)).astype(np.float16)
            idx = rng.choice(s.size, 200, replace=False)
            s.flat[idx] = (rng.integers(1, 1024, 200) * 2.0**-24).astype(np.float16)
            scales = mx.array(s)
            f = mx.array(factor, dtype=mx.float16)
            rebuilt = scales * f
            for dev in self.implied_bias_devices():
                with self.subTest(bits=bits, factor=factor, dev=dev):
                    self.assert_implied_bias_matches(
                        mx.dequantize, (w, scales), rebuilt, f, gs, bits, stream=dev
                    )
                    for M in [1, 2, 3, 8, 64]:
                        x = mx.random.normal((M, K)).astype(mx.float16)
                        self.assert_implied_bias_matches(
                            mx.quantized_matmul,
                            (x, w, scales),
                            rebuilt,
                            f,
                            True,
                            gs,
                            bits,
                            stream=dev,
                        )

    def test_implied_bias_dequantize(self):
        key = mx.random.key(5)
        for dev, dtype, (bits, gs, factor) in product(
            self.implied_bias_devices(),
            self.IMPLIED_BIAS_DTYPES,
            self.IMPLIED_BIAS_CASES,
        ):
            with self.subTest(dev=dev, dtype=dtype, bits=bits, gs=gs):
                w = mx.random.normal(shape=(64, 512), key=key).astype(dtype)
                w_q, scales, _ = mx.quantize(w, gs, bits, stream=dev)
                full = self.reference_bias(scales, factor)
                f = mx.array(factor, dtype)
                w_full = mx.dequantize(w_q, scales, full, gs, bits, stream=dev)
                w_ib = mx.dequantize(w_q, scales, f, gs, bits, stream=dev)
                self.assert_bitwise_equal(w_ib, w_full)
                # A float32 factor is cast to the scales' dtype.
                w_ib32 = mx.dequantize(
                    w_q, scales, mx.array(factor), gs, bits, stream=dev
                )
                self.assert_bitwise_equal(w_ib32, w_full)
                # Embedding-style gather: dequantize gathered rows with the
                # scalar factor (no gather of a bias table).
                idx = mx.array([3, 0, 63, 3])
                e_full = mx.dequantize(
                    w_q[idx], scales[idx], full[idx], gs, bits, stream=dev
                )
                e_ib = mx.dequantize(w_q[idx], scales[idx], f, gs, bits, stream=dev)
                self.assert_bitwise_equal(e_ib, e_full)

    def gather_implied_weights(self, rng, dtype, bits, gs, factor, E, N, K):
        # Random codes and scales: what a lattice export holds, with the
        # explicit bias `scales * T(f)` and the 0-d factor that stands for it.
        w = mx.array(
            rng.integers(0, 2**32, (E, N, K * bits // 32), dtype=np.uint64).astype(
                np.uint32
            )
        )
        scales = mx.array(
            rng.uniform(0.01, 1.0, (E, N, K // gs)).astype(np.float32)
        ).astype(dtype)
        return w, scales, self.reference_bias(scales, factor), mx.array(factor, dtype)

    def assert_same_bits(self, a, b):
        # assert_bitwise_equal, and the same bit patterns: the sign of a zero
        # counts too.
        self.assert_bitwise_equal(a, b)
        bits_t = mx.uint32 if a.dtype == mx.float32 else mx.uint16
        self.assertTrue(mx.array_equal(a.view(bits_t), b.view(bits_t)).item())

    @staticmethod
    def traced(dev, fn):
        # fn()'s result, evaluated, and the names of the kernels that ran it
        # (on the GPU; none elsewhere).
        tracing = dev == mx.gpu
        if tracing:
            mx.metal._kernel_trace_start()
        try:
            y = fn()
            mx.eval(y)
        finally:
            names = mx.metal._kernel_trace_stop() if tracing else []
        return y, names

    def traced_gather_qmm(self, dev, *args, **kwargs):
        # The result and the names of the gather kernels that served it (GPU).
        y, names = self.traced(dev, lambda: mx.gather_qmm(*args, stream=dev, **kwargs))
        return y, [n for n in names if "_gather_q" in n]

    @staticmethod
    def is_ib_kernel(name):
        # "_ib" ends the name, or comes before the sorted kernels' alignment
        # constants.
        return name.endswith("_ib") or "_ib_align_" in name

    def assert_gather_implied(
        self, dev, x, w, scales, full, f, route, implied_kernel, **kwargs
    ):
        # The factor must reproduce the explicit bias bit for bit. On the GPU
        # the trace proves which kernel did it: `route` names the kernel
        # family, and its name ends "_ib" only where the op keeps the factor.
        y_ib, k_ib = self.traced_gather_qmm(dev, x, w, scales, f, **kwargs)
        y_full, k_full = self.traced_gather_qmm(dev, x, w, scales, full, **kwargs)
        self.assert_same_bits(y_ib, y_full)
        if dev != mx.gpu:
            return
        self.assertEqual(len(k_ib), 1, k_ib)
        self.assertEqual(len(k_full), 1, k_full)
        self.assertIn(route, k_ib[0])
        if route == "gather_qmv_":
            self.assertNotIn("gather_qmv_fast", k_ib[0])
        self.assertFalse(self.is_ib_kernel(k_full[0]), k_full)
        self.assertEqual(self.is_ib_kernel(k_ib[0]), implied_kernel, k_ib)
        self.assertEqual(k_ib[0].replace("_ib", ""), k_full[0])

    @staticmethod
    def gather_sorted(x, indices):
        # mlx-lm's _gather_sort: one row per (token, expert), experts in order.
        m = indices.shape[-1]
        indices = indices.flatten()
        order = mx.argsort(indices)
        return x.flatten(0, -3)[order // m], indices[order]

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_implied_bias_gather_routes(self):
        # numen-tech/gemma4-qat#1833: with one row per matrix, transposed, on
        # the GPU and with unpromoted scales (what mlx-lm's SwitchGLU presents
        # for MoE experts) the op keeps the 0-d factor and the gather_qmv,
        # gather_qmv_fast and sorted gather_qmm_rhs kernels form the bias.
        if mx.default_device() != mx.gpu:
            self.skipTest("requires the GPU")
        rng = np.random.default_rng(1833)
        dev = mx.gpu

        def tokens(L, K, dtype):
            return mx.array(
                rng.standard_normal((L, 1, 1, K)).astype(np.float32) / K**0.5
            ).astype(dtype)

        def check(wts, L, sort, route, top_k=8, gs=32, bits=4):
            # wts: (w, scales, explicit bias, factor), as gather_implied_weights
            # returns them. L tokens of top_k random experts each, in the
            # scales' dtype; K and the expert count are read off w.
            w, scales = wts[:2]
            K, E = w.shape[-1] * 32 // bits, w.shape[0]
            x = tokens(L, K, scales.dtype)
            idx = mx.array(rng.integers(0, E, (L, top_k)).astype(np.uint32))
            if sort:
                x, idx = self.gather_sorted(x, idx)
            self.assert_gather_implied(
                dev,
                x,
                *wts,
                route,
                True,
                rhs_indices=idx,
                transpose=True,
                group_size=gs,
                bits=bits,
                sorted_indices=sort,
            )

        # The 26B expert shapes at 4-bit, group size 32, fp16: gate/up
        # (K % 256 == 0: gather_qmv_fast) and down (gather_qmv). Rows are
        # tokens x top-k. Unsorted, or sorted below B >= 16 and B / E >= 4,
        # takes gather_qmv; sorted above it gather_qmm_rhs (with a ragged
        # last row tile at 1064 rows).
        for K, N, qmv in [
            (2816, 704, "gather_qmv_fast"),
            (704, 2816, "gather_qmv_"),
            (2816, 708, "gather_qmv_"),  # N % 8 != 0, and a ragged rhs column tile
        ]:
            wts = self.gather_implied_weights(rng, mx.float16, 4, 32, -8.0, 8, N, K)
            for L, sort, route in [
                (1, False, qmv),  # decode
                (3, False, qmv),  # verify widths
                (5, False, qmv),
                (1, True, qmv),  # sorted, B < 16
                (3, True, qmv),  # sorted, B / E < 4
                (8, True, "gather_qmm_rhs"),  # prefill widths
                (64, True, "gather_qmm_rhs"),
                (133, True, "gather_qmm_rhs"),
            ]:
                with self.subTest(K=K, N=N, L=L, sort=sort):
                    check(wts, L, sort, route)

        # 128 experts, top-8: sorted rows take gather_qmm_rhs from 64 tokens.
        wts = self.gather_implied_weights(rng, mx.float16, 4, 32, -8.0, 128, 64, 512)
        for L, route in [
            (8, "gather_qmv_fast"),
            (63, "gather_qmv_fast"),
            (64, "gather_qmm_rhs"),
            (67, "gather_qmm_rhs"),
        ]:
            with self.subTest(E=128, L=L):
                check(wts, L, True, route)

        # Experts sliced out of a larger stack (non-zero buffer offsets), and
        # lhs_indices picking the rows.
        w, scales, full, f = self.gather_implied_weights(
            rng, mx.float16, 4, 32, -8.0, 9, 64, 512
        )
        for sort, L, route in [(False, 3, "gather_qmv_fast"), (True, 16, "rhs")]:
            with self.subTest(sliced=True, sort=sort):
                check((w[1:], scales[1:], full[1:], f), L, sort, route)
        with self.subTest(lhs=True):
            self.assert_gather_implied(
                dev,
                tokens(3, 512, mx.float16).squeeze(1),
                w,
                scales,
                full,
                f,
                "gather_qmv_fast",
                True,
                lhs_indices=mx.array([2, 0, 1, 1, 2], mx.uint32),
                rhs_indices=mx.array([8, 3, 3, 0, 5], mx.uint32),
                transpose=True,
                group_size=32,
                bits=4,
            )

        # Scales and weights in every layout the kernels take without a copy:
        # gather_qmv reads them through their strides, the sorted kernel
        # copies what is not row contiguous.
        for name, *wts in self.gather_layouts(rng, mx.float16, 4, 32, -8.0, 4, 64, 512):
            if wts[0].ndim == 4:
                continue  # rhs_indices over a 2-d expert grid: the replay test
            for L, sort, route in [(3, False, "gather_qmv_fast"), (16, True, "rhs")]:
                with self.subTest(layout=name, sort=sort):
                    check(wts, L, sort, route, top_k=2)

        # The other formats and dtypes with a prebuilt "_ib" kernel.
        for dtype, (bits, gs, factor) in product(
            self.IMPLIED_BIAS_DTYPES,
            self.IMPLIED_BIAS_CASES + [(4, 32, -8.0), (4, 128, -8.0)],
        ):
            wts = self.gather_implied_weights(rng, dtype, bits, gs, factor, 4, 72, 1024)
            for L, sort, route in [
                (3, False, "gather_qmv_fast"),
                (8, True, "gather_qmm_rhs"),
                (37, True, "gather_qmm_rhs"),
            ]:
                with self.subTest(dtype=dtype, bits=bits, gs=gs, L=L, sort=sort):
                    check(wts, L, sort, route, top_k=2, gs=gs, bits=bits)

        # A lattice export stores a block whose largest code is +2^(bits-1)
        # steps with a negative scale (and a positive bias), and an all-zero
        # block may hold a -0.0 scale, whose bias is +0.0: no kernel may
        # assume scales > 0. Mixed signs, both zeros and whole zero rows.
        for dtype in [mx.float16, mx.bfloat16, mx.float32]:
            for K, N, qmv in [
                (2816, 704, "gather_qmv_fast"),
                (704, 2816, "gather_qmv_"),
            ]:
                w, scales, _, f = self.gather_implied_weights(
                    rng, dtype, 4, 32, -8.0, 8, N, K
                )
                s = np.array(scales.astype(mx.float32))
                s *= rng.choice([-1.0, 1.0], s.shape).astype(np.float32)
                flat = s.reshape(-1)
                zeros = rng.choice(flat.size, 400, replace=False)
                flat[zeros[:200]] = -0.0
                flat[zeros[200:]] = 0.0
                s[0, 0] = -0.0
                s[1, 1] = 0.0
                scales = mx.array(s).astype(dtype)
                full = self.reference_bias(scales, -8.0)
                neg_zero = mx.array(np.signbit(s) & (s == 0))
                # -8 * -0.0 is +0.0: the explicit bias of a -0.0 scale.
                self.assertFalse(
                    mx.any(
                        mx.array(np.signbit(np.array(full.astype(mx.float32))))
                        & neg_zero
                    ).item()
                )
                for L, sort, route in [
                    (1, False, qmv),
                    (5, False, qmv),
                    (3, True, qmv),
                    (8, True, "gather_qmm_rhs"),
                    (67, True, "gather_qmm_rhs"),
                ]:
                    with self.subTest(signed=True, dtype=dtype, K=K, L=L, sort=sort):
                        check((w, scales, full, f), L, sort, route)

        # A factor that is no power of two: T(f) and the product both round,
        # so a kernel that ignored the factor it was given, or formed the
        # product in another type, would not reproduce `scales * T(f)`.
        for dtype, factor in product(self.IMPLIED_BIAS_DTYPES, [-5.3, 0.7]):
            for K, N, qmv in [(2816, 72, "gather_qmv_fast"), (704, 72, "gather_qmv_")]:
                wts = self.gather_implied_weights(rng, dtype, 4, 32, factor, 4, N, K)
                _, scales, full, _ = wts
                self.assertFalse(
                    mx.array_equal(full, self.reference_bias(scales, -8.0)).item()
                )
                for L, sort, route in [
                    (3, False, qmv),
                    (8, True, "gather_qmm_rhs"),
                    (37, True, "gather_qmm_rhs"),
                ]:
                    with self.subTest(dtype=dtype, factor=factor, K=K, L=L):
                        check(wts, L, sort, route, top_k=2)

        # Subnormal fp16 scales: the product must round through fp16.
        w, scales, full, f = self.gather_implied_weights(
            rng, mx.float16, 2, 64, -1.0, 4, 72, 512
        )
        s = np.array(scales)
        pick = rng.choice(s.size, 200, replace=False)
        s.flat[pick] = (rng.integers(1, 1024, 200) * 2.0**-24).astype(np.float16)
        scales = mx.array(s)
        f = mx.array(-0.5, mx.float16)
        full = scales * f
        for L, sort, route in [(3, False, "gather_qmv"), (8, True, "gather_qmm_rhs")]:
            with self.subTest(subnormal=True, sort=sort):
                check((w, scales, full, f), L, sort, route, top_k=2, gs=64, bits=2)

    def test_implied_bias_gather_qmm(self):
        # Every gather route without an implied-bias kernel still gets a
        # materialized bias from the op: the matrix gather_qmm, gather_qvm,
        # the non-transposed sorted kernel, more than one row per matrix,
        # promoted scales, a bit width with no prebuilt "_ib" kernel, and the
        # CPU. The result equals the per-group-bias call bit for bit and no
        # "_ib" kernel runs.
        rng = np.random.default_rng(7)
        lhs = mx.array([0, 1, 1])
        rhs = mx.array([1, 0, 1])
        for dev, dtype, (bits, gs, factor) in product(
            self.implied_bias_devices(),
            self.IMPLIED_BIAS_DTYPES,
            self.IMPLIED_BIAS_CASES + [(4, 32, -8.0)],
        ):
            E, N, K = 2, 64, 512
            w, scales, full, f = self.gather_implied_weights(
                rng, dtype, bits, gs, factor, E, N, K
            )

            def x(*shape):
                return mx.array(
                    rng.standard_normal(shape).astype(np.float32) / shape[-1] ** 0.5
                ).astype(dtype)

            sorted_rhs = mx.array(np.sort(rng.integers(0, E, 64)).astype(np.uint32))
            cases = [
                # name, x, kwargs, kernel family
                (
                    "rows_3",
                    x(2, 3, K),
                    dict(lhs_indices=lhs, rhs_indices=rhs, transpose=True),
                    "gather_qmv",
                ),
                (
                    "matrix",
                    x(2, 64, K),
                    dict(lhs_indices=lhs, rhs_indices=rhs, transpose=True),
                    "gather_qmm_t",
                ),
                (
                    "matrix_n",
                    x(2, 64, N),
                    dict(lhs_indices=lhs, rhs_indices=rhs, transpose=False),
                    "gather_qmm_n",
                ),
                (
                    "qvm",
                    x(2, 1, N),
                    dict(lhs_indices=lhs, rhs_indices=rhs, transpose=False),
                    "gather_qvm",
                ),
                (
                    "sorted_n",
                    x(64, 1, N),
                    dict(rhs_indices=sorted_rhs, transpose=False, sorted_indices=True),
                    "gather_qmm_rhs",
                ),
            ]
            for name, xv, kwargs, route in cases:
                with self.subTest(dev=dev, dtype=dtype, bits=bits, gs=gs, case=name):
                    self.assert_gather_implied(
                        dev,
                        xv,
                        w,
                        scales,
                        full,
                        f,
                        route,
                        False,
                        group_size=gs,
                        bits=bits,
                        **kwargs,
                    )

        # One row per matrix, transposed: the factor is kept on the GPU only
        # with unpromoted scales and a prebuilt width.
        for dev in self.implied_bias_devices():
            for x_dtype, s_dtype, bits in [
                (mx.float32, mx.float16, 4),
                (mx.float32, mx.bfloat16, 4),
                (mx.float16, mx.float16, 3),
                (mx.float16, mx.float16, 6),
            ]:
                factor = -float(2 ** (bits - 1))
                w, scales, full, f = self.gather_implied_weights(
                    rng, s_dtype, bits, 32, factor, 4, 64, 512
                )
                for L, sort in [(3, False), (32, True)]:
                    xv = mx.array(
                        rng.standard_normal((L, 1, 512)).astype(np.float32) / 16
                    ).astype(x_dtype)
                    idx = mx.array(np.sort(rng.integers(0, 4, L)).astype(np.uint32))
                    with self.subTest(
                        dev=dev, x_dtype=x_dtype, s_dtype=s_dtype, bits=bits, L=L
                    ):
                        # The factor in the scales' dtype, and a float32 one,
                        # which is cast to it.
                        for fac in (f, mx.array(factor)):
                            self.assert_gather_implied(
                                dev,
                                xv,
                                w,
                                scales,
                                full,
                                fac,
                                "gather_q",
                                False,
                                rhs_indices=idx,
                                transpose=True,
                                group_size=32,
                                bits=bits,
                                sorted_indices=sort,
                            )

    def test_implied_bias_gather_grad(self):
        # The reverse products of a gather_qmm that keeps the factor equal
        # those of the explicit `scales * f` graph: the factor is a constant
        # and the scales gradient carries its f * d(bias) term.
        rng = np.random.default_rng(11)
        for dev, dtype, (bits, gs, factor) in product(
            self.implied_bias_devices(),
            [mx.float32, mx.float16],
            [(4, 32, -8.0), (2, 64, -2.0), (8, 64, -128.0)],
        ):
            if dev == mx.cpu and dtype != mx.float32:
                continue  # the CPU's GatherMM (the scales gradient) is fp32 only
            E, N, K = 4, 64, 512
            w, scales, _, f = self.gather_implied_weights(
                rng, dtype, bits, gs, factor, E, N, K
            )
            for L, sort in [(3, False), (32, True)]:
                x = mx.array(
                    rng.standard_normal((L, 1, K)).astype(np.float32) / K**0.5
                ).astype(dtype)
                idx = mx.array(np.sort(rng.integers(0, E, L)).astype(np.uint32))
                cotan = mx.array(
                    rng.standard_normal((L, 1, N)).astype(np.float32)
                ).astype(dtype)

                def run(x, scales, b):
                    return mx.gather_qmm(
                        x,
                        w,
                        scales,
                        b,
                        rhs_indices=idx,
                        transpose=True,
                        group_size=gs,
                        bits=bits,
                        sorted_indices=sort,
                        stream=dev,
                    )

                def f_ib(x, scales):
                    return run(x, scales, f)

                def f_full(x, scales):
                    return run(x, scales, scales * f)

                with self.subTest(dev=dev, dtype=dtype, bits=bits, L=L, sort=sort):
                    (y_ib,), (dx_ib, ds_ib) = mx.vjp(f_ib, [x, scales], [cotan])
                    (y_full,), (dx_full, ds_full) = mx.vjp(f_full, [x, scales], [cotan])
                    self.assert_bitwise_equal(y_ib, y_full)
                    self.assert_bitwise_equal(dx_ib, dx_full)
                    self.assert_bitwise_equal(ds_ib, ds_full)
                    _, (df,) = mx.vjp(lambda f: run(x, scales, f), [f], [cotan])
                    self.assert_bitwise_equal(df, mx.zeros_like(f))
                    # The factor is a constant for forward mode too, and for
                    # whatever produced it: sqrt'(0) must not be reached.
                    (out,), (tf,) = mx.jvp(
                        lambda f: run(x, scales, f), [f], [mx.array(1.0, dtype)]
                    )
                    self.assert_bitwise_equal(tf, mx.zeros_like(out))
                    g = mx.grad(
                        lambda t: (
                            run(x, scales, mx.sqrt(t).astype(dtype))
                            .astype(mx.float32)
                            .sum()
                        )
                    )(mx.array(0.0))
                    self.assertEqual(g.item(), 0.0)

    def gather_layouts(self, rng, dtype, bits, gs, factor, E, N, K):
        # The same E experts in every layout the backends keep without a copy
        # (each matrix contiguous): contiguous, every other expert of a larger
        # stack (strided), a 2-d expert grid with its axes swapped (permuted
        # strides), and one expert's scales broadcast over all (zero stride).
        # Each layout: name, w, scales, the explicit bias and the factor.
        w, scales, _, f = self.gather_implied_weights(
            rng, dtype, bits, gs, factor, 2 * E, N, K
        )
        mx.eval(w, scales)
        layouts = [
            ("contiguous", w[:E], scales[:E]),
            ("strided", w[::2], scales[::2]),
            (
                "permuted",
                w[:E].reshape(2, E // 2, N, -1).swapaxes(0, 1),
                scales[:E].reshape(2, E // 2, N, -1).swapaxes(0, 1),
            ),
            ("broadcast", w[:E], mx.broadcast_to(scales[:1], (E, N, K // gs))),
        ]
        out = []
        for name, wl, sl in layouts:
            # Evaluated views: the strides are what the kernels see.
            mx.eval(wl, sl)
            full = sl * f
            mx.eval(full)
            out.append((name, wl, sl, full, f))
        return out

    def test_implied_bias_gather_shapeless_replay(self):
        # A shapeless compile replays the traced primitive with other shapes:
        # the factor the op kept for one row per matrix then reaches routes
        # with no implied-bias kernel, where the backend forms the bias, for
        # scales in any layout (strided scales take the general binary
        # kernel, which reads every operand's strides). On the CPU the op
        # materializes at trace time, so the replay is the plain graph.
        rng = np.random.default_rng(13)
        E, N, K = 4, 64, 512
        for dev in self.implied_bias_devices():
            for name, w, scales, full, f in self.gather_layouts(
                rng, mx.float16, 4, 32, -8.0, E, N, K
            ):
                grid = w.ndim == 4
                lhs = mx.array([0, 1, 1], mx.uint32)
                rhs = mx.array([3, 0, 2], mx.uint32)

                def gather(x, b):
                    return mx.gather_qmm(
                        x,
                        w,
                        scales,
                        b,
                        lhs_indices=lhs,
                        rhs_indices=rhs,
                        transpose=True,
                        group_size=32,
                        bits=4,
                        stream=dev,
                    )

                compiled = mx.compile(lambda x: gather(x, f), shapeless=True)
                for M, route, implied in [
                    (1, "gather_qmv_fast", True),  # the trace
                    (3, "gather_qmv_fast", False),  # verify widths
                    (5, "gather_qmv_fast", False),
                    (64, "gather_qmm_t", False),  # a prefill width
                    (1, "gather_qmv_fast", True),
                ]:
                    x = mx.array(
                        rng.standard_normal((2, M, K)).astype(np.float32) / K**0.5
                    ).astype(mx.float16)
                    mx.eval(x)
                    y, names = self.traced(dev, lambda: compiled(x))
                    with self.subTest(dev=dev, layout=name, M=M, grid=grid):
                        self.assert_same_bits(y, gather(x, full))
                        if dev != mx.gpu:
                            continue
                        gathers = [n for n in names if "_gather_q" in n]
                        self.assertEqual(len(gathers), 1, names)
                        self.assertIn(route, gathers[0])
                        self.assertEqual(gathers[0].endswith("_ib"), implied, names)
                        # The backend's own multiply, only on the replays that
                        # left the "_ib" routes.
                        self.assertEqual(
                            any("Multiply" in n for n in names), not implied, names
                        )

    def test_implied_bias_grad(self):
        # Reverse products wrt x and scales use the implied-bias kernels and
        # the same closed form as the explicit-bias call.
        key = mx.random.key(6)
        k1, k2, k3 = mx.random.split(key, 3)
        for bits, gs, factor in self.IMPLIED_BIAS_CASES:
            with self.subTest(bits=bits, gs=gs):
                x = mx.random.normal(shape=(2, 512), key=k1)
                w = mx.random.normal(shape=(64, 512), key=k2)
                w_q, scales, _ = mx.quantize(w, gs, bits)
                f = mx.array(factor)
                cotan = mx.random.normal(shape=(2, 64), key=k3)

                def f_ib(x, scales):
                    return mx.quantized_matmul(x, w_q, scales, f, True, gs, bits)

                def f_full(x, scales):
                    return mx.quantized_matmul(
                        x, w_q, scales, factor * scales, True, gs, bits
                    )

                _, vjp_ib = mx.vjp(f_ib, [x, scales], [cotan])
                _, vjp_full = mx.vjp(f_full, [x, scales], [cotan])
                self.assert_bitwise_equal(vjp_ib[0], vjp_full[0])
                # Same closed form, different fp32 accumulation order (the
                # explicit path sums the bias term through the multiply's vjp).
                self.assertTrue(
                    mx.allclose(vjp_ib[1], vjp_full[1], rtol=1e-4, atol=1e-3)
                )

                _, jvp_ib = mx.jvp(lambda x: f_ib(x, scales), [x], [x])
                _, jvp_full = mx.jvp(lambda x: f_full(x, scales), [x], [x])
                self.assert_bitwise_equal(jvp_ib[0], jvp_full[0])

    def test_implied_bias_factor_grad(self):
        # The factor gradient is zero where the implied-bias kernels run (4-bit,
        # unpromoted scales) and where the bias is materialized (e.g. 3-bit).
        key = mx.random.key(8)
        k1, k2 = mx.random.split(key)
        lhs = mx.array([0, 1])
        rhs = mx.array([1, 0])
        configs = [
            (mx.float32, mx.float32, 4),
            (mx.float16, mx.float16, 4),
            (mx.float32, mx.float16, 4),
            (mx.float32, mx.float32, 3),
        ]
        for dev, (x_dtype, s_dtype, bits) in product(
            self.implied_bias_devices(), configs
        ):
            factor = -float(2 ** (bits - 1))
            w = mx.random.normal(shape=(2, 64, 512), key=k2).astype(s_dtype)
            w_q, scales, _ = mx.quantize(w, 64, bits, stream=dev)
            x = mx.random.normal(shape=(2, 1, 512), key=k1).astype(x_dtype)
            ops = {
                "quantized_matmul": lambda f: mx.quantized_matmul(
                    x[0], w_q[0], scales[0], f, True, 64, bits, stream=dev
                ),
                "gather_qmm": lambda f: mx.gather_qmm(
                    x, w_q, scales, f, lhs, rhs, True, 64, bits, stream=dev
                ),
                "dequantize": lambda f: mx.dequantize(
                    w_q[0], scales[0], f, 64, bits, stream=dev
                ),
            }
            for name, op in ops.items():
                with self.subTest(
                    dev=dev, x_dtype=x_dtype, s_dtype=s_dtype, bits=bits, op=name
                ):
                    f = mx.array(factor)
                    out = op(f)
                    _, (df,) = mx.vjp(op, [f], [mx.ones_like(out)])
                    self.assert_bitwise_equal(df, mx.zeros_like(f))
                    self.assertEqual(
                        mx.grad(lambda f: op(f).astype(mx.float32).sum())(f).item(),
                        0.0,
                    )
            with self.subTest(dev=dev, x_dtype=x_dtype, s_dtype=s_dtype, bits=bits):
                op = ops["quantized_matmul"]
                f = mx.array(factor)
                (out,), (tf,) = mx.jvp(op, [f], [mx.array(1.0)])
                self.assert_bitwise_equal(tf, mx.zeros_like(out))

    def test_implied_bias_factor_mixed_grad(self):
        # The scales and x gradients must not depend on the factor either, on
        # the kernel paths (4-bit) and on the materialized path (3-bit).
        key = mx.random.key(9)
        k1, k2 = mx.random.split(key)
        configs = [
            (mx.float32, mx.float32, 4),
            (mx.float16, mx.float16, 4),
            (mx.float32, mx.float32, 3),
        ]
        for dev, (x_dtype, s_dtype, bits) in product(
            self.implied_bias_devices(), configs
        ):
            factor = -float(2 ** (bits - 1))
            w = mx.random.normal(shape=(64, 512), key=k2).astype(s_dtype)
            w_q, scales, _ = mx.quantize(w, 64, bits, stream=dev)
            x = mx.random.normal(shape=(1, 512), key=k1).astype(x_dtype)

            def loss(x, scales, f):
                out = mx.quantized_matmul(x, w_q, scales, f, True, 64, bits, stream=dev)
                return out.astype(mx.float32).sum()

            def dscales_sum(f):
                return mx.grad(loss, argnums=1)(x, scales, f).astype(mx.float32).sum()

            def dx_sum(f):
                return mx.grad(loss, argnums=0)(x, scales, f).astype(mx.float32).sum()

            f = mx.array(factor)
            for name, fn in [("scales", dscales_sum), ("x", dx_sum)]:
                with self.subTest(
                    dev=dev, x_dtype=x_dtype, s_dtype=s_dtype, bits=bits, grad=name
                ):
                    self.assertEqual(mx.grad(fn)(f).item(), 0.0)

    def test_implied_bias_dtype_promotion(self):
        # fp32 activations over fp16 scales promote the kernel's scales to fp32.
        # The bias must still be the fp16 reference `scales * T(f)`: with the
        # smallest fp16 subnormal 2^-24 and f = -0.5, that is 0, whereas an
        # fp32-formed bias is -2^-25.
        tiny = 2.0**-24
        scales = mx.full((8, 1), tiny, dtype=mx.float16)
        w = mx.zeros((8, 16), dtype=mx.uint32)
        x = mx.ones((1, 128), dtype=mx.float32)
        for dev in self.implied_bias_devices():
            for f in [mx.array(-0.5, mx.float16), mx.array(-0.5)]:
                with self.subTest(dev=dev, fdtype=f.dtype):
                    y_full = mx.quantized_matmul(
                        x,
                        w,
                        scales,
                        self.reference_bias(scales, -0.5),
                        True,
                        128,
                        4,
                        stream=dev,
                    )
                    y_ib = mx.quantized_matmul(
                        x, w, scales, f, True, 128, 4, stream=dev
                    )
                    self.assertEqual(y_ib.dtype, mx.float32)
                    self.assert_bitwise_equal(y_ib, y_full)
                    self.assert_bitwise_equal(y_ib, mx.zeros((1, 8)))
        # bf16 activations over fp16 scales promote to fp32 as well.
        xb = mx.ones((1, 128), dtype=mx.bfloat16)
        for dev in self.implied_bias_devices():
            y_ib = mx.quantized_matmul(
                xb, w, scales, mx.array(-0.5, mx.float16), True, 128, 4, stream=dev
            )
            self.assertEqual(y_ib.dtype, mx.float32)
            self.assert_bitwise_equal(y_ib, mx.zeros((1, 8)))

    def test_implied_bias_cpu_dequantize_rounding(self):
        # The CPU fallback forms the bias in the scales' dtype even when the
        # factor arrives as fp32: every code is 1, so the reference is
        # 2^-24 + T(-0.5 * 2^-24) = 2^-24; an fp32 product would leave 2^-25,
        # which the final fp16 cast rounds to 0.
        tiny = 2.0**-24
        scales = mx.full((8, 1), tiny, dtype=mx.float16)
        w = mx.full((8, 16), 0x11111111, dtype=mx.uint32)
        expected = mx.full((8, 128), tiny, dtype=mx.float16)
        for dev in self.implied_bias_devices():
            with self.subTest(dev=dev):
                d_full = mx.dequantize(
                    w, scales, self.reference_bias(scales, -0.5), 128, 4, stream=dev
                )
                d_ib = mx.dequantize(w, scales, mx.array(-0.5), 128, 4, stream=dev)
                self.assert_bitwise_equal(d_ib, d_full)
                self.assert_bitwise_equal(d_ib, expected)

    def test_implied_bias_non_representable_factor(self):
        # A factor that is not exactly representable in T is outside the
        # contract; the documented result is still `scales * T(f)`: with the
        # fp16 scale 0.10009765625 and f = 1.0004, T(f) = 1 so the bias is the
        # scale itself (T(f * scale) would be the next fp16 value up).
        scale = 0.10009765625
        scales = mx.full((8, 1), scale, dtype=mx.float16)
        f = mx.array(1.0004)
        rebuilt = self.reference_bias(scales, 1.0004)
        self.assert_bitwise_equal(rebuilt, scales)
        other = (scales.astype(mx.float32) * f).astype(mx.float16)
        self.assertFalse(mx.array_equal(rebuilt, other).item())
        w = mx.zeros((8, 16), dtype=mx.uint32)
        x = mx.ones((1, 128), dtype=mx.float16)
        for dev in self.implied_bias_devices():
            with self.subTest(dev=dev):
                d_ib = mx.dequantize(w, scales, f, 128, 4, stream=dev)
                self.assert_bitwise_equal(
                    d_ib, mx.full((8, 128), scale, dtype=mx.float16)
                )
                self.assert_implied_bias_matches(
                    mx.quantized_matmul,
                    (x, w, scales),
                    rebuilt,
                    f,
                    True,
                    128,
                    4,
                    stream=dev,
                )

    def test_implied_bias_non_power_of_two_bits(self):
        # 3/5/6-bit have no `_ib` kernels in the non-JIT Metal library, so the
        # op materializes the bias and the ordinary kernels run -- on the GPU
        # (where a missing kernel name would throw at dispatch) and the CPU.
        key = mx.random.key(190)
        k1, k2 = mx.random.split(key)
        for dev, dtype, bits in product(
            self.implied_bias_devices(), self.IMPLIED_BIAS_DTYPES, [3, 5, 6]
        ):
            factor = -float(2 ** (bits - 1))
            w = mx.random.normal(shape=(72, 512), key=k2).astype(dtype)
            w_q, scales, _ = mx.quantize(w, 64, bits)
            full = self.reference_bias(scales, factor)
            f = mx.array(factor, dtype)
            with self.subTest(dev=dev, dtype=dtype, bits=bits, op="dequantize"):
                self.assert_implied_bias_matches(
                    mx.dequantize, (w_q, scales), full, f, 64, bits, stream=dev
                )
            for M, transpose in [
                (1, True),
                (3, True),
                (8, True),
                (33, True),
                (2, False),
            ]:
                with self.subTest(
                    dev=dev, dtype=dtype, bits=bits, M=M, qvm=not transpose
                ):
                    shape = (M, 512) if transpose else (M, 72)
                    x = mx.random.normal(shape=shape, key=k1).astype(dtype)
                    self.assert_implied_bias_matches(
                        mx.quantized_matmul,
                        (x, w_q, scales),
                        full,
                        f,
                        transpose,
                        64,
                        bits,
                        stream=dev,
                    )

    def test_implied_bias_throws(self):
        x = mx.random.normal(shape=(2, 512))
        w = mx.random.normal(shape=(64, 512))
        w_q, scales, _ = mx.quantize(w, 64, 4)
        f = mx.array(-8.0)
        # Only a 0-d biases array is a factor; a 1-d one is a shape error.
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales, mx.array([-8.0]), True, 64, 4)
        # The factor must be a real floating type.
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales, mx.array(-8), True, 64, 4)
        # So must a per-group biases array.
        int_biases = mx.zeros(scales.shape, dtype=mx.int32)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales, int_biases, True, 64, 4)
        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, int_biases, 64, 4)
        # The fp modes have no biases.
        w_q4, scales4 = mx.quantize(w, mode="mxfp4")
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q4, scales4, f, True, mode="mxfp4")

    def test_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 5, 6, 8],  # bits
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                if M < group_size:
                    continue
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = mx.random.normal(shape=x_shape, key=k1)
                w = mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm_splitk(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [128],  # M
            [16384],  # N
            [1, 3],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = 1e-1 * mx.random.normal(shape=x_shape, key=k1)
                w = 1e-1 * mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

        # Test with 1D vector
        group_size = 32
        bits = 8
        N = 2048
        x = 1e-1 * mx.random.normal(shape=(N,), key=k1)
        w = 1e-1 * mx.random.normal(shape=(N, N), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        y_q = mx.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_qvm_splitk_multi_row(self):
        # Test qvm split_k with M > 1 to ensure the x row stride is correct
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [64, 32],  # group_size
            [4, 8],  # bits
            [128],  # out dim (N)
            [2048, 4096],  # in dim (K) >= 1024 to trigger split_k
            [2, 3],  # M (multiple rows)
        )
        for group_size, bits, N, K, M in tests:
            with self.subTest(M=M, K=K, N=N, group_size=group_size, bits=bits):
                x = 1e-1 * mx.random.normal(shape=(M, K), key=k1)
                w = 1e-1 * mx.random.normal(shape=(K, N), key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_fp_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        # Add a splitk
        tests = list(tests)
        tests.append((128, 16384, 0))
        modes = ["mxfp4", "nvfp4", "mxfp8"]

        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (1, N) if B == 0 else (B, 1, N)
                    w_shape = (N, M) if B == 0 else (B, N, M)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode)
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=False,
                        mode=mode,
                    )
                    y_hat = x @ w_hat
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    FP_TAIL_TOLS = {mx.float32: 2e-3, mx.float16: 5e-3, mx.bfloat16: 4e-2}

    def _fp_tail_dtypes(self):
        # The half types only run on the GPU
        if mx.default_device() == mx.gpu:
            return [mx.float32, mx.float16, mx.bfloat16]
        return [mx.float32]

    def test_fp_qmm_non_multiple_of_32(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        def check(M, K, N, transpose, mode, dtype):
            with self.subTest(
                M=M, K=K, N=N, transpose=transpose, mode=mode, dtype=dtype
            ):
                w_shape = (N, K) if transpose else (K, N)
                x = (mx.random.normal(shape=(M, K), key=k1) / K**0.5).astype(dtype)
                w = mx.random.normal(shape=w_shape, key=k2).astype(dtype)
                w_q, scales = mx.quantize(w, mode=mode)
                w_hat = mx.dequantize(w_q, scales, mode=mode)

                y_q = mx.quantized_matmul(
                    x, w_q, scales, transpose=transpose, mode=mode
                )
                # An unbounded store of the N tail spills onto the next output
                # row and races the threadgroup that owns it, so two identical
                # runs must agree bit for bit.
                y_q2 = mx.quantized_matmul(
                    x, w_q, scales, transpose=transpose, mode=mode
                )
                y_hat = x @ (mx.swapaxes(w_hat, -1, -2) if transpose else w_hat)

                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertTrue(mx.array_equal(y_q, y_q2))
                self.assertLess((y_q - y_hat).abs().max(), self.FP_TAIL_TOLS[dtype])

        # (M, K, N, transpose). M is kept clear of the qmv batch limit
        # (get_qmv_batch_limit caps at 32) so these route to the tiled kernel.
        nvfp4_shapes = [
            (50, 1040, 128, True),  # K % 32 == 16
            (33, 528, 64, True),  # K % 32 == 16, M just over one BM tile
            (50, 512, 1040, False),  # N % 32 == 16
            (50, 1040, 1040, False),  # both dims carry a tail
            (50, 16, 128, True),  # K is a single partial tile
        ]
        control_shapes = [
            (50, 1056, 128, True),  # K % 32 == 0
            (50, 512, 1056, False),  # N % 32 == 0
        ]

        for dtype in self._fp_tail_dtypes():
            for M, K, N, transpose in nvfp4_shapes:
                check(M, K, N, transpose, "nvfp4", dtype)
            for mode in ["mxfp4", "mxfp8"]:
                for M, K, N, transpose in control_shapes:
                    check(M, K, N, transpose, mode, dtype)

    def test_fp_qmm_non_multiple_of_32_vjp(self):
        # The backward of quantized_matmul runs the same kernels with the
        # transpose flipped, so cover both directions at a tail shape.
        key = mx.random.key(0)
        k1, k2, k3 = mx.random.split(key, 3)
        M, K, N = 50, 1040, 128

        for dtype in self._fp_tail_dtypes():
            for transpose in [True, False]:
                with self.subTest(transpose=transpose, dtype=dtype):
                    w_shape = (N, K) if transpose else (K, N)
                    x = (mx.random.normal(shape=(M, K), key=k1) / K**0.5).astype(dtype)
                    w = mx.random.normal(shape=w_shape, key=k2).astype(dtype)
                    w_q, scales = mx.quantize(w, mode="nvfp4")
                    w_hat = mx.dequantize(w_q, scales, mode="nvfp4")
                    cot = (mx.random.normal(shape=(M, N), key=k3) / N**0.5).astype(
                        dtype
                    )

                    def f(x):
                        return mx.quantized_matmul(
                            x, w_q, scales, transpose=transpose, mode="nvfp4"
                        )

                    (out,), (grad,) = mx.vjp(f, (x,), (cot,))
                    tol = self.FP_TAIL_TOLS[dtype]

                    # The forward runs the tail kernel for this transpose; the
                    # backward runs it for the other one, we might as well check both.
                    out_hat = x @ (mx.swapaxes(w_hat, -1, -2) if transpose else w_hat)
                    self.assertEqual(out.shape, out_hat.shape)
                    self.assertLess((out - out_hat).abs().max(), tol)

                    grad_hat = cot @ (
                        w_hat if transpose else mx.swapaxes(w_hat, -1, -2)
                    )
                    self.assertEqual(grad.shape, grad_hat.shape)
                    self.assertLess((grad - grad_hat).abs().max(), tol)

    def test_fp_gather_qmm_non_multiple_of_32(self):
        # Both gather_qmm kernels: the sorted-indices (MoE prefill) kernel and
        # the tiled gather kernel.
        key = mx.random.key(0)
        k1, k2, k3 = mx.random.split(key, 3)
        mode = "nvfp4"
        E = 4

        def check(K, N, transpose, sorted_indices, dtype):
            with self.subTest(
                K=K,
                N=N,
                transpose=transpose,
                sorted_indices=sorted_indices,
                dtype=dtype,
            ):
                w_shape = (E, N, K) if transpose else (E, K, N)
                w = mx.random.normal(shape=w_shape, key=k2).astype(dtype)
                w_q, scales = mx.quantize(w, mode=mode)
                w_hat = mx.dequantize(w_q, scales, mode=mode)

                if sorted_indices:
                    T = 64
                    x = (mx.random.normal(shape=(T, 1, K), key=k1) / K**0.5).astype(
                        dtype
                    )
                    rhs = mx.sort(mx.random.randint(0, E, shape=(T,), key=k3)).astype(
                        mx.uint32
                    )
                    lhs = None
                else:
                    B, M = 8, 50
                    x = (mx.random.normal(shape=(B, M, K), key=k1) / K**0.5).astype(
                        dtype
                    )
                    rhs = mx.random.randint(0, E, shape=(B,), key=k3).astype(mx.uint32)
                    lhs = mx.arange(B, dtype=mx.uint32)

                y_q = mx.gather_qmm(
                    x,
                    w_q,
                    scales,
                    lhs_indices=lhs,
                    rhs_indices=rhs,
                    transpose=transpose,
                    mode=mode,
                    sorted_indices=sorted_indices,
                )
                w_sel = mx.take(w_hat, rhs, axis=0)
                y_hat = x @ (mx.swapaxes(w_sel, -1, -2) if transpose else w_sel)

                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), self.FP_TAIL_TOLS[dtype])

        # (K, N, transpose): transpose puts the tail on K, else on N.
        shapes = [
            (1040, 128, True),  # K % 32 == 16
            (512, 1040, False),  # N % 32 == 16
        ]
        for dtype in self._fp_tail_dtypes():
            for K, N, transpose in shapes:
                for sorted_indices in [True, False]:
                    check(K, N, transpose, sorted_indices, dtype)

    def test_mode_error_cases(self):
        w = mx.random.normal(shape=(256, 256))
        x = mx.random.normal(shape=(1, 256))

        # Invalid mode
        with self.assertRaises(ValueError):
            mx.quantize(w, mode="xyz")

        wq, scales, biases = mx.quantize(w, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, biases, bits=4, group_size=32, mode="xyz")

        with self.assertRaises(ValueError):
            mx.quantized_matmul(
                x, wq, scales, biases, bits=4, group_size=32, mode="xyz"
            )

        rhs_indices = mx.array(0)
        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x,
                wq,
                scales,
                biases,
                rhs_indices=rhs_indices,
                bits=4,
                group_size=32,
                mode="xyz",
            )

        # Only quantize floating point types
        with self.assertRaises(ValueError):
            mx.quantize(mx.zeros((128, 128), mx.int32))

        with self.assertRaises(ValueError):
            mx.quantize(mx.zeros((128, 128), mx.int32), mode="mxfp4")

        # Must have bias for affine
        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x, wq, scales, None, rhs_indices=rhs_indices, bits=4, group_size=32
            )

        # Must be floating point
        x = mx.zeros(shape=(256,), dtype=mx.int32)
        scales = mx.zeros(scales.shape, dtype=mx.int32)
        biases = mx.zeros(scales.shape, dtype=mx.int32)
        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x, wq, scales, biases, rhs_indices=rhs_indices, bits=4, group_size=32
            )

    def test_throw(self):
        x = mx.random.normal(shape=(10, 512))
        w = mx.random.normal(shape=(32, 512))
        w_q, scales, biases = mx.quantize(w)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q.T, scales, biases)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q.T, scales.T, biases)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales, biases, False)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales.T, biases.T)
        y = mx.quantized_matmul(x, w_q, scales, biases, True)
        mx.eval(y)

    def test_small_matrix(self):
        tol = 3e-2 if mx.cuda.is_available() else 1e-3
        for w_shape in [(8, 256), (1, 8, 256), (3, 8, 256)]:
            with self.subTest(w_shape=w_shape):
                w = mx.random.normal(shape=(w_shape))
                w_q, scales, biases = mx.quantize(w)
                w_hat = mx.dequantize(w_q, scales, biases)

                # Test qmv
                for shape in [(3, 1, 256), (3, 4, 256)]:
                    x = mx.random.normal(shape=shape)
                    y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), tol)

                # Test qmm_t
                x = mx.random.normal(shape=(3, 10, 256))
                y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
                y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), tol)

                # Test qvm
                x = mx.random.normal(shape=(3, 1, 8))
                y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), tol)

                # Test qmm
                x = mx.random.normal(shape=(3, 10, 8))
                y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_non_multiples(self):
        tol = 3e-2 if mx.cuda.is_available() else 1e-3
        w = mx.random.normal(shape=(33, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)

        # Test qmv
        x = mx.random.normal(shape=(1, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qmm_t
        x = mx.random.normal(shape=(10, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qvm
        x = mx.random.normal(shape=(1, 33))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qmm
        x = mx.random.normal(shape=(10, 33))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Smaller than 8
        w = mx.random.normal(shape=(3, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)

        # Test qmv
        x = mx.random.normal(shape=(1, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qmm_t
        x = mx.random.normal(shape=(10, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qvm
        x = mx.random.normal(shape=(1, 3))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test qmm
        x = mx.random.normal(shape=(10, 3))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

        # Test with larger than 128 unaligned sizes
        w = mx.random.normal(shape=(99, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)
        x = mx.random.normal(shape=(129, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_qmv_small_non_multiples(self):
        # Test very small K and N dimensions (e.g., [MxK] x [NxK].T = [MxN])
        # Each tuple is (M, K, N) representing input rows, weight cols, weight rows
        test_cases = [
            (1, 32, 3),
            (2, 32, 10),
            (1, 32, 5),
            (4, 32, 7),
        ]

        # Test different quantization settings (bits, group_size, mode)
        quantization_settings = [
            (4, 32, "affine"),
            (6, 32, "affine"),
            (4, 16, "nvfp4"),
        ]

        for M, K, N in test_cases:
            for bits, group_size, mode in quantization_settings:
                # Test without batch dimension
                with self.subTest(
                    M=M,
                    K=K,
                    N=N,
                    batch=None,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                ):
                    w = mx.random.normal(shape=(N, K))
                    w_q, *sb = mx.quantize(
                        w,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    w_hat = mx.dequantize(
                        w_q,
                        *sb,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )

                    # Test qmv/qmm_t (transpose=True): [MxK] @ [NxK].T = [MxN]
                    x = mx.random.normal(shape=(M, K))
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        *sb,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_gather_qmm(self):
        def quantize(w, transpose=True, group_size=None, bits=None, mode="affine"):
            if mode == "affine":
                qw, s, b = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
            else:
                qw, s = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
                b = None
            w_hat = mx.dequantize(qw, s, b, group_size=group_size, bits=bits, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def test_shape(
            M,
            N,
            K,
            dtype=mx.float32,
            batch_A=(),
            batch_B=(),
            lhs_indices=None,
            rhs_indices=None,
            transpose=True,
            group_size=None,
            bits=None,
            mode="affine",
            noncontiguous_x=False,
            rtol=1e-5,
            atol=1e-4,
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                dtype=dtype,
                batch_A=batch_A,
                batch_B=batch_B,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                noncontiguous_x=noncontiguous_x,
                rtol=rtol,
                atol=atol,
            ):
                x = mx.random.normal(shape=batch_A + (M, K)).astype(dtype)
                w = mx.random.normal(
                    shape=batch_B + ((N, K) if transpose else (K, N))
                ).astype(dtype)
                w_hat, qw, s, b = quantize(w, transpose, group_size, bits, mode=mode)

                if noncontiguous_x:
                    x = mx.concatenate([x, x], axis=-1)[..., : x.shape[-1]]

                if lhs_indices is not None:
                    lhs_indices = mx.array(lhs_indices)
                if rhs_indices is not None:
                    rhs_indices = mx.array(rhs_indices)

                c1 = mx.gather_mm(x, w_hat, lhs_indices, rhs_indices)
                c2 = mx.gather_qmm(
                    x,
                    qw,
                    s,
                    b,
                    lhs_indices,
                    rhs_indices,
                    transpose=transpose,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                self.assertTrue(mx.allclose(c1, c2, rtol=rtol, atol=atol))

        inputs = (
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (1,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (2,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (3,),
                "lhs_indices": (0, 2),
                "batch_B": (1,),
                "rhs_indices": (0,),
            },
            {
                "batch_A": (5,),
                "lhs_indices": (0, 2),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (4, 2),
                "lhs_indices": (
                    (7, 6),
                    (5, 4),
                    (1, 2),
                ),
                "batch_B": (4, 1),
                "rhs_indices": ((2,), (0,), (1,)),
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "nvfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp8",
            },
        )

        for kwargs in inputs:
            test_shape(1, 32, 128, **kwargs)
            test_shape(32, 32, 256, **kwargs)
            test_shape(1, 32, 256, **kwargs)
            test_shape(32, 256, 32, transpose=False, **kwargs)
            test_shape(1, 256, 32, transpose=False, **kwargs)
            test_shape(32, 32, 512, **kwargs)
            test_shape(1, 32, 512, **kwargs)
            test_shape(32, 512, 32, transpose=False, **kwargs)
            test_shape(1, 512, 32, transpose=False, **kwargs)

        # Long K with many gathered vectors uses the nvfp4 gather qmv kernel on
        # the GPU. Low precision outputs need a tolerance of about one ulp.
        on_gpu = mx.default_device() == mx.gpu
        bf16 = dict(dtype=mx.bfloat16 if on_gpu else mx.float32, rtol=1e-2, atol=1e-2)
        f16 = dict(dtype=mx.float16 if on_gpu else mx.float32, rtol=1e-3, atol=1e-3)
        test_shape(
            1,
            32,
            1024,
            batch_A=(1,),
            batch_B=(3,),
            lhs_indices=(0,),
            rhs_indices=(2, 1, 0, 2, 1, 0, 2, 1, 0),
            mode="nvfp4",
            **bf16,
        )
        test_shape(
            1,
            33,
            1024,
            batch_A=(2, 2),
            batch_B=(2,),
            lhs_indices=((0,), (3,)),
            rhs_indices=((1, 0, 1),),
            mode="nvfp4",
            noncontiguous_x=True,
            **f16,
        )
        # K = 1056 leaves one warp lane active in the second reduction step.
        test_shape(
            1,
            32,
            1056,
            batch_A=(1,),
            batch_B=(3,),
            lhs_indices=(0,),
            rhs_indices=(2, 1, 0, 2, 1, 0, 2, 1, 0),
            mode="nvfp4",
        )
        for mode in ("mxfp4", "mxfp8"):
            test_shape(
                1,
                33,
                1024,
                batch_A=(1,),
                batch_B=(3,),
                lhs_indices=(0,),
                rhs_indices=(2, 1, 0, 2, 1, 0, 2, 1, 0),
                mode=mode,
                **bf16,
            )
        for mode in ("nvfp4", "mxfp4", "mxfp8"):
            for K, tol in ((448, 1e-1), (512, 1e-2), (704, 1e-2)):
                test_shape(
                    1,
                    33,
                    K,
                    batch_A=(1,),
                    batch_B=(3,),
                    lhs_indices=(0,),
                    rhs_indices=(2, 1, 0, 2, 1, 0, 2, 1, 0),
                    mode=mode,
                    **{**bf16, "rtol": tol, "atol": tol},
                )

    def test_gather_qqmm(self):
        if mx.default_device() == mx.cpu:
            self.skipTest("Not implemented for CPU")
            return

        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        batches = (
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (1,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (2,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (3,),
                "lhs_indices": (0, 2),
                "batch_B": (1,),
                "rhs_indices": (0,),
            },
            {
                "batch_A": (5,),
                "lhs_indices": (0, 2),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
        )
        tests = product(
            batches,
            [1, 32],  # M
            [32, 256],  # N
            [32, 256],  # K
            ["nvfp4", "mxfp8"],  # mode
        )

        for batch, M, N, K, mode in tests:
            with self.subTest(shape=(M, N, K), mode=mode, **batch):
                batch_A, lhs_indices, batch_B, rhs_indices = batch.values()
                x_shape = (*batch_A, M, K)
                w_shape = (*batch_B, N, K)

                x = mx.random.normal(shape=x_shape, key=k1)
                global_scale_x = mx.max(mx.abs(x)) if mode == "nvfp4" else None
                x_hat = mx.dequantize(
                    *mx.quantize(x, mode=mode, global_scale=global_scale_x),
                    mode=mode,
                    dtype=mx.float32,
                    global_scale=global_scale_x,
                )

                w = mx.random.normal(shape=w_shape, key=k2)
                global_scale_w = mx.max(mx.abs(w)) if mode == "nvfp4" else None
                w_q, scales = mx.quantize(w, mode=mode, global_scale=global_scale_w)
                w_hat = mx.dequantize(
                    w_q,
                    scales,
                    mode=mode,
                    global_scale=global_scale_w,
                    dtype=mx.float32,
                )

                if lhs_indices is not None:
                    lhs_indices = mx.array(lhs_indices)
                if rhs_indices is not None:
                    rhs_indices = mx.array(rhs_indices)

                y_q = mx.gather_qqmm(
                    x,
                    w_q,
                    scales,
                    lhs_indices,
                    rhs_indices,
                    mode=mode,
                    global_scale_x=global_scale_x,
                    global_scale_w=global_scale_w,
                )
                y_hat = mx.gather_mm(
                    x_hat, mx.swapaxes(w_hat, -1, -2), lhs_indices, rhs_indices
                )
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_gather_qqmm_global_scale_matrix_paths(self):
        if mx.default_device() == mx.cpu:
            self.skipTest("Not implemented for CPU")

        E, N, K = 3, 64, 256
        w = mx.random.normal((E, N, K), key=mx.random.key(10)).astype(mx.bfloat16)
        global_scale_w = mx.max(mx.abs(w), axis=(1, 2)).astype(mx.float32)
        quantized = [
            mx.quantize(w[e], mode="nvfp4", global_scale=global_scale_w[e])
            for e in range(E)
        ]
        w_q = mx.stack([q for q, _ in quantized])
        scales_w = mx.stack([scales for _, scales in quantized])
        w_hat = mx.stack(
            [
                mx.dequantize(
                    q,
                    scales,
                    mode="nvfp4",
                    dtype=mx.bfloat16,
                    global_scale=global_scale_w[e],
                )
                for e, (q, scales) in enumerate(quantized)
            ]
        )

        def check(x, rhs, lhs=None, sorted_indices=False):
            global_scale_x = mx.max(mx.abs(x)).astype(mx.float32)
            x_hat = mx.dequantize(
                *mx.quantize(x, mode="nvfp4", global_scale=global_scale_x),
                mode="nvfp4",
                dtype=mx.bfloat16,
                global_scale=global_scale_x,
            )
            actual = mx.gather_qqmm(
                x,
                w_q,
                scales_w,
                lhs,
                rhs,
                mode="nvfp4",
                global_scale_x=global_scale_x,
                global_scale_w=global_scale_w,
                sorted_indices=sorted_indices,
            )
            expected = mx.gather_mm(
                x_hat,
                mx.swapaxes(w_hat, -1, -2),
                lhs,
                rhs,
                sorted_indices=sorted_indices,
            )
            error = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
            scale = mx.maximum(mx.abs(expected.astype(mx.float32)).max(), 1e-20)
            tol = 1e-1 if mx.cuda.is_available() else 3e-2
            self.assertLess(error.max() / scale, tol)

        # Sorted indices select the RHS-grouped matrix kernel.
        x = mx.random.normal((24, 1, K), key=mx.random.key(20)).astype(mx.bfloat16)
        rhs = mx.array([0] * 8 + [1] * 8 + [2] * 8)
        check(x, rhs, sorted_indices=True)

        # A large M selects the gathered matrix kernel.
        x = mx.random.normal((2, 64, K), key=mx.random.key(30)).astype(mx.bfloat16)
        lhs = mx.array([0, 1, 0, 1])
        rhs = mx.array([0, 1, 2, 0])
        check(x, rhs, lhs)

    def test_qmm_fp_type(self):
        indices = mx.array([[2], [0], [1]], dtype=mx.uint32)

        modes = ["mxfp8", "mxfp4"]
        for mode in modes:
            for t in [mx.bfloat16, mx.float16, mx.float32]:
                x = mx.random.normal((32, 256)).astype(t)

                w = mx.random.normal((32, 256))
                wq, s = mx.quantize(w, mode=mode)
                out = mx.quantized_matmul(x, wq, s, mode=mode)
                self.assertEqual(out.dtype, t)

                w = mx.random.normal((4, 32, 256))
                wq, s = mx.quantize(w, mode=mode)

                out = mx.gather_qmm(x, wq, s, rhs_indices=indices, mode=mode)
                self.assertEqual(out.dtype, t)

    def test_gather_matmul_grad(self):
        def quantize(w, transpose=True, group_size=64, bits=4):
            qw, s, b = mx.quantize(w, group_size=group_size, bits=bits)
            w_hat = mx.dequantize(qw, s, b, group_size=group_size, bits=bits)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        lhs_indices = mx.array([[7, 6], [4, 1], [0, 2]], dtype=mx.uint32)
        rhs_indices = mx.array([[2], [0], [1]], dtype=mx.uint32)

        x = mx.random.normal((4, 2, 32, 256))
        w = mx.random.normal((4, 1, 32, 256))
        w_hat, qw, s, b = quantize(w)

        def f_ref(x, w, i1, i2):
            return mx.gather_mm(x, w, i1, i2).sum()

        def f_test(x, qw, s, b, i1, i2):
            return mx.gather_qmm(x, qw, s, b, i1, i2, transpose=True).sum()

        r1 = f_ref(x, w_hat, lhs_indices, rhs_indices)
        r2 = f_test(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(mx.allclose(r1, r2, atol=1e-4))

        g1 = mx.grad(f_ref)(x, w_hat, lhs_indices, rhs_indices)
        g2 = mx.grad(f_test)(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(mx.allclose(g1, g2, atol=1e-4))

    def test_gather_qmm_matrix_path(self):
        # Regression test for matrix-size gather_qmm with half precision
        # inputs: on NAX devices the kernel name was built with bk = 32
        # while the kernels are only instantiated with bk = 64, so the
        # kernel lookup failed with "Unable to load kernel".
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        dtype = mx.bfloat16 if (mx.default_device() == mx.gpu) else mx.float32
        E, M, N, K = 4, 64, 512, 512
        lhs_indices = mx.array([0, 1, 2], dtype=mx.uint32)
        rhs_indices = mx.array([0, 2, 3], dtype=mx.uint32)
        for mode in ["affine", "mxfp4"]:
            with self.subTest(mode=mode):
                x = (mx.random.normal(shape=(3, M, K), key=k1) / K**0.5).astype(dtype)
                # Keep w in the same dtype so affine scales do not promote
                # the matmul to float32 (which would skip the NAX route).
                w = mx.random.normal(shape=(E, N, K), key=k2).astype(dtype)
                w_q, *wargs = mx.quantize(w, mode=mode)
                w_hat = mx.dequantize(w_q, *wargs, mode=mode)
                y_q = mx.gather_qmm(
                    x,
                    w_q,
                    *wargs,
                    lhs_indices=lhs_indices,
                    rhs_indices=rhs_indices,
                    transpose=True,
                    mode=mode,
                )
                y_hat = mx.stack(
                    [
                        x[i].astype(mx.float32) @ w_hat[int(rhs_indices[i])].T
                        for i in range(3)
                    ]
                ).astype(dtype)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-1)

    @unittest.skipIf(
        is_m1_mac() and not mx.metal.is_available(),
        "Accelerate bug https://github.com/ml-explore/mlx/pull/3563",
    )
    def test_gather_qmm_sorted(self):
        def quantize(w, transpose=True, group_size=None, mode="affine"):
            if mode == "affine":
                qw, s, b = mx.quantize(w, group_size=group_size, mode=mode)
            else:
                qw, s = mx.quantize(w, mode=mode)
                b = None

            w_hat = mx.dequantize(qw, s, b, group_size=group_size, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def gather_sort(x, indices):
            N, M = indices.shape
            indices = indices.flatten()
            order = mx.argsort(indices)
            inv_order = mx.argsort(order)
            return x.flatten(0, -3)[order // M], indices[order], inv_order

        def scatter_unsort(x, inv_order, shape=None):
            x = x[inv_order]
            if shape is not None:
                x = mx.unflatten(x, 0, shape)
            return x

        parameters = [
            # L, K, D, E, I, transpose
            (32, 512, 512, 4, 2, True, "affine"),
            (32, 512, 544, 4, 2, True, "mxfp4"),
            (32, 512, 544, 4, 2, True, "nvfp4"),
            (32, 512, 544, 4, 2, True, "mxfp8"),
            (39, 512, 512, 4, 2, True, "affine"),
            (128, 512, 512, 4, 2, True, "affine"),
            (133, 512, 512, 4, 2, True, "affine"),
            (133, 512, 555, 4, 2, True, "affine"),
            (133, 512, 512, 4, 2, True, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
            (64, 512, 544, 4, 2, False, "mxfp4"),
            (64, 512, 544, 4, 2, False, "nvfp4"),
            (64, 512, 544, 4, 2, False, "mxfp8"),
            (133, 512, 512, 4, 2, False, "affine"),
            (133, 512, 544, 4, 2, False, "affine"),
            (133, 512, 555, 4, 2, False, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
        ]

        key = mx.random.key(0)
        k1, k2, k3 = mx.random.split(key, 3)
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32

        for L, K, D, E, num_indices, transpose, mode in parameters:
            with self.subTest(
                L=L, K=K, D=D, E=E, I=num_indices, transpose=transpose, mode=mode
            ):
                if mode != "affine":
                    group_size = None
                    dtype = (
                        mx.bfloat16 if (mx.default_device() == mx.gpu) else mx.float32
                    )
                else:
                    group_size = 64
                    dtype = (
                        mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
                    )

                K, D = (K, D) if transpose else (D, K)
                ishape = (L, num_indices)
                xshape = (L, 1, 1, K)
                wshape = (E, D, K) if transpose else (E, K, D)

                indices = (mx.random.uniform(shape=ishape, key=k1) * E).astype(
                    mx.uint32
                )
                x = mx.random.normal(xshape, key=k2) / K**0.5
                w = mx.random.normal(wshape, key=k3) / K**0.5

                x = x.astype(dtype)
                w = w.astype(dtype)

                w, *wq = quantize(
                    w, group_size=group_size, mode=mode, transpose=transpose
                )

                y1 = mx.gather_mm(x, w, rhs_indices=indices)
                y2 = mx.gather_qmm(
                    x,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    transpose=transpose,
                    rhs_indices=indices,
                )
                xs, idx, inv_order = gather_sort(x, indices)
                y3 = mx.gather_mm(xs, w, rhs_indices=idx, sorted_indices=True)

                y4 = mx.gather_qmm(
                    xs,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    rhs_indices=idx,
                    transpose=transpose,
                    sorted_indices=True,
                )
                y3 = scatter_unsort(y3, inv_order, indices.shape)
                y4 = scatter_unsort(y4, inv_order, indices.shape)

                tol = 1.5e-5 if (dtype == mx.float32) else 1e-3

                self.assertLess((y1 - y2).abs().max(), tol)
                self.assertLess((y1 - y3).abs().max(), tol)
                self.assertLess((y1 - y4).abs().max(), tol)

                self.assertTrue(mx.allclose(y1, y2, atol=tol))
                self.assertTrue(mx.allclose(y1, y3, atol=tol))
                self.assertTrue(mx.allclose(y1, y4, atol=tol))

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_gather_qmm_sorted_nax_large_m(self):
        E, N, K, group_size = 16, 64, 64, 32
        dtype = mx.float16
        mx.random.seed(0)
        w = (mx.random.normal((E, N, K)) * 0.1).astype(dtype)
        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=4)
        w_hat = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=4)

        for M in (32767, 32768, 32769, 32832):
            with self.subTest(M=M):
                x = (mx.random.normal((M, 1, K)) * 0.1).astype(dtype)
                rhs_indices = (mx.arange(M) * E // M).astype(mx.uint32)
                y_hat = mx.gather_mm(
                    x.astype(mx.float32),
                    mx.swapaxes(w_hat, -1, -2).astype(mx.float32),
                    rhs_indices=rhs_indices,
                    sorted_indices=True,
                )
                mx.eval(y_hat)
                mx.synchronize()

                for value in (-31.0, 47.0):
                    poison = mx.full(y_hat.shape, value, dtype=mx.float16)
                    mx.eval(poison)
                    mx.synchronize()
                    del poison

                    y_q = mx.gather_qmm(
                        x,
                        w_q,
                        scales,
                        biases,
                        rhs_indices=rhs_indices,
                        transpose=True,
                        group_size=group_size,
                        bits=4,
                        sorted_indices=True,
                    )
                    max_error = (y_q.astype(mx.float32) - y_hat).abs().max()
                    self.assertLess(float(max_error.item()), 5e-2)
                    del y_q, max_error

    @unittest.skipIf(mx.cuda.is_available(), "Not implemented for CUDA")
    def test_gather_qmm_sorted_sliced_weight(self):
        E, R, D, N = 8, 64, 256, 64
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
        mx.random.seed(0)
        w = (mx.random.normal((E, 2 * R, D)) * 0.05).astype(dtype)
        qw, s, b = mx.quantize(w, group_size=64, bits=4)
        x = (mx.random.normal((N, 1, D)) * 0.5).astype(dtype)
        indices = mx.sort(mx.random.randint(0, E, (N,)).astype(mx.uint32))

        for sl in (slice(0, R), slice(R, 2 * R)):
            view = (qw[:, sl], s[:, sl], b[:, sl])
            copy = tuple(mx.contiguous(a) for a in view)
            kwargs = dict(
                rhs_indices=indices,
                transpose=True,
                group_size=64,
                bits=4,
                sorted_indices=True,
            )
            self.assertTrue(
                mx.allclose(
                    mx.gather_qmm(x, *view, **kwargs),
                    mx.gather_qmm(x, *copy, **kwargs),
                    atol=1e-4,
                )
            )

    def test_gather_qmm_sorted_unaligned_k(self):
        # Every case in test_gather_qmm_sorted uses K % 64 == 0, so the tail of
        # the K loop is never exercised. When K is not a multiple of the block
        # size the sorted path has to zero the partial tile correctly, otherwise
        # it silently drops the tail contribution for part of the output.
        L, D, E = 128, 256, 4
        key = mx.random.key(0)
        k1, k2, k3 = mx.random.split(key, 3)
        on_gpu = mx.default_device() == mx.gpu

        dtype = mx.float16 if on_gpu else mx.float32

        for mode, group_size in (
            ("affine", 32),
            ("mxfp4", None),
            ("mxfp8", None),
            ("nvfp4", None),
        ):
            for K in (160, 288, 544):
                with self.subTest(mode=mode, K=K):
                    indices = mx.sort(
                        (mx.random.uniform(shape=(L,), key=k1) * E).astype(mx.uint32)
                    )
                    x = (mx.random.normal((L, 1, K), key=k2) / K**0.5).astype(dtype)
                    w = (mx.random.normal((E, D, K), key=k3) / K**0.5).astype(dtype)

                    if mode == "affine":
                        wq = mx.quantize(w, group_size=group_size, mode=mode)
                    else:
                        wq = mx.quantize(w, mode=mode)

                    kwargs = dict(
                        group_size=group_size,
                        mode=mode,
                        transpose=True,
                        rhs_indices=indices,
                    )
                    y_sorted = mx.gather_qmm(x, *wq, sorted_indices=True, **kwargs)
                    y_unsorted = mx.gather_qmm(x, *wq, sorted_indices=False, **kwargs)

                    tol = 1e-3 if on_gpu else 1.5e-5
                    self.assertLess((y_sorted - y_unsorted).abs().max(), tol)

    @unittest.skipIf(not mx.cuda.is_available(), "CUDA kernel path only")
    def test_gather_qmm_sorted_rhs_cuda(self):
        # Sorted rows with a shared LHS take the sorted-RHS matrix kernel. Row
        # counts cover the 16/32/64-row tiles and a row tail; random routing
        # gives runs of unequal length that cross tile edges.
        mx.random.seed(0)
        shapes = [(64, 4, 2816, 704)] + [
            (rows, 8, 256, 256) for rows in (16, 24, 40, 64, 100, 200, 1000)
        ]
        for rows, experts, n, k in shapes:
            for dtype in (mx.bfloat16, mx.float16):
                x = (mx.random.normal((rows, 1, 1, k)) / k**0.5).astype(dtype)
                w = mx.random.normal((experts, n, k)).astype(dtype)
                indices = mx.sort(mx.random.randint(0, experts, (rows,))).astype(
                    mx.uint32
                )[:, None]
                for mode in ("nvfp4", "mxfp4", "mxfp8"):
                    with self.subTest(rows=rows, dtype=dtype, mode=mode):
                        qw, scales = mx.quantize(w, mode=mode)
                        w_hat = mx.dequantize(qw, scales, mode=mode)
                        reference = mx.gather_mm(
                            x, w_hat.swapaxes(-1, -2), rhs_indices=indices
                        )
                        expected = mx.gather_qmm(
                            x,
                            qw,
                            scales,
                            rhs_indices=indices,
                            transpose=True,
                            mode=mode,
                            sorted_indices=False,
                        )
                        mx.eval(reference, expected)
                        # Successive runs in one CTA reuse its shared memory;
                        # repeat to catch a race between them.
                        for _ in range(3):
                            actual = mx.gather_qmm(
                                x,
                                qw,
                                scales,
                                rhs_indices=indices,
                                transpose=True,
                                mode=mode,
                                sorted_indices=True,
                            )
                            mx.eval(actual)
                            self.assertLess(
                                mx.max(mx.abs(expected - actual)).item(), 1e-2
                            )
                            self.assertTrue(
                                mx.allclose(reference, actual, rtol=2e-2, atol=2e-2)
                            )

        # One x row broadcast over the sorted indices stays on the generic
        # gathered kernel.
        rows, experts, n, k = 16, 4, 256, 256
        x = mx.random.normal((1, 1, k)).astype(mx.bfloat16)
        w = mx.random.normal((experts, n, k)).astype(mx.bfloat16)
        qw, scales = mx.quantize(w, mode="mxfp8")
        indices = mx.repeat(mx.arange(experts), rows // experts).astype(mx.uint32)
        kwargs = dict(rhs_indices=indices, transpose=True, mode="mxfp8")
        expected = mx.gather_qmm(x, qw, scales, sorted_indices=False, **kwargs)
        actual = mx.gather_qmm(x, qw, scales, sorted_indices=True, **kwargs)
        self.assertLess(mx.max(mx.abs(expected - actual)).item(), 1e-3)

    def test_gather_qmm_grad(self):
        def gather_qmm_ref(x, w, s, b, lhs, rhs, trans, sort):
            if lhs is not None:
                x = x[lhs]
            if rhs is not None:
                w = w[rhs]
                s = s[rhs]
                b = b[rhs]
            return mx.quantized_matmul(x, w, s, b, transpose=trans)

        def gather_qmm(x, w, s, b, lhs, rhs, trans, sort):
            return mx.gather_qmm(
                x,
                w,
                s,
                b,
                transpose=trans,
                lhs_indices=lhs,
                rhs_indices=rhs,
                sorted_indices=sort,
            )

        key = mx.random.key(0)
        k1, k2, k3, k4 = mx.random.split(key, 4)
        dtype = mx.float32

        x = mx.random.normal((16, 1, 256), key=k1).astype(dtype)
        w, s, b = mx.quantize(mx.random.normal((4, 256, 256), key=k2).astype(dtype))
        indices = mx.sort(mx.random.randint(0, 4, shape=(16,), key=k3))
        cotan = mx.random.normal((16, 1, 256), key=k4).astype(dtype)

        (o1,), (dx1, ds1, db1) = mx.vjp(
            lambda x, s, b: gather_qmm_ref(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )
        (o2,), (dx2, ds2, db2) = mx.vjp(
            lambda x, s, b: gather_qmm(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )

        self.assertLess((o1 - o2).abs().max(), 1e-4)
        self.assertTrue(mx.allclose(o1, o2, atol=1e-4))
        self.assertTrue(mx.allclose(dx1, dx2, atol=1e-4))
        tol = 5e-1 if mx.cuda.is_available() else 1e-3
        self.assertTrue(mx.allclose(ds1, ds2, atol=tol))
        self.assertTrue(mx.allclose(db1, db2, atol=tol))

    def test_vjp_scales_biases(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(2, 2, 512))
        w = mx.random.normal(shape=(512, 512))
        wq, s, b = mx.quantize(w, bits=4, group_size=64)

        def mm(sb, x, wq):
            return mx.quantized_matmul(x, wq, *sb, bits=4, group_size=64).sum()

        params = (s, b)
        dparams = mx.grad(mm)((s, b), x, wq)

        eps = 8e-3
        delta = 5e-1 if mx.cuda.is_available() else 2e-2
        # numerical grad check with a few indices
        indices = [(0, 0), (11, 4), (22, 7)]
        for idx in indices:
            for p in [0, 1]:
                params[p][idx] += eps
                out_up = mm(params, x, wq)
                params[p][idx] -= 2 * eps
                out_down = mm(params, x, wq)
                params[p][idx] += eps
                num_ds = (out_up - out_down) / (2 * eps)
                self.assertAlmostEqual(dparams[p][idx], num_ds, delta=delta)

    def test_fp_vjp_scales_throws(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(2, 512))
        w = mx.random.normal(shape=(512, 512))
        for mode in ["mxfp4", "mxfp8", "nvfp4"]:
            wq, s = mx.quantize(w, mode=mode)

            def mm(s, x, wq):
                return mx.quantized_matmul(x, wq, s, mode=mode).sum()

            # Should raise
            with self.assertRaises(ValueError):
                mx.grad(mm)(s, x, wq)

            rhs_indices = mx.array(0)
            with self.assertRaises(ValueError):

                def gmm(s, x, wq):
                    return mx.gather_qmm(
                        x,
                        wq,
                        s,
                        rhs_indices=rhs_indices,
                        mode=mode,
                    ).sum()

                _ds = mx.grad(gmm)(s, x, wq)

    @unittest.skipIf(
        not mx.is_available(mx.gpu), "Global scale is only supported on the GPU"
    )
    def test_gather_qmm_global_scale(self):
        mx.random.seed(0)
        N, K = 128, 256

        def rel_err(got, expected):
            # Per row
            d = mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
            scale = mx.abs(expected.astype(mx.float32))
            axes = tuple(range(1, d.ndim))
            return (d.max(axis=axes) / mx.maximum(scale.max(axis=axes), 1e-20)).max()

        def quantize_experts(w):
            """One tensor scale per expert."""
            E = w.shape[0]
            gs = mx.stack([mx.abs(w[e]).max().astype(mx.float32) for e in range(E)])
            qs = [mx.quantize(w[e], mode="nvfp4", global_scale=gs[e]) for e in range(E)]
            w_hat = mx.stack(
                [
                    mx.dequantize(
                        q, sc, mode="nvfp4", global_scale=gs[e], dtype=w.dtype
                    )
                    for e, (q, sc) in enumerate(qs)
                ]
            )
            return (
                mx.stack([q for q, _ in qs]),
                mx.stack([sc for _, sc in qs]),
                gs,
                w_hat,
            )

        tests = product(
            [4, 128, 234],  # E
            [mx.float32, mx.float16, mx.bfloat16],  # dtype
            [True, False],  # transpose
            [(32, 2, False), (1, 2, False), (256, 4, True)],  # M, B, sort
        )
        for E, dtype, transpose, (M, B, sort) in tests:
            with self.subTest(E=E, dtype=dtype, transpose=transpose, M=M, B=B):
                # Each expert has a different scale, so we multiply by a factor
                # to make the experts have different magnitudes.
                factors = mx.array(
                    [[1, 2, 4, 8][e % 4] for e in range(E)], mx.float32
                ).reshape((E, 1, 1))
                wshape = (E, N, K) if transpose else (E, K, N)
                w = (mx.random.normal(wshape) * factors).astype(dtype)
                wq, s, gs, w_hat = quantize_experts(w)
                self.assertEqual(gs.shape, (E,))

                x = mx.random.normal((B, M, K)).astype(dtype)
                indices = mx.random.randint(0, E, (B,))
                if sort:
                    indices = mx.sort(indices)

                wg = w_hat[indices]
                expected = x @ (wg.swapaxes(-1, -2) if transpose else wg)
                kwargs = dict(
                    rhs_indices=indices,
                    transpose=transpose,
                    mode="nvfp4",
                    sorted_indices=sort,
                )

                out = mx.gather_qmm(x, wq, s, global_scale=gs, **kwargs)
                tol = (
                    1e-5 if dtype == mx.float32 and not mx.cuda.is_available() else 3e-2
                )
                self.assertLess(rel_err(out, expected), tol)

                # Each expert uses its own scale, not a neighbour's
                rotated = mx.concatenate([gs[1:], gs[:1]])
                wrong = mx.gather_qmm(x, wq, s, global_scale=rotated, **kwargs)
                self.assertGreater(rel_err(wrong, expected), 0.5)

        # Repeated and out-of-order experts with explicit lhs indices on the
        # M=1 vector path; each output must pick its own expert's scale. The
        # long K is the decode shape that long-K vector kernels take.
        E = 6
        for K in (K, 2048):
            with self.subTest(K=K):
                w = (mx.random.normal((E, N, K)) / K**0.5).astype(mx.bfloat16)
                wq, s, gs, w_hat = quantize_experts(w)
                indices = mx.array([3, 0, 5, 3, 1, 0], mx.uint32)
                x = (mx.random.normal((2, 1, K)) / K**0.5).astype(mx.bfloat16)
                lhs = mx.array([0, 1, 0, 1, 0, 1], mx.uint32)
                expected = x[lhs] @ w_hat[indices].swapaxes(-1, -2)
                out = mx.gather_qmm(
                    x,
                    wq,
                    s,
                    lhs_indices=lhs,
                    rhs_indices=indices,
                    mode="nvfp4",
                    global_scale=gs,
                )
                self.assertLess(rel_err(out, expected), 3e-2)

    def test_quantize_strided(self):
        N = 64
        mode = "nvfp4"
        w = mx.random.normal(shape=(N, N))
        w_q, scales = mx.quantize(w, mode="nvfp4")

        scales = mx.broadcast_to(mx.array(56, mx.uint8), scales.shape)
        w_hat = mx.dequantize(w_q, scales, mode=mode)
        expected = mx.dequantize(w_q, mx.contiguous(scales), mode=mode)
        self.assertTrue(mx.allclose(w_hat, expected))

    # The row-stride route (numen-tech/gemma4-qat#1973): under
    # mx.metal._set_quantized_row_stride(True) or MLX_QMM_ROW_STRIDE=1 an affine
    # 4/8-bit quantized_matmul with an explicit bias reads w, scales and biases
    # whose rows are a row stride apart in place ("_rs" kernels) instead of
    # copying them. Every comparison is on bit patterns against mx.contiguous
    # copies of the same views, and the kernel trace proves the route.

    ROW_STRIDE_ENV = "MLX_QMM_ROW_STRIDE"

    @staticmethod
    def is_copy_kernel(name):
        return "_copy" in name

    @staticmethod
    def is_rs_kernel(name):
        return name.endswith("_rs")

    def row_stride_run(self, on, fn):
        # fn()'s result and kernel names with the process flag set to `on` and
        # the variable out of the way.
        saved = os.environ.pop(self.ROW_STRIDE_ENV, None)
        mx.metal._set_quantized_row_stride(on)
        try:
            return self.traced(mx.gpu, fn)
        finally:
            mx.metal._set_quantized_row_stride(False)
            if saved is not None:
                os.environ[self.ROW_STRIDE_ENV] = saved

    @staticmethod
    def row_stride_views(key, dtype, bits, gs, lead, L, W, D, rot):
        # A packed [*lead, L + spare, W] triple and two overlapping D-wide
        # views of its first L rows, columns [0, D) and [rot, W): the K and V
        # views of a compact K=V KV row (W = D + rot).
        rows = mx.random.normal(lead + (L + 96, W), key=key).astype(dtype)
        wq, scales, biases = mx.quantize(rows, group_size=gs, bits=bits)
        mx.eval(wq, scales, biases)
        wpv = 32 // bits  # values per word

        def view(lo, hi):
            return (
                wq[..., :L, lo // wpv : hi // wpv],
                scales[..., :L, lo // gs : hi // gs],
                biases[..., :L, lo // gs : hi // gs],
            )

        return view(0, D), view(rot, W)

    @staticmethod
    def packed(triple):
        out = tuple(mx.contiguous(a) for a in triple)
        mx.eval(*out)
        return out

    def assert_row_stride(self, x, view, transpose, gs, bits, taught=True):
        # The view read in place gives the bits of its packed copy through the
        # same kernels with "_rs" names and no copy; with the switch off it is
        # copied and takes the stock kernels. `taught` False: the route has no
        # twin, so the view is copied with the switch on too. Returns the "_rs"
        # kernels that ran.
        def mm(triple):
            return lambda: mx.quantized_matmul(
                x, *triple, transpose=transpose, group_size=gs, bits=bits
            )

        dense = self.packed(view)
        y_ref, t_ref = self.row_stride_run(True, mm(dense))
        y_on, t_on = self.row_stride_run(True, mm(view))
        y_off, t_off = self.row_stride_run(False, mm(view))
        self.assert_same_bits(y_on, y_ref)
        self.assert_same_bits(y_off, y_ref)
        self.assertFalse(any(map(self.is_rs_kernel, t_ref)), t_ref)
        if view[0].ndim > 2 and view[0].shape[-2] == 1:
            # One row broadcast over the batch dims has row stride 0
            # (broadcast_to), which eval_gpu copies even when the row is
            # packed; the route leaves that alone.
            self.assertEqual(t_on, t_off)
            self.assertEqual(t_on, t_ref)
            return []
        self.assertFalse(any(map(self.is_copy_kernel, t_ref)), t_ref)
        # Off: the stock kernels after one copy per array.
        self.assertEqual([n for n in t_off if not self.is_copy_kernel(n)], t_ref)
        self.assertEqual(sum(map(self.is_copy_kernel, t_off)), 3, t_off)
        if not taught:
            self.assertEqual(t_on, t_off)
            return []
        rs = [n for n in t_on if self.is_rs_kernel(n)]
        self.assertEqual(len(rs), 1, t_on)
        self.assertFalse(any(map(self.is_copy_kernel, t_on)), t_on)
        self.assertEqual([n[:-3] if self.is_rs_kernel(n) else n for n in t_on], t_ref)
        return rs

    @staticmethod
    def row_stride_family(name):
        # "affine_qmv_fast_float16_t_gs_..." -> "qmv_fast"
        return re.match(r"affine_(q[a-z_]+?)_(?:float|bfloat)", name).group(1)

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_row_stride_kv_views(self):
        # The quantized attention pair over a compact K=V KV row at 8-bit g64:
        # 640-value rows (160 words, 10 groups), K = columns [0, 512), V =
        # columns [128, 640); queries [1, Hkv, 8, M, 512] against K transposed,
        # scores [1, Hkv, 8, M, L] against V.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        arch = mx.metal.device_info()["architecture"]
        gen = self._gpu_gen()
        families = set()
        k = mx.random.key(1973)
        cases = [(2, L, M) for L in (1, 63, 640, 1024, 2304) for M in range(1, 9)]
        cases += [(2, L, M) for L in (63, 640, 2304) for M in (13, 33, 512)]
        cases += [(4, L, M) for L in (640, 2304) for M in (1, 2, 4, 8, 33, 512)]
        for Hkv, L, M in cases:
            k, k1, k2, k3 = mx.random.split(k, 4)
            kview, vview = self.row_stride_views(
                k1, mx.float16, 8, 64, (1, Hkv, 1), L, 640, 512, 128
            )
            q = mx.random.normal((1, Hkv, 8, M, 512), key=k2).astype(mx.float16)
            p = mx.random.normal((1, Hkv, 8, M, L), key=k3).astype(mx.float16)
            mx.eval(q, p)
            with self.subTest(Hkv=Hkv, L=L, M=M):
                rs = self.assert_row_stride(q, kview, True, 64, 8)
                rs += self.assert_row_stride(p, vview, False, 64, 8)
                families.update(map(self.row_stride_family, rs))
        nax = any("_nax_" in f + "_" for f in families)
        expected = {"qmv_fast", "qvm", "qvm_split_k"}
        expected |= {"qmm_t_nax", "qmm_n_nax"} if nax else {"qmm_t", "qmm_n"}
        if gen >= 15:
            expected.add("qmv_wide_rr")
        self.assertTrue(expected <= families, (arch, sorted(families)))

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_row_stride_formats(self):
        # 4- and 8-bit at g32/64/128, fp32, fp16 and bf16, batched and 2-D:
        # K % 256 != 0 takes the plain qmv, a transposed 2-D product at or
        # above the qmv batch limit is the split-K qmm, which has no twin.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        families = set()
        k = mx.random.key(73)
        for bits, gs, dtype in product(
            (4, 8), (32, 64, 128), (mx.float32, mx.float16, mx.bfloat16)
        ):
            for lead, W, D, rot, L in (
                ((1, 2, 1), 640, 512, 128, 700),
                ((), 640, 512, 128, 1100),
                ((3,), 512, 384, 128, 72),
                ((), 512, 384, 128, 250),
            ):
                for M in (1, 3, 6, 40):
                    k, k1, k2, k3 = mx.random.split(k, 4)
                    kview, vview = self.row_stride_views(
                        k1, dtype, bits, gs, lead, L, W, D, rot
                    )
                    xl = lead[:-1] + (8,) if len(lead) == 3 else lead
                    q = mx.random.normal(xl + (M, D), key=k2).astype(dtype)
                    p = mx.random.normal(xl + (M, L), key=k3).astype(dtype)
                    mx.eval(q, p)
                    with self.subTest(
                        bits=bits, gs=gs, dtype=dtype, lead=lead, D=D, L=L, M=M
                    ):
                        # A 2-D w folds every leading dim of x into the rows.
                        rows = M * (lead[0] if len(lead) == 1 else 1)
                        splitk = len(lead) == 0 and rows >= self._qmv_batch_limit(D, L)
                        rs = self.assert_row_stride(
                            q, kview, True, gs, bits, taught=not splitk
                        )
                        rs += self.assert_row_stride(p, vview, False, gs, bits)
                        families.update(map(self.row_stride_family, rs))
        self.assertIn("qmv", families)
        if self._gpu_gen() >= 15:
            self.assertIn("qmv_wide", families)

    @staticmethod
    def _qmv_batch_limit(K, N):
        # get_qmv_batch_limit (quantized.cpp) for K, N <= 2048.
        assert K <= 2048 and N <= 2048
        arch = mx.metal.device_info()["architecture"]
        gen = int("".join(c for c in arch.split("_g")[-1] if c.isdigit()))
        ultra = arch[-1] == "d"
        if gen >= 17 and not ultra:
            return 33
        if gen >= 15 and not ultra:
            return 13
        if ultra:
            return 32
        return 14 if gen >= 13 else 18

    @unittest.skipIf(not mx.metal.is_available(), "requires Metal")
    def test_row_stride_copies(self):
        # What the route does not take keeps copying, switch on: a column
        # stride, qmv_quad (K = 64 or 128), 2-bit, an implied (0-d) bias, a
        # bias-free call, an fp mode. And the variable decides alone.
        if mx.default_device() == mx.cpu:
            self.skipTest("Covers GPU kernels only")
        k = mx.random.key(7)
        k1, k2, k3 = mx.random.split(k, 3)
        dtype = mx.float16

        def copies(names):
            return sum(map(self.is_copy_kernel, names))

        def check(x, view, dense, n_copies, transpose=True, **kwargs):
            def mm(triple):
                return lambda: mx.quantized_matmul(
                    x, *triple, transpose=transpose, **kwargs
                )

            y_ref, t_ref = self.row_stride_run(True, mm(dense))
            y_on, t_on = self.row_stride_run(True, mm(view))
            self.assert_same_bits(y_on, y_ref)
            self.assertFalse(any(map(self.is_rs_kernel, t_on)), t_on)
            self.assertEqual(copies(t_on) - copies(t_ref), n_copies, t_on)

        # Evaluated up front: the traces below must hold the product alone.
        x = mx.random.normal((3, 512), key=k2).astype(dtype)
        x32 = x.astype(mx.float32)
        rows = mx.random.normal((200, 1024), key=k1).astype(dtype)
        wq, scales, biases = mx.quantize(rows, group_size=64, bits=8)
        mx.eval(x, x32, wq, scales, biases)

        with self.subTest("column stride"):
            view = (wq[:, ::2], scales[:, ::2], biases[:, ::2])
            check(x, view, self.packed(view), 3, group_size=64, bits=8)

        with self.subTest("qmv_quad"):
            for K in (64, 128):
                view = (wq[:, : K // 4], scales[:, : K // 64], biases[:, : K // 64])
                check(x[:, :K], view, self.packed(view), 3, group_size=64, bits=8)

        with self.subTest("implied bias"):
            factor = mx.array(-128.0, dtype)
            view = (wq[:, :128], scales[:, :8], factor)
            dense = (mx.contiguous(view[0]), mx.contiguous(view[1]), factor)
            mx.eval(*dense)
            check(x, view, dense, 2, group_size=64, bits=8)

        with self.subTest("2-bit"):
            w2, s2, b2 = mx.quantize(rows, group_size=64, bits=2)
            mx.eval(w2, s2, b2)
            view = (w2[:, :32], s2[:, :8], b2[:, :8])
            check(x, view, self.packed(view), 3, group_size=64, bits=2)
            with self.subTest("bias-free"):
                view = (w2[:, :32], s2[:, :8], None)
                dense = (mx.contiguous(w2[:, :32]), mx.contiguous(s2[:, :8]), None)
                mx.eval(dense[0], dense[1])
                check(x[:1], view, dense, 2, group_size=64, bits=2)

        with self.subTest("nvfp4"):
            wf, sf = mx.quantize(rows.astype(mx.float32), mode="nvfp4")[:2]
            mx.eval(wf, sf)
            view = (wf[:, :64], sf[:, :32], None)
            dense = (mx.contiguous(view[0]), mx.contiguous(view[1]), None)
            mx.eval(dense[0], dense[1])
            check(x32, view, dense, 2, group_size=16, bits=4, mode="nvfp4")

        with self.subTest("environment"):
            view = (wq[:, :128], scales[:, :8], biases[:, :8])

            def run(env, flag):
                saved = os.environ.get(self.ROW_STRIDE_ENV)
                if env is None:
                    os.environ.pop(self.ROW_STRIDE_ENV, None)
                else:
                    os.environ[self.ROW_STRIDE_ENV] = env
                mx.metal._set_quantized_row_stride(flag)
                try:
                    _, names = self.traced(
                        mx.gpu,
                        lambda: mx.quantized_matmul(
                            x, *view, transpose=True, group_size=64, bits=8
                        ),
                    )
                finally:
                    mx.metal._set_quantized_row_stride(False)
                    if saved is None:
                        os.environ.pop(self.ROW_STRIDE_ENV, None)
                    else:
                        os.environ[self.ROW_STRIDE_ENV] = saved
                return any(map(self.is_rs_kernel, names))

            self.assertFalse(run(None, False))  # off by default
            self.assertTrue(run(None, True))
            self.assertTrue(run("1", False))
            self.assertFalse(run("0", True))  # the variable decides alone

    def test_quantized_matmul_batched_split_k(self):
        tol = 3e-2 if mx.cuda.is_available() else 1e-3
        K, N = 2048, 512
        w = mx.random.normal(shape=(K, N))
        w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
        w_hat = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)

        shapes = [
            (2, K),
            (2, 1, K),
            (3, 1, K),
            (2, 1, 1, K),
        ]
        for shape in shapes:
            x = mx.random.normal(shape=shape)
            y_q = mx.quantized_matmul(
                x, w_q, scales, biases, transpose=False, group_size=64, bits=4
            )
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), tol)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
