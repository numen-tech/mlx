# Copyright © 2023-2026 Apple Inc.

import math
import os
import platform
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

    def _rr_env(self, disabled):
        # MLX_DISABLE_QMV_WIDE_RR is read on every qmv_wide() call (gemma4-qat#313).
        if disabled:
            os.environ["MLX_DISABLE_QMV_WIDE_RR"] = "1"
        else:
            os.environ.pop("MLX_DISABLE_QMV_WIDE_RR", None)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr(self):
        # Row-reuse qmv_wide (gemma4-qat#313): affine 2/4/8-bit, K % 256 == 0, M in
        # [2, vector_limit). These (M, N, B) route to affine_qmv_wide_rr on gen >= 15
        # (2-bit M = 2 stays on qmv); the reference is dequantize-then-matmul, tol as
        # test_qmv_wide. N = 67 is a ragged 8-row tile, M = 9 a token tail (5 + 4),
        # B = 3 the batched variant.
        key = mx.random.key(313)
        k1, k2 = mx.random.split(key)
        for group_size, bits, K in product([32, 64, 128], [2, 4, 8], [256, 1536]):
            for M, N, B, dtype in product(
                [2, 3, 5, 7, 8, 9],
                [256, 67],
                [1, 3],
                [mx.float16, mx.bfloat16, mx.float32],
            ):
                with self.subTest(
                    M=M, N=N, K=K, B=B, group_size=group_size, bits=bits, dtype=dtype
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
                        x, w_q, scales, biases, True, group_size, bits
                    )
                    y_hat = x.astype(mx.float32) @ mx.swapaxes(w_hat, -1, -2)
                    rel = {mx.float16: 2e-2, mx.bfloat16: 5e-2, mx.float32: 1e-3}
                    tol = rel[dtype] * mx.abs(y_hat).max().item() + 1e-3
                    self.assertEqual(y_q.shape, y_hat.shape)
                    err = (y_q.astype(mx.float32) - y_hat).abs().max().item()
                    self.assertLess(err, tol)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr_no_worse_than_wide(self):
        # #178 no-worse rule: on the same weights, rr's error against the fp32
        # reference is at most qmv_wide's plus one ulp of the largest reference
        # value. Same process, two routes.
        self.addCleanup(self._rr_env, False)
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
                self._rr_env(disabled=True)
                y_wide = mx.quantized_matmul(x, w_q, s, b, True, group_size, bits)
                mx.eval(y_wide)
                self._rr_env(disabled=False)
                y_rr = mx.quantized_matmul(x, w_q, s, b, True, group_size, bits)
                mx.eval(y_rr)
                err_wide = (y_wide.astype(mx.float32) - ref).abs().max().item()
                err_rr = (y_rr.astype(mx.float32) - ref).abs().max().item()
                self.assertLessEqual(err_rr, err_wide + ulp)

    @unittest.skipUnless(mx.metal.is_available(), "qmv_wide_rr is a Metal kernel")
    def test_qmv_wide_rr_implied_bias_matches_materialized(self):
        # The 0-d factor must be applied as affine_bias does (f * scale in T), so
        # rr on the implied form is bitwise equal to rr on the materialized biases,
        # including the subnormal-scale fp16 case from
        # test_implied_bias_dtype_promotion.
        for bits, group_size in product([2, 4, 8], [32, 64, 128]):
            for scale_value in (0.05, 2.0**-24):
                with self.subTest(bits=bits, group_size=group_size, scale=scale_value):
                    K, N, M = 512, 72, 5
                    x = mx.random.normal(shape=(M, K), key=mx.random.key(1))
                    x = x.astype(mx.float16)
                    w_q = mx.random.randint(
                        0, 2**bits, (N, K * bits // 32), dtype=mx.uint32
                    )
                    s = mx.full((N, K // group_size), scale_value, mx.float16)
                    f = mx.array(-(2.0 ** (bits - 1)), mx.float16)
                    y_implied = mx.quantized_matmul(
                        x, w_q, s, f, True, group_size, bits
                    )
                    y_full = mx.quantized_matmul(
                        x, w_q, s, s * f, True, group_size, bits
                    )
                    self.assert_bitwise_equal(y_implied, y_full)

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

    def test_implied_bias_gather_qmm(self):
        # gather_qmm has no implied-bias kernels: the op materializes the bias,
        # so the result equals the per-group-bias call bit for bit.
        key = mx.random.key(7)
        k1, k2 = mx.random.split(key)
        lhs = mx.array([0, 1, 1])
        rhs = mx.array([1, 0, 1])
        for dev, dtype, (bits, gs, factor) in product(
            self.implied_bias_devices(),
            self.IMPLIED_BIAS_DTYPES,
            self.IMPLIED_BIAS_CASES,
        ):
            with self.subTest(dev=dev, dtype=dtype, bits=bits, gs=gs):
                x = mx.random.normal(shape=(2, 3, 512), key=k1).astype(dtype)
                w = mx.random.normal(shape=(2, 64, 512), key=k2).astype(dtype)
                w_q, scales, _ = mx.quantize(w, gs, bits)
                full = self.reference_bias(scales, factor)
                f = mx.array(factor, dtype)
                self.assert_implied_bias_matches(
                    mx.gather_qmm,
                    (x, w_q, scales),
                    full,
                    f,
                    lhs,
                    rhs,
                    True,
                    gs,
                    bits,
                    stream=dev,
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
