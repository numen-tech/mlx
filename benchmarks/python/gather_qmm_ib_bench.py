# Copyright © 2026 Apple Inc.

"""Gather quantized matmul, explicit biases vs the implied-bias factor.

The shapes are the Gemma 4 26B MoE experts as mlx-lm's SwitchGLU presents them
(numen-tech/gemma4-qat#1833): E = 128 experts, top-8, 4-bit, group size 32,
fp16, one row per expert; gate/up K 2816 -> N 704, down K 704 -> N 2816. Up to
7 tokens run unsorted (gather_qmv / gather_qmv_fast), 8 tokens and more are
sorted (gather_qmm_rhs from B / E >= 4).

  python gather_qmm_ib_bench.py [--save out.npz] [--check ref.npz]

--save writes the explicit-bias output of every cell as fp16 bit patterns, plus
two routes with no implied-bias kernel (the matrix gather_qmm and gather_qvm);
--check compares them bit for bit with an earlier --save, the proof that a core
change left the explicit-bias path alone.
"""

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

E, TOP_K, BITS, GS = 128, 8, 4, 32
FACTOR = -float(2 ** (BITS - 1))
DTYPE = mx.float16


def make_weights(rng, E, N, K):
    w = mx.array(
        rng.integers(0, 2**32, (E, N, K * BITS // 32), dtype=np.uint64).astype(
            np.uint32
        )
    )
    scales = mx.array(rng.uniform(0.002, 0.02, (E, N, K // GS))).astype(DTYPE)
    biases = scales * mx.array(FACTOR, DTYPE)
    mx.eval(w, scales, biases)
    return w, scales, biases


def gqmm(x, w, scales, b, idx, transpose=True, sorted_indices=False):
    return mx.gather_qmm(
        x,
        w,
        scales,
        b,
        rhs_indices=idx,
        transpose=transpose,
        group_size=GS,
        bits=BITS,
        sorted_indices=sorted_indices,
    )


def gather_sort(x, indices):
    _, m = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    return x.flatten(0, -3)[order // m], indices[order]


def trace(fn):
    start = getattr(mx.metal, "_kernel_trace_start", None)
    if start is None:
        return fn(), []
    start()
    out = fn()
    mx.eval(out)
    return out, mx.metal._kernel_trace_stop()


def timeit(fn, iters):
    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - tic) / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--save")
    p.add_argument("--check")
    args = p.parse_args()

    rng = np.random.default_rng(1833)
    saved = {}
    ref = np.load(args.check) if args.check else None
    f = mx.array(FACTOR, DTYPE)
    print(f"mlx {mx.__version__} on {mx.metal.device_info()['architecture']}")
    print(
        "| layer | K | N | tokens | rows | sorted | kernel (implied) |"
        " explicit us | implied us | implied / explicit |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for layer, K, N in [("gate/up", 2816, 704), ("down", 704, 2816)]:
        w, scales, biases = make_weights(rng, E, N, K)
        for tokens in [1, 3, 5, 8, 64, 512]:
            x = mx.array(rng.standard_normal((tokens, 1, 1, K)) / K**0.5).astype(DTYPE)
            idx = mx.array(
                np.stack(
                    [rng.choice(E, TOP_K, replace=False) for _ in range(tokens)]
                ).astype(np.uint32)
            )
            do_sort = idx.size >= 64
            if do_sort:
                x, idx = gather_sort(x, idx)
            mx.eval(x, idx)

            def run(b):
                return gqmm(x, w, scales, b, idx, sorted_indices=do_sort)

            y_full, _ = trace(lambda: run(biases))
            y_ib, names = trace(lambda: run(f))
            assert mx.array_equal(
                y_full.view(mx.uint16), y_ib.view(mx.uint16)
            ).item(), (layer, tokens)
            name = f"{layer}.t{tokens}"
            saved[name] = np.array(y_full.view(mx.uint16))
            if ref is not None:
                assert np.array_equal(ref[name], saved[name]), name
            kernel = next((n for n in names if "gather_q" in n), "-")
            for _ in range(args.warmup):
                mx.eval(run(biases), run(f))
            full_us, ib_us = [], []
            for _ in range(args.repeats):
                full_us.append(timeit(lambda: run(biases), args.iters))
                ib_us.append(timeit(lambda: run(f), args.iters))
            a, b = statistics.median(full_us), statistics.median(ib_us)
            print(
                f"| {layer} | {K} | {N} | {tokens} | {tokens * TOP_K} | "
                f"{'yes' if do_sort else 'no'} | {kernel} | {a:.1f} | {b:.1f} | "
                f"{b / a:.3f} |"
            )

    # Routes with no implied-bias kernel: explicit biases only, for --save.
    w, scales, biases = make_weights(rng, 4, 512, 512)
    x = mx.array(rng.standard_normal((4, 64, 512)) / 512**0.5).astype(DTYPE)
    idx = mx.array([3, 0, 2, 2], mx.uint32)
    for name, xs, transpose in [("matrix", x, True), ("qvm", x[:, :1], False)]:
        y = gqmm(xs, w, scales, biases, idx, transpose)
        saved[name] = np.array(y.view(mx.uint16))
    if ref is not None:
        for name in ("matrix", "qvm"):
            assert np.array_equal(ref[name], saved[name]), name
        print(
            f"explicit-bias outputs equal {args.check} bit for bit "
            f"({len(saved)} arrays)"
        )
    if args.save:
        np.savez(args.save, **saved)
        print(f"saved {len(saved)} arrays to {args.save}")


if __name__ == "__main__":
    main()
