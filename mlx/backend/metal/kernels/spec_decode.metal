// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Greedy speculative-decode verify, one thread per row: n_accepted is the
// matching draft prefix length, committed is that prefix then the target token.
[[kernel]] void spec_decode_verify(
    const device int* draft [[buffer(0)]],
    const device int* target [[buffer(1)]],
    device int* n_accepted [[buffer(2)]],
    device int* committed [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& B [[buffer(5)]],
    uint b [[thread_position_in_grid]]) {
  if (b >= uint(B)) {
    return;
  }
  int row = int(b);
  const device int* d = draft + row * K;
  const device int* t = target + row * (K + 1);
  device int* c = committed + row * (K + 1);

  int n = K;
  for (int j = 0; j < K; ++j) {
    if (d[j] != t[j]) {
      n = j;
      break;
    }
  }
  n_accepted[row] = n;
  for (int j = 0; j < K + 1; ++j) {
    c[j] = (j < n) ? d[j] : ((j == n) ? t[n] : 0);
  }
}
