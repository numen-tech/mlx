// Copyright © 2026 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"

namespace mlx::core {

inline constexpr short get_pack_factor(int bits, int wsize = 8) {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

inline constexpr short get_bytes_per_pack(int bits, int wsize = 8) {
  bool power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

// Implied bias (affine mode): a 0-d `biases` array holds a single factor f
// standing for the per-group bias `scales * T(f)` in the scales' dtype T.
inline bool is_implied_bias(const array& biases) {
  return biases.ndim() == 0;
}

inline bool is_implied_bias(const std::optional<array>& biases) {
  return biases.has_value() && is_implied_bias(*biases);
}

} // namespace mlx::core
