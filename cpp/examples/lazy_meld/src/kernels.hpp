/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cudf::experimental::lazy::detail {
void normalize_mask(std::uint32_t const* input,
                    std::int64_t offset,
                    std::int64_t rows,
                    std::uint32_t* output,
                    cudaStream_t stream);
void finalize_sum(double const* partials,
                  std::uint32_t const* validity,
                  std::int64_t const* count,
                  double* output,
                  bool* output_valid,
                  cudaStream_t stream);
}  // namespace cudf::experimental::lazy::detail
