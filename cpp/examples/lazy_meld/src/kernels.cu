/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernels.hpp"

#include <cudf/utilities/error.hpp>

#include <cub/block/block_reduce.cuh>

namespace cudf::experimental::lazy::detail {
namespace {
constexpr int block_size = 256;

__global__ void normalize_mask_kernel(std::uint32_t const* input,
                                      std::int64_t offset,
                                      std::int64_t rows,
                                      std::uint32_t* output)
{
  auto const word = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word * 32 >= rows) { return; }
  auto const source_word = offset / 32 + word;
  auto const shift       = offset % 32;
  auto bits              = input[source_word] >> shift;
  // Do not read past the last source word, even if the allocation has no cuDF padding.
  if (shift != 0 && (source_word + 1) * 32 < offset + rows) {
    bits |= input[source_word + 1] << (32 - shift);
  }
  output[word] = bits;
}

struct partial_sum {
  double value;
  int seen;
};
struct combine {
  __device__ partial_sum operator()(partial_sum lhs, partial_sum rhs) const
  {
    return {lhs.value + rhs.value, lhs.seen | rhs.seen};
  }
};

__global__ void finalize_sum_kernel(double const* partials,
                                    std::uint32_t const* validity,
                                    std::int64_t const* count,
                                    double* output,
                                    bool* output_valid)
{
  using reducer = cub::BlockReduce<partial_sum, block_size>;
  __shared__ typename reducer::TempStorage storage;
  partial_sum local{0.0, 0};
  for (std::int64_t row = threadIdx.x; row < *count; row += blockDim.x) {
    if ((validity[row / 32] >> (row % 32)) & 1u) {
      local.value += partials[row];
      local.seen = 1;
    }
  }
  auto const total = reducer(storage).Reduce(local, combine{});
  if (threadIdx.x == 0) {
    *output       = total.value;
    *output_valid = total.seen != 0;
  }
}
}  // namespace

void normalize_mask(std::uint32_t const* input,
                    std::int64_t offset,
                    std::int64_t rows,
                    std::uint32_t* output,
                    cudaStream_t stream)
{
  if (rows == 0) { return; }
  auto const blocks = static_cast<unsigned>((rows + 32 * block_size - 1) / (32 * block_size));
  normalize_mask_kernel<<<blocks, block_size, 0, stream>>>(input, offset, rows, output);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void finalize_sum(double const* partials,
                  std::uint32_t const* validity,
                  std::int64_t const* count,
                  double* output,
                  bool* output_valid,
                  cudaStream_t stream)
{
  finalize_sum_kernel<<<1, block_size, 0, stream>>>(
    partials, validity, count, output, output_valid);
  CUDF_CUDA_TRY(cudaGetLastError());
}
}  // namespace cudf::experimental::lazy::detail
