/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/experimental/lazy/plan.hpp>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace cudf::experimental::lazy {
/** Select Meld's native CPU compiler. This header/library has no CUDA or libcudf dependency. */
struct cpu_target {};
inline constexpr cpu_target cpu{};

/**
 * Borrowed typed host memory for one column. `values` is already sliced; validity_offset is the
 * corresponding bit offset into the optional little-endian Arrow/cuDF-style validity bitmap.
 * An empty bitmap means all values are valid. GPU device pointers are not accepted as host memory.
 */
struct host_column_view {
  host_values_view values;
  std::span<std::uint8_t const> validity{};
  std::size_t validity_offset{};
};
using host_table_view = std::span<host_column_view const>;
using host_scalar     = std::optional<double>;

namespace detail {
struct cpu_program_state;
}

/** Reusable native machine code generated from the same lazy graph used by the GPU adapter. */
class cpu_program {
 public:
  /**
   * Execute on the calling CPU thread. Borrows inputs for this call, with no materialized filter
   * or projection columns. Unaligned validity slices need a temporary bitmap. Empty/all-null SUM
   * returns nullopt. Independent calls can run concurrently on the same program.
   */
  host_scalar run(host_table_view input) const;
  /**
   * Execute on a new CPU worker using std::async. Descriptors and the program are retained, but
   * callers must keep input buffers unchanged and alive until completion. Dropping an unfinished
   * future can block. This POC has no persistent worker pool or automatic parallel partitioning.
   */
  std::future<host_scalar> submit(host_table_view input) const;
  std::string const& plan_ir() const;

 private:
  explicit cpu_program(std::shared_ptr<detail::cpu_program_state const> state);
  std::shared_ptr<detail::cpu_program_state const> _state;
  friend cpu_program compile(reduction const& query, cpu_target);
};

/** Compile through Meld's CPU JIT. No GPU initialization, PTX loading or CUDA fallback occurs. */
cpu_program compile(reduction const& query, cpu_target);
}  // namespace cudf::experimental::lazy
