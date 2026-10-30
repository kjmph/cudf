/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/experimental/lazy/plan.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace cudf::experimental::lazy {
namespace detail {
struct program_state;
struct pending_state;
}  // namespace detail

/**
 * Owns the output and launch resources until completion. Inputs are borrowed: the caller must keep
 * their allocations unchanged and alive until ready()/wait() reports completion. Keep the CUDA
 * context, stream and memory resource alive through destruction of this object and its output.
 *
 * Destruction waits for this execution if still pending. Keep this handle to overlap useful work;
 * immediately discarding it is an explicit ownership boundary that can block.
 */
class pending_result {
 public:
  ~pending_result();
  pending_result(pending_result&&) noexcept;
  pending_result& operator=(pending_result&&) noexcept;
  pending_result(pending_result const&)            = delete;
  pending_result& operator=(pending_result const&) = delete;

  bool ready() const;
  void wait() const;
  /**
   * Order a consumer stream after this result without waiting on the host.
   * Keep this handle/output alive until that consumer finishes; this does not transfer ownership.
   */
  void wait_on(cuda::stream_ref consumer) const;
  /** Device-resident cuDF scalar; usable immediately by downstream work on the submission stream.
   */
  cudf::numeric_scalar<double> const& result() const;
  /** Wait, then transfer ownership of the scalar to the caller. */
  std::unique_ptr<cudf::numeric_scalar<double>> take_result();
  /** Device scratch, including argument metadata and any normalized input masks, excluding output.
   */
  std::size_t scratch_bytes() const;

 private:
  explicit pending_result(std::unique_ptr<detail::pending_state> state);
  std::unique_ptr<detail::pending_state> _state;
  friend class meld_program;
};

/** An immutable compiled fragment, reusable across input batches and concurrent submissions. */
class meld_program {
 public:
  /**
   * Enqueue the fused Meld kernel and GPU finalization on `stream`, using `mr` for all device
   * allocations. No input/output payload is copied to the host and no completion wait is issued.
   * Allocation/JIT driver behavior is not a guarantee of nonblocking host API latency.
   *
   * Input schema must match the typed scan: FLOAT64 or DATE32 (cuDF TIMESTAMP_DAYS).
   * Nulls and slices are supported; a nullable slice
   * whose offset is not byte-aligned needs an additional bitmap normalization kernel/allocation.
   * Call on the CUDA context used by compile(). Establish input stream dependencies before calling.
   */
  pending_result submit(cudf::table_view input,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr) const;
  std::string const& plan_ir() const;
  std::string const& ptx() const;

 private:
  explicit meld_program(std::shared_ptr<detail::program_state> state);
  std::shared_ptr<detail::program_state> _state;
  friend meld_program compile(reduction const& query);
};

/** Compile and load for the current CUDA device/context. Cold compilation is deliberately explicit.
 */
meld_program compile(reduction const& query);
}  // namespace cudf::experimental::lazy
