/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/execution/graph.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::experimental::execution {
namespace detail {
struct gpu_state;
struct gpu_pending;
}  // namespace detail
struct gpu_dispatch_counts {
  std::size_t meld{}, cudf{};
};
/** Owns result and all intermediate resources through completion.
 * Input buffers are borrowed until wait()/ready(). Keep context, stream and memory resource alive.
 * Destruction waits for pending producer work. Downstream consumers must finish before destruction
 * too: wait_on establishes ordering, not shared ownership of output buffers.
 */
class pending_table {
 public:
  ~pending_table();
  pending_table(pending_table&&) noexcept;
  pending_table& operator=(pending_table&&) noexcept;
  pending_table(pending_table const&)            = delete;
  pending_table& operator=(pending_table const&) = delete;
  bool ready() const;
  void wait() const;
  void wait_on(cuda::stream_ref) const;
  cudf::table_view result() const;
  std::unique_ptr<cudf::table> take_result();
  gpu_dispatch_counts dispatched() const;

 private:
  explicit pending_table(std::unique_ptr<detail::gpu_pending>);
  std::unique_ptr<detail::gpu_pending> _state;
  friend class gpu_program;
};
class gpu_program {
 public:
  /** Run on the compiling CUDA context. Establish input stream dependencies before calling.
   * Uses the supplied stream and RMM resource. cuDF operations can synchronize internally for
   * cardinality/validity; this is not a promise of nonblocking host submission. No CPU fallback.
   */
  pending_table submit(cudf::table_view, cuda::stream_ref, rmm::device_async_resource_ref) const;
  std::string explain() const;

 private:
  explicit gpu_program(std::shared_ptr<detail::gpu_state const>);
  std::shared_ptr<detail::gpu_state const> _state;
  friend gpu_program compile_gpu(execution_plan const&);
};
/** Requires a GPU plan. Compiles Meld regions and loads their modules for the current device.
 * cuDF regions use ordinary library/AST kernels. The adapter remains GPU-host validation work.
 */
gpu_program compile_gpu(execution_plan const&);
}  // namespace cudf::experimental::execution
