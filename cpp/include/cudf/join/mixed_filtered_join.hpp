/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 */

namespace detail {
class mixed_filtered_join;
}  // namespace detail

/**
 * @brief Reusable mixed semi/anti join that builds an equality index from the right table.
 *
 * The equality build table is supplied at construction time. Subsequent `semi_join` and
 * `anti_join` calls may probe the index with different left tables, right conditional views, and
 * predicates without rebuilding it.
 *
 * A row matches when its equality columns match and `binary_predicate` evaluates to true for the
 * corresponding pair of conditional rows. A predicate result of null does not constitute a match.
 *
 * @note The `mixed_filtered_join` object must not outlive the table viewed by `build_equality`,
 * else behavior is undefined.
 * @note The probe functions are thread-safe and may execute concurrently after work submitted to
 * the construction stream is visible to their streams.
 * @note All NaNs are considered equal.
 */
class mixed_filtered_join {
 public:
  mixed_filtered_join() = delete;
  ~mixed_filtered_join();
  mixed_filtered_join(mixed_filtered_join const&)            = delete;
  mixed_filtered_join(mixed_filtered_join&&)                 = delete;
  mixed_filtered_join& operator=(mixed_filtered_join const&) = delete;
  mixed_filtered_join& operator=(mixed_filtered_join&&)      = delete;

  /**
   * @brief Constructs a reusable mixed join from a right-side equality table.
   *
   * @throw cudf::logic_error If `build_equality` has no columns.
   *
   * @param build_equality The right table containing equality join columns
   * @param compare_nulls Whether null equality-key values compare equal
   * @param stream CUDA stream used to construct the equality index
   */
  mixed_filtered_join(table_view const& build_equality,
                      null_equality compare_nulls  = null_equality::EQUAL,
                      rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Returns left row indices having at least one right row for which equality and the
   * conditional predicate match.
   *
   * @throw cudf::data_type_error If `binary_predicate` produces a non-boolean result.
   * @throw cudf::logic_error If `probe_equality` and `probe_conditional` have different row counts.
   * @throw cudf::logic_error If `build_conditional` and the equality build table have different row
   * counts.
   * @throw cudf::logic_error If the probe and build equality tables have different column counts.
   * @throw cudf::data_type_error If the probe and build equality tables have different column
   * types.
   *
   * @param probe_equality The left table containing equality join columns
   * @param probe_conditional The left table containing columns referenced by `binary_predicate`
   * @param build_conditional The right table containing columns referenced by `binary_predicate`
   * @param binary_predicate The condition evaluated for equality-matching row pairs
   * @param stream CUDA stream used for device operations
   * @param mr Device memory resource used to allocate the returned indices
   * @return Left row indices having at least one matching right row
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> semi_join(
    table_view const& probe_equality,
    table_view const& probe_conditional,
    table_view const& build_conditional,
    ast::expression const& binary_predicate,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns left row indices having no right row for which equality and the conditional
   * predicate match.
   *
   * @throw cudf::data_type_error If `binary_predicate` produces a non-boolean result.
   * @throw cudf::logic_error If `probe_equality` and `probe_conditional` have different row counts.
   * @throw cudf::logic_error If `build_conditional` and the equality build table have different row
   * counts.
   * @throw cudf::logic_error If the probe and build equality tables have different column counts.
   * @throw cudf::data_type_error If the probe and build equality tables have different column
   * types.
   *
   * @param probe_equality The left table containing equality join columns
   * @param probe_conditional The left table containing columns referenced by `binary_predicate`
   * @param build_conditional The right table containing columns referenced by `binary_predicate`
   * @param binary_predicate The condition evaluated for equality-matching row pairs
   * @param stream CUDA stream used for device operations
   * @param mr Device memory resource used to allocate the returned indices
   * @return Left row indices having no matching right row
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> anti_join(
    table_view const& probe_equality,
    table_view const& probe_conditional,
    table_view const& build_conditional,
    ast::expression const& binary_predicate,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<cudf::detail::mixed_filtered_join> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
