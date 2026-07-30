/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_filter_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/mixed_filtered_join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>

#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Retains every build row in the reusable mixed-semi hash set.
 *
 * Build row indices are unique, so using index equality prevents the static set from discarding
 * duplicate equality keys. Probe operations replace this comparator with equality-key and
 * conditional-expression evaluation.
 */
struct distinct_row_index_equality {
  __device__ constexpr bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    return lhs == rhs;
  }
};

using reusable_hash_set_type = mixed_semi_hash_set_type<distinct_row_index_equality>;

template <typename HashTable>
void build_hash_table(table_view const& build,
                      HashTable& hash_table,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream)
{
  auto const rows = cuda::counting_iterator<size_type>{0};

  if (compare_nulls == null_equality::EQUAL or not nullable(build)) {
    hash_table.insert_async(rows, rows + build.num_rows(), stream.value());
    return;
  }

  auto const [row_bitmask, _] =
    cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref());
  hash_table.insert_if_async(rows,
                             rows + build.num_rows(),
                             cuda::counting_iterator<size_type>{0},
                             row_is_valid{static_cast<bitmask_type const*>(row_bitmask.data())},
                             stream.value());
}

template <size_type block_size, bool has_nulls, typename SetRef>
CUDF_KERNEL void __launch_bounds__(block_size)
  mixed_filtered_join_probe(table_device_view probe_conditional,
                            table_device_view build_conditional,
                            row_equality equality_probe,
                            SetRef set_ref,
                            cudf::device_span<bool> keep_mask,
                            ast::detail::expression_device_view expression_data)
{
  extern __shared__ char raw_intermediate_storage[];
  auto* intermediate_storage =
    reinterpret_cast<ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto* thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * expression_data.num_intermediates);

  auto const evaluator = ast::detail::expression_evaluator<has_nulls>{
    probe_conditional, build_conditional, expression_data};
  auto const expression_equal = single_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, false, equality_probe};
  auto const filtered_ref = set_ref.rebind_key_eq(expression_equal);

  auto const start  = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size>();
  for (auto probe_index = start; probe_index < probe_conditional.num_rows();
       probe_index += stride) {
    keep_mask[probe_index] = filtered_ref.contains(probe_index);
  }
}

template <bool has_nulls, typename SetRef>
void launch_probe(table_device_view const& probe_conditional,
                  table_device_view const& build_conditional,
                  row_equality equality_probe,
                  SetRef set_ref,
                  cudf::device_span<bool> keep_mask,
                  ast::detail::expression_device_view expression_data,
                  detail::grid_1d const& config,
                  std::size_t shared_memory_bytes,
                  rmm::cuda_stream_view stream)
{
  mixed_filtered_join_probe<DEFAULT_JOIN_BLOCK_SIZE, has_nulls>
    <<<config.num_blocks, config.num_threads_per_block, shared_memory_bytes, stream.value()>>>(
      probe_conditional, build_conditional, equality_probe, set_ref, keep_mask, expression_data);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace

class mixed_filtered_join {
 public:
  mixed_filtered_join(table_view const& build_equality,
                      null_equality compare_nulls,
                      double load_factor,
                      rmm::cuda_stream_view stream,
                      cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _build_equality{build_equality}, _compare_nulls{compare_nulls}
  {
    CUDF_EXPECTS(_build_equality.num_columns() > 0,
                 "The build equality table must have at least one column.");
    CUDF_EXPECTS(load_factor > 0 and load_factor <= 1,
                 "Invalid load factor: must be greater than 0 and less than or equal to 1.",
                 std::invalid_argument);
    if (_build_equality.num_rows() == 0) { return; }

    _preprocessed_build        = row::equality::preprocessed_table::create(_build_equality, stream);
    auto const build_has_nulls = cudf::nullate::DYNAMIC{cudf::has_nulls(_build_equality)};
    auto const build_hasher =
      row::hash::row_hasher{_preprocessed_build}.device_hasher(build_has_nulls);
    _hash_table.reset(
      new reusable_hash_set_type{{static_cast<std::size_t>(_build_equality.num_rows())},
                                 load_factor,
                                 cuco::empty_key{JoinNoMatch},
                                 distinct_row_index_equality{},
                                 {build_hasher},
                                 {},
                                 {},
                                 rmm::mr::polymorphic_allocator<char>{std::move(mr)},
                                 {stream.value()}});
    build_hash_table(_build_equality, *_hash_table, _compare_nulls, stream);
  }

  std::unique_ptr<rmm::device_uvector<size_type>> semi_anti_join(
    table_view const& probe_equality,
    table_view const& probe_conditional,
    table_view const& build_conditional,
    ast::expression const& binary_predicate,
    join_kind join_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_EXPECTS(join_type == join_kind::LEFT_SEMI_JOIN or join_type == join_kind::LEFT_ANTI_JOIN,
                 "Only left semi and left anti joins are supported.");
    CUDF_EXPECTS(probe_conditional.num_rows() == probe_equality.num_rows(),
                 "The probe conditional and equality tables must have the same number of rows.");
    CUDF_EXPECTS(build_conditional.num_rows() == _build_equality.num_rows(),
                 "The build conditional and equality tables must have the same number of rows.");

    CUDF_EXPECTS(probe_equality.num_columns() == _build_equality.num_columns(),
                 "The probe and build equality tables must have the same number of columns.");
    CUDF_EXPECTS(cudf::have_same_types(probe_equality, _build_equality),
                 "The probe and build equality tables must have the same column types.",
                 cudf::data_type_error);

    auto const has_nulls = cudf::nullate::DYNAMIC{
      cudf::has_nulls(probe_equality) or cudf::has_nulls(_build_equality) or
      binary_predicate.may_evaluate_null(probe_conditional, build_conditional, stream)};
    auto const parser = ast::detail::expression_parser{
      binary_predicate, probe_conditional, build_conditional, has_nulls, stream, mr};
    CUDF_EXPECTS(parser.output_type().id() == type_id::BOOL8,
                 "The expression must produce a boolean output.",
                 cudf::data_type_error);

    if (_build_equality.num_rows() == 0) {
      if (join_type == join_kind::LEFT_ANTI_JOIN) {
        return get_trivial_left_join_indices(probe_conditional, stream, mr).first;
      }
      return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
    }
    if (probe_equality.num_rows() == 0) {
      return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
    }

    auto const probe_conditional_view = table_device_view::create(probe_conditional, stream);
    auto const build_conditional_view = table_device_view::create(build_conditional, stream);
    auto const preprocessed_probe =
      row::equality::preprocessed_table::create(probe_equality, stream);
    auto const equality_probe =
      row::equality::two_table_comparator{preprocessed_probe, _preprocessed_build}.equal_to<false>(
        has_nulls, _compare_nulls);
    auto const probe_hasher = row::hash::row_hasher{preprocessed_probe}.device_hasher(has_nulls);

    auto keep_mask = rmm::device_uvector<bool>(probe_equality.num_rows(), stream);
    detail::grid_1d const config(probe_equality.num_rows(), DEFAULT_JOIN_BLOCK_SIZE);
    auto const shared_memory_bytes = parser.shmem_per_thread * config.num_threads_per_block;
    auto const set_ref = _hash_table->ref(cuco::contains).rebind_hash_function(probe_hasher);
    if (has_nulls) {
      launch_probe<true>(*probe_conditional_view,
                         *build_conditional_view,
                         equality_probe,
                         set_ref,
                         cudf::device_span<bool>{keep_mask},
                         parser.device_expression_data,
                         config,
                         shared_memory_bytes,
                         stream);
    } else {
      launch_probe<false>(*probe_conditional_view,
                          *build_conditional_view,
                          equality_probe,
                          set_ref,
                          cudf::device_span<bool>{keep_mask},
                          parser.device_expression_data,
                          config,
                          shared_memory_bytes,
                          stream);
    }

    auto result =
      std::make_unique<rmm::device_uvector<size_type>>(probe_equality.num_rows(), stream, mr);
    auto const result_end = cudf::detail::copy_if(
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{probe_equality.num_rows()},
      keep_mask.begin(),
      result->begin(),
      [join_type] __device__(bool keep) {
        return keep == (join_type == join_kind::LEFT_SEMI_JOIN);
      },
      stream);
    result->resize(cuda::std::distance(result->begin(), result_end), stream);
    return result;
  }

 private:
  table_view _build_equality;
  null_equality _compare_nulls;
  std::shared_ptr<row::equality::preprocessed_table> _preprocessed_build;
  std::unique_ptr<reusable_hash_set_type> _hash_table;
};

}  // namespace detail

mixed_filtered_join::~mixed_filtered_join() = default;

mixed_filtered_join::mixed_filtered_join(table_view const& build_equality,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         rmm::cuda_stream_view stream,
                                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<detail::mixed_filtered_join>(
      build_equality, compare_nulls, load_factor, stream, std::move(mr))}
{
}

mixed_filtered_join::mixed_filtered_join(table_view const& build_equality,
                                         null_equality compare_nulls,
                                         rmm::cuda_stream_view stream,
                                         cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : mixed_filtered_join(
      build_equality, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr))
{
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_filtered_join::semi_join(
  table_view const& probe_equality,
  table_view const& probe_conditional,
  table_view const& build_conditional,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->semi_anti_join(probe_equality,
                               probe_conditional,
                               build_conditional,
                               binary_predicate,
                               join_kind::LEFT_SEMI_JOIN,
                               stream,
                               mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> mixed_filtered_join::anti_join(
  table_view const& probe_equality,
  table_view const& probe_conditional,
  table_view const& build_conditional,
  ast::expression const& binary_predicate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->semi_anti_join(probe_equality,
                               probe_conditional,
                               build_conditional,
                               binary_predicate,
                               join_kind::LEFT_ANTI_JOIN,
                               stream,
                               mr);
}

}  // namespace cudf
