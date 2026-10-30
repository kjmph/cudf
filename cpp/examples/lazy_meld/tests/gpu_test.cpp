/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "q6_workload.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/experimental/lazy/meld.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lazy   = cudf::experimental::lazy;
using clock_type = std::chrono::steady_clock;
constexpr cudf::data_type float64{cudf::type_id::FLOAT64};

namespace {
void require(bool condition, char const* message)
{
  if (!condition) { throw std::runtime_error(message); }
}
template <typename F>
void rejects(F operation)
{
  try {
    operation();
  } catch (std::invalid_argument const&) {
    return;
  }
  throw std::runtime_error("unsupported input was accepted");
}

struct stream_owner {
  cudaStream_t value{};
  stream_owner() { CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~stream_owner() { cudaStreamDestroy(value); }
  stream_owner(stream_owner const&)            = delete;
  stream_owner& operator=(stream_owner const&) = delete;
  cuda::stream_ref ref() const { return cuda::stream_ref{value}; }
};

struct resource_scope {
  cuda::mr::any_resource<cuda::mr::device_accessible> previous;
  explicit resource_scope(rmm::device_async_resource_ref resource)
    : previous(cudf::set_current_device_resource(resource))
  {
  }
  ~resource_scope() { cudf::set_current_device_resource(std::move(previous)); }
};

using lazy_test::host_table;
using lazy_test::make_data;
using lazy_test::q6;
using lazy_test::reference;

std::unique_ptr<cudf::table> upload(host_table const& host,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (std::size_t c = 0; c < host.size(); ++c) {
    auto const& input = host[c];
    auto const rows   = static_cast<cudf::size_type>(host[0].values.size());
    auto column       = cudf::make_fixed_width_column(
      c == 3 ? cudf::data_type{cudf::type_id::TIMESTAMP_DAYS} : float64,
      rows,
      input.valid.empty() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_VALID,
      stream,
      mr);
    if (rows != 0) {
      void* destination  = c == 3
                             ? static_cast<void*>(column->mutable_view().data<cudf::timestamp_D>())
                             : static_cast<void*>(column->mutable_view().data<double>());
      void const* source = c == 3 ? static_cast<void const*>(input.dates.data())
                                  : static_cast<void const*>(input.values.data());
      CUDF_CUDA_TRY(cudaMemcpyAsync(destination,
                                    source,
                                    rows * (c == 3 ? sizeof(lazy::date32) : sizeof(double)),
                                    cudaMemcpyDefault,
                                    stream.get()));
    }
    if (!input.valid.empty()) {
      std::vector<cudf::bitmask_type> mask((rows + 31) / 32, 0);
      cudf::size_type nulls{};
      for (cudf::size_type row = 0; row < rows; ++row) {
        if (input.valid[row]) {
          mask[row / 32] |= cudf::bitmask_type{1} << (row % 32);
        } else {
          ++nulls;
        }
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().null_mask(),
                                    mask.data(),
                                    mask.size() * sizeof(cudf::bitmask_type),
                                    cudaMemcpyDefault,
                                    stream.get()));
      // Test-data upload is a boundary, outside all timed regions. Keep host mask storage alive.
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      column->set_null_count(nulls);
    }
    columns.push_back(std::move(column));
  }
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return std::make_unique<cudf::table>(std::move(columns));
}

void check(cudf::scalar const& result, std::optional<double> expected, cuda::stream_ref stream)
{
  require(result.is_valid(stream) == expected.has_value(), "SUM validity disagrees with reference");
  if (!expected) { return; }
  auto value = static_cast<cudf::numeric_scalar<double> const&>(result).value(stream);
  if (std::isnan(*expected)) {
    require(std::isnan(value), "valid NaN did not propagate");
  } else if (std::isinf(*expected)) {
    require(value == *expected, "infinity did not propagate");
  } else {
    require(std::abs(value - *expected) <= 1e-10 * std::max(1.0, std::abs(*expected)),
            "SUM payload disagrees with reference");
  }
}

// Retain the eager path's literal scalars and AST between runs too. Do not penalize it for
// reconstructing scalar allocations that the lazy compiled program already retains.
class eager_query {
 public:
  eager_query(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    : _limit(24.0, true, stream, mr),
      _low(0.05, true, stream, mr),
      _high(0.07, true, stream, mr),
      _from(cudf::timestamp_D{cudf::duration_D{8766}}, true, stream, mr),
      _to(cudf::timestamp_D{cudf::duration_D{9131}}, true, stream, mr)
  {
  }

  std::unique_ptr<cudf::scalar> run(cudf::table_view input,
                                    bool jit,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
  {
    auto compute = [&](cudf::table_view table, cudf::ast::expression const& expr) {
      return jit ? cudf::compute_column_jit(table, expr, stream, mr)
                 : cudf::compute_column(table, expr, stream, mr);
    };
    auto mask = compute(input, _predicate);
    // Only the two columns needed downstream are compacted.
    auto selected = cudf::apply_retention_mask(
      cudf::table_view{{input.column(1), input.column(2)}}, mask->view(), stream, mr);
    auto values = compute(selected->view(), _product);
    return cudf::reduce(
      values->view(), *cudf::make_sum_aggregation<cudf::reduce_aggregation>(), float64, stream, mr);
  }

 private:
  using op = cudf::ast::ast_operator;
  cudf::numeric_scalar<double> _limit, _low, _high;
  cudf::timestamp_scalar<cudf::timestamp_D> _from, _to;
  cudf::ast::literal _limit_expr{_limit}, _low_expr{_low}, _high_expr{_high};
  cudf::ast::literal _from_expr{_from}, _to_expr{_to};
  cudf::ast::column_reference _quantity{0}, _discount{2}, _selected_price{0}, _selected_discount{1};
  cudf::ast::column_reference _shipdate{3};
  cudf::ast::operation _small{op::LESS, _quantity, _limit_expr};
  cudf::ast::operation _above{op::GREATER_EQUAL, _discount, _low_expr};
  cudf::ast::operation _below{op::LESS_EQUAL, _discount, _high_expr};
  cudf::ast::operation _first_and{op::NULL_LOGICAL_AND, _small, _above};
  cudf::ast::operation _numeric{op::NULL_LOGICAL_AND, _first_and, _below};
  cudf::ast::operation _after{op::GREATER_EQUAL, _shipdate, _from_expr};
  cudf::ast::operation _before{op::LESS, _shipdate, _to_expr};
  cudf::ast::operation _dates{op::NULL_LOGICAL_AND, _after, _before};
  cudf::ast::operation _predicate{op::NULL_LOGICAL_AND, _numeric, _dates};
  cudf::ast::operation _product{op::MUL, _selected_price, _selected_discount};
};

void correctness(lazy::meld_program const& program,
                 eager_query& eager,
                 cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr)
{
  auto verify = [&](host_table const& host) {
    auto input    = upload(host, stream, mr);
    auto expected = reference(host, 0, host[0].values.size());
    auto pending  = program.submit(input->view(), stream, mr);
    pending.wait();
    require(pending.ready(), "completed result is not ready");
    check(pending.result(), expected, stream);
    check(*eager.run(input->view(), false, stream, mr), expected, stream);
    check(*eager.run(input->view(), true, stream, mr), expected, stream);
  };
  for (std::size_t rows : {0, 1, 31, 32, 33, 255, 256, 257, 4099}) {
    verify(make_data(rows, false));
    verify(make_data(rows, true));
  }
  auto boundaries = lazy_test::boundary_data();
  verify(boundaries);
  auto boundary_input  = upload(boundaries, stream, mr);
  auto boundary_result = program.submit(boundary_input->view(), stream, mr);
  check(boundary_result.result(), 37.0, stream);
  auto none = make_data(257, false);
  std::fill(none[0].values.begin(), none[0].values.end(), 100.0);
  verify(none);
  for (auto column : {0, 1, 2, 3}) {
    auto all_null = make_data(257, true);
    std::fill(all_null[column].valid.begin(), all_null[column].valid.end(), false);
    verify(all_null);
  }
  auto special         = make_data(257, false);
  special[3].dates[5]  = lazy::date(1994, 6, 15);
  special[1].values[5] = std::numeric_limits<double>::quiet_NaN();
  verify(special);
  special[1].values[5] = std::numeric_limits<double>::infinity();
  verify(special);
  special[1].values[5] = 1.0;
  special[2].values[5] = std::numeric_limits<double>::quiet_NaN();
  verify(special);

  auto host  = make_data(1031, true);
  auto input = upload(host, stream, mr);
  for (cudf::size_type offset : {1, 7, 8, 31, 32, 33}) {
    auto sliced  = cudf::slice(input->view(), {offset, input->num_rows()}, stream).front();
    auto pending = program.submit(sliced, stream, mr);
    check(pending.result(), reference(host, offset, host[0].values.size()), stream);
  }
  // Reuse one program across overlapping executions, including on a second nondefault stream.
  stream_owner other;
  auto a = program.submit(input->view(), stream, mr);
  auto b = program.submit(input->view(), other.ref(), mr);  // Upload already completed.
  a.wait_on(other.ref());
  check(a.result(), reference(host, 0, host[0].values.size()), other.ref());
  check(b.result(), reference(host, 0, host[0].values.size()), other.ref());

  // An execution retains its compiled module even after the program handle is destroyed.
  auto retained = [&] { return lazy::compile(q6()).submit(input->view(), stream, mr); }();
  auto scalar   = retained.take_result();
  check(*scalar, reference(host, 0, host[0].values.size()), stream);

  rejects([&] { program.submit(cudf::table_view{{input->get_column(0).view()}}, stream, mr); });
  auto wrong_type = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1031, cudf::mask_state::UNALLOCATED, stream, mr);
  rejects([&] {
    program.submit(cudf::table_view{{wrong_type->view(),
                                     input->get_column(1).view(),
                                     input->get_column(2).view(),
                                     input->get_column(3).view()}},
                   stream,
                   mr);
  });
  std::cout << "GPU checks passed: eager/JIT/CPU parity, nulls, empty, NaN/Inf, slices, reuse, "
               "streams, lifetime\n";
}

template <typename Run>
void measure(char const* name,
             Run run,
             cuda::stream_ref stream,
             rmm::device_async_resource_ref upstream,
             int iterations)
{
  // Warm both compilation/cache paths and allocator state before collecting samples.
  for (int i = 0; i < 3; ++i) {
    auto output = run(upstream);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  std::vector<double> wall_ms, gpu_ms;
  std::size_t peak_bytes{};
  for (int i = 0; i < iterations; ++i) {
    rmm::mr::statistics_resource_adaptor stats{upstream};
    auto mr = rmm::device_async_resource_ref{stats};
    resource_scope current{mr};  // Include cuDF's internal temporary allocations too.
    cudaEvent_t begin{}, end{};
    CUDF_CUDA_TRY(cudaEventCreate(&begin));
    CUDF_CUDA_TRY(cudaEventCreate(&end));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    auto const start = clock_type::now();
    CUDF_CUDA_TRY(cudaEventRecord(begin, stream.get()));
    {
      auto output = run(mr);
      CUDF_CUDA_TRY(cudaEventRecord(end, stream.get()));
      CUDF_CUDA_TRY(cudaEventSynchronize(end));
    }  // Include output/scratch destruction in wall time for both implementations.
    wall_ms.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - start).count());
    float elapsed{};
    CUDF_CUDA_TRY(cudaEventElapsedTime(&elapsed, begin, end));
    gpu_ms.push_back(elapsed);
    peak_bytes = std::max(peak_bytes, static_cast<std::size_t>(stats.get_bytes_counter().peak));
    CUDF_CUDA_TRY(cudaEventDestroy(begin));
    CUDF_CUDA_TRY(cudaEventDestroy(end));
  }
  std::sort(wall_ms.begin(), wall_ms.end());
  std::sort(gpu_ms.begin(), gpu_ms.end());
  std::cout << name << ",wall_median_ms=" << wall_ms[wall_ms.size() / 2]
            << ",stream_median_ms=" << gpu_ms[gpu_ms.size() / 2]
            << ",peak_incremental_device_bytes=" << peak_bytes << '\n';
}
}  // namespace

int main(int argc, char** argv)
try {
  bool const benchmark = argc >= 2 && std::string{argv[1]} == "--benchmark";
  if (argc > 1 && !benchmark) {
    throw std::invalid_argument("usage: lazy_gpu_test [--benchmark [rows [iterations]]]");
  }
  auto const rows_arg  = argc >= 3 ? std::stoll(argv[2]) : 1'000'000;
  int const iterations = argc >= 4 ? std::stoi(argv[3]) : 20;
  require(rows_arg > 0 && rows_arg <= std::numeric_limits<cudf::size_type>::max() && iterations > 0,
          "invalid benchmark rows or iteration count");
  rmm::mr::cuda_async_memory_resource allocator;
  auto mr = rmm::device_async_resource_ref{allocator};
  resource_scope current{mr};
  stream_owner owned_stream;
  auto stream  = owned_stream.ref();
  auto start   = clock_type::now();
  auto program = lazy::compile(q6());
  std::cout << "cold_meld_compile_and_load_ms="
            << std::chrono::duration<double, std::milli>(clock_type::now() - start).count() << '\n';
  eager_query eager(stream, mr);
  correctness(program, eager, stream, mr);
  if (benchmark) {
    int device{};
    cudaDeviceProp properties{};
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, device));
    std::cout << "gpu=" << properties.name << ",rows=" << rows_arg << ",iterations=" << iterations
              << '\n';
    for (bool nullable : {false, true}) {
      auto host    = make_data(static_cast<std::size_t>(rows_arg), nullable);
      auto input   = upload(host, stream, mr);
      auto pending = program.submit(input->view(), stream, mr);
      check(pending.result(), reference(host, 0, host[0].values.size()), stream);
      std::cout << "nullable=" << nullable << ",lazy_scratch_bytes=" << pending.scratch_bytes()
                << '\n';
      // Check benchmark data as well as the smaller edge cases before reporting timings.
      check(*eager.run(input->view(), false, stream, mr),
            reference(host, 0, host[0].values.size()),
            stream);
      check(*eager.run(input->view(), true, stream, mr),
            reference(host, 0, host[0].values.size()),
            stream);
      measure(
        "eager_ast",
        [&](auto resource) { return eager.run(input->view(), false, stream, resource); },
        stream,
        mr,
        iterations);
      measure(
        "eager_jit",
        [&](auto resource) { return eager.run(input->view(), true, stream, resource); },
        stream,
        mr,
        iterations);
      measure(
        "lazy_meld",
        [&](auto resource) { return program.submit(input->view(), stream, resource); },
        stream,
        mr,
        iterations);
    }
  }
  return 0;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
