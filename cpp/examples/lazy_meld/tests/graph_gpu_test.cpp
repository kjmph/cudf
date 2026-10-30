/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "q6_host_binding.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/experimental/execution/cpu.hpp>
#include <cudf/experimental/execution/gpu.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ex = cudf::experimental::execution;
void require(bool b, char const* m)
{
  if (!b) throw std::runtime_error(m);
}
bool valid(ex::host_column const& c, std::size_t i)
{
  return c.validity.empty() || ((c.validity[i / 8] >> (i % 8)) & 1u);
}
std::unique_ptr<cudf::table> upload(ex::host_table const& host,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& c : host.columns) {
    auto column = cudf::make_fixed_width_column(
      cudf::data_type{c.values.type() == ex::data_type::date32 ? cudf::type_id::TIMESTAMP_DAYS
                                                               : cudf::type_id::FLOAT64},
      c.values.size(),
      c.validity.empty() ? cudf::mask_state::UNALLOCATED : cudf::mask_state::ALL_VALID,
      stream,
      mr);
    void* pointer = c.values.type() == ex::data_type::date32
                      ? static_cast<void*>(column->mutable_view().data<cudf::timestamp_D>())
                      : static_cast<void*>(column->mutable_view().data<double>());
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(pointer,
                      c.values.data(),
                      c.values.size() * cudf::experimental::lazy::size_of(c.values.type()),
                      cudaMemcpyDefault,
                      stream.get()));
    if (!c.validity.empty()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().null_mask(),
                                    c.validity.data(),
                                    c.validity.size(),
                                    cudaMemcpyDefault,
                                    stream.get()));
      cudf::size_type nulls = 0;
      for (std::size_t i = 0; i < c.values.size(); ++i)
        nulls += !valid(c, i);
      column->set_null_count(nulls);
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    columns.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}
void compare(cudf::table_view device, ex::host_table const& expected, cuda::stream_ref stream)
{
  require(device.num_columns() == static_cast<cudf::size_type>(expected.columns.size()) &&
            device.num_rows() == static_cast<cudf::size_type>(expected.num_rows()),
          "GPU result shape mismatch");
  for (cudf::size_type i = 0; i < device.num_columns(); ++i) {
    auto column = device.column(i);
    ex::host_column actual{std::vector<double>(device.num_rows()), {}};
    CUDF_CUDA_TRY(cudaMemcpyAsync(actual.values.data(),
                                  column.data<double>(),
                                  actual.values.size() * sizeof(double),
                                  cudaMemcpyDefault,
                                  stream.get()));
    if (column.nullable()) {
      actual.validity.resize((device.num_rows() + 7) / 8);
      CUDF_CUDA_TRY(cudaMemcpyAsync(actual.validity.data(),
                                    column.null_mask(),
                                    actual.validity.size(),
                                    cudaMemcpyDefault,
                                    stream.get()));
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    for (std::size_t row = 0; row < actual.values.size(); ++row) {
      require(valid(actual, row) == valid(expected.columns[i], row), "GPU result null mismatch");
      if (!valid(actual, row)) continue;
      auto a = actual.values[row], b = expected.columns[i].values[row];
      require(a == b || (std::isnan(a) && std::isnan(b)), "GPU result value mismatch");
    }
  }
}
int main()
try {
  using ex::col;
  using ex::lit;
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  auto nan    = std::numeric_limits<double>::quiet_NaN();
  ex::host_table fixture{{{{1, 1, 2, 2, 0, 0, -0.0, nan, nan}, {0xef, 0x01}},
                          {{1, 2, 1, 2, 1, 0, 0, 1, 1}, {}},
                          {{10, 20, 30, 0, 50, 6, 7, 8, 0}, {0xf7, 0x00}}}};
  ex::host_table groups{{{{0, 2, 3, 4, nan, 0}, {0x1f}}, {{26, 20, 100, 0, 16, 100}, {0x37}}}};
  auto device  = upload(fixture, stream, mr);
  auto query   = ex::input(3).group_by({col(0) + col(1)}).sum(col(2) * lit(2)).sort_by({0});
  auto grouped = ex::compile_gpu(ex::plan(query)).submit(device->view(), stream, mr);
  grouped.wait();
  compare(grouped.result(), groups, stream);
  auto mixed = ex::compile_gpu(ex::plan(query.sum(col(1))));
  auto baseline =
    ex::compile_gpu(ex::plan(query.sum(col(1)), {ex::target::gpu, ex::policy::kernels_only}));
  auto result = mixed.submit(device->view(), stream, mr);
  auto eager  = baseline.submit(device->view(), stream, mr);
  result.wait();
  eager.wait();
  ex::host_table total{{{{262}, {}}}};
  compare(result.result(), total, stream);
  compare(eager.result(), total, stream);
  auto counts = result.dispatched();
  require(counts.meld == 1 && counts.cudf == 2, "mixed GPU fragments were not dispatched");
  require(eager.dispatched().meld == 0 && eager.dispatched().cudf == 3,
          "forced cuDF path was not used");
  auto owned = result.take_result();
  compare(owned->view(), total, stream);
  // A compiled program is retained by pending work even after its public handle is destroyed.
  auto future =
    ex::compile_gpu(ex::plan(ex::input(3).sum(col(2)))).submit(device->view(), stream, mr);
  future.wait();
  ex::host_table plain{{{{131}, {}}}};
  compare(future.result(), plain, stream);
  auto full = lazy_test::boundary_data();
  lazy_test::host_binding binding{full};
  ex::host_table full_host;
  for (std::size_t c = 0; c < 4; ++c)
    full_host.columns.push_back(
      {c == 3 ? ex::host_column_values{full[c].dates} : ex::host_column_values{full[c].values},
       binding.masks[c]});
  auto full_device = upload(full_host, stream, mr);
  ex::host_table full_expected{{{{37}, {}}}};
  for (auto policy : {ex::policy::automatic, ex::policy::kernels_only}) {
    auto full_program =
      ex::compile_gpu(ex::plan(ex::from_lazy(lazy_test::q6()), {ex::target::gpu, policy}));
    auto full_result = full_program.submit(full_device->view(), stream, mr);
    full_result.wait();
    compare(full_result.result(), full_expected, stream);
  }
  std::cout << "graph GPU execution=PASS (full Q6 DATE32 predicates, cuDF expression keys, "
               "null/NaN/zero groups, "
               "mixed Meld dispatch, forced cuDF, ownership)\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
