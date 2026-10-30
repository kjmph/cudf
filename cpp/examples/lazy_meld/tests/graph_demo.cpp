/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/experimental/execution/cpu.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace ex     = cudf::experimental::execution;
using clock_type = std::chrono::steady_clock;
int main(int argc, char** argv)
try {
  using ex::col;
  using ex::lit;
  std::size_t rows = argc > 1 ? std::stoull(argv[1]) : 100000;
  if (rows == 0 || rows > 10000000) throw std::invalid_argument("rows must be in [1, 10000000]");
  auto query = ex::input(3)
                 .filter(col(0) < lit(24))
                 .group_by({col(0) + col(1)})
                 .sum(col(2) * lit(2))
                 .sort_by({0})
                 .sum(col(1));
  auto physical = ex::plan(query, {ex::target::cpu});
  std::cout << physical.explain() << "\nGPU proposal (planning only):\n"
            << ex::plan(query).explain();
  auto start   = clock_type::now();
  auto program = ex::compile_cpu(physical);
  std::cout << "cold_compile_ms="
            << std::chrono::duration<double, std::milli>(clock_type::now() - start).count() << '\n';
  ex::host_table data;
  for (int c = 0; c < 3; ++c)
    data.columns.push_back({std::vector<double>(rows), {}});
  double expected = 0;
  for (std::size_t i = 0; i < rows; ++i) {
    data.columns[0].values[i] = i % 50;
    data.columns[1].values[i] = i % 7;
    data.columns[2].values[i] = (i % 1000) * 0.125;
    if (i % 50 < 24) expected += 2 * data.columns[2].values[i];
  }
  auto input = data.view();
  for (int i = 0; i < 3; ++i)
    program.run(input);
  std::vector<double> samples;
  ex::execution_metrics metrics;
  ex::host_table result;
  for (int i = 0; i < 20; ++i) {
    start  = clock_type::now();
    result = program.run(input, &metrics);
    samples.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - start).count());
    if (result.num_rows() != 1 || result.columns[0].values[0] != expected)
      throw std::runtime_error("result does not match independent reference");
  }
  std::sort(samples.begin(), samples.end());
  std::cout << "execution_target=cpu\nrows=" << rows
            << "\niterations=20\nresult=" << result.columns[0].values[0]
            << "\nreference_result=" << expected << "\nmeld_fragments=" << metrics.meld_fragments
            << "\nhost_reference_fragments=" << metrics.reference_fragments
            << "\nintermediate_logical_bytes=" << metrics.intermediate_bytes
            << "\nwarm_median_ms=" << (samples[9] + samples[10]) / 2 << "\ncorrectness=PASS\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
