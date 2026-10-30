/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "q6_host_binding.hpp"

#include <cudf/experimental/lazy/cpu.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace lazy   = cudf::experimental::lazy;
using clock_type = std::chrono::steady_clock;

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
  throw std::runtime_error("invalid host binding was accepted");
}

using lazy_test::host_binding;

void check(lazy::host_scalar actual, lazy::host_scalar expected)
{
  require(actual.has_value() == expected.has_value(), "CPU SUM validity disagrees with reference");
  if (!expected) { return; }
  if (std::isnan(*expected)) {
    require(std::isnan(*actual), "CPU SUM lost valid NaN");
  } else if (std::isinf(*expected)) {
    require(*actual == *expected, "CPU SUM lost infinity");
  } else {
    require(std::abs(*actual - *expected) <= 1e-10 * std::max(1.0, std::abs(*expected)),
            "CPU SUM payload disagrees with reference");
  }
}

void correctness(lazy::cpu_program const& program)
{
  auto verify = [&](lazy_test::host_table const& data) {
    host_binding binding{data};
    auto expected = lazy_test::reference(data, 0, data[0].values.size());
    check(program.run(binding.columns), expected);
    check(program.submit(binding.columns).get(), expected);
  };
  for (std::size_t rows : {0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 4099}) {
    verify(lazy_test::make_data(rows, false));
    verify(lazy_test::make_data(rows, true));
  }
  auto boundaries = lazy_test::boundary_data();
  host_binding boundary_binding{boundaries};
  check(program.run(boundary_binding.columns), 37.0);
  check(lazy_test::reference(boundaries, 0, 10), 37.0);
  auto data = lazy_test::make_data(257, true);
  std::fill(data[0].values.begin(), data[0].values.end(), 100.0);
  verify(data);
  for (std::size_t column : {0, 1, 2, 3}) {
    data = lazy_test::make_data(257, true);
    std::fill(data[column].valid.begin(), data[column].valid.end(), false);
    verify(data);
  }
  data             = lazy_test::make_data(257, false);
  data[3].dates[5] = lazy::date(1994, 6, 15);
  for (double special :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    data[1].values[5] = special;
    verify(data);
  }
  data[1].values[5] = 1.0;
  data[2].values[5] = std::numeric_limits<double>::quiet_NaN();
  verify(data);

  data = lazy_test::make_data(1031, true);
  for (std::size_t offset : {1, 7, 8, 31, 32, 33}) {
    for (std::size_t count : {0, 1, 7, 8, 257}) {
      host_binding slice{data, offset, count};
      check(program.run(slice.columns), lazy_test::reference(data, offset, offset + count));
    }
  }
  host_binding binding{data};
  auto first  = program.submit(binding.columns);
  auto second = program.submit(binding.columns);
  check(first.get(), lazy_test::reference(data, 0, data[0].values.size()));
  check(second.get(), lazy_test::reference(data, 0, data[0].values.size()));
  auto retained = lazy::compile(lazy_test::q6(), lazy::cpu).submit(binding.columns);
  check(retained.get(), lazy_test::reference(data, 0, data[0].values.size()));

  // Exercise generated CPU code after a projection changes column positions and nullability.
  using lazy::col;
  using lazy::lit;
  auto projected = lazy::compile(
    lazy::scan(lazy_test::q6_schema()).project({col(2), lit(1.25)}).filter(col(1) > lit(0)).sum(1),
    lazy::cpu);
  check(projected.run(binding.columns), 1.25 * data[0].values.size());
  auto null_total = lazy::compile(lazy::scan(lazy_test::q6_schema()).sum(0), lazy::cpu);
  auto empty      = lazy_test::make_data(0, false);
  host_binding empty_binding{empty};
  check(null_total.run(empty_binding.columns), std::nullopt);

  rejects([&] { program.run(std::span{binding.columns}.first(2)); });
  auto malformed = binding.columns;
  malformed[3]   = binding.columns[0];
  rejects([&] { program.run(malformed); });
  malformed           = binding.columns;
  malformed[1].values = malformed[1].values.first(10);
  rejects([&] { program.submit(malformed); });
  malformed             = binding.columns;
  malformed[0].validity = malformed[0].validity.first(1);
  rejects([&] { program.run(malformed); });
  malformed                    = binding.columns;
  malformed[0].validity_offset = std::numeric_limits<std::size_t>::max();
  rejects([&] { program.run(malformed); });
  std::cout << "cpu_correctness=PASS (full Q6, date boundaries, nulls, empty, NaN/Inf, slices, "
               "reuse, async, typed schema)\n";
}

template <typename Run>
double median_ms(Run run, lazy::host_scalar expected, int iterations)
{
  for (int i = 0; i < 3; ++i) {
    check(run(), expected);
  }
  std::vector<double> samples;
  for (int i = 0; i < iterations; ++i) {
    auto const start  = clock_type::now();
    auto const result = run();
    auto const elapsed =
      std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    check(result, expected);  // Validate every result, outside the timed interval.
    samples.push_back(elapsed);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
}  // namespace

int main(int argc, char** argv)
try {
  bool const benchmark = argc >= 2 && std::string{argv[1]} == "--benchmark";
  if (argc > 4 || (argc > 1 && !benchmark)) {
    throw std::invalid_argument("usage: lazy_cpu_test [--benchmark [rows [iterations]]]");
  }
  auto const rows      = argc >= 3 ? std::stoll(argv[2]) : 1'000'000;
  int const iterations = argc >= 4 ? std::stoi(argv[3]) : 20;
  require(rows > 0 && rows <= std::numeric_limits<std::int32_t>::max() && iterations > 0,
          "invalid benchmark rows or iteration count");
  std::cout << std::setprecision(12);
  auto const query = lazy_test::q6();
  auto const start = clock_type::now();
  auto program     = lazy::compile(query, lazy::cpu);
  std::cout << "cold_meld_cpu_compile_ms="
            << std::chrono::duration<double, std::milli>(clock_type::now() - start).count() << '\n';
  correctness(program);
  if (benchmark) {
    std::cout << "query=full_q6\nshipdate_type=date32\ndataset=synthetic\nexecution_target="
                 "cpu\ncpu_threads_per_execution=1\nrows="
              << rows << "\niterations=" << iterations << '\n';
    for (bool nullable : {false, true}) {
      auto data = lazy_test::make_data(static_cast<std::size_t>(rows), nullable);
      host_binding binding{data};
      auto const expected = lazy_test::reference(data, 0, static_cast<std::size_t>(rows));
      auto const actual   = program.run(binding.columns);
      check(actual, expected);
      std::cout << "nullable=" << std::boolalpha << nullable << "\nresult=";
      if (actual) {
        std::cout << *actual;
      } else {
        std::cout << "null";
      }
      std::cout << "\nreference_result=";
      if (expected) {
        std::cout << *expected;
      } else {
        std::cout << "null";
      }
      std::cout << "\nwarm_meld_cpu_median_ms="
                << median_ms([&] { return program.run(binding.columns); }, expected, iterations)
                << "\nwarm_meld_cpu_submit_wait_median_ms="
                << median_ms(
                     [&] { return program.submit(binding.columns).get(); }, expected, iterations)
                << '\n';
    }
  }
  return 0;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
