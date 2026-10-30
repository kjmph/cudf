/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "q6_host_binding.hpp"

#include <cudf/experimental/execution/cpu.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace ex = cudf::experimental::execution;
namespace {
double number(std::string_view field)
{
  double value;
  auto [end, error] = std::from_chars(field.data(), field.data() + field.size(), value);
  if (error != std::errc{} || end != field.data() + field.size() || !std::isfinite(value))
    throw std::invalid_argument("invalid numeric field in lineitem.tbl");
  return value;
}
int integer(std::string_view field)
{
  int value;
  auto [end, error] = std::from_chars(field.data(), field.data() + field.size(), value);
  if (error != std::errc{} || end != field.data() + field.size())
    throw std::invalid_argument("invalid date field");
  return value;
}
lazy_test::host_table load_lineitem(std::string const& path)
{
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open lineitem.tbl: " + path);
  lazy_test::host_table data;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) throw std::invalid_argument("empty line in lineitem.tbl");
    std::vector<std::string_view> fields;
    std::string_view rest{line};
    while (true) {
      auto pipe = rest.find('|');
      fields.push_back(rest.substr(0, pipe));
      if (pipe == std::string_view::npos) break;
      rest.remove_prefix(pipe + 1);
    }
    if (fields.size() == 17 && fields.back().empty()) fields.pop_back();
    if (fields.size() != 16)
      throw std::invalid_argument("lineitem.tbl requires 16 pipe-delimited fields");
    auto ship = fields[10];
    if (ship.size() != 10 || ship[4] != '-' || ship[7] != '-')
      throw std::invalid_argument("shipdate must be YYYY-MM-DD");
    auto day =
      ex::date(integer(ship.substr(0, 4)), integer(ship.substr(5, 2)), integer(ship.substr(8, 2)));
    data[0].values.push_back(number(fields[4]));
    data[1].values.push_back(number(fields[5]));
    data[2].values.push_back(number(fields[6]));
    data[3].dates.push_back(day);
  }
  if (file.bad()) throw std::runtime_error("error reading lineitem.tbl");
  return data;
}
void check(ex::host_table const& actual, std::optional<double> expected)
{
  if (actual.num_rows() != 1 || actual.columns.size() != 1)
    throw std::runtime_error("Q6 result shape mismatch");
  auto const& c = actual.columns[0];
  bool valid    = c.validity.empty() || (c.validity[0] & 1u);
  if (valid != expected.has_value())
    throw std::runtime_error("Q6 result validity does not match independent reference");
  if (!expected) return;
  double value = c.values[0];
  bool equal   = std::isnan(*expected) ? std::isnan(value)
                 : std::isinf(*expected)
                   ? value == *expected
                   : std::isfinite(value) &&
                     std::abs(value - *expected) <= 1e-10 * std::max(1.0, std::abs(*expected));
  if (!equal) throw std::runtime_error("Q6 does not match independent reference");
}
}  // namespace
int main(int argc, char** argv)
try {
  bool file = argc > 1 && std::string_view{argv[1]} == "--lineitem";
  if ((file && argc != 3) || (!file && argc > 2))
    throw std::invalid_argument(
      "usage: q6_demo [synthetic_rows] | q6_demo --lineitem path/to/lineitem.tbl");
  std::size_t count = file ? 0 : argc == 2 ? std::stoull(argv[1]) : 1000000;
  if (!file && (count == 0 || count > 10000000))
    throw std::invalid_argument("synthetic_rows must be in [1,10000000]");
  auto data = file ? load_lineitem(argv[2]) : lazy_test::make_data(count, false);
  lazy_test::host_binding input{data};
  auto query    = ex::from_lazy(lazy_test::q6());
  auto physical = ex::plan(query, {ex::target::cpu});
  std::cout << physical.explain() << std::setprecision(15);
  auto start   = std::chrono::steady_clock::now();
  auto program = ex::compile_cpu(physical);
  std::cout
    << "cold_compile_ms="
    << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
    << '\n';
  auto expected = lazy_test::reference(data, 0, data[0].values.size());
  for (int i = 0; i < 3; ++i)
    check(program.run(input.columns), expected);
  std::vector<double> samples;
  ex::host_table result;
  for (int i = 0; i < 20; ++i) {
    start  = std::chrono::steady_clock::now();
    result = program.run(input.columns);
    samples.push_back(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    check(result, expected);
  }
  check(program.submit(input.columns).get(), expected);
  std::sort(samples.begin(), samples.end());
  std::cout
    << "query=full_q6\nexecution_target=cpu\nshipdate_type=date32\n"
    << "shipdate_range=[1994-01-01,1995-01-01)\ndiscount_range=[0.05,0.07]\nquantity_lt=24\n"
    << "dataset=" << (file ? "lineitem.tbl" : "synthetic") << "\nrows=" << data[0].values.size()
    << "\niterations=20\nresult=";
  if (expected)
    std::cout << result.columns[0].values[0];
  else
    std::cout << "null";
  std::cout << "\nreference_result=";
  if (expected)
    std::cout << *expected;
  else
    std::cout << "null";
  std::cout << "\nwarm_median_ms=" << (samples[9] + samples[10]) / 2 << "\ncorrectness=PASS\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
