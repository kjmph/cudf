/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/experimental/lazy/plan.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <optional>
#include <vector>

// The CPU and GPU executables share the graph, dataset and independent reference computation.
namespace lazy_test {
struct host_column {
  std::vector<double> values;
  std::vector<bool> valid;  // Empty means no mask, rather than an all-valid allocated mask.
  // Only the fourth column uses this buffer. It is physically DATE32, never encoded as doubles.
  std::vector<cudf::experimental::lazy::date32> dates;
};
using host_table = std::array<host_column, 4>;

inline std::vector<cudf::experimental::lazy::data_type> q6_schema()
{
  using T = cudf::experimental::lazy::data_type;
  return {T::float64, T::float64, T::float64, T::date32};
}

inline host_table make_data(std::size_t rows, bool nullable)
{
  host_table result;
  for (std::size_t c = 0; c < result.size(); ++c) {
    if (c == 3)
      result[c].dates.resize(rows);
    else
      result[c].values.resize(rows);
    if (nullable) { result[c].valid.resize(rows, true); }
    for (std::size_t i = 0; i < rows; ++i) {
      if (c == 3) {
        // Mixed years and exact boundary days; no prefiltering by shipdate.
        result[c].dates[i] = {static_cast<std::int32_t>(
          (std::chrono::sys_days{std::chrono::year{1993} / 1 / 1}).time_since_epoch().count() +
          i % 1096)};
      } else
        result[c].values[i] = c == 0   ? 1 + i % 50
                              : c == 1 ? 10.0 + (i % 1000) * 0.25
                                       : static_cast<double>(i % 11) / 100.0;
      if (nullable) { result[c].valid[i] = (i + c * 3) % 17 != 0; }
    }
  }
  return result;
}

/** Independent hand-calculated fixture: qualifying revenues are 5 + 14 + 18 = 37. */
inline host_table boundary_data()
{
  auto result = make_data(10, true);
  auto days   = [](int y, unsigned m, unsigned d) {
    return cudf::experimental::lazy::date32{static_cast<std::int32_t>(
      std::chrono::sys_days{std::chrono::year{y} / m / d}.time_since_epoch().count())};
  };
  for (std::size_t row = 0; row < 10; ++row) {
    for (auto& column : result)
      column.valid[row] = true;
    result[0].values[row] = 23;
    result[1].values[row] = 100;
    result[2].values[row] = 0.06;
    result[3].dates[row]  = days(1994, 6, 15);
  }
  result[3].dates[0]  = days(1993, 12, 31);  // numeric predicates pass; date must reject.
  result[3].dates[1]  = days(1994, 1, 1);
  result[2].values[1] = 0.05;
  result[3].dates[2]  = days(1994, 12, 31);
  result[2].values[2] = 0.07;
  result[1].values[2] = 200;
  result[3].dates[3]  = days(1995, 1, 1);  // exclusive upper date boundary.
  result[0].values[4] = 24;
  result[2].values[5] = 0.049;
  result[2].values[6] = 0.071;
  result[1].values[7] = 300;
  result[3].valid[8]  = false;
  result[1].valid[9]  = false;
  return result;
}

inline std::optional<double> reference(host_table const& input, std::size_t begin, std::size_t end)
{
  std::optional<double> total;
  auto valid = [&](std::size_t c, std::size_t row) {
    return input[c].valid.empty() || input[c].valid[row];
  };
  for (auto row = begin; row < end; ++row) {
    if (!valid(0, row) || !valid(1, row) || !valid(2, row) || !valid(3, row)) { continue; }
    double const quantity = input[0].values[row], price = input[1].values[row],
                 discount = input[2].values[row];
    // Independent calendar calculation: no use of the wrapper's date()/literal implementation.
    auto ship = std::chrono::sys_days{std::chrono::days{input[3].dates[row].days}};
    if (ship >= std::chrono::sys_days{std::chrono::year{1994} / 1 / 1} &&
        ship < std::chrono::sys_days{std::chrono::year{1995} / 1 / 1} && quantity < 24 &&
        discount >= 0.05 && discount <= 0.07) {
      total = total.value_or(0.0) + price * discount;
    }
  }
  return total;
}

inline cudf::experimental::lazy::reduction q6()
{
  namespace lazy = cudf::experimental::lazy;
  using lazy::col;
  using lazy::date;
  using lazy::lit;
  return lazy::scan(q6_schema())
    .filter((col(3) >= lit(date(1994, 1, 1))) && (col(3) < lit(date(1995, 1, 1))) &&
            (col(0) < lit(24)) && (col(2) >= lit(0.05)) && (col(2) <= lit(0.07)))
    .project({col(1) * col(2)})
    .sum(0);
}
}  // namespace lazy_test
