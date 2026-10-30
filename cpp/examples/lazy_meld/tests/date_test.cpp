/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "q6_host_binding.hpp"

#include <cudf/experimental/execution/cpu.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ex   = cudf::experimental::execution;
namespace lazy = cudf::experimental::lazy;
using ex::col;
using ex::date;
using ex::lit;
void require(bool b, char const* message)
{
  if (!b) throw std::runtime_error(message);
}
template <class F>
void rejects(F f)
{
  try {
    f();
  } catch (std::invalid_argument const&) {
    return;
  }
  throw std::runtime_error("invalid date/schema/binding was accepted");
}
bool valid(ex::host_column const& c, std::size_t i)
{
  return c.validity.empty() || ((c.validity[i / 8] >> (i % 8)) & 1u);
}
void equal(ex::host_table const& a, ex::host_table const& b)
{
  require(a.num_rows() == b.num_rows() && a.columns.size() == b.columns.size(),
          "date result shape mismatch");
  for (std::size_t c = 0; c < a.columns.size(); ++c) {
    auto const& x = a.columns[c];
    auto const& y = b.columns[c];
    require(x.values.type() == y.values.type(), "date result dtype mismatch");
    for (std::size_t i = 0; i < a.num_rows(); ++i) {
      require(valid(x, i) == valid(y, i), "date result validity mismatch");
      if (!valid(x, i)) continue;
      if (x.values.type() == ex::data_type::date32)
        require(x.values.as<ex::date32>()[i] == y.values.as<ex::date32>()[i],
                "date payload mismatch");
      else
        require(std::abs(x.values[i] - y.values[i]) <= 1e-10 * std::max(1.0, std::abs(y.values[i])),
                "sum mismatch");
    }
  }
}
ex::cpu_program compile(ex::relation const& q, ex::policy p = ex::policy::automatic)
{
  return ex::compile_cpu(ex::plan(q, {ex::target::cpu, p}));
}
int main()
try {
  require(date(1970, 1, 1).days == 0 && date(1969, 12, 31).days == -1, "date epoch mismatch");
  require(date(1994, 1, 1).days == 8766 && date(1995, 1, 1).days == 9131,
          "Q6 date literal encoding wrong");
  require(date(2000, 3, 1).days - date(2000, 2, 28).days == 2, "leap year calculation wrong");
  rejects([] { date(1900, 2, 29); });
  rejects([] { date(1994, 2, 29); });
  rejects([] { date(2000, 0, 1); });
  rejects([] { date(2000, 1, 0); });
  rejects([] { date(0, 1, 1); });
  rejects([] { date(10000, 1, 1); });
  auto typed = ex::input({ex::data_type::float64, ex::data_type::date32});
  rejects([&] { typed.filter(col(1) >= lit(8766.0)); });
  rejects([&] { typed.project({col(1) + col(1)}); });
  rejects([&] { typed.sum(col(1)); });
  rejects([] { lazy::scan({lazy::data_type::date32}).sum(0); });
  auto q6    = ex::from_lazy(lazy_test::q6());
  auto fused = compile(q6), reference = compile(q6, ex::policy::kernels_only);
  auto data = lazy_test::boundary_data();
  lazy_test::host_binding bound{data};
  auto actual = fused.run(bound.columns);
  require(valid(actual.columns[0], 0) && actual.columns[0].values[0] == 37.0,
          "full Q6 golden result wrong");
  equal(actual, reference.run(bound.columns));
  // A numeric-only implementation would incorrectly retain these rows.
  for (auto d : {date(1993, 12, 31),
                 date(1995, 1, 1),
                 ex::date32{std::numeric_limits<std::int32_t>::min()},
                 ex::date32{std::numeric_limits<std::int32_t>::max()}}) {
    std::fill(data[3].dates.begin(), data[3].dates.end(), d);
    auto result = fused.run(bound.columns);
    require(!valid(result.columns[0], 0), "out-of-range shipdates survived");
    equal(result, reference.run(bound.columns));
  }
  auto many = lazy_test::make_data(4099, true);
  lazy_test::host_binding many_bound{many};
  equal(fused.run(many_bound.columns), reference.run(many_bound.columns));
  for (auto offset : {1, 7, 8, 31, 32, 33}) {
    lazy_test::host_binding slice{many, static_cast<std::size_t>(offset), 257};
    equal(fused.run(slice.columns), reference.run(slice.columns));
  }
  auto wrong = many_bound.columns;
  wrong[3]   = wrong[0];
  rejects([&] { fused.submit(wrong); });
  // Dates must also survive materialized boundaries, projections, and grouped output as DATE32.
  ex::host_table rows{{{{1, 2, 3, 4, 5}, {}},
                       {std::vector<ex::date32>{date(1994, 1, 1),
                                                date(1993, 12, 31),
                                                date(1994, 1, 1),
                                                date(1995, 1, 1),
                                                date(1969, 12, 31)},
                        {0x0f}}}};
  auto values    = rows.view();
  auto projects  = typed.project({col(1), col(0), lit(date(2000, 2, 29))}).sort_by({0});
  auto projected = compile(projects).run(values);
  equal(projected, compile(projects, ex::policy::kernels_only).run(values));
  require(projected.columns[0].values.type() == ex::data_type::date32 &&
            projected.columns[2].values.type() == ex::data_type::date32,
          "DATE32 became FLOAT64");
  auto groups = typed.group_by({col(1)}).sum(col(0)).sort_by({0});
  equal(compile(groups).run(values), compile(groups, ex::policy::kernels_only).run(values));
  rows.columns[1].validity.clear();
  values = rows.view();
  equal(compile(groups).run(values), compile(groups, ex::policy::kernels_only).run(values));
  auto filtered = typed.filter(col(1) >= lit(date(1994, 1, 1))).sort_by({1}).sum(col(0));
  require(compile(filtered).run(values).columns[0].values[0] == 8,
          "date filter across boundary wrong");
  auto future = fused.submit(many_bound.columns);
  equal(future.get(), reference.run(many_bound.columns));
  std::cout << "DATE32/full Q6=PASS (golden revenue=37; exact boundaries, null dates, signed day "
               "counts, typed materialization/grouping, sliced masks, async)\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
