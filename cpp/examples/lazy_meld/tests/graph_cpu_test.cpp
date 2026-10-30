/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/experimental/execution/cpu.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace ex = cudf::experimental::execution;
using ex::col;
using ex::lit;
void require(bool b, char const* m)
{
  if (!b) throw std::runtime_error(m);
}
template <class F>
void rejects(F f)
{
  try {
    f();
  } catch (std::invalid_argument const&) {
    return;
  }
  throw std::runtime_error("bad input accepted");
}
bool valid(ex::host_column const& c, std::size_t i)
{
  return c.validity.empty() || ((c.validity[i / 8] >> (i % 8)) & 1u);
}
void equal(ex::host_table const& a, ex::host_table const& b)
{
  require(a.columns.size() == b.columns.size() && a.num_rows() == b.num_rows(), "shape mismatch");
  for (std::size_t c = 0; c < a.columns.size(); ++c)
    for (std::size_t i = 0; i < a.num_rows(); ++i) {
      require(valid(a.columns[c], i) == valid(b.columns[c], i), "validity mismatch");
      if (!valid(a.columns[c], i)) continue;
      auto x = a.columns[c].values[i], y = b.columns[c].values[i];
      require((std::isnan(x) && std::isnan(y)) || x == y ||
                (std::isfinite(x) && std::isfinite(y) &&
                 std::abs(x - y) <= 1e-11 * std::max(1.0, std::abs(y))),
              "value mismatch");
    }
}
ex::cpu_program compile(ex::relation const& q, ex::policy p = ex::policy::automatic)
{
  return ex::compile_cpu(ex::plan(q, {ex::target::cpu, p}));
}
int main()
try {
  auto source    = ex::input(3);
  auto grouped   = source.group_by({col(0) + col(1)}).sum(col(2) * lit(2));
  auto sorted    = grouped.sort_by({0});
  auto automatic = compile(sorted), reference = compile(sorted, ex::policy::kernels_only);
  auto nan = std::numeric_limits<double>::quiet_NaN();
  ex::host_table fixture{{{{1, 1, 2, 2, 0, 0, -0.0, nan, nan}, {0xef, 0x01}},
                          {{1, 2, 1, 2, 1, 0, 0, 1, 1}, {}},
                          {{10, 20, 30, 0, 50, 6, 7, 8, 0}, {0xf7, 0x00}}}};
  auto view   = fixture.view();
  auto result = automatic.run(view);
  equal(result, reference.run(view));
  // Independent golden groups: 0=>26, 2=>20, 3=>100, 4=>NULL, NaN=>16, NULL=>100.
  ex::host_table expected{{{{0, 2, 3, 4, nan, 0}, {0x1f}}, {{26, 20, 100, 0, 16, 100}, {0x37}}}};
  equal(result, expected);
  auto fused = compile(grouped, ex::policy::meld_only);
  ex::execution_metrics metrics;
  require(fused.run(view, &metrics).num_rows() == 6, "fused group count wrong");
  require(metrics.meld_fragments == 1 && metrics.intermediate_bytes == 0,
          "fused group materialized a stage");
  auto chained = compile(sorted.sum(col(1)));
  auto total   = chained.run(view, &metrics);
  require(total.columns[0].values[0] == 262 && valid(total.columns[0], 0),
          "group->sort->sum wrong");
  require(metrics.meld_fragments == 2 && metrics.reference_fragments == 1,
          "mixed execution was not dispatched");
  auto multi = source.filter(col(0) < lit(17))
                 .project({col(2), col(1), col(0)})
                 .group_by({col(1) + lit(1), col(2)})
                 .sum(col(0))
                 .sort_by({0, 1});
  auto multi_native = compile(multi), multi_ref = compile(multi, ex::policy::kernels_only);
  auto rows_graph = source.filter((col(0) < lit(17)) && (col(1) >= lit(2)))
                      .project({col(2) * lit(1.25), lit(7)})
                      .filter(col(0) > lit(0));
  auto rows_native = compile(rows_graph), rows_ref = compile(rows_graph, ex::policy::kernels_only);
  for (auto n : {0, 1, 7, 8, 9, 31, 32, 33, 257, 4099}) {
    ex::host_table t;
    for (int c = 0; c < 3; ++c) {
      ex::host_column column{std::vector<double>(n), std::vector<std::uint8_t>((n + 7) / 8)};
      for (int i = 0; i < n; ++i) {
        column.values[i] = c == 0 ? i % 29 : c == 1 ? i % 7 : i * 0.125;
        if ((i + c * 3) % 17 != 0) column.validity[i / 8] |= std::uint8_t(1u << (i % 8));
      }
      t.columns.push_back(std::move(column));
    }
    auto input = t.view();
    equal(automatic.run(input), reference.run(input));
    equal(multi_native.run(input), multi_ref.run(input));
    equal(rows_native.run(input), rows_ref.run(input));
    for (auto offset : {1, 7, 8, 31, 32, 33})
      if (offset <= n) {
        auto sliced = input;
        for (auto& c : sliced) {
          c.values          = c.values.subspan(offset);
          c.validity_offset = offset;
        }
        equal(multi_native.run(sliced), multi_ref.run(sliced));
      }
  }
  ex::host_table distinct;
  for (int c = 0; c < 3; ++c)
    distinct.columns.push_back({std::vector<double>(2003), {}});
  for (std::size_t i = 0; i < 2003; ++i) {
    distinct.columns[0].values[i] = static_cast<double>(i);
    distinct.columns[2].values[i] = 1;
  }
  auto distinct_view = distinct.view();
  require(automatic.run(distinct_view).num_rows() == 2003, "high-cardinality keys were lost");
  equal(automatic.run(distinct_view), reference.run(distinct_view));
  fixture.columns[0].values[8] = std::bit_cast<double>(std::uint64_t{0x7ff8000000001234});
  fixture.columns[2].values[0] = nan;
  fixture.columns[2].values[1] = std::numeric_limits<double>::infinity();
  view                         = fixture.view();
  equal(automatic.run(view), reference.run(view));
  // Empty/all-filtered scalar and grouped output, including an aggregate used as another input.
  auto empty     = source.filter(col(0) < lit(-100));
  auto empty_sum = compile(empty.sum(col(1))).run(view);
  require(empty_sum.num_rows() == 1 && !valid(empty_sum.columns[0], 0), "empty SUM contract wrong");
  require(compile(empty.group_by({col(0)}).sum(col(1))).run(view).num_rows() == 0,
          "empty groups exist");
  for (auto& c : fixture.columns)
    c.validity = std::vector<std::uint8_t>(2, 0);
  view            = fixture.view();
  auto null_group = automatic.run(view);
  require(null_group.num_rows() == 1 && !valid(null_group.columns[0], 0) &&
            !valid(null_group.columns[1], 0),
          "all-null group wrong");
  auto a = automatic.submit(view), b = automatic.submit(view);
  equal(a.get(), b.get());
  auto future = compile(sorted).submit(view);
  equal(future.get(), null_group);
  auto bad      = view;
  bad[0].values = bad[0].values.first(1);
  rejects([&] { automatic.run(bad); });
  bad             = view;
  bad[0].validity = bad[0].validity.first(1);
  rejects([&] { automatic.run(bad); });
  bad                    = view;
  bad[0].validity_offset = std::numeric_limits<std::size_t>::max();
  rejects([&] { automatic.submit(bad); });
  rejects([&] { automatic.run({}); });
  rejects([&] { ex::compile_cpu(ex::plan(sorted)); });
  std::cout
    << "graph CPU execution=PASS (expression keys, null/NaN/zero keys, golden groups, "
       "mixed dispatch, rows, empty, masks/slices, schema, reuse, async, invalid bindings)\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
