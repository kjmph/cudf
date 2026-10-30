/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/experimental/execution/graph.hpp>

#include <iostream>
#include <stdexcept>
namespace ex = cudf::experimental::execution;
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
  throw std::runtime_error("invalid graph/selection was accepted");
}
int main()
try {
  using ex::col;
  using ex::lit;
  auto source  = ex::input(3);
  auto grouped = source.filter(col(0) < lit(24)).group_by({col(0) + col(1)}).sum(col(2) * lit(2));
  auto q       = grouped.sort_by({0}).sum(col(1));
  auto cpu     = ex::plan(q, {ex::target::cpu});
  auto gpu     = ex::plan(q, {ex::target::gpu});
  require(cpu.fragments().size() == 3, "CPU phase partition is wrong");
  require(cpu.fragments()[0].backend == ex::implementation::meld &&
            cpu.fragments()[1].backend == ex::implementation::host_reference &&
            cpu.fragments()[2].backend == ex::implementation::meld,
          "CPU mixed execution missing");
  require(gpu.fragments().size() == 4, "GPU phase partition is wrong");
  require(gpu.fragments()[1].backend == ex::implementation::cudf &&
            gpu.fragments()[3].backend == ex::implementation::meld,
          "GPU groupby/SUM selection wrong");
  require(gpu.explain().find("hook not implemented") != std::string::npos, "hidden capability gap");
  auto fused = ex::plan(grouped, {ex::target::cpu, ex::policy::meld_only});
  require(fused.fragments().size() == 1, "group expressions were materialized");
  rejects([&] { ex::plan(q, {ex::target::cpu, ex::policy::meld_only}); });
  rejects([&] { ex::plan(grouped, {ex::target::gpu, ex::policy::meld_only}); });
  rejects([&] { ex::plan(source.project({col(0)}), {ex::target::gpu, ex::policy::meld_only}); });
  auto inherited = ex::from_lazy(cudf::experimental::lazy::scan(3).project({col(2)}).sum(0));
  require(ex::plan(inherited).fragments().size() == 1, "legacy capture did not reach planner");
  require(ex::plan(source.sum(col(0))).fragments()[0].operations.size() == 1,
          "branch mutated source");
  require(ex::plan(q, {ex::target::cpu, ex::policy::kernels_only}).fragments().size() == 4,
          "forced reference did not retain operator boundaries");
  rejects([&] { source.group_by({}).sum(col(0)); });
  rejects([&] { source.group_by({col(3)}).sum(col(0)); });
  rejects([&] { source.group_by({col(0) > lit(0)}).sum(col(1)); });
  rejects([&] { source.sum(col(0) > lit(0)); });
  rejects([&] { source.sort_by({3}); });
  rejects([&] { source.sort_by({}); });
  rejects([&] { ex::plan(source); });
  std::cout << cpu.explain() << gpu.explain() << "graph planning=PASS\n";
} catch (std::exception const& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
