/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compiler.hpp"
#include "graph_detail.hpp"
#include "q6_workload.hpp"

#include <cudf/experimental/lazy/plan.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace lazy = cudf::experimental::lazy;

int main()
try {
  using lazy::col;
  using lazy::lit;
  auto make_query = [](double limit) {
    return lazy::scan(3)
      .filter((col(0) < lit(limit)) && (col(2) >= lit(0.05)))
      .filter(col(2) <= lit(0.07))
      .project({col(1) * col(2)})
      .sum(0);
  };
  lazy::detail::compiled_artifact first, second;
  std::string error;
  auto dates = lazy_test::q6().to_meld_ir();
  if (dates.find("#meld.date<\"1994-01-01\">") == std::string::npos ||
      dates.find("!meld.nullable<!meld.date>") == std::string::npos)
    throw std::runtime_error("Q6 lost typed date literals or input schema");
  if (!lazy::detail::compile_meld(dates, "sm_80", second, error)) throw std::runtime_error(error);
  if (!lazy::detail::compile_meld(make_query(24).to_meld_ir(), "sm_80", first, error) ||
      !lazy::detail::compile_meld(make_query(42).to_meld_ir(), "sm_80", second, error)) {
    throw std::runtime_error(error);
  }
  if (first.ptx.find(".entry meld_unit_0") == std::string::npos) {
    throw std::runtime_error("Meld did not produce the expected kernel entry point");
  }
  if (first.ptx != second.ptx || first.literals == second.literals || first.literals.empty()) {
    throw std::runtime_error("literal binding and compiled-code reuse disagree");
  }
  auto projected = lazy::scan(2).project({col(1), lit(1.25)}).filter(col(1) > lit(0)).sum(1);
  if (!lazy::detail::compile_meld(projected.to_meld_ir(), "sm_80", second, error)) {
    throw std::runtime_error(error);
  }
  if (!lazy::detail::compile_meld(lazy::scan(1).sum(0).to_meld_ir(), "sm_80", second, error)) {
    throw std::runtime_error(error);
  }
  namespace ex = cudf::experimental::execution;
  auto mixed =
    ex::input(3).group_by({col(0) + col(1)}).sum(col(2)).sort_by({0}).sum(col(1) * lit(2));
  auto planned = ex::plan(mixed);
  for (auto const& region : ex::detail::graph_access::state(planned).regions) {
    if (region.backend != ex::implementation::meld) continue;
    if (!lazy::detail::compile_meld(
          lazy::detail::emit_meld_ir(region.nodes), "sm_80", second, error))
      throw std::runtime_error(error);
  }
  if (lazy::detail::compile_meld("this is not Meld IR", "sm_80", second, error) || error.empty()) {
    throw std::runtime_error("compiler diagnostics were lost");
  }
  std::cout << "Meld parsing, lowering, PTX generation and literal binding passed\n";
  return 0;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
