/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/experimental/lazy/plan.hpp>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace lazy = cudf::experimental::lazy;

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
  throw std::runtime_error("invalid plan was accepted");
}

int main(int argc, char** argv)
try {
  using lazy::col;
  using lazy::lit;
  auto input = lazy::scan(3);
  auto filtered =
    input.filter((col(0) < lit(24.0)) && (col(2) >= lit(0.05))).filter(col(2) <= lit(0.07));
  auto query = filtered.project({col(1) * col(2)}).sum(0);
  auto ir    = query.to_meld_ir();
  if (argc == 2 && std::string{argv[1]} == "--emit-ir") {
    std::cout << ir;
    return 0;
  }
  require(query.input_columns() == 3, "lost scan schema");
  require(ir == query.to_meld_ir(), "IR emission is not deterministic");
  require(ir.find("meld.filter") < ir.find("meld.project"), "operation order changed");
  require(ir.find("meld.project") < ir.find("meld.aggregate"), "SUM is not terminal");
  require(ir.find("0x4038000000000000") != std::string::npos, "literal precision lost");
  // Branching a deferred handle must not mutate the shared input or its other consumers.
  require(input.sum(0).to_meld_ir().find("meld.filter") == std::string::npos,
          "filter mutated its input");
  require(filtered.num_columns() == 3, "project mutated its input");
  auto reordered    = input.project({col(2), lit(1.25)}).filter(col(1) > lit(0)).sum(1);
  auto reordered_ir = reordered.to_meld_ir();
  require(reordered_ir.find("!meld.relation<!meld.nullable<f64>, f64>") != std::string::npos,
          "projection schema/nullability was lost");
  require(reordered_ir.find("meld.measure all @sum(%c1) : (f64)") != std::string::npos,
          "SUM did not bind to the projected schema");
  rejects([] { lazy::scan(0); });
  rejects([&] { input.filter(col(0)); });
  rejects([&] { input.filter(col(3) > lit(0)); });
  rejects([&] { input.project({}); });
  rejects([&] { input.project({col(0) > lit(0)}); });
  rejects([&] { input.project({(col(0) > lit(0)) + lit(1)}); });
  rejects([&] { input.filter(col(0) && col(1)); });
  rejects([&] { input.project({col(1)}).sum(1); });
  rejects([&] { input.project({col(1)}).filter(col(2) > lit(0)); });
  rejects([] { lit(std::numeric_limits<double>::infinity()); });
  std::cout << "lazy graph/schema checks passed\n";
  return 0;
} catch (std::exception const& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
