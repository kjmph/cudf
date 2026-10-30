/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/lazy/plan.hpp>

#include <stdexcept>
#include <utility>

namespace cudf::experimental::lazy::detail {
enum class operation {
  column,
  literal,
  date_literal,
  add,
  subtract,
  multiply,
  lt,
  lte,
  gt,
  gte,
  equal,
  logical_and
};
struct expression_node {
  operation op;
  std::size_t index{};
  double value{};
  std::shared_ptr<expression_node const> left, right;
  date32 date_value{};
};
struct field {
  bool predicate{false};
  bool nullable{true};
  data_type type{data_type::float64};
};
struct frame_node {
  enum class kind { scan, filter, project, sum, group_sum, sort } op;
  std::shared_ptr<frame_node const> input;
  // group_sum stores key expressions followed by its SUM argument.
  std::vector<expression> expressions;
  std::vector<field> schema;
  std::vector<std::size_t> sort_keys{};
};
using node_ptr = std::shared_ptr<frame_node const>;
struct plan_access {
  static auto const& node(expression const& e)
  {
    if (!e._node) throw std::invalid_argument("moved-from lazy expression");
    return e._node;
  }
  static expression make(expression_node node)
  {
    return expression{std::make_shared<expression_node const>(std::move(node))};
  }
  static node_ptr node(frame const& f)
  {
    f.num_columns();
    return f._node;
  }
  static node_ptr node(reduction const& r)
  {
    r.input_columns();
    return std::make_shared<frame_node const>(
      frame_node{frame_node::kind::sum, r._input, {col(r._column)}, {{false, true}}});
  }
  static frame as_frame(node_ptr n) { return frame{std::move(n)}; }
};
field infer_expression(expression const&, std::vector<field> const&);
// Each region starts with a scan; aggregations, if any, terminate the region.
std::string emit_meld_ir(std::vector<node_ptr> const& chain);
}  // namespace cudf::experimental::lazy::detail
