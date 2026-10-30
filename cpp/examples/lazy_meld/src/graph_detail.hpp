/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "plan_nodes.hpp"

#include <cudf/experimental/execution/graph.hpp>

namespace cudf::experimental::execution::detail {
using lazy::detail::frame_node;
using lazy::detail::node_ptr;
struct region {
  implementation backend;
  std::vector<node_ptr> nodes;  // First node is a schema-only scan.
};
struct plan_state {
  plan_options options;
  std::size_t input_columns;
  std::vector<region> regions;
  std::vector<fragment> fragments;
};
struct graph_access {
  static node_ptr node(relation const& r)
  {
    if (!r._node) throw std::invalid_argument("moved-from execution graph");
    return r._node;
  }
  static relation make(node_ptr n) { return relation{std::move(n)}; }
  static execution_plan make(std::shared_ptr<plan_state const> s)
  {
    return execution_plan{std::move(s)};
  }
  static plan_state const& state(execution_plan const& p)
  {
    if (!p._state) throw std::invalid_argument("moved-from execution plan");
    return *p._state;
  }
};
}  // namespace cudf::experimental::execution::detail
