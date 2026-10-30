/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "graph_detail.hpp"

#include <algorithm>
#include <sstream>

namespace cudf::experimental::execution {
namespace {
using namespace detail;
using lazy::detail::field;
using lazy::detail::infer_expression;
using lazy::detail::plan_access;
using kind = frame_node::kind;

char const* name(kind k)
{
  switch (k) {
    case kind::scan: return "input";
    case kind::filter: return "filter";
    case kind::project: return "project";
    case kind::sum: return "sum(expression)";
    case kind::group_sum: return "group_by(expressions).sum(expression)";
    case kind::sort: return "stable_sort";
  }
  throw std::logic_error("unknown graph operation");
}
relation aggregate(relation const& input, std::vector<expression> keys, expression value)
{
  auto n = graph_access::node(input);
  if (keys.size() > 64) throw std::invalid_argument("at most 64 grouping keys are supported");
  std::vector<field> schema;
  for (auto const& e : keys) {
    auto f = infer_expression(e, n->schema);
    if (f.predicate) throw std::invalid_argument("grouping keys must be FLOAT64 or DATE32");
    schema.push_back(f);
  }
  auto measure_type = infer_expression(value, n->schema);
  if (measure_type.predicate || measure_type.type != data_type::float64)
    throw std::invalid_argument("SUM requires a FLOAT64 expression");
  auto op = keys.empty() ? kind::sum : kind::group_sum;
  keys.push_back(std::move(value));
  schema.push_back({false, true});
  return graph_access::make(
    std::make_shared<frame_node const>(frame_node{op, n, std::move(keys), std::move(schema)}));
}
}  // namespace

relation::relation(std::shared_ptr<lazy::detail::frame_node const> n) : _node(std::move(n)) {}
std::size_t relation::num_columns() const { return graph_access::node(*this)->schema.size(); }
relation input(std::size_t columns) { return from_lazy(lazy::scan(columns)); }
relation input(std::vector<data_type> types) { return from_lazy(lazy::scan(std::move(types))); }
relation from_lazy(lazy::frame const& f) { return graph_access::make(plan_access::node(f)); }
relation from_lazy(lazy::reduction const& r) { return graph_access::make(plan_access::node(r)); }
relation relation::filter(expression e) const
{
  return from_lazy(plan_access::as_frame(graph_access::node(*this)).filter(std::move(e)));
}
relation relation::project(std::vector<expression> e) const
{
  return from_lazy(plan_access::as_frame(graph_access::node(*this)).project(std::move(e)));
}
relation relation::sum(expression e) const { return aggregate(*this, {}, std::move(e)); }
grouped_relation relation::group_by(std::vector<expression> keys) const
{
  if (keys.empty())
    throw std::invalid_argument("group_by needs keys; use sum for a scalar aggregate");
  if (keys.size() > 64) throw std::invalid_argument("at most 64 grouping keys are supported");
  for (auto const& key : keys)
    if (infer_expression(key, graph_access::node(*this)->schema).predicate)
      throw std::invalid_argument("grouping keys must be FLOAT64 or DATE32");
  return grouped_relation{*this, std::move(keys)};
}
grouped_relation::grouped_relation(relation input, std::vector<expression> keys)
  : _input(std::move(input)), _keys(std::move(keys))
{
}
relation grouped_relation::sum(expression value) const
{
  return aggregate(_input, _keys, std::move(value));
}
relation relation::sort_by(std::vector<std::size_t> columns) const
{
  auto n = graph_access::node(*this);
  if (columns.empty()) throw std::invalid_argument("sort_by needs at least one key");
  for (auto c : columns)
    if (c >= n->schema.size()) throw std::invalid_argument("sort key out of range");
  return graph_access::make(std::make_shared<frame_node const>(
    frame_node{kind::sort, n, {}, n->schema, std::move(columns)}));
}

execution_plan::execution_plan(std::shared_ptr<detail::plan_state const> s) : _state(std::move(s))
{
}
std::vector<fragment> const& execution_plan::fragments() const
{
  return graph_access::state(*this).fragments;
}
plan_options execution_plan::options() const { return graph_access::state(*this).options; }
std::size_t execution_plan::input_columns() const
{
  return graph_access::state(*this).input_columns;
}
std::string execution_plan::explain() const
{
  auto const& s = graph_access::state(*this);
  std::ostringstream out;
  out << "target=" << (s.options.device == target::cpu ? "cpu" : "gpu") << " selection="
      << (s.options.selection == policy::automatic      ? "automatic"
          : s.options.selection == policy::kernels_only ? "kernels_only"
                                                        : "meld_only")
      << " (capability rules; no cost model)\n";
  for (auto const& f : s.fragments) {
    out << "fragment " << f.id << " ["
        << (f.backend == implementation::meld   ? "meld"
            : f.backend == implementation::cudf ? "cudf"
                                                : "host_reference")
        << "]: ";
    for (std::size_t i = 0; i < f.operations.size(); ++i)
      out << (i ? " -> " : "") << f.operations[i];
    out << "\n  " << f.reason << "\n";
  }
  return out.str();
}

execution_plan plan(relation const& graph, plan_options options)
{
  if (options.device != target::cpu && options.device != target::gpu)
    throw std::invalid_argument("unknown execution target");
  if (options.selection != policy::automatic && options.selection != policy::kernels_only &&
      options.selection != policy::meld_only)
    throw std::invalid_argument("unknown selection policy");
  std::vector<node_ptr> chain;
  for (auto n = graph_access::node(graph); n; n = n->input)
    chain.push_back(n);
  std::reverse(chain.begin(), chain.end());
  if (chain.size() == 1) throw std::invalid_argument("graph must contain an operation");
  auto s           = std::make_shared<plan_state>();
  s->options       = options;
  s->input_columns = chain.front()->schema.size();
  auto library =
    options.device == target::cpu ? implementation::host_reference : implementation::cudf;
  auto add = [&](std::vector<node_ptr> const& ops, implementation backend, std::string reason) {
    if (ops.empty()) return;
    if (options.selection == policy::meld_only && backend != implementation::meld)
      throw std::invalid_argument(std::string("meld_only cannot execute ") + name(ops.front()->op) +
                                  " on this target: " + reason);
    region r{backend,
             {std::make_shared<frame_node const>(
               frame_node{kind::scan, {}, {}, ops.front()->input->schema})}};
    fragment info{s->fragments.size(), backend, {}, std::move(reason)};
    for (auto const& n : ops) {
      auto copy  = *n;
      copy.input = r.nodes.back();
      r.nodes.push_back(std::make_shared<frame_node const>(std::move(copy)));
      info.operations.push_back(name(n->op));
    }
    s->regions.push_back(std::move(r));
    s->fragments.push_back(std::move(info));
  };
  std::vector<node_ptr> pending;
  auto flush = [&] {
    if (options.device == target::cpu && options.selection != policy::kernels_only) {
      add(pending,
          implementation::meld,
          "Fused row expressions; materialize only the region output.");
    } else {
      for (auto const& n : pending)
        add({n},
            library,
            "Materialize operator output; this GPU Meld adapter requires a terminal scalar SUM.");
    }
    pending.clear();
  };
  for (std::size_t i = 1; i < chain.size(); ++i) {
    auto n = chain[i];
    if (options.selection == policy::kernels_only) {
      add({n},
          library,
          n->op == kind::group_sum
            ? "Materialize key/measure expressions, then group; null keys included."
            : "Forced operator-by-operator execution; materialize output.");
    } else if (n->op == kind::sort) {
      flush();
      add({n}, library, "Stable ordering requires all rows; materialize the sorted output.");
    } else if (n->op == kind::group_sum && options.device == target::gpu) {
      flush();
      add({n},
          library,
          "Materialize key/measure expressions for cuDF groupby; inline device-expression hook not "
          "implemented.");
    } else {
      pending.push_back(n);
      if (n->op == kind::sum || n->op == kind::group_sum) {
        add(pending,
            implementation::meld,
            n->op == kind::group_sum ? "Fuse filters, keys and measure into Meld hash aggregation; "
                                       "no per-input-row key columns; materialize group results."
                                     : "Fuse row expressions into scalar SUM; no filter/project "
                                       "columns; materialize aggregate result.");
        pending.clear();
      }
    }
  }
  flush();
  return graph_access::make(std::move(s));
}
}  // namespace cudf::experimental::execution
