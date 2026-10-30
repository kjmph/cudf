/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/lazy/plan.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf::experimental::execution {
using lazy::col;
using lazy::data_type;
using lazy::date;
using lazy::date32;
using lazy::expression;
using lazy::lit;
namespace detail {
struct graph_access;
struct plan_state;
}  // namespace detail
class grouped_relation;

/** Immutable, single-input operation graph. Columns can be FLOAT64 or DATE32.
 * Expressions refer to the immediate input schema. Construction never reads data or executes work.
 * SUM skips nulls and returns null for no valid values. Group keys include nulls, merge NaNs and
 * signed zero, and have unspecified output order. Expressions use the existing lazy SQL semantics.
 */
class relation {
 public:
  relation filter(expression predicate) const;
  relation project(std::vector<expression> columns) const;
  relation sum(expression value) const;
  grouped_relation group_by(std::vector<expression> keys) const;
  /** Stable ascending sort, nulls last, NaNs after numbers; an explicit execution boundary. */
  relation sort_by(std::vector<std::size_t> columns) const;
  std::size_t num_columns() const;

 private:
  explicit relation(std::shared_ptr<lazy::detail::frame_node const> node);
  std::shared_ptr<lazy::detail::frame_node const> _node;
  friend struct detail::graph_access;
};
/** Deferred grouping specification; no key columns are created by this handle. */
class grouped_relation {
 public:
  /** Output: keys in argument order, followed by one nullable SUM column. */
  relation sum(expression value) const;

 private:
  grouped_relation(relation input, std::vector<expression> keys);
  relation _input;
  std::vector<expression> _keys;
  friend class relation;
};
/** Schema placeholder: nullable FLOAT64 columns. Bind actual buffers at execution. */
relation input(std::size_t columns);
relation input(std::vector<data_type> types);
/** Existing lazy calls and engine-built graphs feed the same planner. */
relation from_lazy(lazy::frame const&);
relation from_lazy(lazy::reduction const&);

enum class target { cpu, gpu };
enum class policy { automatic, kernels_only, meld_only };
enum class implementation { meld, host_reference, cudf };
struct plan_options {
  target device{target::gpu};
  policy selection{policy::automatic};
};
struct fragment {
  std::size_t id{};
  implementation backend{};
  std::vector<std::string> operations;
  std::string reason;
};
/** Immutable physical plan. Planning is independent of LLVM, CUDA, and input data.
 * Rules describe adapter capabilities, not a calibrated cost model. meld_only rejects regions
 * this adapter cannot execute. On CPU, kernels_only selects an untuned reference implementation,
 * never libcudf. GPU planning does not establish GPU availability or successful compilation.
 */
class execution_plan {
 public:
  std::vector<fragment> const& fragments() const;
  std::string explain() const;
  plan_options options() const;
  std::size_t input_columns() const;

 private:
  explicit execution_plan(std::shared_ptr<detail::plan_state const> state);
  std::shared_ptr<detail::plan_state const> _state;
  friend struct detail::graph_access;
};
execution_plan plan(relation const&, plan_options = {});
}  // namespace cudf::experimental::execution
