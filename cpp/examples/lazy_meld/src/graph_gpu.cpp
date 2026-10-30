/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "graph_detail.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/experimental/execution/gpu.hpp>
#include <cudf/experimental/lazy/meld.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <optional>
#include <unordered_map>

namespace cudf::experimental::execution {
namespace detail {
struct gpu_state {
  execution_plan plan;
  CUcontext context{};
  std::vector<std::optional<lazy::meld_program>> programs;
};
struct expression_owner {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> literals;
  std::unordered_map<lazy::detail::expression_node const*, cudf::ast::expression const*> nodes;
};
struct gpu_pending {
  std::shared_ptr<gpu_state const> program;
  cuda::stream_ref stream;
  cudaEvent_t complete{};
  bool recorded = false;
  bool enqueued = false;
  gpu_dispatch_counts counts;
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  std::vector<std::unique_ptr<expression_owner>> expressions;
  std::vector<lazy::pending_result> compiled;
  gpu_pending(std::shared_ptr<gpu_state const> p, cuda::stream_ref s)
    : program(std::move(p)), stream(s)
  {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
  }
  ~gpu_pending()
  {
    // Also covers exceptions between launches and event recording: drain before releasing owners.
    if (recorded)
      cudaEventSynchronize(complete);
    else if (enqueued)
      cudaStreamSynchronize(stream.get());
    if (complete) cudaEventDestroy(complete);
  }
};
}  // namespace detail
namespace {
using namespace lazy::detail;
using kind = frame_node::kind;
cudf::ast::ast_operator ast_op(operation op)
{
  using A = cudf::ast::ast_operator;
  switch (op) {
    case operation::add: return A::ADD;
    case operation::subtract: return A::SUB;
    case operation::multiply: return A::MUL;
    case operation::lt: return A::LESS;
    case operation::lte: return A::LESS_EQUAL;
    case operation::gt: return A::GREATER;
    case operation::gte: return A::GREATER_EQUAL;
    case operation::equal: return A::EQUAL;
    case operation::logical_and: return A::NULL_LOGICAL_AND;
    default: throw std::logic_error("unsupported AST operation");
  }
}
cudf::ast::expression const& emit(expression_node const& e,
                                  detail::expression_owner& o,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  if (auto found = o.nodes.find(&e); found != o.nodes.end()) return *found->second;
  cudf::ast::expression const* result;
  if (e.op == operation::column)
    result = &o.tree.push(cudf::ast::column_reference(e.index));
  else if (e.op == operation::literal) {
    auto scalar = std::make_unique<cudf::numeric_scalar<double>>(e.value, true, stream, mr);
    auto& typed = *scalar;
    o.literals.push_back(std::move(scalar));
    result = &o.tree.push(cudf::ast::literal(typed));
  } else if (e.op == operation::date_literal) {
    auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
      cudf::timestamp_D{cudf::duration_D{e.date_value.days}}, true, stream, mr);
    auto& typed = *scalar;
    o.literals.push_back(std::move(scalar));
    result = &o.tree.push(cudf::ast::literal(typed));
  } else {
    auto& a = emit(*e.left, o, stream, mr);
    auto& b = emit(*e.right, o, stream, mr);
    result  = &o.tree.push(cudf::ast::operation(ast_op(e.op), a, b));
  }
  o.nodes.emplace(&e, result);
  return *result;
}
cudf::column_view evaluate(expression const& e,
                           cudf::table_view input,
                           detail::gpu_pending& state,
                           rmm::device_async_resource_ref mr)
{
  state.expressions.push_back(std::make_unique<detail::expression_owner>());
  // Reserve literal ownership before any scalar construction enqueues an upload.
  auto count = [](auto const& self, expression_node const& n) -> std::size_t {
    return 1 + (n.left ? self(self, *n.left) : 0) + (n.right ? self(self, *n.right) : 0);
  };
  state.expressions.back()->literals.reserve(count(count, *plan_access::node(e)));
  auto const& ast = emit(*plan_access::node(e), *state.expressions.back(), state.stream, mr);
  state.columns.push_back(cudf::compute_column(input, ast, state.stream, mr));
  return state.columns.back()->view();
}
std::unique_ptr<cudf::table> scalar_table(cudf::scalar const& scalar,
                                          detail::gpu_pending& state,
                                          rmm::device_async_resource_ref mr)
{
  state.columns.push_back(cudf::make_column_from_scalar(scalar, 1, state.stream, mr));
  // Retain the source owner through every potentially throwing allocation/copy.
  return std::make_unique<cudf::table>(
    cudf::table_view{{state.columns.back()->view()}}, state.stream, mr);
}
std::unique_ptr<cudf::table> library(frame_node const& n,
                                     cudf::table_view input,
                                     detail::gpu_pending& state,
                                     rmm::device_async_resource_ref mr)
{
  auto stream = state.stream;
  if (n.op == kind::filter) {
    auto mask = evaluate(n.expressions[0], input, state, mr);
    return cudf::apply_retention_mask(input, mask, stream, mr);
  }
  if (n.op == kind::sort) {
    std::vector<cudf::column_view> keys;
    for (auto c : n.sort_keys)
      keys.push_back(input.column(c));
    return cudf::stable_sort_by_key(
      input,
      cudf::table_view{keys},
      {},
      std::vector<cudf::null_order>(keys.size(), cudf::null_order::AFTER),
      stream,
      mr);
  }
  if (n.op == kind::sum) {
    auto values = evaluate(n.expressions[0], input, state, mr);
    state.scalars.push_back(cudf::reduce(values,
                                         *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::FLOAT64},
                                         stream,
                                         mr));
    return scalar_table(*state.scalars.back(), state, mr);
  }
  std::vector<cudf::column_view> values;
  for (auto const& e : n.expressions)
    values.push_back(evaluate(e, input, state, mr));
  if (n.op == kind::project)
    return std::make_unique<cudf::table>(cudf::table_view{values}, stream, mr);
  auto measure = values.back();
  values.pop_back();
  cudf::groupby::groupby group(cudf::table_view{values}, cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = measure;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  // Keep group result owners alive if later host allocation fails while work is pending.
  auto grouped = [&] {
    try {
      return group.aggregate(requests, stream, mr);
    } catch (...) {
      cudaStreamSynchronize(stream.get());
      throw;
    }
  }();
  try {
    std::vector<cudf::column_view> output;
    for (auto const& c : grouped.first->view())
      output.push_back(c);
    output.push_back(grouped.second[0].results[0]->view());
    auto result = std::make_unique<cudf::table>(cudf::table_view{output}, stream, mr);
    state.tables.push_back(std::move(grouped.first));
    state.columns.push_back(std::move(grouped.second[0].results[0]));
    return result;
  } catch (...) {
    cudaStreamSynchronize(stream.get());
    throw;
  }
}
void context_is(CUcontext expected)
{
  CUcontext current{};
  if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != expected)
    throw std::invalid_argument("graph GPU execution requires its compiling CUDA context");
}
}  // namespace

gpu_program::gpu_program(std::shared_ptr<detail::gpu_state const> s) : _state(std::move(s)) {}
gpu_program compile_gpu(execution_plan const& p)
{
  if (p.options().device != target::gpu)
    throw std::invalid_argument("compile_gpu requires a GPU plan");
  CUDF_CUDA_TRY(cudaFree(nullptr));
  auto state = std::make_shared<detail::gpu_state>(detail::gpu_state{p, {}, {}});
  if (cuCtxGetCurrent(&state->context) != CUDA_SUCCESS || !state->context)
    throw std::runtime_error("no current CUDA context");
  auto const& plan = detail::graph_access::state(p);
  state->programs.resize(plan.regions.size());
  for (std::size_t i = 0; i < plan.regions.size(); ++i)
    if (plan.regions[i].backend == implementation::meld) {
      auto const& end = *plan.regions[i].nodes.back();
      if (end.op != kind::sum)
        throw std::logic_error("GPU planner selected unsupported Meld output");
      auto frame         = plan_access::as_frame(end.input);
      state->programs[i] = lazy::compile(frame.project({end.expressions[0]}).sum(0));
    }
  return gpu_program{std::move(state)};
}
std::string gpu_program::explain() const
{
  if (!_state) throw std::logic_error("moved-from graph GPU program");
  return _state->plan.explain();
}
pending_table gpu_program::submit(cudf::table_view input,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr) const
{
  if (!_state) throw std::logic_error("moved-from graph GPU program");
  context_is(_state->context);
  auto const& plan = detail::graph_access::state(_state->plan);
  if (static_cast<std::size_t>(input.num_columns()) != plan.input_columns)
    throw std::invalid_argument("input column count does not match graph schema");
  auto const& schema = plan.regions.front().nodes.front()->schema;
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    auto expected =
      schema[i].type == data_type::date32 ? cudf::type_id::TIMESTAMP_DAYS : cudf::type_id::FLOAT64;
    if (input.column(i).type().id() != expected)
      throw std::invalid_argument("GPU input type does not match graph schema");
  }
  auto state = std::make_unique<detail::gpu_pending>(_state, stream);
  // Reserve outer owner containers before submitting work; expression owners retain their
  // scalars separately. Bounds cover expression columns, scalar outputs and group results.
  std::size_t expressions = 0;
  for (auto const& r : plan.regions)
    for (auto const& n : r.nodes)
      expressions += n->expressions.size();
  state->tables.reserve(2 * plan.regions.size());
  state->columns.reserve(expressions + 2 * plan.regions.size());
  state->scalars.reserve(plan.regions.size());
  state->expressions.reserve(expressions);
  state->compiled.reserve(plan.regions.size());
  auto current    = input;
  state->enqueued = true;
  for (std::size_t i = 0; i < plan.regions.size(); ++i) {
    if (_state->programs[i]) {
      state->compiled.push_back(_state->programs[i]->submit(current, stream, mr));
      state->tables.push_back(scalar_table(state->compiled.back().result(), *state, mr));
      ++state->counts.meld;
    } else {
      state->tables.push_back(library(*plan.regions[i].nodes.back(), current, *state, mr));
      ++state->counts.cudf;
    }
    current = state->tables.back()->view();
  }
  CUDF_CUDA_TRY(cudaEventRecord(state->complete, stream.get()));
  state->recorded = true;
  return pending_table{std::move(state)};
}
pending_table::pending_table(std::unique_ptr<detail::gpu_pending> s) : _state(std::move(s)) {}
pending_table::~pending_table()                                   = default;
pending_table::pending_table(pending_table&&) noexcept            = default;
pending_table& pending_table::operator=(pending_table&&) noexcept = default;
cudf::table_view pending_table::result() const
{
  if (!_state || _state->tables.empty() || !_state->tables.back())
    throw std::logic_error("graph result has no output");
  return _state->tables.back()->view();
}
bool pending_table::ready() const
{
  result();
  auto status = cudaEventQuery(_state->complete);
  if (status == cudaErrorNotReady) return false;
  CUDF_CUDA_TRY(status);
  return true;
}
void pending_table::wait() const
{
  result();
  CUDF_CUDA_TRY(cudaEventSynchronize(_state->complete));
}
void pending_table::wait_on(cuda::stream_ref s) const
{
  result();
  CUDF_CUDA_TRY(cudaStreamWaitEvent(s.get(), _state->complete, 0));
}
std::unique_ptr<cudf::table> pending_table::take_result()
{
  wait();
  return std::move(_state->tables.back());
}
gpu_dispatch_counts pending_table::dispatched() const
{
  result();
  return _state->counts;
}
}  // namespace cudf::experimental::execution
