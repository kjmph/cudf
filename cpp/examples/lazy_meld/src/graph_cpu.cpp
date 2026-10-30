/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compiler.hpp"
#include "graph_detail.hpp"
#include "host_input.hpp"

#include <cudf/experimental/execution/cpu.hpp>

#include <meld/Runtime/GroupTable.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace cudf::experimental::execution {
namespace detail {
struct cpu_state {
  execution_plan plan;
  std::vector<lazy::detail::compiled_cpu_artifact> artifacts;
  std::vector<data_type> types;
};
}  // namespace detail
namespace {
using namespace lazy::detail;
using kind  = frame_node::kind;
using value = std::optional<double>;
bool valid(host_column_view const& c, std::size_t i)
{
  auto bit = i + c.validity_offset;
  return c.validity.empty() || ((c.validity[bit / 8] >> (bit % 8)) & 1u);
}
value read(host_column_view const& c, std::size_t i)
{
  if (!valid(c, i)) return std::nullopt;
  // Reference evaluation only: every int32 day count is exactly representable in a double.
  // Actual column storage and generated loads/stores remain typed DATE32.
  return c.values.type() == data_type::date32 ? double(c.values.as<date32>()[i].days)
                                              : c.values.as<double>()[i];
}
value evaluate(expression_node const& e, host_table_view input, std::size_t row)
{
  if (e.op == operation::column) return read(input[e.index], row);
  if (e.op == operation::literal) return e.value;
  if (e.op == operation::date_literal) return double(e.date_value.days);
  auto a = evaluate(*e.left, input, row), b = evaluate(*e.right, input, row);
  if (e.op == operation::logical_and) {
    if ((a && !*a) || (b && !*b)) return 0;
    return a && b ? value{1} : std::nullopt;
  }
  if (!a || !b) return std::nullopt;
  switch (e.op) {
    case operation::add: return *a + *b;
    case operation::subtract: return *a - *b;
    case operation::multiply: return *a * *b;
    case operation::lt: return *a < *b;
    case operation::lte: return *a <= *b;
    case operation::gt: return *a > *b;
    case operation::gte: return *a >= *b;
    case operation::equal: return *a == *b;
    default: throw std::logic_error("invalid scalar expression");
  }
}
host_table allocate(std::vector<field> const& schema, std::size_t rows)
{
  host_table t;
  for (auto f : schema)
    t.columns.push_back(
      {f.type == data_type::date32 ? host_column_values{std::vector<date32>(rows)}
                                   : host_column_values{std::vector<double>(rows)},
       f.nullable ? std::vector<std::uint8_t>(bitmap_bytes(rows)) : std::vector<std::uint8_t>{}});
  return t;
}
void write(host_column& c, std::size_t row, value v)
{
  if (c.values.type() == data_type::date32)
    c.values.as<date32>()[row] = {static_cast<std::int32_t>(v.value_or(0))};
  else
    c.values[row] = v.value_or(0);
  if (!c.validity.empty() && v) c.validity[row / 8] |= std::uint8_t(1u << (row % 8));
}
void resize(host_table& t, std::size_t rows)
{
  for (auto& c : t.columns) {
    c.values.resize(rows);
    if (!c.validity.empty()) c.validity.resize(bitmap_bytes(rows));
  }
}
host_table gather(host_table_view input,
                  std::vector<field> const& schema,
                  std::vector<std::size_t> const& rows)
{
  auto out = allocate(schema, rows.size());
  for (std::size_t c = 0; c < input.size(); ++c)
    for (std::size_t i = 0; i < rows.size(); ++i)
      write(out.columns[c], i, read(input[c], rows[i]));
  return out;
}
double canonical(double x)
{
  return std::isnan(x) ? std::numeric_limits<double>::quiet_NaN() : x == 0 ? 0.0 : x;
}
host_table reference(frame_node const& n, host_table_view input)
{
  auto rows = input.front().values.size();
  auto eval = [&](expression const& e, std::size_t r) {
    return evaluate(*plan_access::node(e), input, r);
  };
  if (n.op == kind::project) {
    auto out = allocate(n.schema, rows);
    for (std::size_t c = 0; c < n.expressions.size(); ++c)
      for (std::size_t i = 0; i < rows; ++i)
        write(out.columns[c], i, eval(n.expressions[c], i));
    return out;
  }
  if (n.op == kind::filter || n.op == kind::sort) {
    std::vector<std::size_t> indices;
    indices.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i)
      if (n.op == kind::sort || eval(n.expressions[0], i).value_or(0)) indices.push_back(i);
    if (n.op == kind::sort)
      std::stable_sort(indices.begin(), indices.end(), [&](auto a, auto b) {
        for (auto c : n.sort_keys) {
          auto av = read(input[c], a), bv = read(input[c], b);
          if (bool(av) != bool(bv)) return bool(av);  // NULL last.
          if (!av) continue;
          if (std::isnan(*av) != std::isnan(*bv)) return !std::isnan(*av);
          if (*av < *bv) return true;
          if (*bv < *av) return false;
        }
        return false;
      });
    return gather(input, n.schema, indices);
  }
  if (n.op == kind::sum) {
    value sum;
    for (std::size_t i = 0; i < rows; ++i)
      if (auto v = eval(n.expressions[0], i)) sum = sum.value_or(0) + *v;
    auto out = allocate(n.schema, 1);
    write(out.columns[0], 0, sum);
    return out;
  }
  // The comparison path deliberately materializes key/measure columns before aggregation.
  std::vector<field> projected_schema;
  for (auto const& e : n.expressions)
    projected_schema.push_back(infer_expression(e, n.input->schema));
  auto projected =
    reference(frame_node{kind::project, n.input, n.expressions, projected_schema}, input);
  auto columns = projected.view();
  using key    = std::vector<std::pair<bool, std::uint64_t>>;
  std::map<key, value> groups;
  for (std::size_t i = 0; i < rows; ++i) {
    key k;
    for (std::size_t c = 0; c + 1 < columns.size(); ++c) {
      auto v = read(columns[c], i);
      k.emplace_back(bool(v), v ? std::bit_cast<std::uint64_t>(canonical(*v)) : 0);
    }
    auto& sum = groups[k];
    if (auto v = read(columns.back(), i)) sum = sum.value_or(0) + *v;
  }
  auto out      = allocate(n.schema, groups.size());
  std::size_t i = 0;
  for (auto const& [k, sum] : groups) {
    for (std::size_t c = 0; c < k.size(); ++c)
      write(
        out.columns[c], i, k[c].first ? value{std::bit_cast<double>(k[c].second)} : std::nullopt);
    write(out.columns.back(), i++, sum);
  }
  return out;
}

host_table native(detail::region const& region,
                  compiled_cpu_artifact const& artifact,
                  host_table_view input)
{
  auto const& last     = *region.nodes.back();
  auto rows            = static_cast<std::int64_t>(input.front().values.size());
  auto grouped         = last.op == kind::group_sum;
  std::size_t capacity = last.op == kind::sum ? 1 : static_cast<std::size_t>(rows);
  if (grouped) {
    if (capacity > (std::size_t{1} << 60))
      throw std::length_error("group table exceeds ABI capacity");
    capacity = std::bit_ceil(std::max(std::size_t{2}, capacity * 2));
  }
  auto output = allocate(last.schema, capacity);
  host_input bound(input);
  void* inputs[]{bound.pointers.data()};
  std::vector<void*> outputs;
  std::vector<MeldColumn> descriptors;
  for (auto& c : output.columns) {
    auto mask = c.validity.empty() ? nullptr : c.validity.data();
    outputs.push_back(c.values.data());
    outputs.push_back(mask);
    descriptors.push_back(
      {c.values.data(), mask, static_cast<std::int32_t>(size_of(c.values.type()))});
  }
  std::vector<std::uint8_t> claims(grouped ? bitmap_bytes(capacity) : 0);
  MeldGroupTable groups{descriptors.data(),
                        static_cast<std::int32_t>(descriptors.size()),
                        static_cast<std::int32_t>(last.expressions.size() - 1),
                        claims.data(),
                        static_cast<std::int64_t>(capacity),
                        static_cast<std::int64_t>(capacity)};
  if (grouped) outputs.push_back(&groups);
  std::int64_t fragments = 1, written = 0;
  MeldStatus status = meldStatusOk;
  artifact.entry(0,
                 1,
                 &fragments,
                 artifact.literals.empty() ? nullptr : artifact.literals.data(),
                 inputs,
                 &rows,
                 nullptr,
                 outputs.data(),
                 &written,
                 static_cast<std::int64_t>(capacity),
                 &status);
  if (status != meldStatusOk)
    throw std::runtime_error("Meld graph execution failed: status=" + std::to_string(status));
  if (grouped) written = mlir::meld::compactGroupTable(groups);
  if (written < 0 || static_cast<std::size_t>(written) > capacity ||
      (last.op == kind::sum && written != 1))
    throw std::runtime_error("Meld graph violated output capacity contract");
  resize(output, static_cast<std::size_t>(written));
  return output;
}
}  // namespace

std::size_t host_table::num_rows() const
{
  return columns.empty() ? 0 : columns.front().values.size();
}
std::vector<host_column_view> host_table::view() const
{
  std::vector<host_column_view> result;
  for (auto const& c : columns)
    result.push_back({c.values.view(), c.validity});
  return result;
}
cpu_program::cpu_program(std::shared_ptr<detail::cpu_state const> s) : _state(std::move(s)) {}
cpu_program compile_cpu(execution_plan const& p)
{
  if (p.options().device != target::cpu)
    throw std::invalid_argument("compile_cpu requires a CPU plan");
  auto const& s = detail::graph_access::state(p);
  auto compiled = std::make_shared<detail::cpu_state>(detail::cpu_state{p, {}, {}});
  for (auto field : s.regions.front().nodes.front()->schema)
    compiled->types.push_back(field.type);
  compiled->artifacts.resize(s.regions.size());
  for (std::size_t i = 0; i < s.regions.size(); ++i)
    if (s.regions[i].backend == implementation::meld) {
      std::string error;
      if (!compile_meld_cpu(emit_meld_ir(s.regions[i].nodes), compiled->artifacts[i], error))
        throw std::runtime_error("Meld compilation of fragment " + std::to_string(i) +
                                 " failed:\n" + error);
    }
  return cpu_program{std::move(compiled)};
}
std::string cpu_program::explain() const
{
  if (!_state) throw std::logic_error("moved-from graph CPU program");
  return _state->plan.explain();
}
host_table cpu_program::run(host_table_view input, execution_metrics* metrics) const
{
  if (!_state) throw std::logic_error("moved-from graph CPU program");
  auto const& s = detail::graph_access::state(_state->plan);
  validate_input(input, _state->types);
  execution_metrics measured;
  host_table result;
  std::vector<host_column_view> current(input.begin(), input.end());
  for (std::size_t i = 0; i < s.regions.size(); ++i) {
    auto const& r = s.regions[i];
    auto output   = r.backend == implementation::meld ? native(r, _state->artifacts[i], current)
                                                      : reference(*r.nodes.back(), current);
    if (r.backend == implementation::meld)
      ++measured.meld_fragments;
    else
      ++measured.reference_fragments;
    if (i + 1 < s.regions.size())
      for (auto const& c : output.columns)
        measured.intermediate_bytes +=
          c.values.size() * size_of(c.values.type()) + c.validity.size();
    result  = std::move(output);
    current = result.view();
  }
  if (metrics) *metrics = measured;
  return result;
}
std::future<host_table> cpu_program::submit(host_table_view input) const
{
  if (!_state) throw std::logic_error("moved-from graph CPU program");
  validate_input(input, _state->types);
  return std::async(std::launch::async,
                    [p = *this, c = std::vector<host_column_view>(input.begin(), input.end())] {
                      return p.run(c);
                    });
}
}  // namespace cudf::experimental::execution
