/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "plan_nodes.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cudf::experimental::lazy {
namespace {
using detail::expression_node;
using detail::field;
using detail::frame_node;
using detail::operation;
using detail::plan_access;

field infer(expression_node const& node, std::vector<field> const& schema)
{
  if (node.op == operation::column) {
    if (node.index >= schema.size()) {
      throw std::invalid_argument("lazy column index out of range");
    }
    return schema[node.index];
  }
  if (node.op == operation::literal) { return {false, false}; }
  if (node.op == operation::date_literal) { return {false, false, data_type::date32}; }
  auto const lhs            = infer(*node.left, schema);
  auto const rhs            = infer(*node.right, schema);
  auto const boolean_inputs = node.op == operation::logical_and;
  if (lhs.predicate != boolean_inputs || rhs.predicate != boolean_inputs || lhs.type != rhs.type) {
    throw std::invalid_argument("lazy expression operand type mismatch");
  }
  auto const arithmetic =
    node.op == operation::add || node.op == operation::subtract || node.op == operation::multiply;
  if (arithmetic && lhs.type != data_type::float64)
    throw std::invalid_argument("date arithmetic is not supported; compare dates with dates");
  return {!arithmetic, lhs.nullable || rhs.nullable};
}

expression binary(operation op, expression lhs, expression rhs)
{
  return plan_access::make({op, 0, 0, plan_access::node(lhs), plan_access::node(rhs)});
}

std::string type(field f)
{
  std::string payload = f.predicate ? "i1" : f.type == data_type::date32 ? "!meld.date" : "f64";
  return f.nullable ? "!meld.nullable<" + payload + ">" : payload;
}

std::string relation_type(std::vector<field> const& schema)
{
  std::string result = "!meld.relation<";
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (i != 0) { result += ", "; }
    result += type(schema[i]);
  }
  return result + ">";
}

char const* function_name(operation op)
{
  switch (op) {
    case operation::add: return "add";
    case operation::subtract: return "subtract";
    case operation::multiply: return "multiply";
    case operation::lt: return "lt";
    case operation::lte: return "lte";
    case operation::gt: return "gt";
    case operation::gte: return "gte";
    case operation::equal: return "equal";
    case operation::logical_and: return "and";
    default: throw std::logic_error("not a binary lazy expression");
  }
}

void block_arguments(std::ostream& out, std::vector<field> const& schema)
{
  out << "    ^bb0(";
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (i != 0) { out << ", "; }
    out << "%c" << i << ": " << type(schema[i]);
  }
  out << "):\n";
}

struct emitted_value {
  std::string name;
  field dtype;
};

std::string date_text(date32 value)
{
  using namespace std::chrono;
  auto point = sys_days{days{value.days}};
  if (point < sys_days{year{1} / January / 1} || point > sys_days{year{9999} / December / 31})
    throw std::invalid_argument("date literals require a year in [1, 9999]");
  year_month_day ymd{point};
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setfill('0') << std::setw(4) << int(ymd.year()) << '-' << std::setw(2)
      << unsigned(ymd.month()) << '-' << std::setw(2) << unsigned(ymd.day());
  return out.str();
}

class expression_writer {
 public:
  expression_writer(std::ostream& out, std::vector<field> const& schema)
    : _out(out), _schema(schema)
  {
  }

  emitted_value emit(expression_node const& node)
  {
    if (auto found = _values.find(&node); found != _values.end()) { return found->second; }
    auto const dtype = infer(node, _schema);
    if (node.op == operation::column) { return {"%c" + std::to_string(node.index), dtype}; }
    auto const name = "%e" + std::to_string(_next++);
    if (node.op == operation::literal) {
      // MLIR's hexadecimal float spelling preserves every bit, independent of locale.
      _out << "      " << name << " = meld.literal 0x" << std::hex << std::setfill('0')
           << std::setw(16) << std::bit_cast<std::uint64_t>(node.value) << std::dec << " : f64\n";
    } else if (node.op == operation::date_literal) {
      _out << "      " << name << " = meld.literal #meld.date<\"" << date_text(node.date_value)
           << "\">\n";
    } else {
      auto const lhs = emit(*node.left);
      auto const rhs = emit(*node.right);
      _out << "      " << name << " = meld.call @" << function_name(node.op)
           << (lhs.dtype.type == data_type::date32 ? "_date" : "") << "(" << lhs.name << ", "
           << rhs.name << ") : (" << type(lhs.dtype) << ", " << type(rhs.dtype) << ") -> "
           << type(dtype) << "\n";
    }
    return _values.emplace(&node, emitted_value{name, dtype}).first->second;
  }

 private:
  std::ostream& _out;
  std::vector<field> const& _schema;
  std::unordered_map<expression_node const*, emitted_value> _values;
  std::size_t _next{};
};
}  // namespace

expression::expression(std::shared_ptr<expression_node const> node) : _node(std::move(node)) {}

expression col(std::size_t index)
{
  return plan_access::make({operation::column, index, 0, {}, {}});
}
expression lit(double value)
{
  if (!std::isfinite(value)) {
    throw std::invalid_argument("lazy literals must be finite FLOAT64");
  }
  return plan_access::make({operation::literal, 0, value, {}, {}});
}
date32 date(int y, unsigned m, unsigned d)
{
  using namespace std::chrono;
  if (y < 1 || y > 9999 || m < 1 || m > 12 || d < 1 || d > 31)
    throw std::invalid_argument("invalid calendar date");
  year_month_day ymd{year{y}, month{m}, day{d}};
  if (!ymd.ok()) throw std::invalid_argument("invalid calendar date");
  return {static_cast<std::int32_t>(sys_days{ymd}.time_since_epoch().count())};
}
expression lit(date32 value)
{
  date_text(value);
  return plan_access::make({operation::date_literal, 0, 0, {}, {}, value});
}
expression operator+(expression a, expression b) { return binary(operation::add, a, b); }
expression operator-(expression a, expression b) { return binary(operation::subtract, a, b); }
expression operator*(expression a, expression b) { return binary(operation::multiply, a, b); }
expression operator<(expression a, expression b) { return binary(operation::lt, a, b); }
expression operator<=(expression a, expression b) { return binary(operation::lte, a, b); }
expression operator>(expression a, expression b) { return binary(operation::gt, a, b); }
expression operator>=(expression a, expression b) { return binary(operation::gte, a, b); }
expression operator==(expression a, expression b) { return binary(operation::equal, a, b); }
expression operator&&(expression a, expression b) { return binary(operation::logical_and, a, b); }

frame::frame(std::shared_ptr<frame_node const> node) : _node(std::move(node)) {}
frame scan(std::size_t columns)
{
  return scan(std::vector<data_type>(columns, data_type::float64));
}
frame scan(std::vector<data_type> types)
{
  auto columns = types.size();
  if (columns == 0) { throw std::invalid_argument("lazy scan needs at least one column"); }
  std::vector<field> schema;
  for (auto t : types) {
    size_of(t);
    schema.push_back({false, true, t});
  }
  return frame{std::make_shared<frame_node const>(
    frame_node{frame_node::kind::scan, {}, {}, std::move(schema)})};
}
std::size_t frame::num_columns() const
{
  if (!_node) { throw std::invalid_argument("moved-from lazy frame"); }
  return _node->schema.size();
}
frame frame::filter(expression predicate) const
{
  num_columns();
  if (!infer(*plan_access::node(predicate), _node->schema).predicate) {
    throw std::invalid_argument("lazy filter requires a boolean expression");
  }
  return frame{std::make_shared<frame_node const>(
    frame_node{frame_node::kind::filter, _node, {std::move(predicate)}, _node->schema})};
}
frame frame::project(std::vector<expression> expressions) const
{
  num_columns();
  if (expressions.empty()) {
    throw std::invalid_argument("lazy project needs at least one column");
  }
  std::vector<field> schema;
  for (auto const& expr : expressions) {
    auto const dtype = infer(*plan_access::node(expr), _node->schema);
    if (dtype.predicate) {
      throw std::invalid_argument("lazy project does not materialize boolean results");
    }
    schema.push_back(dtype);
  }
  return frame{std::make_shared<frame_node const>(
    frame_node{frame_node::kind::project, _node, std::move(expressions), std::move(schema)})};
}
reduction frame::sum(std::size_t column) const
{
  if (column >= num_columns()) {
    throw std::invalid_argument("lazy SUM column index out of range");
  }
  if (_node->schema[column].type != data_type::float64)
    throw std::invalid_argument("SUM requires FLOAT64 input");
  return reduction{_node, column};
}

reduction::reduction(std::shared_ptr<frame_node const> input, std::size_t column)
  : _input(std::move(input)), _column(column)
{
}
std::size_t reduction::input_columns() const
{
  if (!_input) { throw std::invalid_argument("moved-from lazy reduction"); }
  auto node = _input;
  while (node->input) {
    node = node->input;
  }
  return node->schema.size();
}
std::vector<data_type> reduction::input_types() const
{
  input_columns();
  auto node = _input;
  while (node->input)
    node = node->input;
  std::vector<data_type> types;
  for (auto f : node->schema)
    types.push_back(f.type);
  return types;
}

std::string reduction::to_meld_ir() const
{
  std::vector<detail::node_ptr> chain;
  for (auto node = plan_access::node(*this); node; node = node->input)
    chain.push_back(node);
  std::reverse(chain.begin(), chain.end());
  return detail::emit_meld_ir(chain);
}

field detail::infer_expression(expression const& e, std::vector<field> const& schema)
{
  return infer(*plan_access::node(e), schema);
}

std::string detail::emit_meld_ir(std::vector<node_ptr> const& chain)
{
  if (chain.size() < 2 || chain.front()->op != frame_node::kind::scan)
    throw std::invalid_argument("Meld region requires a scan and an operation");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "meld.plan version \"0.99.0\" {\n";
  for (auto family : {"arithmetic", "comparison", "boolean"}) {
    out << "  meld.extension_urn @" << family
        << " \"https://github.com/substrait-io/substrait/blob/main/extensions/functions_" << family
        << ".yaml\"\n";
  }
  for (auto name : {"add", "subtract", "multiply"}) {
    out << "  meld.function @" << name << " in @arithmetic \"" << name << ":fp64_fp64\"\n";
  }
  for (auto name : {"lt", "lte", "gt", "gte", "equal"}) {
    out << "  meld.function @" << name << " in @comparison \"" << name << ":fp64_fp64\"\n";
    out << "  meld.function @" << name << "_date in @comparison \"" << name << ":date_date\"\n";
  }
  out << "  meld.function @and in @boolean \"and:bool\"\n"
      << "  meld.function @sum in @arithmetic \"sum:fp64\"\n"
      << "  meld.root names [";
  for (std::size_t i = 0; i < chain.back()->schema.size(); ++i) {
    if (i) out << ", ";
    out << "\"out" << i << "\"";
  }
  out << "] {\n    %r0 = meld.named_table [\"input\"] columns [";
  for (std::size_t i = 0; i < chain.front()->schema.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"c" << i << "\"";
  }
  out << "] : " << relation_type(chain.front()->schema) << "\n";
  auto yield = [&](std::vector<emitted_value> const& values) {
    out << "      meld.yield";
    for (std::size_t i = 0; i < values.size(); ++i)
      out << (i ? ", " : " ") << values[i].name;
    for (std::size_t i = 0; i < values.size(); ++i)
      out << (i ? ", " : " : ") << type(values[i].dtype);
    out << "\n";
  };
  for (std::size_t i = 1; i < chain.size(); ++i) {
    auto const& node   = *chain[i];
    auto const& schema = chain[i - 1]->schema;
    bool const aggregate =
      node.op == frame_node::kind::sum || node.op == frame_node::kind::group_sum;
    bool const filter = node.op == frame_node::kind::filter;
    if (node.op == frame_node::kind::sort || node.op == frame_node::kind::scan ||
        (aggregate && i + 1 != chain.size()))
      throw std::invalid_argument("unsupported operation inside Meld region");
    out << "    %r" << i << " = meld."
        << (aggregate ? "aggregate"
            : filter  ? "filter"
                      : "project")
        << " %r" << i - 1 << (aggregate ? " keys {\n" : " {\n");
    block_arguments(out, schema);
    expression_writer writer(out, schema);
    std::vector<emitted_value> values;
    auto count = aggregate ? node.expressions.size() - 1 : node.expressions.size();
    for (std::size_t j = 0; j < count; ++j)
      values.push_back(writer.emit(*plan_access::node(node.expressions[j])));
    yield(values);
    if (aggregate) {
      out << "    } measures {\n";
      block_arguments(out, schema);
      expression_writer measure_writer(out, schema);
      auto value = measure_writer.emit(*plan_access::node(node.expressions.back()));
      out << "      %sum = meld.measure all @sum(" << value.name << ") : (" << type(value.dtype)
          << ") -> !meld.nullable<f64>\n";
      yield({{"%sum", {false, true}}});
    }
    out << "    }";
    if (!filter) out << " : " << relation_type(node.schema);
    out << "\n";
  }
  out << "    meld.yield %r" << chain.size() - 1 << " : " << relation_type(chain.back()->schema)
      << "\n  }\n}\n";
  return out.str();
}
}  // namespace cudf::experimental::lazy
