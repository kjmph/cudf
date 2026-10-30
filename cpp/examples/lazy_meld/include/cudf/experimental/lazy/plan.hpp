/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/lazy/types.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cudf::experimental::lazy {
namespace detail {
struct expression_node;
struct frame_node;
struct plan_access;
}  // namespace detail

/** An immutable row expression. Column indices refer to the immediate input of an operation. */
class expression {
 public:
  expression(expression const&)            = default;
  expression(expression&&)                 = default;
  expression& operator=(expression const&) = default;
  expression& operator=(expression&&)      = default;

 private:
  explicit expression(std::shared_ptr<detail::expression_node const> node);
  std::shared_ptr<detail::expression_node const> _node;
  friend struct detail::plan_access;
};

expression col(std::size_t index);
expression lit(double value);
expression lit(date32 value);
expression operator+(expression lhs, expression rhs);
expression operator-(expression lhs, expression rhs);
expression operator*(expression lhs, expression rhs);
expression operator<(expression lhs, expression rhs);
expression operator<=(expression lhs, expression rhs);
expression operator>(expression lhs, expression rhs);
expression operator>=(expression lhs, expression rhs);
expression operator==(expression lhs, expression rhs);
/** SQL three-valued AND. This builds an expression; it does not short-circuit C++ evaluation. */
expression operator&&(expression lhs, expression rhs);

class reduction;

/** Deferred operations over FLOAT64 and DATE32 columns. Construction does no GPU work. */
class frame {
 public:
  frame filter(expression predicate) const;
  frame project(std::vector<expression> expressions) const;
  reduction sum(std::size_t column) const;
  std::size_t num_columns() const;

 private:
  explicit frame(std::shared_ptr<detail::frame_node const> node);
  std::shared_ptr<detail::frame_node const> _node;
  friend frame scan(std::size_t columns);
  friend frame scan(std::vector<data_type> types);
  friend struct detail::plan_access;
};

/** Bind a schema of `columns` nullable FLOAT64 fields; actual buffers are bound at submit(). */
frame scan(std::size_t columns);
/** Bind an explicitly typed, nullable schema. */
frame scan(std::vector<data_type> types);

/** A terminal scalar SUM. Empty or all-null input produces an invalid scalar, as in libcudf. */
class reduction {
 public:
  std::size_t input_columns() const;
  std::vector<data_type> input_types() const;
  /** Inspectable compiler input, generated from the chain rather than supplied by the caller. */
  std::string to_meld_ir() const;

 private:
  reduction(std::shared_ptr<detail::frame_node const> input, std::size_t column);
  std::shared_ptr<detail::frame_node const> _input;
  std::size_t _column;
  friend class frame;
  friend struct detail::plan_access;
};
}  // namespace cudf::experimental::lazy
