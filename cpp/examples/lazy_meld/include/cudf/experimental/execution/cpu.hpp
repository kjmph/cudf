/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/execution/graph.hpp>
#include <cudf/experimental/lazy/cpu.hpp>

namespace cudf::experimental::execution {
using lazy::host_column_view;
using lazy::host_table_view;
/** Owning, typed output values. Numeric operator[] remains available for FLOAT64 examples;
 * use as<date32>() for dates. An incorrect accessor throws rather than reinterpreting storage.
 */
class host_column_values {
 public:
  host_column_values() = default;
  host_column_values(std::vector<double> v) : _values(std::move(v)) {}
  host_column_values(std::vector<date32> v) : _values(std::move(v)) {}
  host_column_values(std::initializer_list<double> v) : _values(std::vector<double>(v)) {}
  template <class T>
  std::vector<T>& as()
  {
    return std::get<std::vector<T>>(_values);
  }
  template <class T>
  std::vector<T> const& as() const
  {
    return std::get<std::vector<T>>(_values);
  }
  double& operator[](std::size_t i) { return as<double>()[i]; }
  double const& operator[](std::size_t i) const { return as<double>()[i]; }
  std::size_t size() const
  {
    return std::visit([](auto const& v) { return v.size(); }, _values);
  }
  bool empty() const { return size() == 0; }
  void resize(std::size_t n)
  {
    std::visit([&](auto& v) { v.resize(n); }, _values);
  }
  data_type type() const { return _values.index() == 0 ? data_type::float64 : data_type::date32; }
  void* data()
  {
    return std::visit([](auto& v) -> void* { return v.data(); }, _values);
  }
  void const* data() const
  {
    return std::visit([](auto const& v) -> void const* { return v.data(); }, _values);
  }
  lazy::host_values_view view() const
  {
    return std::visit([](auto const& v) { return lazy::host_values_view{v}; }, _values);
  }

 private:
  std::variant<std::vector<double>, std::vector<date32>> _values;
};
/** Owning FLOAT64/DATE32 output. An empty validity bitmap means all valid. */
struct host_column {
  host_column_values values;
  std::vector<std::uint8_t> validity;
};
struct host_table {
  std::vector<host_column> columns;
  std::size_t num_rows() const;
  std::vector<host_column_view> view() const;
};
struct execution_metrics {
  std::size_t meld_fragments{};
  std::size_t reference_fragments{};
  /** Sum of logical buffers materialized between fragments; not peak memory or allocator traffic.
   */
  std::size_t intermediate_bytes{};
};
namespace detail {
struct cpu_state;
}
class cpu_program {
 public:
  /** Single-threaded, reusable and safe for concurrent independent invocations.
   * Input buffers are borrowed for this call; intermediate and result buffers are owned.
   * This is Meld/host-reference execution, not a CPU port of libcudf.
   */
  host_table run(host_table_view, execution_metrics* = nullptr) const;
  /** Retains program/descriptors, borrows buffers until completion. Uses a new std::async worker.
   */
  std::future<host_table> submit(host_table_view) const;
  std::string explain() const;

 private:
  explicit cpu_program(std::shared_ptr<detail::cpu_state const>);
  std::shared_ptr<detail::cpu_state const> _state;
  friend cpu_program compile_cpu(execution_plan const&);
};
/** Requires a CPU plan. Compile once, then bind compatible batches repeatedly. */
cpu_program compile_cpu(execution_plan const&);
}  // namespace cudf::experimental::execution
