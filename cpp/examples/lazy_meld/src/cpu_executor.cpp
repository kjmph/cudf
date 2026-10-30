/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compiler.hpp"
#include "host_input.hpp"

#include <cudf/experimental/lazy/cpu.hpp>

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::experimental::lazy {
namespace detail {
struct cpu_program_state {
  std::vector<data_type> types;
  std::string ir;
  compiled_cpu_artifact artifact;
};
}  // namespace detail

cpu_program::cpu_program(std::shared_ptr<detail::cpu_program_state const> state)
  : _state(std::move(state))
{
}

cpu_program compile(reduction const& query, cpu_target)
{
  auto state   = std::make_shared<detail::cpu_program_state>();
  state->types = query.input_types();
  state->ir    = query.to_meld_ir();
  std::string error;
  if (!detail::compile_meld_cpu(state->ir, state->artifact, error)) {
    throw std::runtime_error("Meld CPU compilation failed:\n" + error);
  }
  return cpu_program{std::move(state)};
}

std::string const& cpu_program::plan_ir() const
{
  if (!_state) { throw std::logic_error("moved-from Meld CPU program"); }
  return _state->ir;
}

host_scalar cpu_program::run(host_table_view input) const
{
  if (!_state) { throw std::logic_error("moved-from Meld CPU program"); }
  detail::validate_input(input, _state->types);
  std::int64_t const rows = static_cast<std::int64_t>(input.front().values.size());
  detail::host_input bound(input);
  void* inputs[]{bound.pointers.data()};
  double value{};
  std::uint8_t valid{};
  void* outputs[]{&value, &valid};
  std::int64_t const fragments = 1;
  std::int64_t output_rows{};
  MeldStatus status    = meldStatusOk;
  auto const& artifact = _state->artifact;
  artifact.entry(0,
                 1,
                 &fragments,
                 artifact.literals.empty() ? nullptr : artifact.literals.data(),
                 inputs,
                 &rows,
                 nullptr,
                 outputs,
                 &output_rows,
                 1,
                 &status);
  if (status != meldStatusOk || output_rows != 1) {
    throw std::runtime_error("Meld CPU scalar reduction violated its output contract");
  }
  return (valid & 1u) ? host_scalar{value} : std::nullopt;
}

std::future<host_scalar> cpu_program::submit(host_table_view input) const
{
  if (!_state) { throw std::logic_error("moved-from Meld CPU program"); }
  detail::validate_input(input, _state->types);
  return std::async(
    std::launch::async,
    [program = *this, columns = std::vector<host_column_view>{input.begin(), input.end()}] {
      return program.run(columns);
    });
}
}  // namespace cudf::experimental::lazy
