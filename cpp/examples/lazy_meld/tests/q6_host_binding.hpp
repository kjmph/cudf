/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "q6_workload.hpp"

#include <cudf/experimental/lazy/cpu.hpp>

namespace lazy_test {
/** Owns packed validity masks; payload buffers remain borrowed from the fixture. */
struct host_binding {
  std::array<std::vector<std::uint8_t>, 4> masks;
  std::array<cudf::experimental::lazy::host_column_view, 4> columns;
  explicit host_binding(host_table const& data,
                        std::size_t offset = 0,
                        std::size_t count  = std::dynamic_extent)
  {
    namespace lazy = cudf::experimental::lazy;
    for (std::size_t c = 0; c < columns.size(); ++c) {
      auto const& source = data[c];
      if (!source.valid.empty()) {
        masks[c].resize((source.valid.size() + 7) / 8);
        for (std::size_t row = 0; row < source.valid.size(); ++row)
          if (source.valid[row]) masks[c][row / 8] |= std::uint8_t(1u << (row % 8));
      }
      auto values =
        c == 3 ? lazy::host_values_view{source.dates} : lazy::host_values_view{source.values};
      columns[c] = {values.subspan(offset, count), masks[c], offset};
    }
  }
};
}  // namespace lazy_test
