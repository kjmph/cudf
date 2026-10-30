/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/experimental/lazy/cpu.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

namespace cudf::experimental::lazy::detail {
inline std::size_t bitmap_bytes(std::size_t bits) { return bits / 8 + (bits % 8 != 0); }
inline void validate_input(host_table_view input, std::span<data_type const> types)
{
  if (input.size() != types.size() || input.empty())
    throw std::invalid_argument("input column count does not match graph schema");
  auto rows = input.front().values.size();
  if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("row count exceeds Meld ABI");
  for (std::size_t i = 0; i < input.size(); ++i) {
    auto const& c = input[i];
    if (c.values.type() != types[i])
      throw std::invalid_argument("input type does not match graph schema");
    if (c.values.size() != rows) throw std::invalid_argument("input columns have unequal lengths");
    if (!c.validity.empty() &&
        (c.validity_offset > std::numeric_limits<std::size_t>::max() - rows ||
         bitmap_bytes(c.validity_offset + rows) > c.validity.size()))
      throw std::invalid_argument("validity bitmap does not cover input view");
  }
}
struct host_input {
  std::vector<void*> pointers;
  std::vector<std::vector<std::uint8_t>> masks;
  explicit host_input(host_table_view input)
  {
    pointers.reserve(2 * input.size());
    masks.reserve(input.size());
    for (auto const& c : input) {
      void const* mask = nullptr;
      if (!c.validity.empty() && !c.values.empty()) {
        auto offset = c.validity_offset;
        if (offset % 8 == 0)
          mask = c.validity.data() + offset / 8;
        else {
          auto& m = masks.emplace_back(bitmap_bytes(c.values.size()));
          for (std::size_t byte = 0; byte < m.size(); ++byte) {
            auto source   = offset / 8 + byte;
            unsigned bits = c.validity[source] >> (offset % 8);
            if (source + 1 < bitmap_bytes(offset + c.values.size()))
              bits |= unsigned(c.validity[source + 1]) << (8 - offset % 8);
            m[byte] = static_cast<std::uint8_t>(bits);
          }
          mask = m.data();
        }
      }
      pointers.push_back(const_cast<void*>(c.values.data()));
      pointers.push_back(const_cast<void*>(mask));
    }
  }
};
}  // namespace cudf::experimental::lazy::detail
