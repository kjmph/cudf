/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace cudf::experimental::lazy {
enum class data_type { float64, date32 };
/** Days since 1970-01-01, without a time of day or time zone. Matches Meld DATE/cuDF
 * TIMESTAMP_DAYS. */
struct date32 {
  std::int32_t days{};
  friend bool operator==(date32, date32) = default;
};
static_assert(sizeof(date32) == sizeof(std::int32_t));
/** Construct a calendar date. Reject invalid dates and years outside [1, 9999]. */
date32 date(int year, unsigned month, unsigned day);
inline std::size_t size_of(data_type type)
{
  switch (type) {
    case data_type::float64: return sizeof(double);
    case data_type::date32: return sizeof(date32);
  }
  throw std::invalid_argument("unsupported experimental column type");
}

/** Borrowed, typed host values. Dates remain four-byte day counts; no conversion to doubles. */
class host_values_view {
 public:
  host_values_view() = default;
  template <class T, std::size_t N>
    requires(std::is_same_v<std::remove_const_t<T>, double> ||
             std::is_same_v<std::remove_const_t<T>, date32>)
  host_values_view(std::span<T, N> values) : _values(std::span<std::add_const_t<T>>{values})
  {
  }
  template <std::ranges::contiguous_range R>
    requires std::ranges::sized_range<R> &&
             (std::is_same_v<std::ranges::range_value_t<R>, double> ||
              std::is_same_v<std::ranges::range_value_t<R>, date32>)
  host_values_view(R const& values)
    : host_values_view(std::span<std::ranges::range_value_t<R> const>{std::ranges::data(values),
                                                                      std::ranges::size(values)})
  {
  }
  std::size_t size() const
  {
    return std::visit([](auto v) { return v.size(); }, _values);
  }
  bool empty() const { return size() == 0; }
  data_type type() const { return _values.index() == 0 ? data_type::float64 : data_type::date32; }
  void const* data() const
  {
    return std::visit([](auto v) -> void const* { return v.data(); }, _values);
  }
  template <class T>
  std::span<T const> as() const
  {
    return std::get<std::span<T const>>(_values);
  }
  host_values_view subspan(std::size_t offset, std::size_t count = std::dynamic_extent) const
  {
    if (offset > size() || (count != std::dynamic_extent && count > size() - offset))
      throw std::out_of_range("host values slice out of range");
    return std::visit([&](auto v) { return host_values_view{v.subspan(offset, count)}; }, _values);
  }
  host_values_view first(std::size_t count) const { return subspan(0, count); }

 private:
  std::variant<std::span<double const>, std::span<date32 const>> _values;
};
}  // namespace cudf::experimental::lazy
