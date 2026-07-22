/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/join/mixed_filtered_join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace {

using join_result = std::unique_ptr<rmm::device_uvector<cudf::size_type>>;

std::vector<cudf::size_type> indices_to_host(join_result const& result)
{
  auto const result_view = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                             static_cast<cudf::size_type>(result->size()),
                                             result->data(),
                                             nullptr,
                                             0};
  auto const host_result = cudf::test::to_host<cudf::size_type>(result_view).first;
  return {host_result.begin(), host_result.end()};
}

void expect_indices(join_result const& result,
                    std::vector<cudf::size_type> expected,
                    rmm::cuda_stream_view stream = cudf::test::get_default_stream())
{
  std::vector<cudf::size_type> actual;
  actual.reserve(result->size());
  for (std::size_t i = 0; i < result->size(); ++i) {
    actual.push_back(result->element(i, stream));
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(actual, expected);
}

void expect_same_indices(join_result const& actual, join_result const& expected)
{
  auto actual_host   = indices_to_host(actual);
  auto expected_host = indices_to_host(expected);
  std::sort(actual_host.begin(), actual_host.end());
  std::sort(expected_host.begin(), expected_host.end());
  EXPECT_EQ(actual_host, expected_host);
}

struct MixedFilteredJoinTest : public cudf::test::BaseFixture {};

TEST_F(MixedFilteredJoinTest, ReusesBuildWithDuplicateEqualityKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{10, 30, 25};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto join = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};

  auto const left_value  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);
  auto const greater =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_value, right_value);

  // The first build row for key 1 fails 20 < 10, but the duplicate passes 20 < 30.
  cudf::test::fixed_width_column_wrapper<int32_t> first_probe_key{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> first_probe_value{20, 20, 20};
  auto const first_probe_equality    = cudf::table_view{{first_probe_key}};
  auto const first_probe_conditional = cudf::table_view{{first_probe_value}};
  expect_indices(
    join.semi_join(first_probe_equality, first_probe_conditional, build_conditional, less), {0, 1});
  expect_indices(
    join.anti_join(first_probe_equality, first_probe_conditional, build_conditional, less), {2});
  expect_indices(
    join.semi_join(first_probe_equality, first_probe_conditional, build_conditional, greater), {0});

  cudf::test::fixed_width_column_wrapper<int32_t> second_probe_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> second_probe_value{40, 10};
  expect_indices(join.semi_join(cudf::table_view{{second_probe_key}},
                                cudf::table_view{{second_probe_value}},
                                build_conditional,
                                less),
                 {1});
}

TEST_F(MixedFilteredJoinTest, ReusesBuildWithAlternateConditionalTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> first_build_value{10, 30, 25};
  cudf::test::fixed_width_column_wrapper<int32_t> second_build_value{50, 5, 0};
  auto const build_equality = cudf::table_view{{build_key}};
  auto join = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};

  cudf::test::fixed_width_column_wrapper<int32_t> probe_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value{20, 10};
  auto const probe_equality    = cudf::table_view{{probe_key}};
  auto const probe_conditional = cudf::table_view{{probe_value}};
  auto const left_value        = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value       = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);

  expect_indices(
    join.semi_join(probe_equality, probe_conditional, cudf::table_view{{first_build_value}}, less),
    {0, 1});
  expect_indices(
    join.semi_join(probe_equality, probe_conditional, cudf::table_view{{second_build_value}}, less),
    {0});
}

TEST_F(MixedFilteredJoinTest, MultiColumnStringKeysAndCompoundPredicate)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_id{1, 1, 1, 2, 2};
  cudf::test::strings_column_wrapper build_region{"us", "eu", "us", "us", "eu"};
  cudf::test::fixed_width_column_wrapper<int32_t> build_min{10, 30, 20, 5, 40};
  cudf::test::fixed_width_column_wrapper<int32_t> build_max{100, 40, 50, 20, 80};
  auto const build_equality    = cudf::table_view{{build_id, build_region}};
  auto const build_conditional = cudf::table_view{{build_min, build_max}};
  auto join = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};

  cudf::test::fixed_width_column_wrapper<int32_t> probe_id{1, 1, 2, 2, 3};
  cudf::test::strings_column_wrapper probe_region{"us", "eu", "eu", "ap", "us"};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_amount{25, 35, 50, 10, 20};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_cap{60, 50, 70, 30, 30};
  auto const probe_equality    = cudf::table_view{{probe_id, probe_region}};
  auto const probe_conditional = cudf::table_view{{probe_amount, probe_cap}};

  auto const left_amount = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_min   = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const left_cap    = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const right_max   = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto const above_min =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_amount, right_min);
  auto const below_max = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_cap, right_max);
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, above_min, below_max);

  expect_indices(join.semi_join(probe_equality, probe_conditional, build_conditional, predicate),
                 {0, 2});
  expect_indices(join.anti_join(probe_equality, probe_conditional, build_conditional, predicate),
                 {1, 3, 4});
}

TEST_F(MixedFilteredJoinTest, NullPredicateResultsAreNotMatches)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{{1, 1, 2, 0},
                                                            {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{{0, 30, 0, 25},
                                                              {false, true, false, true}};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto join = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};

  cudf::test::fixed_width_column_wrapper<int32_t> probe_key{{1, 2, 0, 1},
                                                            {true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value{{20, 20, 20, 0},
                                                              {true, true, true, false}};
  auto const probe_equality    = cudf::table_view{{probe_key}};
  auto const probe_conditional = cudf::table_view{{probe_value}};
  auto const left_value        = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value       = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);

  expect_indices(join.semi_join(probe_equality, probe_conditional, build_conditional, less), {0});
  expect_indices(join.anti_join(probe_equality, probe_conditional, build_conditional, less),
                 {1, 2, 3});
}

TEST_F(MixedFilteredJoinTest, NullEqualityPolicy)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{{0, 1}, {false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{30, 30};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_key{{0, 1}, {false, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value{20, 20};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto const probe_equality    = cudf::table_view{{probe_key}};
  auto const probe_conditional = cudf::table_view{{probe_value}};
  auto const left_value        = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value       = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);

  auto nulls_equal = cudf::mixed_filtered_join{build_equality, cudf::null_equality::EQUAL};
  expect_indices(nulls_equal.semi_join(probe_equality, probe_conditional, build_conditional, less),
                 {0, 1});

  auto nulls_unequal = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};
  expect_indices(
    nulls_unequal.semi_join(probe_equality, probe_conditional, build_conditional, less), {1});
}

TEST_F(MixedFilteredJoinTest, EmptyBuildAndProbe)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_build_key{};
  cudf::test::fixed_width_column_wrapper<int32_t> empty_build_value{};
  auto const empty_build_equality    = cudf::table_view{{empty_build_key}};
  auto const empty_build_conditional = cudf::table_view{{empty_build_value}};
  auto join = cudf::mixed_filtered_join{empty_build_equality, cudf::null_equality::UNEQUAL};

  cudf::test::fixed_width_column_wrapper<int32_t> probe_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value{10, 20};
  auto const probe_equality    = cudf::table_view{{probe_key}};
  auto const probe_conditional = cudf::table_view{{probe_value}};
  auto const left_value        = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value       = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const equal = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, left_value, right_value);

  expect_indices(join.semi_join(probe_equality, probe_conditional, empty_build_conditional, equal),
                 {});
  expect_indices(join.anti_join(probe_equality, probe_conditional, empty_build_conditional, equal),
                 {0, 1});
  expect_indices(
    join.semi_join(empty_build_equality, empty_build_conditional, empty_build_conditional, equal),
    {});
  expect_indices(
    join.anti_join(empty_build_equality, empty_build_conditional, empty_build_conditional, equal),
    {});
}

TEST_F(MixedFilteredJoinTest, NaNEqualityKeys)
{
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  cudf::test::fixed_width_column_wrapper<double> build_key{nan, nan, 1.0};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{10, 30, 20};
  cudf::test::fixed_width_column_wrapper<double> probe_key{nan, 1.0, 2.0};
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value{20, 10, 10};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto const probe_equality    = cudf::table_view{{probe_key}};
  auto const probe_conditional = cudf::table_view{{probe_value}};
  auto const left_value        = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value       = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);

  for (auto const compare_nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    auto join = cudf::mixed_filtered_join{build_equality, compare_nulls};
    expect_indices(join.semi_join(probe_equality, probe_conditional, build_conditional, less),
                   {0, 1});
    expect_indices(join.anti_join(probe_equality, probe_conditional, build_conditional, less), {2});
  }
}

TEST_F(MixedFilteredJoinTest, MatchesExistingMixedJoinAcrossMultipleBlocks)
{
  constexpr cudf::size_type build_rows = 777;
  constexpr cudf::size_type probe_rows = 2053;
  std::vector<int32_t> build_key_0(build_rows);
  std::vector<int32_t> build_key_1(build_rows);
  std::vector<int32_t> build_value_0(build_rows);
  std::vector<int32_t> build_value_1(build_rows);
  std::vector<bool> build_validity(build_rows);
  for (cudf::size_type i = 0; i < build_rows; ++i) {
    build_key_0[i]    = i % 31;
    build_key_1[i]    = (i / 3) % 7;
    build_value_0[i]  = (i * 17) % 101;
    build_value_1[i]  = (i * 31) % 137;
    build_validity[i] = (i % 23) != 0;
  }

  std::vector<int32_t> probe_key_0(probe_rows);
  std::vector<int32_t> probe_key_1(probe_rows);
  std::vector<int32_t> probe_value_0(probe_rows);
  std::vector<int32_t> probe_value_1(probe_rows);
  std::vector<bool> probe_validity(probe_rows);
  for (cudf::size_type i = 0; i < probe_rows; ++i) {
    probe_key_0[i]    = i % 37;
    probe_key_1[i]    = (i / 5) % 7;
    probe_value_0[i]  = (i * 19) % 113;
    probe_value_1[i]  = (i * 7) % 149;
    probe_validity[i] = (i % 29) != 0;
  }

  cudf::test::fixed_width_column_wrapper<int32_t> build_key_0_col(
    build_key_0.begin(), build_key_0.end(), build_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> build_key_1_col(build_key_1.begin(),
                                                                  build_key_1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> build_value_0_col(build_value_0.begin(),
                                                                    build_value_0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> build_value_1_col(build_value_1.begin(),
                                                                    build_value_1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> probe_key_0_col(
    probe_key_0.begin(), probe_key_0.end(), probe_validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> probe_key_1_col(probe_key_1.begin(),
                                                                  probe_key_1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value_0_col(probe_value_0.begin(),
                                                                    probe_value_0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> probe_value_1_col(probe_value_1.begin(),
                                                                    probe_value_1.end());

  auto const build_equality    = cudf::table_view{{build_key_0_col, build_key_1_col}};
  auto const build_conditional = cudf::table_view{{build_value_0_col, build_value_1_col}};
  auto const probe_equality    = cudf::table_view{{probe_key_0_col, probe_key_1_col}};
  auto const probe_conditional = cudf::table_view{{probe_value_0_col, probe_value_1_col}};

  auto const left_0    = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_0   = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const left_1    = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
  auto const right_1   = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto const first     = cudf::ast::operation(cudf::ast::ast_operator::GREATER, left_0, right_0);
  auto const second    = cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, left_1, right_1);
  auto const predicate = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, first, second);

  for (auto const compare_nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    auto join = cudf::mixed_filtered_join{build_equality, compare_nulls};
    auto actual_semi =
      join.semi_join(probe_equality, probe_conditional, build_conditional, predicate);
    auto actual_anti =
      join.anti_join(probe_equality, probe_conditional, build_conditional, predicate);
    auto expected_semi = cudf::mixed_left_semi_join(probe_equality,
                                                    build_equality,
                                                    probe_conditional,
                                                    build_conditional,
                                                    predicate,
                                                    compare_nulls);
    auto expected_anti = cudf::mixed_left_anti_join(probe_equality,
                                                    build_equality,
                                                    probe_conditional,
                                                    build_conditional,
                                                    predicate,
                                                    compare_nulls);

    EXPECT_EQ(actual_semi->size() + actual_anti->size(), probe_rows);
    expect_same_indices(actual_semi, expected_semi);
    expect_same_indices(actual_anti, expected_anti);
  }
}

TEST_F(MixedFilteredJoinTest, ConcurrentProbesShareBuildIndex)
{
  auto const parent_stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{10, 30, 25};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto join =
    cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL, parent_stream};

  cudf::test::fixed_width_column_wrapper<int32_t> first_key{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> first_value{20, 20, 20};
  cudf::test::fixed_width_column_wrapper<int32_t> second_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> second_value{40, 10};
  auto const left_value  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const less = cudf::ast::operation(cudf::ast::ast_operator::LESS, left_value, right_value);

  auto streams = cudf::detail::fork_streams(parent_stream, 2);
  auto first   = join.semi_join(cudf::table_view{{first_key}},
                              cudf::table_view{{first_value}},
                              build_conditional,
                              less,
                              streams[0]);
  auto second  = join.anti_join(cudf::table_view{{second_key}},
                               cudf::table_view{{second_value}},
                               build_conditional,
                               less,
                               streams[1]);
  cudf::detail::join_streams(streams, parent_stream);

  expect_indices(first, {0, 1}, parent_stream);
  expect_indices(second, {0}, parent_stream);
}

TEST_F(MixedFilteredJoinTest, ValidatesTableShapesAndPredicateType)
{
  cudf::test::fixed_width_column_wrapper<int32_t> build_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> build_value{10, 20};
  cudf::test::fixed_width_column_wrapper<int32_t> short_build_value{10};
  auto const build_equality    = cudf::table_view{{build_key}};
  auto const build_conditional = cudf::table_view{{build_value}};
  auto join = cudf::mixed_filtered_join{build_equality, cudf::null_equality::UNEQUAL};

  cudf::test::fixed_width_column_wrapper<int32_t> probe_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int64_t> wrong_type_probe_key{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> short_probe_value{10};
  auto const probe_equality = cudf::table_view{{probe_key}};
  auto const non_boolean    = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);

  EXPECT_THROW((void)cudf::mixed_filtered_join(cudf::table_view{}, cudf::null_equality::UNEQUAL),
               cudf::logic_error);
  EXPECT_THROW(
    (void)join.semi_join(
      probe_equality, cudf::table_view{{short_probe_value}}, build_conditional, non_boolean),
    cudf::logic_error);
  EXPECT_THROW(
    (void)join.semi_join(
      probe_equality, build_conditional, cudf::table_view{{short_build_value}}, non_boolean),
    cudf::logic_error);
  EXPECT_THROW(
    (void)join.semi_join(probe_equality, build_conditional, build_conditional, non_boolean),
    cudf::data_type_error);

  auto const left_value  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto const right_value = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto const equal = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, left_value, right_value);
  EXPECT_THROW(
    (void)join.semi_join(
      cudf::table_view{{probe_key, probe_key}}, build_conditional, build_conditional, equal),
    cudf::logic_error);
  EXPECT_THROW(
    (void)join.semi_join(
      cudf::table_view{{wrong_type_probe_key}}, build_conditional, build_conditional, equal),
    cudf::data_type_error);
}

}  // namespace
