/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using device_read_request = cudf::io::datasource::device_read_request;

template <typename T>
std::future<T> make_ready_future(T value)
{
  std::promise<T> promise;
  promise.set_value(std::move(value));
  return promise.get_future();
}

class datasource_base : public cudf::io::datasource {
 public:
  explicit datasource_base(std::vector<uint8_t> data = std::vector<uint8_t>(1024))
    : data_{std::move(data)}
  {
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const read_size = offset < data_.size() ? std::min(size, data_.size() - offset) : 0;
    std::vector<uint8_t> result(read_size);
    if (read_size != 0) { std::memcpy(result.data(), data_.data() + offset, read_size); }
    return buffer::create(std::move(result));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = offset < data_.size() ? std::min(size, data_.size() - offset) : 0;
    if (read_size != 0) { std::memcpy(dst, data_.data() + offset, read_size); }
    return read_size;
  }

  [[nodiscard]] size_t size() const override { return data_.size(); }

 protected:
  std::vector<uint8_t> data_;
};

class recording_single_datasource final : public datasource_base {
 public:
  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view) override
  {
    reads.emplace_back(offset, size, dst);
    return make_ready_future(size);
  }

  std::vector<std::tuple<size_t, size_t, uint8_t*>> reads;
};

class recording_batch_datasource final : public datasource_base,
                                         public cudf::io::device_read_batch_source {
 public:
  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    ++single_read_calls;
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const> requests, rmm::cuda_stream_view) override
  {
    ++batch_read_calls;
    recorded_requests.assign(requests.begin(), requests.end());
    std::vector<size_t> sizes;
    sizes.reserve(requests.size());
    std::transform(
      requests.begin(), requests.end(), std::back_inserter(sizes), [](auto const& request) {
        return request.size;
      });
    return make_ready_future(std::move(sizes));
  }

  int batch_read_calls{};
  int single_read_calls{};
  std::vector<device_read_request> recorded_requests;
};

struct scheduling_error : std::runtime_error {
  scheduling_error() : std::runtime_error{"scheduling error"} {}
};

struct completion_error : std::runtime_error {
  completion_error() : std::runtime_error{"completion error"} {}
};

struct gate_state {
  gate_state() : release_future{release.get_future().share()} {}

  std::promise<void> entered;
  std::promise<void> release;
  std::shared_future<void> release_future;
  std::atomic<bool> completed{};
};

class gated_batch_datasource final : public datasource_base,
                                     public cudf::io::device_read_batch_source {
 public:
  explicit gated_batch_datasource(std::shared_ptr<gate_state> state) : state_{std::move(state)} {}

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const> requests, rmm::cuda_stream_view) override
  {
    std::vector<size_t> results;
    results.reserve(requests.size());
    std::transform(
      requests.begin(), requests.end(), std::back_inserter(results), [](auto const& request) {
        return request.size;
      });
    return std::async(std::launch::async, [state = state_, results = std::move(results)]() mutable {
      state->entered.set_value();
      state->release_future.wait();
      state->completed = true;
      return std::move(results);
    });
  }

 private:
  std::shared_ptr<gate_state> state_;
};

class scheduling_failure_batch_datasource final : public datasource_base,
                                                  public cudf::io::device_read_batch_source {
 public:
  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const>, rmm::cuda_stream_view) override
  {
    throw scheduling_error{};
  }
};

class scheduling_failure_datasource final : public datasource_base {
 public:
  explicit scheduling_failure_datasource(std::shared_ptr<gate_state> state)
    : state_{std::move(state)}
  {
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t,
                                        size_t size,
                                        uint8_t*,
                                        rmm::cuda_stream_view) override
  {
    if (calls_++ != 0) { throw scheduling_error{}; }
    return std::async(std::launch::async, [state = state_, size] {
      state->entered.set_value();
      state->release_future.wait();
      state->completed = true;
      return size;
    });
  }

 private:
  std::shared_ptr<gate_state> state_;
  int calls_{};
};

class completion_failure_datasource final : public datasource_base {
 public:
  explicit completion_failure_datasource(std::shared_ptr<gate_state> state)
    : state_{std::move(state)}
  {
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t,
                                        size_t size,
                                        uint8_t*,
                                        rmm::cuda_stream_view) override
  {
    if (calls_++ == 0) {
      std::promise<size_t> failed;
      failed.set_exception(std::make_exception_ptr(completion_error{}));
      return failed.get_future();
    }
    return std::async(std::launch::async, [state = state_, size] {
      state->entered.set_value();
      state->release_future.wait();
      state->completed = true;
      return size;
    });
  }

 private:
  std::shared_ptr<gate_state> state_;
  int calls_{};
};

class gated_completion_failure_datasource final : public datasource_base {
 public:
  explicit gated_completion_failure_datasource(std::shared_ptr<gate_state> state)
    : state_{std::move(state)}
  {
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    return std::async(std::launch::async, [state = state_]() -> size_t {
      state->entered.set_value();
      state->release_future.wait();
      state->completed = true;
      throw completion_error{};
    });
  }

 private:
  std::shared_ptr<gate_state> state_;
};

class gated_datasource final : public datasource_base {
 public:
  explicit gated_datasource(std::shared_ptr<gate_state> state) : state_{std::move(state)} {}

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t,
                                        size_t size,
                                        uint8_t*,
                                        rmm::cuda_stream_view) override
  {
    return std::async(std::launch::async, [state = state_, size] {
      state->entered.set_value();
      state->release_future.wait();
      state->completed = true;
      return size;
    });
  }

 private:
  std::shared_ptr<gate_state> state_;
};

class copying_batch_datasource final : public datasource_base,
                                       public cudf::io::device_read_batch_source {
 public:
  explicit copying_batch_datasource(std::string data, size_t host_read_size = 0)
    : datasource_base{std::vector<uint8_t>(data.begin(), data.end())},
      host_read_size_{host_read_size}
  {
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return size != host_read_size_;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    ++host_read_calls;
    return datasource_base::host_read(offset, size);
  }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    ++single_read_calls;
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const> requests, rmm::cuda_stream_view stream) override
  {
    ++batch_read_calls;
    batch_sizes.push_back(requests.size());
    std::vector<size_t> results;
    results.reserve(requests.size());
    for (auto const& request : requests) {
      auto const read_size =
        request.offset < data_.size() ? std::min(request.size, data_.size() - request.offset) : 0;
      if (read_size != 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(request.dst,
                                      data_.data() + request.offset,
                                      read_size,
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
      }
      results.push_back(read_size);
    }
    stream.synchronize();
    return make_ready_future(std::move(results));
  }

  int batch_read_calls{};
  int single_read_calls{};
  int host_read_calls{};
  std::vector<size_t> batch_sizes;

 private:
  size_t host_read_size_{};
};

class wrong_batch_result_datasource final : public datasource_base,
                                            public cudf::io::device_read_batch_source {
 public:
  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const>, rmm::cuda_stream_view) override
  {
    return make_ready_future(std::vector<size_t>{});
  }
};

class short_batch_result_datasource final : public datasource_base,
                                            public cudf::io::device_read_batch_source {
 public:
  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t, size_t, uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::logic_error("individual device read was not expected");
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const> requests, rmm::cuda_stream_view) override
  {
    std::vector<size_t> results;
    results.reserve(requests.size());
    std::transform(
      requests.begin(), requests.end(), std::back_inserter(results), [](auto const& request) {
        return request.size == 0 ? 0 : request.size - 1;
      });
    return make_ready_future(std::move(results));
  }
};

struct DatasourceTest : public cudf::test::BaseFixture {};

TEST_F(DatasourceTest, DefaultDeviceBatchPreservesOrderAndAcceptsEmptyNullDestination)
{
  recording_single_datasource source;
  std::array<uint8_t, 32> destinations{};
  std::array<device_read_request, 3> requests{
    {{100, 0, nullptr}, {200, 7, &destinations[0]}, {300, 11, &destinations[8]}}};

  auto completion = cudf::io::device_read_batch_async(
    source,
    cudf::host_span<device_read_request const>{requests.data(), requests.size()},
    cudf::get_default_stream());
  EXPECT_EQ(completion.get(), (std::vector<size_t>{0, 7, 11}));
  ASSERT_EQ(source.reads.size(), 2);
  EXPECT_EQ(std::get<0>(source.reads[0]), 200);
  EXPECT_EQ(std::get<0>(source.reads[1]), 300);
}

TEST_F(DatasourceTest, NativeDeviceBatchAcceptsEmptyBatch)
{
  recording_batch_datasource source;
  auto completion = cudf::io::device_read_batch_async(
    source, cudf::host_span<device_read_request const>{}, cudf::get_default_stream());

  EXPECT_TRUE(completion.get().empty());
  EXPECT_EQ(source.batch_read_calls, 1);
  EXPECT_EQ(source.single_read_calls, 0);
  EXPECT_TRUE(source.recorded_requests.empty());
}

TEST_F(DatasourceTest, DefaultDeviceBatchRejectsNonemptyNullDestinationBeforeScheduling)
{
  recording_single_datasource source;
  std::array<uint8_t, 1> destination{};
  std::array<device_read_request, 2> requests{{{0, 1, destination.data()}, {1, 1, nullptr}}};

  EXPECT_THROW(cudf::io::device_read_batch_async(
                 source,
                 cudf::host_span<device_read_request const>{requests.data(), requests.size()},
                 cudf::get_default_stream()),
               cudf::logic_error);
  EXPECT_TRUE(source.reads.empty());
}

TEST_F(DatasourceTest, DeviceBatchRejectsInvalidRangesBeforeScheduling)
{
  recording_single_datasource source;
  std::array<uint8_t, 4> destination{};

  std::array<device_read_request, 1> source_overflow{{
    {std::numeric_limits<size_t>::max(), 2, destination.data()},
  }};
  EXPECT_THROW(cudf::io::device_read_batch_async(source,
                                                 cudf::host_span<device_read_request const>{
                                                   source_overflow.data(), source_overflow.size()},
                                                 cudf::get_default_stream()),
               cudf::logic_error);

  auto const destination_address = reinterpret_cast<std::uintptr_t>(destination.data());
  std::array<device_read_request, 1> destination_overflow{{
    {0, std::numeric_limits<std::uintptr_t>::max() - destination_address + 1, destination.data()},
  }};
  EXPECT_THROW(
    cudf::io::device_read_batch_async(source,
                                      cudf::host_span<device_read_request const>{
                                        destination_overflow.data(), destination_overflow.size()},
                                      cudf::get_default_stream()),
    cudf::logic_error);

  std::array<device_read_request, 2> overlapping{{
    {0, 2, destination.data()},
    {2, 2, destination.data() + 1},
  }};
  EXPECT_THROW(cudf::io::device_read_batch_async(
                 source,
                 cudf::host_span<device_read_request const>{overlapping.data(), overlapping.size()},
                 cudf::get_default_stream()),
               cudf::logic_error);
  EXPECT_TRUE(source.reads.empty());
}

TEST_F(DatasourceTest, UserDatasourceWrapperForwardsDeviceBatch)
{
  recording_batch_datasource source;
  auto wrapper = cudf::io::datasource::create(&source);
  std::array<uint8_t, 1> destination{};
  std::array<device_read_request, 2> requests{{{3, 0, nullptr}, {9, 1, destination.data()}}};

  auto completion = cudf::io::device_read_batch_async(
    *wrapper,
    cudf::host_span<device_read_request const>{requests.data(), requests.size()},
    cudf::get_default_stream());
  EXPECT_EQ(completion.get(), (std::vector<size_t>{0, 1}));
  EXPECT_EQ(source.batch_read_calls, 1);
  EXPECT_EQ(source.single_read_calls, 0);
  ASSERT_EQ(source.recorded_requests.size(), requests.size());
  EXPECT_EQ(source.recorded_requests[1].offset, 9);
}

TEST_F(DatasourceTest, UserDatasourceWrapperFallsBackToIndividualDeviceReads)
{
  recording_single_datasource source;
  auto wrapper = cudf::io::datasource::create(&source);
  std::array<uint8_t, 2> destinations{};
  std::array<device_read_request, 3> requests{
    {{3, 0, nullptr}, {9, 1, &destinations[0]}, {17, 1, &destinations[1]}}};

  auto completion = cudf::io::device_read_batch_async(
    *wrapper,
    cudf::host_span<device_read_request const>{requests.data(), requests.size()},
    cudf::get_default_stream());
  EXPECT_EQ(completion.get(), (std::vector<size_t>{0, 1, 1}));
  ASSERT_EQ(source.reads.size(), 2);
  EXPECT_EQ(std::get<0>(source.reads[0]), 9);
  EXPECT_EQ(std::get<0>(source.reads[1]), 17);
}

TEST_F(DatasourceTest, DefaultDeviceBatchDrainsBeforeRethrowingSchedulingFailure)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  scheduling_failure_datasource source{state};
  std::array<uint8_t, 2> destinations{};
  std::array<device_read_request, 2> requests{{{0, 1, &destinations[0]}, {1, 1, &destinations[1]}}};

  auto invocation = std::async(std::launch::async, [&]() -> std::exception_ptr {
    try {
      std::ignore = cudf::io::device_read_batch_async(
        source,
        cudf::host_span<device_read_request const>{requests.data(), requests.size()},
        cudf::get_default_stream());
      return nullptr;
    } catch (...) {
      return std::current_exception();
    }
  });
  ASSERT_EQ(entered.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(invocation.wait_for(0s), std::future_status::timeout);
  state->release.set_value();
  auto const error = invocation.get();
  ASSERT_NE(error, nullptr);
  EXPECT_THROW(std::rethrow_exception(error), scheduling_error);
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, DefaultDeviceBatchDrainsAfterCompletionFailure)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  completion_failure_datasource source{state};
  std::array<uint8_t, 2> destinations{};
  std::array<device_read_request, 2> requests{{{0, 1, &destinations[0]}, {1, 1, &destinations[1]}}};

  auto completion = cudf::io::device_read_batch_async(
    source,
    cudf::host_span<device_read_request const>{requests.data(), requests.size()},
    cudf::get_default_stream());
  auto get_result = std::async(std::launch::async, [&] { return completion.get(); });
  ASSERT_EQ(entered.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(get_result.wait_for(0s), std::future_status::timeout);
  state->release.set_value();
  EXPECT_THROW(std::ignore = get_result.get(), completion_error);
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, DiscardedDefaultDeviceBatchDrainsReads)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  gated_datasource source{state};
  std::array<uint8_t, 1> destination{};
  std::array<device_read_request, 1> requests{{{0, 1, destination.data()}}};

  auto discard = std::async(std::launch::async, [&] {
    auto completion = cudf::io::device_read_batch_async(
      source,
      cudf::host_span<device_read_request const>{requests.data(), requests.size()},
      cudf::get_default_stream());
    completion = std::future<std::vector<size_t>>{};
  });
  ASSERT_EQ(entered.wait_for(10s), std::future_status::ready);
  EXPECT_EQ(discard.wait_for(0s), std::future_status::timeout);
  state->release.set_value();
  EXPECT_NO_THROW(discard.get());
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, ParquetUtilitySubmitsOneDeviceBatchPerDatasource)
{
  copying_batch_datasource first{"abcdefghijklmnopqrstuv"};
  copying_batch_datasource second{"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  std::array<std::reference_wrapper<cudf::io::datasource>, 2> sources{first, second};
  std::vector<std::vector<cudf::io::text::byte_range_info>> ranges{{{1, 3}, {8, 2}, {13, 4}},
                                                                   {{2, 5}, {12, 3}}};

  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{sources.data(),
                                                                        sources.size()},
    cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{ranges.data(),
                                                                        ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  EXPECT_NO_THROW(completion.get());
  EXPECT_EQ(first.batch_read_calls, 1);
  EXPECT_EQ(second.batch_read_calls, 1);
  EXPECT_EQ(first.single_read_calls, 0);
  EXPECT_EQ(second.single_read_calls, 0);
  EXPECT_EQ(first.batch_sizes, (std::vector<size_t>{3}));
  EXPECT_EQ(second.batch_sizes, (std::vector<size_t>{2}));

  std::array<std::vector<std::string>, 2> expected{std::vector<std::string>{"bcd", "ij", "nopq"},
                                                   std::vector<std::string>{"CDEFG", "MNO"}};
  ASSERT_EQ(spans.size(), expected.size());
  for (std::size_t source_index = 0; source_index < spans.size(); ++source_index) {
    ASSERT_EQ(spans[source_index].size(), expected[source_index].size());
    for (std::size_t range_index = 0; range_index < spans[source_index].size(); ++range_index) {
      std::string actual(spans[source_index][range_index].size(), '\0');
      CUDF_CUDA_TRY(cudaMemcpy(actual.data(),
                               spans[source_index][range_index].data(),
                               actual.size(),
                               cudaMemcpyDeviceToHost));
      EXPECT_EQ(actual, expected[source_index][range_index]);
    }
  }
}

TEST_F(DatasourceTest, DiscardedParquetCompletionDrainsNativeDeviceBatch)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  gated_batch_datasource source{state};
  std::array<cudf::io::text::byte_range_info, 1> ranges{{{0, 16}}};

  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    source,
    cudf::host_span<cudf::io::text::byte_range_info const>{ranges.data(), ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto discard = std::async(std::launch::async, [completion = std::move(completion)]() mutable {
    completion = std::future<void>{};
  });

  auto const entered_status = entered.wait_for(10s);
  if (entered_status != std::future_status::ready) { state->release.set_value(); }
  ASSERT_EQ(entered_status, std::future_status::ready);
  EXPECT_EQ(discard.wait_for(0s), std::future_status::timeout);

  state->release.set_value();
  EXPECT_NO_THROW(discard.get());
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, ParquetUtilitySchedulingFailureOutranksEarlierCompletionFailure)
{
  auto completion_state   = std::make_shared<gate_state>();
  auto completion_entered = completion_state->entered.get_future();
  auto scheduling_state   = std::make_shared<gate_state>();
  auto scheduling_entered = scheduling_state->entered.get_future();
  gated_completion_failure_datasource first{completion_state};
  scheduling_failure_datasource second{scheduling_state};
  std::array<std::reference_wrapper<cudf::io::datasource>, 2> sources{first, second};
  std::vector<std::vector<cudf::io::text::byte_range_info>> ranges{{{0, 1}}, {{0, 1}, {2, 1}}};

  auto invocation = std::async(std::launch::async, [&]() -> std::exception_ptr {
    try {
      std::ignore = cudf::io::parquet::fetch_byte_ranges_to_device_async(
        cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{sources.data(),
                                                                            sources.size()},
        cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{ranges.data(),
                                                                            ranges.size()},
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());
      return nullptr;
    } catch (...) {
      return std::current_exception();
    }
  });

  auto const completion_status = completion_entered.wait_for(10s);
  auto const scheduling_status = scheduling_entered.wait_for(10s);
  if (completion_status != std::future_status::ready or
      scheduling_status != std::future_status::ready) {
    scheduling_state->release.set_value();
    completion_state->release.set_value();
  }
  ASSERT_EQ(completion_status, std::future_status::ready);
  ASSERT_EQ(scheduling_status, std::future_status::ready);
  EXPECT_EQ(invocation.wait_for(0s), std::future_status::timeout);

  // The second source must drain the read it scheduled before its synchronous scheduling failure
  // can reach Parquet. Parquet must then drain the already-scheduled first source, even though that
  // earlier completion also fails, before preserving the scheduling failure as primary.
  scheduling_state->release.set_value();
  EXPECT_EQ(invocation.wait_for(0s), std::future_status::timeout);
  completion_state->release.set_value();

  auto const error = invocation.get();
  ASSERT_NE(error, nullptr);
  EXPECT_THROW(std::rethrow_exception(error), scheduling_error);
  EXPECT_TRUE(scheduling_state->completed);
  EXPECT_TRUE(completion_state->completed);
}

TEST_F(DatasourceTest, NativeBatchSchedulingFailureDrainsEarlierSource)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  gated_batch_datasource first{state};
  scheduling_failure_batch_datasource second;
  std::array<std::reference_wrapper<cudf::io::datasource>, 2> sources{first, second};
  std::vector<std::vector<cudf::io::text::byte_range_info>> ranges{{{0, 1}}, {{0, 1}}};

  auto invocation = std::async(std::launch::async, [&]() -> std::exception_ptr {
    try {
      std::ignore = cudf::io::parquet::fetch_byte_ranges_to_device_async(
        cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{sources.data(),
                                                                            sources.size()},
        cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{ranges.data(),
                                                                            ranges.size()},
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref());
      return nullptr;
    } catch (...) {
      return std::current_exception();
    }
  });

  auto const entered_status = entered.wait_for(10s);
  if (entered_status != std::future_status::ready) { state->release.set_value(); }
  ASSERT_EQ(entered_status, std::future_status::ready);
  EXPECT_EQ(invocation.wait_for(0s), std::future_status::timeout);

  state->release.set_value();
  auto const error = invocation.get();
  ASSERT_NE(error, nullptr);
  EXPECT_THROW(std::rethrow_exception(error), scheduling_error);
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, ParquetUtilityPreservesMixedHostAndDeviceReads)
{
  copying_batch_datasource source{"abcdefghijklmnopqrstuvwxyz", 4};
  std::array<cudf::io::text::byte_range_info, 3> ranges{{{1, 3}, {8, 4}, {15, 2}}};

  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    source,
    cudf::host_span<cudf::io::text::byte_range_info const>{ranges.data(), ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  EXPECT_NO_THROW(completion.get());
  EXPECT_EQ(source.batch_read_calls, 1);
  EXPECT_EQ(source.single_read_calls, 0);
  EXPECT_EQ(source.host_read_calls, 1);
  EXPECT_EQ(source.batch_sizes, (std::vector<size_t>{2}));

  std::array<std::string, 3> expected{"bcd", "ijkl", "pq"};
  ASSERT_EQ(spans.size(), expected.size());
  for (std::size_t range_index = 0; range_index < spans.size(); ++range_index) {
    std::string actual(spans[range_index].size(), '\0');
    CUDF_CUDA_TRY(
      cudaMemcpy(actual.data(), spans[range_index].data(), actual.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected[range_index]);
  }
}

TEST_F(DatasourceTest, ParquetUtilityRejectsWrongBatchResultCardinality)
{
  wrong_batch_result_datasource source;
  std::array<cudf::io::text::byte_range_info, 1> ranges{{{0, 16}}};
  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    source,
    cudf::host_span<cudf::io::text::byte_range_info const>{ranges.data(), ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  EXPECT_THROW(completion.get(), cudf::logic_error);
}

TEST_F(DatasourceTest, WrongBatchResultDrainsLaterNativeBatch)
{
  auto state   = std::make_shared<gate_state>();
  auto entered = state->entered.get_future();
  wrong_batch_result_datasource first;
  gated_batch_datasource second{state};
  std::array<std::reference_wrapper<cudf::io::datasource>, 2> sources{first, second};
  std::vector<std::vector<cudf::io::text::byte_range_info>> ranges{{{0, 1}}, {{0, 1}}};

  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{sources.data(),
                                                                        sources.size()},
    cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{ranges.data(),
                                                                        ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto waiter = std::async(std::launch::async, [completion = std::move(completion)]() mutable {
    try {
      completion.get();
      return std::exception_ptr{};
    } catch (...) {
      return std::current_exception();
    }
  });

  auto const entered_status = entered.wait_for(10s);
  if (entered_status != std::future_status::ready) { state->release.set_value(); }
  ASSERT_EQ(entered_status, std::future_status::ready);
  EXPECT_EQ(waiter.wait_for(0s), std::future_status::timeout);

  state->release.set_value();
  auto const error = waiter.get();
  ASSERT_NE(error, nullptr);
  EXPECT_THROW(std::rethrow_exception(error), cudf::logic_error);
  EXPECT_TRUE(state->completed);
}

TEST_F(DatasourceTest, ParquetUtilityRejectsShortBatchRead)
{
  short_batch_result_datasource source;
  std::array<cudf::io::text::byte_range_info, 1> ranges{{{0, 16}}};
  auto [buffers, spans, completion] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    source,
    cudf::host_span<cudf::io::text::byte_range_info const>{ranges.data(), ranges.size()},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  EXPECT_THROW(completion.get(), cudf::logic_error);
}

}  // namespace
