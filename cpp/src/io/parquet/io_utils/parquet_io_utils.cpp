/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/utilities/future_drain_guard.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/async_device_io.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

namespace {

/**
 * @brief Dispatches the fetch task for each source index and collects the results
 *
 * Dispatches sequentially or using host worker pool depending on the number of sources.
 *
 * @tparam Task Callable invocable as `fetch_task(std::size_t source_idx)`
 * @param num_sources Number of sources to process
 * @param fetch_task Task to run for each source index
 * @return Vector of results, one per source, in source order
 */
template <typename Task>
auto dispatch_fetch_tasks(std::size_t num_sources, Task fetch_task)
{
  using result_type = std::invoke_result_t<Task, std::size_t>;

  auto constexpr parallel_threshold = 32;

  std::vector<result_type> results;
  results.reserve(num_sources);

  if (num_sources < parallel_threshold) {
    // Run sequentially to avoid task dispatch overhead
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_sources),
                  [&](std::size_t source_idx) { results.emplace_back(fetch_task(source_idx)); });
  } else {
    // Dispatch the tasks to the host worker pool
    cudf::io::detail::future_drain_guard<result_type> tasks{num_sources};
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_sources),
                  [&](std::size_t source_idx) {
                    tasks.push(cudf::detail::host_worker_pool().submit_task(
                      [&fetch_task, source_idx]() { return fetch_task(source_idx); }));
                  });
    tasks.consume_all([&results](result_type result) { results.emplace_back(std::move(result)); });
  }
  return results;
}

/**
 * @copydoc cudf::io::parquet::fetch_footers_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  // Look up runtime configuration once, as late as possible.
  auto const metadata_size_hint = cudf::io::parquet::metadata_size_hint();
  // Helper to fetch footer from a datasource
  auto const fetch_footer = [metadata_size_hint](cudf::io::datasource& datasource) {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);
    size_t const len          = datasource.size();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");

    auto const speculative_read_size =
      std::min(len, std::max(metadata_size_hint, static_cast<size_t>(ender_len)));
    auto const speculative_read_offset = len - speculative_read_size;

    auto speculative_buffer = datasource.host_read(speculative_read_offset, speculative_read_size);
    CUDF_EXPECTS(speculative_buffer->size() == speculative_read_size,
                 "Failed to read Parquet speculative metadata bytes");

    auto const ender = reinterpret_cast<file_ender_s const*>(
      speculative_buffer->data() + speculative_buffer->size() - ender_len);

    if (speculative_read_offset == 0) {
      auto const header = reinterpret_cast<file_header_s const*>(speculative_buffer->data());
      CUDF_EXPECTS(header->magic == detail::parquet_magic, "Corrupted header");
    }

    CUDF_EXPECTS(ender->magic == detail::parquet_magic, "Corrupted footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    auto const footer_offset = len - ender->footer_len - ender_len;
    if (footer_offset >= speculative_read_offset) {
      // fastpath: the speculative read includes the full footer.
      auto const footer_start_offset = footer_offset - speculative_read_offset;
      CUDF_EXPECTS(footer_start_offset + ender->footer_len <= speculative_buffer->size(),
                   "Speculative metadata read did not include full footer bytes");
      std::vector<uint8_t> footer_bytes(ender->footer_len);
      std::memcpy(
        footer_bytes.data(), speculative_buffer->data() + footer_start_offset, ender->footer_len);
      return cudf::io::datasource::buffer::create(std::move(footer_bytes));
    }

    // The speculative read only got part of the footer. Read the missing prefix, then stitch.
    auto const missing_prefix_size = speculative_read_offset - footer_offset;
    auto missing_prefix            = datasource.host_read(footer_offset, missing_prefix_size);
    CUDF_EXPECTS(missing_prefix->size() == missing_prefix_size,
                 "Failed to read the missing footer prefix bytes");
    std::vector<uint8_t> footer_bytes(ender->footer_len);
    std::memcpy(footer_bytes.data(), missing_prefix->data(), missing_prefix_size);
    auto const footer_suffix_size = ender->footer_len - missing_prefix_size;
    std::memcpy(
      footer_bytes.data() + missing_prefix_size, speculative_buffer->data(), footer_suffix_size);
    return cudf::io::datasource::buffer::create(std::move(footer_bytes));
  };

  return dispatch_fetch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_footer(datasources[source_idx].get());
  });
}

/**
 * @copydoc cudf::io::parquet::fetch_page_indexes_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_EXPECTS(datasources.size() == page_index_bytes_per_source.size(),
               "Encountered mismatch in number of datasources and page index byte ranges");

  // Helper to fetch page index bytes from a datasource
  auto const fetch_page_index = [](cudf::io::datasource& datasource,
                                   cudf::io::text::byte_range_info const& page_index_bytes) {
    CUDF_EXPECTS(
      page_index_bytes.offset() >= 0 and
        std::cmp_less_equal(page_index_bytes.offset() + page_index_bytes.size(), datasource.size()),
      std::format("Invalid page index byte range: offset={}, size={}, datasource_size={}",
                  page_index_bytes.offset(),
                  page_index_bytes.size(),
                  datasource.size()),
      std::out_of_range);
    return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
  };

  return dispatch_fetch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_page_index(datasources[source_idx].get(), page_index_bytes_per_source[source_idx]);
  });
}

using host_read_buffer = std::unique_ptr<cudf::io::datasource::buffer>;

/**
 * @brief Drains device reads and establishes host-to-device copy completion before rethrowing.
 *
 * A datasource future is required to stop accessing its destination before it becomes ready, even
 * exceptionally. A failed CUDA stream fence is followed by a device-wide fence. If that last-resort
 * fence also fails, returning would make it possible to destroy storage while DMA is still in
 * flight, so the process is terminated. The original enqueue or synchronization exception remains
 * primary whenever recovery is possible.
 */
[[noreturn]] void finish_failed_host_copy(
  std::exception_ptr primary_exception,
  bool stream_fence_required,
  rmm::cuda_stream_view stream,
  int device,
  cudf::io::detail::future_drain_guard<std::vector<size_t>>& device_read_tasks)
{
  try {
    device_read_tasks.wait_all();
  } catch (...) {
    // The enqueue or stream error remains primary. The datasource completion contract guarantees
    // that an exceptional ready future has nevertheless stopped accessing its destination.
  }

  if (stream_fence_required) {
    try {
      cudf::io::detail::synchronize_stream(stream, device);
    } catch (...) {
      // The enqueue error remains primary. synchronize_stream established completion.
    }
  }

  std::rethrow_exception(primary_exception);
}

[[noreturn]] void finish_failed_device_schedule(
  std::exception_ptr primary_exception,
  cudf::io::detail::future_drain_guard<std::vector<size_t>>& device_read_tasks)
{
  try {
    device_read_tasks.wait_all();
  } catch (...) {
    // The synchronous scheduling error remains primary. The datasource completion contract
    // guarantees that an exceptional ready future has stopped accessing its destination.
  }

  std::rethrow_exception(primary_exception);
}

using device_spans_per_source_type = std::vector<cudf::device_span<uint8_t const>>;

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<device_spans_per_source_type>,
           std::future<void>>
fetch_byte_ranges_to_device_async_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::host_span<cudf::io::text::byte_range_info const> const>
    byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static std::mutex host_read_mutex;
  static std::mutex device_read_mutex;

  auto const num_sources = datasources.size();

  CUDF_EXPECTS(num_sources == byte_ranges_per_source.size(),
               "Encountered mismatch in number of datasources and the number of byte range spans");

  // Total number of byte ranges across all sources
  auto const total_byte_ranges =
    std::accumulate(byte_ranges_per_source.begin(),
                    byte_ranges_per_source.end(),
                    std::size_t{0},
                    [](auto acc, auto const& ranges) { return acc + ranges.size(); });

  // IO descriptors
  std::vector<size_t> io_source_indices;
  std::vector<size_t> io_offsets;
  std::vector<size_t> io_sizes;
  std::vector<uint8_t*> destinations;
  io_source_indices.reserve(total_byte_ranges);
  io_offsets.reserve(total_byte_ranges);
  io_sizes.reserve(total_byte_ranges);
  destinations.reserve(total_byte_ranges);

  // Allocate one device buffer per byte ranges of a datasource
  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.reserve(num_sources);

  // Column chunk device spans, one per byte range per datasource
  std::vector<device_spans_per_source_type> column_chunk_data_per_source(num_sources);

  std::for_each(
    cuda::counting_iterator<cudf::size_type>(0),
    cuda::counting_iterator<cudf::size_type>(num_sources),
    [&](auto const source_idx) {
      auto const& byte_ranges = byte_ranges_per_source[source_idx];

      // Total buffer size required for column chunks of this source
      auto const buffer_size = std::accumulate(
        byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [](auto acc, auto const& range) {
          return acc + range.size();
        });

      // Buffer needs to be padded. Required by `gpuDecodePageData`.
      column_chunk_buffers.emplace_back(
        cudf::util::round_up_safe(buffer_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
        stream,
        mr);

      auto buffer_data = static_cast<uint8_t*>(column_chunk_buffers.back().data());

      // Build device spans for each byte range in this source
      auto& column_chunk_data = column_chunk_data_per_source[source_idx];
      column_chunk_data.reserve(byte_ranges.size());
      std::ignore = std::accumulate(
        byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
          column_chunk_data.emplace_back(buffer_data + acc, static_cast<size_t>(range.size()));
          return acc + range.size();
        });

      // Coalesce contiguous byte ranges within this source into single IO request
      for (size_t chunk = 0; chunk < byte_ranges.size();) {
        auto const io_offset = static_cast<size_t>(byte_ranges[chunk].offset());
        auto io_size         = static_cast<size_t>(byte_ranges[chunk].size());
        size_t next_chunk    = chunk + 1;
        while (next_chunk < byte_ranges.size()) {
          size_t const next_offset = byte_ranges[next_chunk].offset();
          if (next_offset != io_offset + io_size) { break; }
          io_size += byte_ranges[next_chunk].size();
          next_chunk++;
        }
        if (io_size != 0) {
          io_source_indices.push_back(source_idx);
          io_offsets.push_back(io_offset);
          io_sizes.push_back(io_size);
          destinations.push_back(const_cast<uint8_t*>(column_chunk_data[chunk].data()));
        }
        chunk = next_chunk;
      }
    });

  CUDF_EXPECTS(io_offsets.size() == io_sizes.size() and io_sizes.size() == destinations.size() and
                 io_source_indices.size() == io_offsets.size(),
               "Unexpected number of IO source indices, offsets, sizes, or destinations");

  using device_read_request = cudf::io::datasource::device_read_request;
  std::vector<bool> device_read_preferred;
  device_read_preferred.reserve(io_offsets.size());
  std::vector<size_t> device_read_counts(num_sources);
  std::vector<std::vector<device_read_request>> device_read_batches(num_sources);
  std::vector<size_t> device_batch_source_indices;
  device_batch_source_indices.reserve(num_sources);
  auto expected_device_read_sizes = std::make_shared<std::vector<std::vector<size_t>>>();
  expected_device_read_sizes->reserve(num_sources);

  // Guards to drain every issued read before its source or destination can unwind.
  cudf::io::detail::future_drain_guard<host_read_buffer> host_read_tasks{io_offsets.size()};
  int device;
  CUDF_CUDA_TRY(cudf::io::detail::get_stream_device(stream, &device));

  // Vectors to store intermediate host buffers and relevant pointers
  std::vector<host_read_buffer> host_buffers{};
  std::vector<void const*> copy_srcs{};
  std::vector<void*> copy_dsts{};
  std::vector<size_t> copy_sizes{};
  copy_dsts.reserve(io_offsets.size());
  copy_sizes.reserve(io_offsets.size());

  // Evaluate each preference exactly once and schedule host reads while holding the existing host
  // serialization lock. This keeps one caller's preference decisions and host scheduling from
  // interleaving with another caller, while avoiding the old second, potentially inconsistent
  // preference evaluation during device scheduling.
  {
    std::scoped_lock<std::mutex> lock(host_read_mutex);

    for (std::size_t io_index = 0; io_index < io_offsets.size(); ++io_index) {
      auto const source_index = io_source_indices[io_index];
      auto const preferred =
        datasources[source_index].get().is_device_read_preferred(io_sizes[io_index]);
      device_read_preferred.push_back(preferred);
      if (preferred) { ++device_read_counts[source_index]; }
    }

    // Complete all grouping allocations before a datasource can start accessing a destination.
    for (std::size_t source_index = 0; source_index < num_sources; ++source_index) {
      device_read_batches[source_index].reserve(device_read_counts[source_index]);
    }
    for (std::size_t io_index = 0; io_index < io_offsets.size(); ++io_index) {
      if (device_read_preferred[io_index]) {
        device_read_batches[io_source_indices[io_index]].push_back(
          {io_offsets[io_index], io_sizes[io_index], destinations[io_index]});
      }
    }
    for (std::size_t source_index = 0; source_index < num_sources; ++source_index) {
      auto const& batch = device_read_batches[source_index];
      if (batch.empty()) { continue; }
      device_batch_source_indices.push_back(source_index);
      auto& expected_sizes = expected_device_read_sizes->emplace_back();
      expected_sizes.reserve(batch.size());
      std::transform(
        batch.begin(), batch.end(), std::back_inserter(expected_sizes), [](auto const& r) {
          return r.size;
        });
    }

    for (std::size_t io_index = 0; io_index < io_offsets.size(); ++io_index) {
      if (not device_read_preferred[io_index]) {
        auto& datasource       = datasources[io_source_indices[io_index]].get();
        auto const io_offset   = io_offsets[io_index];
        auto const io_size     = io_sizes[io_index];
        auto const destination = destinations[io_index];
        // Asynchronously read column chunk data to a host buffer
        host_read_tasks.push(cudf::detail::host_worker_pool().submit_task(
          [&datasource, io_offset, io_size]() -> host_read_buffer {
            return datasource.host_read(io_offset, io_size);
          }));
        copy_dsts.push_back(static_cast<void*>(destination));
        copy_sizes.push_back(io_size);
      }
    }
  }

  auto device_read_tasks =
    std::make_shared<cudf::io::detail::future_drain_guard<std::vector<size_t>>>(
      device_batch_source_indices.size());

  // Construct the returned completion state before any device read is issued. A failure to allocate
  // that state therefore cannot strand an issued read. The shared guard also drains reads if the
  // returned deferred future is discarded without being consumed.
  auto read_completion =
    std::async(std::launch::deferred, [device_read_tasks, expected_device_read_sizes] {
      device_read_tasks->consume_all_indexed(
        [&expected_device_read_sizes](std::size_t task_index, std::vector<size_t> bytes_read) {
          auto const& expected_sizes = (*expected_device_read_sizes)[task_index];
          CUDF_EXPECTS(bytes_read.size() == expected_sizes.size(),
                       "Unexpected number of device reads completed.");
          for (std::size_t request_index = 0; request_index < bytes_read.size(); ++request_index) {
            CUDF_EXPECTS(bytes_read[request_index] == expected_sizes[request_index],
                         "Unexpected discrepancy in bytes read.");
          }
        });
    });

  // Complete host reads
  if (not host_read_tasks.empty()) {
    copy_srcs.reserve(host_read_tasks.size());
    host_buffers.reserve(host_read_tasks.size());

    host_read_tasks.consume_all_indexed([&](std::size_t task_index, host_read_buffer host_buffer) {
      CUDF_EXPECTS(host_buffer != nullptr, "Datasource returned a null host buffer.");
      CUDF_EXPECTS(host_buffer->size() == copy_sizes[task_index],
                   "Unexpected discrepancy in bytes read.");
      host_buffers.emplace_back(std::move(host_buffer));
      copy_srcs.push_back(host_buffers.back().get()->data());
    });
  }

  // Datasource device reads are not guaranteed to follow stream-ordering (see datasource API docs)
  cudf::io::detail::synchronize_stream(stream, device);

  std::exception_ptr device_schedule_error;
  std::exception_ptr host_copy_enqueue_error;
  // Schedule device reads holding the `device_read_mutex` so that all reads
  // for a caller thread are scheduled without interleaving with reads from
  // other threads, yielding better pipelining.
  {
    std::scoped_lock<std::mutex> lock(device_read_mutex);

    try {
      for (auto const source_index : device_batch_source_indices) {
        auto& datasource  = datasources[source_index].get();
        auto const& batch = device_read_batches[source_index];
        device_read_tasks->push(cudf::io::device_read_batch_async(
          datasource,
          cudf::host_span<device_read_request const>{batch.data(), batch.size()},
          stream));
      }
    } catch (...) {
      device_schedule_error = std::current_exception();
    }

    // Schedule a batched memcpy from host buffers to device
    if (device_schedule_error == nullptr and not host_buffers.empty()) {
      try {
        CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
          copy_dsts.data(), copy_srcs.data(), copy_sizes.data(), copy_dsts.size(), stream));
      } catch (...) {
        host_copy_enqueue_error = std::current_exception();
      }
    }
  }

  if (device_schedule_error != nullptr) {
    finish_failed_device_schedule(device_schedule_error, *device_read_tasks);
  }

  if (host_copy_enqueue_error != nullptr) {
    finish_failed_host_copy(host_copy_enqueue_error, true, stream, device, *device_read_tasks);
  }

  // Synchronize stream if `memcpy_batch_async` was called to safely discard the host buffers
  if (not host_buffers.empty()) {
    try {
      cudf::io::detail::synchronize_stream(stream, device);
    } catch (...) {
      finish_failed_host_copy(std::current_exception(), false, stream, device, *device_read_tasks);
    }
  }

  return {std::move(column_chunk_buffers),
          std::move(column_chunk_data_per_source),
          std::move(read_completion)};
}

}  // namespace

[[nodiscard]] std::size_t metadata_size_hint()
{
  static constexpr auto default_metadata_size_hint = std::size_t{64} * 1024;
  return cudf::detail::getenv_or<std::size_t>("LIBCUDF_PARQUET_METADATA_SIZE_HINT",
                                              default_metadata_size_hint);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  CUDF_FUNC_RANGE();
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  auto footer_buffers = fetch_footers_to_host_impl({datasources.data(), datasources.size()});
  return std::move(footer_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  CUDF_FUNC_RANGE();
  return fetch_footers_to_host_impl(datasources);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::io::text::byte_range_info, 1> page_index_bytes_per_source{page_index_bytes};

  auto page_index_buffers = fetch_page_indexes_to_host_impl(
    {datasources.data(), datasources.size()},
    {page_index_bytes_per_source.data(), page_index_bytes_per_source.size()});
  return std::move(page_index_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_FUNC_RANGE();
  return fetch_page_indexes_to_host_impl(datasources, page_index_bytes_per_source);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::io::datasource& datasource,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::host_span<cudf::io::text::byte_range_info const>, 1> byte_ranges_per_source{
    byte_ranges};

  auto [buffers, fetched_byte_ranges, fut] = fetch_byte_ranges_to_device_async_impl(
    {datasources.data(), datasources.size()},
    {byte_ranges_per_source.data(), byte_ranges_per_source.size()},
    stream,
    mr);

  return {std::move(buffers), std::move(fetched_byte_ranges.front()), std::move(fut)};
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<std::vector<cudf::device_span<uint8_t const>>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<std::vector<cudf::io::text::byte_range_info> const> byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Convert input vectors into host spans for the implementation
  std::vector<cudf::host_span<cudf::io::text::byte_range_info const>> byte_range_spans_per_source;
  byte_range_spans_per_source.reserve(byte_ranges_per_source.size());
  for (auto const& ranges : byte_ranges_per_source) {
    byte_range_spans_per_source.emplace_back(ranges);
  }
  return fetch_byte_ranges_to_device_async_impl(
    datasources,
    {byte_range_spans_per_source.data(), byte_range_spans_per_source.size()},
    stream,
    mr);
}

}  // namespace cudf::io::parquet
