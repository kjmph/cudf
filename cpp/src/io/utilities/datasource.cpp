/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/future_drain_guard.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/async_device_io.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/mmap.hpp>

#include <rmm/device_buffer.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <regex>
#include <utility>
#include <vector>

#ifdef CUDF_KVIKIO_REMOTE_IO
#include <kvikio/remote_handle.hpp>
#endif

namespace cudf {
namespace io {
namespace {

/**
 * @brief Base class for kvikIO-based data sources.
 */
template <typename HandleT>
class kvikio_source : public datasource {
  class kvikio_initializer {
   public:
    kvikio_initializer() { kvikio_integration::set_up_kvikio(); }
  };

  std::pair<std::vector<uint8_t>, std::future<size_t>> clamped_read_to_vector(size_t offset,
                                                                              size_t size)
  {
    // Clamp length to available data
    auto const read_size = std::min(size, this->size() - offset);
    std::vector<uint8_t> v(read_size);
    auto v_data = v.data();
    return {std::move(v), _kvikio_handle.pread(v_data, read_size, offset)};
  }

 public:
  kvikio_source(HandleT&& h) : _kvikio_handle(std::move(h)) {}
  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto [v, fut] = clamped_read_to_vector(offset, size);
    fut.get();
    return buffer::create(std::move(v));
  }

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override
  {
    auto clamped_read = clamped_read_to_vector(offset, size);
    return std::async(std::launch::deferred, [cr = std::move(clamped_read)]() mutable {
      cr.second.get();
      return buffer::create(std::move(cr.first));
    });
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    return host_read_async(offset, size, dst).get();
  }

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override
  {
    // Clamp length to available data
    auto const read_size = std::min(size, this->size() - offset);
    return _kvikio_handle.pread(dst, read_size, offset);
  }

  ~kvikio_source() override = default;

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return supports_device_read();
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");
    auto const read_size = std::min(size, this->size() - offset);
    stream.synchronize();
    return _kvikio_handle.pread(dst, read_size, offset);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override
  {
    rmm::device_buffer out_data(size, stream);
    size_t const read =
      device_read(offset, size, reinterpret_cast<uint8_t*>(out_data.data()), stream);
    out_data.resize(read, stream);
    return datasource::buffer::create(std::move(out_data));
  }

  [[nodiscard]] size_t size() const override { return _kvikio_handle.nbytes(); }

  kvikio_initializer _;

 protected:
  HandleT _kvikio_handle;
};

/**
 * @brief A class representing a file source using kvikIO.
 *
 * This class is derived from `kvikio_source` and is used to handle file operations
 * using kvikIO library.
 */
class file_source : public kvikio_source<kvikio::FileHandle> {
 public:
  explicit file_source(char const* filepath) : kvikio_source{kvikio::FileHandle(filepath, "r")}
  {
    CUDF_EXPECTS(!_kvikio_handle.closed(), "KvikIO did not open the file successfully.");
    CUDF_LOG_INFO(
      "Reading a file using kvikIO, with compatibility mode %s.",
      _kvikio_handle.get_compat_mode_manager().is_compat_mode_preferred() ? "on" : "off");
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    CUDF_EXPECTS(supports_device_read(), "Device reads are not supported for this file.");
    auto const read_size = std::min(size, this->size() - offset);
    stream.synchronize();
    return _kvikio_handle.pread(dst,
                                read_size,
                                offset,
                                kvikio::defaults::task_size(),
                                kvikio::defaults::gds_threshold(),
                                false /* not to sync_default_stream */);
  }
};

/**
 * @brief Implementation class for reading from a file using memory mapped access.
 *
 * Unlike Arrow's memory mapped IO class, this implementation allows memory mapping a subset of the
 * file where the starting offset may not be zero.
 */
class memory_mapped_source : public kvikio_source<kvikio::MmapHandle> {
 public:
  explicit memory_mapped_source(char const* filepath,
                                size_t offset,
                                [[maybe_unused]] size_t max_size_estimate)
    : kvikio_source{kvikio::MmapHandle()}
  {
    // Since the superclass kvikio_source is initialized with an empty mmap handle, `this->size()`
    // returns 0 at this point. Use `kvikio::get_file_size()` instead.
    auto const file_size = kvikio::get_file_size(filepath);
    if (file_size != 0) {
      CUDF_EXPECTS(offset < file_size, "Offset is past end of file", std::overflow_error);
      _kvikio_handle =
        kvikio::MmapHandle(filepath, "r", std::nullopt, 0, kvikio::FileHandle::m644, MAP_SHARED);
    }
  }
};

/**
 * @brief Implementation class for reading from a device buffer source
 */
class device_buffer_source final : public datasource {
 public:
  explicit device_buffer_source(cudf::device_span<std::byte const> d_buffer) : _d_buffer{d_buffer}
  {
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count  = std::min(size, this->size() - offset);
    auto const stream = cudf::detail::global_cuda_stream_pool().get_stream();
    cudf::detail::cuda_memcpy(host_span<uint8_t>{dst, count},
                              device_span<uint8_t const>{
                                reinterpret_cast<uint8_t const*>(_d_buffer.data() + offset), count},
                              stream);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count  = std::min(size, this->size() - offset);
    auto const stream = cudf::detail::global_cuda_stream_pool().get_stream();
    auto h_data       = cudf::detail::make_host_vector_async(
      cudf::device_span<std::byte const>{_d_buffer.data() + offset, count}, stream);
    stream.synchronize();
    return std::make_unique<owning_buffer<cudf::detail::host_vector<std::byte>>>(std::move(h_data));
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto const count = std::min(size, this->size() - offset);
    int device;
    CUDF_CUDA_TRY(cudf::io::detail::get_stream_device(stream, &device));

    auto completion_event = std::make_shared<cudf::io::detail::stream_completion_event>(device);
    auto completion       = std::async(std::launch::deferred, [completion_event, count] {
      completion_event->synchronize();
      return count;
    });

    try {
      auto const device_scope = cudf::io::detail::impl::scoped_cuda_device{device};
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, _d_buffer.data() + offset, count, stream));
      completion_event->record(stream);
    } catch (...) {
      auto const primary_exception = std::current_exception();
      try {
        cudf::io::detail::synchronize_stream(stream, device);
      } catch (...) {
        // The enqueue error remains primary. synchronize_stream has already established completion.
      }
      std::rethrow_exception(primary_exception);
    }

    return completion;
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    return std::make_unique<non_owning_buffer>(
      reinterpret_cast<uint8_t const*>(_d_buffer.data() + offset), size);
  }

  [[nodiscard]] size_t size() const override { return _d_buffer.size(); }

 private:
  cudf::device_span<std::byte const> _d_buffer;  ///< A non-owning view of the existing device data
};

// zero-copy host buffer source
class host_buffer_source final : public datasource {
 public:
  explicit host_buffer_source(cudf::host_span<std::byte const> h_buffer) : _h_buffer{h_buffer} {}

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count = std::min(size, this->size() - offset);
    std::memcpy(dst, _h_buffer.data() + offset, count);
    return count;
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, this->size() - offset);
    return std::make_unique<non_owning_buffer>(
      reinterpret_cast<uint8_t const*>(_h_buffer.data() + offset), count);
  }

  [[nodiscard]] bool supports_device_read() const override { return false; }

  [[nodiscard]] size_t size() const override { return _h_buffer.size(); }

 private:
  cudf::host_span<std::byte const> _h_buffer;  ///< A non-owning view of the existing host data
};

/**
 * @brief Wrapper class for user implemented data sources
 *
 * Holds the user-implemented object with a non-owning pointer; The user object is not deleted
 * when the wrapper object is destroyed.
 * All API calls are forwarded to the user datasource object.
 */
class user_datasource_wrapper : public datasource, public device_read_batch_source {
 public:
  explicit user_datasource_wrapper(datasource* const source) : source(source) {}

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    return source->host_read(offset, size, dst);
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    return source->host_read(offset, size);
  }

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override
  {
    return source->host_read_async(offset, size, dst);
  }

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(size_t offset,
                                                                   size_t size) override
  {
    return source->host_read_async(offset, size);
  }

  [[nodiscard]] bool supports_device_read() const override
  {
    return source->supports_device_read();
  }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override
  {
    return source->is_device_read_preferred(size);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return source->device_read(offset, size, dst, stream);
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    return source->device_read(offset, size, stream);
  }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    return source->device_read_async(offset, size, dst, stream);
  }

  std::future<std::vector<size_t>> device_read_batch_async(
    cudf::host_span<device_read_request const> requests, rmm::cuda_stream_view stream) override
  {
    return cudf::io::device_read_batch_async(*source, requests, stream);
  }

  [[nodiscard]] size_t size() const override { return source->size(); }

  [[nodiscard]] bool is_empty() const override { return source->is_empty(); }

 private:
  datasource* const source;  ///< A non-owning pointer to the user-implemented datasource
};

#ifdef CUDF_KVIKIO_REMOTE_IO
/**
 * @brief Remote file source backed by KvikIO, which handles S3 filepaths seamlessly.
 */
class remote_file_source : public kvikio_source<kvikio::RemoteHandle> {
 public:
  explicit remote_file_source(char const* filepath)
    : kvikio_source{kvikio::RemoteHandle::open(filepath)}
  {
  }

  ~remote_file_source() override = default;

  /**
   * @brief Checks if a path has a URL scheme format that could indicate a remote resource
   *
   * @note Strictly speaking, there is no definitive way to tell if a given file path refers to a
   * remote or local file. For instance, it is legal to have a local directory named `s3:` and its
   * file accessed by `s3://<sub-dir>/<file-name>` (the double slash is collapsed into a single
   * slash), coincidentally taking on the remote S3 format. Here we ignore this special case and use
   * a more practical approach: a file path is considered remote simply if it has a RFC
   * 3986-conformant URL scheme.
   */
  static bool could_be_remote_url(std::string const& filepath)
  {
    // Regular expression to match the URL scheme conforming to RFC 3986
    static std::regex const pattern{R"(^[a-zA-Z][a-zA-Z0-9+.-]*://)", std::regex_constants::icase};
    return std::regex_search(filepath, pattern);
  }
};
#else
/**
 * @brief When KvikIO remote IO is disabled, `is_supported_remote_url()` return false always.
 */
class remote_file_source : public file_source {
 public:
  explicit remote_file_source(char const* filepath) : file_source(filepath) {}
  static constexpr bool could_be_remote_url(std::string const&) { return false; }
};
#endif
}  // namespace

std::unique_ptr<datasource> datasource::create(std::string const& filepath,
                                               size_t offset,
                                               size_t max_size_estimate)
{
  auto const use_memory_mapping = [] {
    auto const policy = cudf::detail::getenv_or("LIBCUDF_MMAP_ENABLED", std::string{"OFF"});

    if (policy == "ON") { return true; }
    if (policy == "OFF") { return false; }

    CUDF_FAIL("Invalid LIBCUDF_MMAP_ENABLED value: " + policy);
  }();

  if (remote_file_source::could_be_remote_url(filepath)) {
    try {
      return std::make_unique<remote_file_source>(filepath.c_str());
    } catch (std::exception const& ex) {
      std::string redacted_msg;
      try {
        // For security reasons, redact the file path if any from KvikIO's exception message
        redacted_msg =
          std::regex_replace(ex.what(), std::regex{filepath}, "<redacted-remote-file-path>");
      } catch (std::exception const& ex) {
        redacted_msg = " unknown due to additional process error";
      }
      CUDF_FAIL("Error accessing the remote file. Reason: " + redacted_msg, std::runtime_error);
    }
  } else if (use_memory_mapping) {
    return std::make_unique<memory_mapped_source>(filepath.c_str(), offset, max_size_estimate);
  } else {
    // Reroute I/O: If the following two env vars are specified, the filepath for an existing local
    // file will be modified (only in-memory, not affecting the original file) such that the first
    // occurrence of local_dir_pattern is replaced by remote_dir_pattern, and a remote file resource
    // will be used instead of a local file resource.
    //
    // For example, let "LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN" be "/mnt/nvme/tmp", and
    // "LIBCUDF_IO_REROUTE_REMOTE_DIR_PATTERN" be
    // "http://example.com:9870/webhdfs/v1/home/ubuntu/data". If a local file with the name
    // "/mnt/nvme/tmp/test.bin" exists, libcudf will create a remote file resource with the URL
    // "http://example.com:9870/webhdfs/v1/home/ubuntu/data/test.bin"
    //
    // This feature can be used as a workaround for PDS-H benchmark using WebHDFS without the need
    // for upstream Polars change.
    auto* local_dir_pattern  = std::getenv("LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN");
    auto* remote_dir_pattern = std::getenv("LIBCUDF_IO_REROUTE_REMOTE_DIR_PATTERN");

    if (local_dir_pattern != nullptr and remote_dir_pattern != nullptr) {
      auto remote_file_path = std::regex_replace(filepath,
                                                 std::regex{local_dir_pattern},
                                                 remote_dir_pattern,
                                                 std::regex_constants::format_first_only);

      // Create a remote file resource only when the pattern is found and replaced; otherwise, still
      // create a local file resource
      if (filepath != remote_file_path) {
        return std::make_unique<remote_file_source>(remote_file_path.c_str());
      }
    }

    // `file_source` reads the file directly, without memory mapping
    return std::make_unique<file_source>(filepath.c_str());
  }
}

std::unique_ptr<datasource> datasource::create(cudf::host_span<std::byte const> buffer)
{
  return std::make_unique<host_buffer_source>(buffer);
}

std::unique_ptr<datasource> datasource::create(cudf::device_span<std::byte const> buffer)
{
  return std::make_unique<device_buffer_source>(buffer);
}

std::unique_ptr<datasource> datasource::create(datasource* source)
{
  // instantiate a wrapper that forwards the calls to the user implementation
  return std::make_unique<user_datasource_wrapper>(source);
}

std::future<std::unique_ptr<datasource::buffer>> datasource::host_read_async(size_t offset,
                                                                             size_t size)
{
  return std::async(std::launch::deferred,
                    [this, offset, size] { return host_read(offset, size); });
}

std::future<size_t> datasource::host_read_async(size_t offset, size_t size, uint8_t* dst)
{
  return std::async(std::launch::deferred,
                    [this, offset, size, dst] { return host_read(offset, size, dst); });
}

std::future<std::vector<size_t>> device_read_batch_async(
  datasource& source,
  cudf::host_span<datasource::device_read_request const> requests,
  rmm::cuda_stream_view stream)
{
  struct batch_state {
    explicit batch_state(std::size_t size) : tasks{size}, results(size)
    {
      request_indices.reserve(size);
    }

    cudf::io::detail::future_drain_guard<size_t> tasks;
    std::vector<size_t> results;
    std::vector<size_t> request_indices;
  };

  std::vector<std::pair<std::uintptr_t, std::uintptr_t>> destination_ranges;
  destination_ranges.reserve(requests.size());
  for (auto const& request : requests) {
    CUDF_EXPECTS(request.size == 0 or request.dst != nullptr,
                 "A nonempty device read requires a non-null destination.");
    CUDF_EXPECTS(request.size <= std::numeric_limits<size_t>::max() - request.offset,
                 "Device read source range overflows.");
    if (request.size != 0) {
      auto const begin = reinterpret_cast<std::uintptr_t>(request.dst);
      CUDF_EXPECTS(request.size <= std::numeric_limits<std::uintptr_t>::max() - begin,
                   "Device read destination range overflows.");
      destination_ranges.emplace_back(begin, begin + request.size);
    }
  }
  std::sort(destination_ranges.begin(), destination_ranges.end());
  for (std::size_t index = 1; index < destination_ranges.size(); ++index) {
    CUDF_EXPECTS(destination_ranges[index].first >= destination_ranges[index - 1].second,
                 "Device read destination ranges overlap.");
  }

  if (auto* batch_source = dynamic_cast<device_read_batch_source*>(&source);
      batch_source != nullptr) {
    return batch_source->device_read_batch_async(requests, stream);
  }

  // Allocate all fallback bookkeeping before issuing a read. The drain guard then owns every issued
  // future until completion, including when scheduling or result consumption throws.
  auto state      = std::make_shared<batch_state>(requests.size());
  auto completion = std::async(std::launch::deferred, [state]() mutable {
    std::exception_ptr completion_error;
    try {
      state->tasks.consume_all_indexed([&state](std::size_t index, std::size_t bytes_read) {
        state->results[state->request_indices[index]] = bytes_read;
      });
    } catch (...) {
      completion_error = std::current_exception();
    }
    if (completion_error != nullptr) { std::rethrow_exception(completion_error); }
    return std::move(state->results);
  });

  try {
    for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
      auto const& request = requests[request_index];
      if (request.size == 0) {
        state->results[request_index] = 0;
      } else {
        state->request_indices.push_back(request_index);
        state->tasks.push(
          source.device_read_async(request.offset, request.size, request.dst, stream));
      }
    }
  } catch (...) {
    auto const scheduling_error = std::current_exception();
    try {
      state->tasks.wait_all();
    } catch (...) {
      // The synchronous scheduling failure remains primary after every previously issued read has
      // stopped accessing its destination.
    }
    std::rethrow_exception(scheduling_error);
  }

  return completion;
}

}  // namespace io
}  // namespace cudf
