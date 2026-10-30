/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compiler.hpp"
#include "kernels.hpp"

#include <cudf/experimental/lazy/meld.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <meld/Execution/ExecutionUnitABI.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::experimental::lazy {
namespace {
void check_driver(CUresult result, char const* operation)
{
  if (result == CUDA_SUCCESS) { return; }
  char const* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string{operation} + ": " +
                           (message ? message : "CUDA driver error"));
}

// This is launch metadata, not an alternative column representation. The pointer tables below
// follow Meld's public ExecutionUnitABI; data and masks point into the original cuDF allocations.
struct alignas(8) launch_header {
  std::int64_t fragment_count{1};
  std::int64_t input_rows{};
  std::int64_t output_rows{};
  MeldStatus status{meldStatusOk};
  std::int32_t padding{};
};
static_assert(sizeof(void*) == 8);
static_assert(mlir::meld::ExecutionUnitABI::pointersPerColumn == 2);
static_assert(mlir::meld::ExecutionUnitABI::valuesPointerOffset == 0);
static_assert(mlir::meld::ExecutionUnitABI::validityPointerOffset == 1);

class pinned_buffer {
 public:
  explicit pinned_buffer(std::size_t size)
  {
    CUDF_CUDA_TRY(cudaMallocHost(&_data, size));
    std::memset(_data, 0, size);
  }
  ~pinned_buffer() { cudaFreeHost(_data); }
  pinned_buffer(pinned_buffer const&)            = delete;
  pinned_buffer& operator=(pinned_buffer const&) = delete;
  void* data() { return _data; }
  template <typename T>
  void put(std::size_t offset, T const& value)
  {
    std::memcpy(static_cast<std::byte*>(_data) + offset, &value, sizeof(T));
  }

 private:
  void* _data{};
};
}  // namespace

namespace detail {
struct program_state {
  std::size_t input_columns{};
  std::vector<data_type> input_types;
  std::string ir;
  compiled_artifact artifact;
  CUcontext context{};
  CUmodule module{};
  CUfunction kernel{};
  unsigned grid_limit{};

  ~program_state()
  {
    if (module && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
      cuModuleUnload(module);
      CUcontext previous{};
      cuCtxPopCurrent(&previous);
    }
  }
};

struct pending_state {
  // The program stays alive even if the caller drops its compiled-program handle.
  std::shared_ptr<program_state> program;
  cuda::stream_ref stream;
  std::unique_ptr<cudf::numeric_scalar<double>> output;
  rmm::device_buffer metadata;
  rmm::device_buffer partials;
  rmm::device_buffer validity;
  std::vector<rmm::device_buffer> normalized_masks;
  pinned_buffer host_metadata;
  cudaEvent_t complete{};
  bool enqueued{false};
  bool recorded{false};

  pending_state(std::shared_ptr<program_state> compiled,
                cuda::stream_ref launch_stream,
                rmm::device_async_resource_ref mr,
                std::size_t metadata_bytes,
                std::size_t partial_rows)
    : program(std::move(compiled)),
      stream(launch_stream),
      output(std::make_unique<cudf::numeric_scalar<double>>(0.0, false, stream, mr)),
      metadata(metadata_bytes, stream, mr),
      partials(std::max<std::size_t>(1, partial_rows) * sizeof(double), stream, mr),
      validity(
        ((std::max<std::size_t>(1, partial_rows) + 31) / 32) * sizeof(std::uint32_t), stream, mr),
      host_metadata(metadata_bytes)
  {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
  }
  ~pending_state()
  {
    // Retain host upload storage, device scratch and module until the queued work is finished.
    // On an exception before event recording, drain only the submission stream for safe cleanup.
    if (recorded) {
      cudaEventSynchronize(complete);
    } else if (enqueued) {
      cudaStreamSynchronize(stream.get());
    }
    if (complete) { cudaEventDestroy(complete); }
  }
  std::size_t scratch_bytes() const
  {
    auto bytes = metadata.size() + partials.size() + validity.size();
    for (auto const& mask : normalized_masks) {
      bytes += mask.size();
    }
    return bytes;
  }
};
}  // namespace detail

meld_program::meld_program(std::shared_ptr<detail::program_state> state) : _state(std::move(state))
{
}
std::string const& meld_program::plan_ir() const { return _state->ir; }
std::string const& meld_program::ptx() const { return _state->artifact.ptx; }

meld_program compile(reduction const& query)
{
  auto state           = std::make_shared<detail::program_state>();
  state->input_columns = query.input_columns();
  state->input_types   = query.input_types();
  state->ir            = query.to_meld_ir();
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  // Context initialization belongs to cold compile, never the repeated submit path.
  CUDF_CUDA_TRY(cudaFree(nullptr));
  check_driver(cuCtxGetCurrent(&state->context), "get current CUDA context");
  cudaDeviceProp properties{};
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&properties, device));
  auto const architecture =
    "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
  state->grid_limit = static_cast<unsigned>(std::max(1, properties.multiProcessorCount) * 4);
  std::string error;
  if (!detail::compile_meld(state->ir, architecture, state->artifact, error)) {
    throw std::runtime_error("Meld compilation failed:\n" + error);
  }
  std::array<char, 8192> log{};
  CUjit_option options[]{CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* values[]{log.data(), reinterpret_cast<void*>(log.size())};
  auto const loaded = cuModuleLoadDataEx(
    &state->module, state->artifact.ptx.c_str(), std::size(options), options, values);
  if (loaded != CUDA_SUCCESS) {
    throw std::runtime_error("CUDA could not load Meld PTX: " + std::string{log.data()} +
                             " (driver status " + std::to_string(loaded) + ")");
  }
  auto const name = std::string{mlir::meld::executionUnitPrefix} + "0";
  check_driver(cuModuleGetFunction(&state->kernel, state->module, name.c_str()),
               "find Meld kernel");
  return meld_program{std::move(state)};
}

pending_result meld_program::submit(cudf::table_view input,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr) const
{
  if (!_state) { throw std::invalid_argument("moved-from Meld program"); }
  if (static_cast<std::size_t>(input.num_columns()) != _state->input_columns) {
    throw std::invalid_argument("input column count does not match the compiled lazy scan");
  }
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    auto expected = _state->input_types[i] == data_type::date32 ? cudf::type_id::TIMESTAMP_DAYS
                                                                : cudf::type_id::FLOAT64;
    if (input.column(i).type().id() != expected) {
      throw std::invalid_argument("GPU input type does not match the compiled lazy schema");
    }
  }
  CUcontext context{};
  check_driver(cuCtxGetCurrent(&context), "get submission context");
  if (context != _state->context) {
    throw std::invalid_argument("submit must use the CUDA context used by compile");
  }

  constexpr unsigned block_size = 256;
  auto const rows               = static_cast<std::int64_t>(input.num_rows());
  auto const grid               = static_cast<unsigned>(
    std::clamp<std::int64_t>((rows + block_size - 1) / block_size, 1, _state->grid_limit));
  // Meld writes one partial per lane that had an input row, regardless of filter selectivity.
  auto const partial_rows =
    std::min<std::int64_t>(rows, static_cast<std::int64_t>(grid) * block_size);
  std::size_t const table_offset   = sizeof(launch_header);
  std::size_t const inputs_offset  = table_offset + 2 * _state->input_columns * sizeof(void*);
  std::size_t const outputs_offset = inputs_offset + sizeof(void*);
  std::size_t const literal_offset = outputs_offset + 2 * sizeof(void*);
  std::size_t const metadata_bytes = literal_offset + _state->artifact.literals.size();
  auto state                       = std::make_unique<detail::pending_state>(
    _state, stream, mr, metadata_bytes, static_cast<std::size_t>(partial_rows));
  auto* device_metadata = static_cast<std::byte*>(state->metadata.data());
  launch_header header;
  header.input_rows = rows;
  state->host_metadata.put(0, header);
  state->host_metadata.put(inputs_offset, device_metadata + table_offset);
  state->host_metadata.put(outputs_offset, state->partials.data());
  state->host_metadata.put(outputs_offset + sizeof(void*), state->validity.data());
  state->enqueued = true;
  for (std::size_t i = 0; i < _state->input_columns; ++i) {
    auto const column = input.column(static_cast<cudf::size_type>(i));
    void const* data  = nullptr;
    if (rows != 0)
      data = _state->input_types[i] == data_type::date32
               ? static_cast<void const*>(column.data<cudf::timestamp_D>())
               : static_cast<void const*>(column.data<double>());
    void const* mask = nullptr;
    if (column.nullable() && rows != 0) {
      if (column.offset() % 8 == 0) {
        mask = reinterpret_cast<std::byte const*>(column.null_mask()) + column.offset() / 8;
      } else {
        state->normalized_masks.emplace_back(
          static_cast<std::size_t>((rows + 31) / 32) * sizeof(std::uint32_t), stream, mr);
        auto* normalized = static_cast<std::uint32_t*>(state->normalized_masks.back().data());
        detail::normalize_mask(column.null_mask(), column.offset(), rows, normalized, stream.get());
        mask = normalized;
      }
    }
    state->host_metadata.put(table_offset + (2 * i) * sizeof(void*), data);
    state->host_metadata.put(table_offset + (2 * i + 1) * sizeof(void*), mask);
  }
  if (!_state->artifact.literals.empty()) {
    std::memcpy(static_cast<std::byte*>(state->host_metadata.data()) + literal_offset,
                _state->artifact.literals.data(),
                _state->artifact.literals.size());
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(state->metadata.data(),
                                state->host_metadata.data(),
                                metadata_bytes,
                                cudaMemcpyDefault,
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemsetAsync(state->validity.data(), 0, state->validity.size(), stream.get()));
  std::int64_t start = 0, stride = 1, capacity = partial_rows;
  void* fragments  = device_metadata + offsetof(launch_header, fragment_count);
  void* literals   = _state->artifact.literals.empty() ? nullptr : device_metadata + literal_offset;
  void* inputs     = device_metadata + inputs_offset;
  void* input_rows = device_metadata + offsetof(launch_header, input_rows);
  void* joins      = nullptr;
  void* outputs    = device_metadata + outputs_offset;
  void* output_rows = device_metadata + offsetof(launch_header, output_rows);
  void* status      = device_metadata + offsetof(launch_header, status);
  // Argument order is exactly meld/Execution/ExecutionUnitABI.h.
  void* arguments[]{&start,
                    &stride,
                    &fragments,
                    &literals,
                    &inputs,
                    &input_rows,
                    &joins,
                    &outputs,
                    &output_rows,
                    &capacity,
                    &status};
  check_driver(cuLaunchKernel(
                 _state->kernel, grid, 1, 1, block_size, 1, 1, 0, stream.get(), arguments, nullptr),
               "launch Meld fragment");
  detail::finalize_sum(static_cast<double const*>(state->partials.data()),
                       static_cast<std::uint32_t const*>(state->validity.data()),
                       static_cast<std::int64_t const*>(output_rows),
                       state->output->data(),
                       state->output->validity_data(),
                       stream.get());
  CUDF_CUDA_TRY(cudaEventRecord(state->complete, stream.get()));
  state->recorded = true;
  return pending_result{std::move(state)};
}

pending_result::pending_result(std::unique_ptr<detail::pending_state> state)
  : _state(std::move(state))
{
}
pending_result::~pending_result()                                    = default;
pending_result::pending_result(pending_result&&) noexcept            = default;
pending_result& pending_result::operator=(pending_result&&) noexcept = default;

bool pending_result::ready() const
{
  if (!_state) { throw std::logic_error("moved-from lazy result"); }
  auto const result = cudaEventQuery(_state->complete);
  if (result == cudaErrorNotReady) { return false; }
  CUDF_CUDA_TRY(result);
  return true;
}
void pending_result::wait() const
{
  if (!_state) { throw std::logic_error("moved-from lazy result"); }
  CUDF_CUDA_TRY(cudaEventSynchronize(_state->complete));
}
void pending_result::wait_on(cuda::stream_ref consumer) const
{
  if (!_state) { throw std::logic_error("moved-from lazy result"); }
  CUDF_CUDA_TRY(cudaStreamWaitEvent(consumer.get(), _state->complete, 0));
}
cudf::numeric_scalar<double> const& pending_result::result() const
{
  if (!_state || !_state->output) { throw std::logic_error("lazy result has no output"); }
  return *_state->output;
}
std::unique_ptr<cudf::numeric_scalar<double>> pending_result::take_result()
{
  result();
  wait();
  return std::move(_state->output);
}
std::size_t pending_result::scratch_bytes() const
{
  if (!_state) { throw std::logic_error("moved-from lazy result"); }
  return _state->scratch_bytes();
}
}  // namespace cudf::experimental::lazy
