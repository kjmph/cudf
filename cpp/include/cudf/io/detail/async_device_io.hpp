/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <exception>

namespace cudf::io::detail {

/**
 * @brief Returns the device associated with a CUDA stream.
 *
 * CUDA 12.8 added an exact runtime query for a stream's device. Older CUDA toolkits supported by
 * cuDF do not expose an equivalent runtime API, so they retain the historical requirement that the
 * stream belong to the calling thread's current device.
 */
inline cudaError_t get_stream_device(rmm::cuda_stream_view stream, int* device) noexcept
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  return cudaStreamGetDevice(stream.value(), device);
#else
  static_cast<void>(stream);
  return cudaGetDevice(device);
#endif
}

namespace impl {

class scoped_cuda_device {
 public:
  explicit scoped_cuda_device(int device) noexcept
  {
    if (cudaGetDevice(&original_device_) != cudaSuccess) { std::terminate(); }
    changed_ = original_device_ != device;
    if (changed_ and cudaSetDevice(device) != cudaSuccess) { std::terminate(); }
  }

  ~scoped_cuda_device() noexcept
  {
    if (changed_ and cudaSetDevice(original_device_) != cudaSuccess) { std::terminate(); }
  }

  scoped_cuda_device(scoped_cuda_device const&)            = delete;
  scoped_cuda_device& operator=(scoped_cuda_device const&) = delete;

 private:
  int original_device_{};
  bool changed_{};
};

}  // namespace impl

/**
 * @brief Producer-recorded completion for asynchronous device I/O.
 *
 * Unlike a CUDA stream handle, a CUDA event retains the identity of work recorded on a
 * per-thread default stream when it is synchronized by another thread. The destructor also waits
 * for a recorded event, so discarding a deferred completion future cannot leave an outstanding
 * access to its destination buffer.
 *
 * Recording and synchronization must not be performed concurrently.
 */
class stream_completion_event {
 public:
  explicit stream_completion_event(int device) : device_{device}
  {
    auto const device_scope = impl::scoped_cuda_device{device_};
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~stream_completion_event() noexcept
  {
    if (event_ == nullptr) { return; }

    auto const device_scope = impl::scoped_cuda_device{device_};
    if (recorded_ and not completed_) {
      auto const event_status = cudaEventSynchronize(event_);
      if (event_status != cudaSuccess and cudaDeviceSynchronize() != cudaSuccess) {
        std::terminate();
      }
    }
    if (cudaEventDestroy(event_) != cudaSuccess) { std::terminate(); }
  }

  stream_completion_event(stream_completion_event const&)            = delete;
  stream_completion_event& operator=(stream_completion_event const&) = delete;
  stream_completion_event(stream_completion_event&&)                 = delete;
  stream_completion_event& operator=(stream_completion_event&&)      = delete;

  /**
   * @brief Records completion after all device I/O has been enqueued on the producer thread.
   */
  void record(rmm::cuda_stream_view stream)
  {
    auto const device_scope = impl::scoped_cuda_device{device_};
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
    recorded_ = true;
  }

  /**
   * @brief Waits for recorded work, with a device-wide fail-stop fallback.
   *
   * If event synchronization fails but device-wide synchronization succeeds, this function
   * rethrows the original event error after completion has been established. If the fallback
   * fence fails, returning would be unsafe and the process is terminated.
   */
  void synchronize()
  {
    CUDF_EXPECTS(recorded_, "Cannot wait for an unrecorded device I/O completion event.");
    if (completed_) { return; }

    auto const device_scope = impl::scoped_cuda_device{device_};
    try {
      CUDF_CUDA_TRY(cudaEventSynchronize(event_));
      completed_ = true;
    } catch (...) {
      auto const primary_exception = std::current_exception();
      if (cudaDeviceSynchronize() != cudaSuccess) { std::terminate(); }
      completed_ = true;
      std::rethrow_exception(primary_exception);
    }
  }

 private:
  int device_{};
  cudaEvent_t event_{};
  bool recorded_{};
  bool completed_{};
};

/**
 * @brief Synchronizes a stream on its device, with a device-wide fail-stop fallback.
 *
 * If stream synchronization fails but device-wide synchronization succeeds, this function rethrows
 * the original stream error after completion has been established. If the fallback fence fails,
 * returning would be unsafe and the process is terminated.
 */
inline void synchronize_stream(rmm::cuda_stream_view stream, int device)
{
  auto const device_scope = impl::scoped_cuda_device{device};
  try {
    stream.synchronize();
  } catch (...) {
    auto const primary_exception = std::current_exception();
    if (cudaDeviceSynchronize() != cudaSuccess) { std::terminate(); }
    std::rethrow_exception(primary_exception);
  }
}

}  // namespace cudf::io::detail
