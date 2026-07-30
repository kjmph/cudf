/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/scan.h>

#include <barrier>
#include <exception>
#include <future>

namespace {

class HashPartitionScanTest : public cudf::test::BaseFixture {};

TEST_F(HashPartitionScanTest, ConcurrentProductionSizedExclusiveScansOnDistinctStreams)
{
  // The failing Q20 hash-partition batch had 256,248,272 rows. With 4,096 rows per block and 52
  // partitions, hash_partition scans 62,561 * 52 = 3,253,172 block-partition sizes, followed by 52
  // global partition sizes. Exercise those two scan shapes concurrently because they select
  // different Blackwell scan configurations while sharing the same kernel instantiation.
  constexpr cudf::size_type large_scan_size = 3'253'172;
  constexpr cudf::size_type small_scan_size = 52;
  constexpr int num_iterations              = 10'000;

  int device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));

  std::barrier dispatch_barrier{2};
  auto const run_scans = [device, &dispatch_barrier](int worker) {
    CUDF_CUDA_TRY(cudaSetDevice(device));
    rmm::cuda_stream stream;
    rmm::device_uvector<cudf::size_type> input(large_scan_size, stream);
    rmm::device_uvector<cudf::size_type> output(large_scan_size, stream);

    auto const mr = cudf::get_current_device_resource_ref();
    thrust::fill(rmm::exec_policy_nosync(stream, mr), input.begin(), input.end(), 0);
    thrust::fill(rmm::exec_policy_nosync(stream, mr), output.begin(), output.end(), -1);
    stream.synchronize();

    std::exception_ptr failure;
    cudf::size_type final_scan_size = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
      auto const run_large_scan = (iteration + worker) % 2 == 0;
      final_scan_size           = run_large_scan ? large_scan_size : small_scan_size;

      dispatch_barrier.arrive_and_wait();
      if (failure == nullptr) {
        try {
          thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                                 input.begin(),
                                 input.begin() + final_scan_size,
                                 output.begin());
        } catch (...) {
          // Keep participating in the barrier so the peer cannot deadlock, then propagate the
          // first launch failure through the future after both workers finish.
          failure = std::current_exception();
        }
      }
      dispatch_barrier.arrive_and_wait();
    }

    if (failure != nullptr) { std::rethrow_exception(failure); }
    stream.synchronize();

    cudf::size_type final_value = -1;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&final_value,
                                  output.data() + final_scan_size - 1,
                                  sizeof(final_value),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
    return final_value;
  };

  auto first_worker  = std::async(std::launch::async, run_scans, 0);
  auto second_worker = std::async(std::launch::async, run_scans, 1);

  EXPECT_EQ(first_worker.get(), 0);
  EXPECT_EQ(second_worker.get(), 0);
}

}  // namespace
