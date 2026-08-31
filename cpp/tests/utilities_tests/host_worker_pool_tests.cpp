/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/host_worker_pool.hpp>

#include <gtest/gtest.h>

TEST(HostWorkerPoolTest, SubmitTaskThroughExportedAccessor)
{
  auto result = cudf::detail::host_worker_pool().submit_task([] { return 42; });

  EXPECT_EQ(result.get(), 42);
}
