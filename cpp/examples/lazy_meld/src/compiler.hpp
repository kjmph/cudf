/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <meld/Execution/ExecutionUnitABI.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The only translation unit that includes LLVM/MLIR is compiler.cpp. This interface is private;
// data engines see neither Meld's IR classes nor its execution runtime.
namespace cudf::experimental::lazy::detail {
struct compiled_artifact {
  std::string ptx;
  std::vector<std::uint8_t> literals;
};

struct compiled_cpu_artifact {
  std::shared_ptr<void> lifetime;  // Retains Meld's JIT without exposing LLVM in the executor.
  mlir::meld::ExecutionUnitFn entry{};
  std::vector<std::uint8_t> literals;
};

bool compile_meld(std::string const& ir,
                  std::string const& architecture,
                  compiled_artifact& artifact,
                  std::string& error);
bool compile_meld_cpu(std::string const& ir, compiled_cpu_artifact& artifact, std::string& error);
}  // namespace cudf::experimental::lazy::detail
