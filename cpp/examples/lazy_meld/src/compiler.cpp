/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "compiler.hpp"

#include <llvm/Support/raw_ostream.h>
#include <meld-builtin/Builtin.h>
#include <meld/Execution/JitEngine.h>
#include <meld/Execution/LiteralBuffer.h>
#include <meld/Execution/PtxCompiler.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <mutex>
#include <utility>

namespace cudf::experimental::lazy::detail {
namespace {
void register_builtins()
{
  static std::once_flag builtins;
  std::call_once(builtins, [] { mlir::meld::registerBuiltinLowering(); });
}
}  // namespace

bool compile_meld(std::string const& ir,
                  std::string const& architecture,
                  compiled_artifact& artifact,
                  std::string& error)
{
  register_builtins();
  mlir::DialectRegistry registry;
  mlir::meld::registerPtxDialects(registry);
  mlir::MLIRContext context(registry);
  error.clear();
  llvm::raw_string_ostream diagnostics(error);
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& diagnostic) {
    diagnostic.print(diagnostics);
    diagnostics << '\n';
    return mlir::success();
  });
  auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
  if (!module) { return false; }
  auto ptx = mlir::meld::compileToPtx(*module, architecture);
  if (mlir::failed(ptx)) { return false; }
  auto literals = mlir::meld::literalBuffersOf(*module);
  if (literals.size() > 1) {
    diagnostics << "lazy Meld backend expected at most one literal buffer\n";
    return false;
  }
  artifact.ptx = std::move(*ptx);
  // Meld omits the module's literals attribute entirely when no unit reads a literal.
  artifact.literals = literals.empty() ? std::vector<std::uint8_t>{} : std::move(literals.front());
  return true;
}

bool compile_meld_cpu(std::string const& ir, compiled_cpu_artifact& artifact, std::string& error)
{
  register_builtins();
  mlir::DialectRegistry registry;
  mlir::meld::registerJitDialects(registry);
  mlir::MLIRContext context(registry);
  error.clear();
  llvm::raw_string_ostream diagnostics(error);
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& diagnostic) {
    diagnostic.print(diagnostics);
    diagnostics << '\n';
    return mlir::success();
  });
  auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &context);
  if (!module) { return false; }
  auto engine = mlir::meld::JitEngine::compile(*module);
  if (mlir::failed(engine)) { return false; }
  auto entry = (*engine)->lookupUnit(0);
  if (mlir::failed(entry)) {
    diagnostics << "Meld CPU JIT did not produce execution unit zero\n";
    return false;
  }
  auto literals = mlir::meld::literalBuffersOf(*module);
  if (literals.size() > 1) {
    diagnostics << "lazy Meld CPU backend expected at most one literal buffer\n";
    return false;
  }
  artifact.lifetime = *engine;
  artifact.entry    = *entry;
  artifact.literals = literals.empty() ? std::vector<std::uint8_t>{} : std::move(literals.front());
  return true;
}
}  // namespace cudf::experimental::lazy::detail
