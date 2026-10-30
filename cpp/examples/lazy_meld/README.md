# Lazy cuDF operations with a Meld backend — proof of concept

The new [experimental graph execution API](EXPERIMENTAL_API.md) extends this POC with explicit
planning, inspectable implementation choices, expression-based grouping, and multiple execution
fragments. Its CPU path runs here; the mixed cuDF/Meld GPU adapter still requires GPU-host validation.

This is an opt-in C++ experiment inside the cuDF checkout. An engine builds a deferred chain of cuDF-shaped operations; this library translates the chain into Meld IR and compiles a reusable CPU or GPU program. The engine never constructs a Meld plan. The GPU adapter accepts ordinary `cudf::table_view` buffers and returns a device-resident `cudf::numeric_scalar<double>`. The CPU adapter accepts host column views and returns `std::optional<double>`.

**Status (2026-10-29):** [full Q6](Q6.md) now includes typed DATE32 shipping-date predicates, using Meld's existing date support. Six host tests pass, including native CPU execution and actual PTX compilation. The full query matches DuckDB's decimal SQL result on 60,175 generated lineitem rows: `1193053.2253`. The CUDA executor and GPU correctness/performance tests remain unbuilt and unrun. [Current validation](results/q6-cpu-aarch64-2026-10-29.txt) records the commands and results.

The earlier [CPU measurements](results/cpu-aarch64-2026-10-27.txt) used the date-free, Q6-shaped graph. The [Meld regression log](results/meld-regressions-2026-10-27.txt) records 103 passes and 18 unsupported GPU/cuDF tests from that earlier revision. No additional Meld changes were needed to expose dates.

**Fork handoff (2026-10-30):** the current research checkout is `kjmph_cudf/` on
`research/meld`, paired with Meld's `kjmph/research/cudf` branch. A fresh native build
from the fork and its validation are recorded in
[the handoff log](results/fork-handoff-2026-10-30.txt). Its executables are under
`/tmp/jit-research-build`. Workspace commands below assume a clone named `cudf/`;
substitute `kjmph_cudf/` when using the fork checkout.

The starting revisions are cuDF `b3378b6246e77ab8c6a2716606becc82b1446383` (26.12 development, local branch `poc/lazy-meld`) and Meld `0bcb442bd37dd1a4dd5bc0a17d39d7249f8d5d0e` (local branch `poc/cudf-lazy`). Keep the local changes in **both** repositories when moving hosts. No change to DBCompiler is needed.

## Engine-facing API

```cpp
#include <cudf/experimental/lazy/meld.hpp>

namespace lazy = cudf::experimental::lazy;
using lazy::col;
using lazy::lit;
using lazy::date;
using T = lazy::data_type;

// Schema only. No allocation, kernel launch, or input data access occurs here.
// quantity, extendedprice, discount, shipdate; all nullable.
auto input = lazy::scan({T::float64, T::float64, T::float64, T::date32});
auto selected = input.filter((col(3) >= lit(date(1994, 1, 1))) &&
                             (col(3) < lit(date(1995, 1, 1))) &&
                             (col(0) < lit(24.0)) &&
                             (col(2) >= lit(0.05)) &&
                             (col(2) <= lit(0.07)));
auto revenue = selected.project({col(1) * col(2)});
auto query = revenue.sum(0);

// Explicit cold boundary: Meld lowering -> LLVM/NVPTX -> driver module load.
auto program = lazy::compile(query);

// Reuse program for each batch. Pass the engine's stream and RMM resource.
auto pending = program.submit(batch_view, stream, memory_resource);
// Work can overlap here. Keep batch buffers unchanged and alive until completion.
auto const& result = pending.result();  // cuDF scalar, still on the device
// A cuDF consumer on `stream` can use result immediately, in stream order.
// For another stream: pending.wait_on(consumer_stream), then enqueue the consumer.
// For host access: pending.wait(); result.value(stream), checking validity first.
```

Column indices are relative to each operation's immediate input; projection changes the schema used by later operations. Handles are immutable, so branching a handle does not mutate its earlier consumers. `query.to_meld_ir()`, `program.plan_ir()`, and `program.ptx()` expose what the compiler receives and produces.

This does **not** intercept existing eager libcudf or Python calls. Engines must opt into these deferred handles. It establishes one compilable fragment ending in a scalar reduction, not an automatic planner for arbitrary cuDF operations or a whole-query optimizer. Existing libcudf binaries and their public eager interfaces are unchanged; the experiment is a separately built library under `cpp/examples/lazy_meld`.

## Run the same graph on CPU

```cpp
#include <cudf/experimental/lazy/cpu.hpp>

auto program = lazy::compile(q6(), lazy::cpu);

// Three double vectors/spans and one lazy::date32 vector/span; optional validity bitmaps.
std::array<lazy::host_column_view, 4> columns{{
  {quantity}, {price}, {discount}, {shipdate}
}};
auto pending = program.submit(columns);
auto result = pending.get();  // std::optional<double>; work ran on a CPU worker
```

`q6()` is exactly the same operation-building function for both targets; the CPU/GPU tests share it in `tests/q6_workload.hpp`, along with the input generator and reference calculation. CPU execution calls Meld's native LLVM JIT entry point over host buffers. It does not interpret the graph, run a hand-coded Q6 substitute, or initialize CUDA. The CPU executable has no libcudf, CUDA runtime/driver, or RMM shared-library dependency. A precise description is **"a lazy cuDF API prototype executing through Meld on CPU"**; this is not evidence that unmodified libcudf executes its existing APIs on CPU.

`cpu_program::run()` executes synchronously on the caller thread. `submit()` uses `std::async` and returns a future; this POC creates one worker per submission and does not yet use a persistent pool or partition a batch across cores. The compiled program is retained by each submission. Input descriptors are copied, but the actual data/bitmap buffers remain borrowed and must outlive completion. Unaligned nullable slices normalize the bitmap on CPU. CPU output is a host scalar, so there is no device-scalar allocation or upload.

The October 27 recorded run used the earlier date-free graph in a Linux ARM64 development container, one CPU thread per execution, one million synthetic rows, and medians of 20 warmed iterations. Cold compilation was 16.12 ms; synchronous execution was 1.19 ms without nulls and 1.39 ms with nulls. Both results matched the independent CPU reference. These are historical local prototype measurements, not full-Q6 results or a comparison against GPU cuDF or a tuned CPU engine. The `cold_meld_cpu_compile_ms` field measures compilation, not query execution. See [Q6.md](Q6.md) for the current graph and reproducible validation.

## What executes

```text
lazy scan -> filter(s) -> project(s) -> scalar SUM
                      |
                generated Meld plan
                      |
                  Meld + LLVM
                 /           \
           native CPU        NVPTX
               |               |
host buffers -> fused loop     cuDF buffers -> fused GPU kernel
               |                              |
          host scalar                  lane partials -> GPU finalizer -> cuDF scalar
```

Neither adapter materializes filter-result or projection-result columns. Input value buffers are borrowed directly. The GPU runtime uses the caller's stream and RMM resource for all device allocations, including launch metadata, partial sums, validity masks, and output. A small pinned host buffer holds launch metadata and literals for an asynchronous upload; input/output payloads never pass through host Arrow. No device-wide synchronization or result readback occurs in the GPU `submit()`.

Meld's existing host-Arrow `CudaDevice` runner is bypassed. `src/compiler.cpp` is the only translation unit that sees LLVM/MLIR; engines' public headers do not. The bridge uses Meld's public `compileToPtx`, literal-buffer API, and execution-unit ABI. Meld's current build still brings Arrow libraries into the compiler's link dependencies; removing that dependency is a later architectural cleanup, not an accomplished property of this prototype.

The accompanying Meld change assigns a scalar reduction's partial row to its GPU lane index. Lane zero writes the active-lane count, eliminating the global atomic row-claim counter on scalar aggregates. Validity bitmap updates still use atomic OR. The active count uses the largest driving fragment, honors the launch's `start`, handles empty input, and bounds stores by output capacity. The execution ABI's argument list is unchanged. Existing filtered row-output and grouped-aggregation paths retain their previous allocation schemes.

Meld's CMake now also respects shared-LLVM linkage when adding NVPTX: it links `LLVM` rather than mixing that shared library with static backend archives. The mixed linkage caused compilation to hang in the conda-based LLVM installation used for this validation. The adapter also accepts Meld's absent literal-buffer metadata for a plan with no literals; CPU and PTX compiler tests exercise that case.

`pending_result` retains the module, scratch, pinned metadata, and output. Its destructor waits for its execution if still pending; keep the handle alive to overlap useful work. `wait()` reports asynchronous CUDA errors. `take_result()` waits and transfers scalar ownership. Inputs remain borrowed, as with cuDF views: the caller owns their lifetime and must establish upstream stream dependencies. `wait_on()` orders a consumer stream but does not transfer ownership: keep the pending handle/output alive until downstream consumers have finished, especially on other streams. Keep the context, streams and memory resource alive through result destruction; create, submit, consume and destroy within the compiling CUDA context. Compilation can initialize the CUDA context and block. Allocation and driver calls can also affect host latency; this is not a promise that every host call is nonblocking.

## Supported semantics and limits

| Area | Current behavior |
| --- | --- |
| Input | One table, one batch per submission; `FLOAT64` and `DATE32` columns, nullable or nonnullable; dates bind to cuDF `TIMESTAMP_DAYS` on GPU |
| Expressions | Finite double and calendar-date literals; numeric `+`, `-`, `*`; same-type ordered `<`, `<=`, `>`, `>=`, `==`; SQL three-valued `&&` |
| Operators | Repeated filters and typed projections, terminated by one numeric scalar `SUM` |
| Nulls | Null filter predicates discard rows; arithmetic propagates null; SUM skips nulls and is invalid for no contributing values |
| NaN/infinity in data | Ordered comparisons and floating-point arithmetic; valid NaNs are not treated as nulls |
| Precision | Numeric data uses double; dates use signed 32-bit days since 1970-01-01. No fast-math flags requested; reduction order differs from eager cuDF, so results are not promised to match bit for bit |
| Slices | Values are borrowed at their offset; masks with byte-aligned offsets are borrowed; other nullable slices normalize only the bitmap on GPU |
| Reuse | Compile once, submit many compatible batches; independent per-submission state supports CPU tasks or GPU streams |
| Parameters | Literals are bound at compilation; changing one builds another program, with Meld's existing PTX cache able to reuse code |
| Unsupported in this scalar API | Other dtypes, date arithmetic, arbitrary UDFs, joins, groupby, sorting, standalone filter/project materialization, multi-GPU scheduling, graph capture, automatic fallback; the graph API adds grouping, sorting and materialization |

Unsupported inputs and malformed expression schemas fail explicitly. There is no silent downcast or CPU fallback. This is deliberately narrower than a transparent lazy cuDF API. A shared input can be branched during construction, but jointly compiling multiple terminal outputs is not implemented.

The launch uses 256 threads per block and up to four blocks per SM, followed by one CUB block reduction over the partials. Those are initial choices, not tuned optima. Per-submit allocations, pinned metadata, per-lane validity atomics, the one-block finalizer, always-nullable scan types, LLVM code quality, and slice-mask normalization are all candidates for measured follow-up. Unaligned nullable slices add bitmap-sized storage/work; they should not be described as a completely zero-copy path.

## Verify without a GPU

From the workspace containing `cudf/` and `meld/`, a compiler and CMake 3.24+ are sufficient for the frontend:

```bash
cmake -S cudf/cpp/examples/lazy_meld -B cudf/cpp/examples/lazy_meld/build-host \
  -DLAZY_MELD_BUILD_COMPILER=OFF -DLAZY_MELD_BUILD_GPU=OFF
cmake --build cudf/cpp/examples/lazy_meld/build-host -j2
ctest --test-dir cudf/cpp/examples/lazy_meld/build-host --output-on-failure
cudf/cpp/examples/lazy_meld/build-host/lazy_plan_test --emit-ir
```

These tests exercise graph immutability, column binding across projections, nullability propagation, literal serialization, and rejected schemas/expressions. They do not substitute for a Meld parser or GPU correctness test.

With LLVM/MLIR 21 (including NVPTX), Arrow and ArrowCompute development packages, and LLVM lit installed, enable `LAZY_MELD_BUILD_COMPILER=ON` while keeping `LAZY_MELD_BUILD_GPU=OFF`. This builds both `lazy_compiler_test` and `lazy_cpu_test`. The compiler test covers actual Meld parsing/lowering/PTX generation, multiple operation chains, diagnostics, and code/literal separation. The CPU test executes generated native code, including null/empty input, NaN/infinity, sliced bitmaps, changed projection schemas, repeated/concurrent submissions, and invalid host bindings. Neither needs a CUDA toolkit nor a GPU.

The working build on the current development host is directly runnable:

```bash
/tmp/jit-poc-build/lazy_cpu_test --benchmark 1000000 20
/tmp/jit-poc-build/q6_demo --lineitem /path/to/lineitem.tbl
```

To reproduce the CPU build with a combined SDK prefix, replace the path below. CMake and Ninja must be on `PATH`:

```bash
CPU_SDK_PREFIX=/path/to/llvm-arrow-sdk
cmake -S cudf/cpp/examples/lazy_meld -B cudf/cpp/examples/lazy_meld/build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLAZY_MELD_BUILD_GPU=OFF \
  -DCMAKE_PREFIX_PATH="$CPU_SDK_PREFIX" \
  -DMLIR_DIR="$CPU_SDK_PREFIX/lib/cmake/mlir" \
  -DLLVM_EXTERNAL_LIT="$CPU_SDK_PREFIX/bin/lit" \
  -DCMAKE_BUILD_RPATH="$CPU_SDK_PREFIX/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CPU_SDK_PREFIX/lib"
cmake --build cudf/cpp/examples/lazy_meld/build-cpu -j2
ctest --test-dir cudf/cpp/examples/lazy_meld/build-cpu --output-on-failure
cudf/cpp/examples/lazy_meld/build-cpu/lazy_cpu_test --benchmark 1000000 20
```

The current SDK is `/tmp/jit-poc-sdk`, created using conda-forge packages `llvmdev=21.1.8`, `mlir=21.1.8`, `libarrow=21`, `libarrow-compute=21`, `lit=21`, `python=3.12`, `cmake>=4`, and `ninja`. CMake and CTest are available in its `bin/` directory. The explicit library search/RPATH ensures this SDK's C++ runtime accompanies its libraries. Meld's full lit suite additionally needs LLVM's `FileCheck`, `not`, and `count` utilities; these were built separately for the historical regression run because this package distribution omitted them.

## Build and validate on the GPU host

Copy both working trees, including their new/untracked files, rather than relying on `git diff` alone. For example, replace `GPU_HOST` with the destination:

```bash
rsync -a cudf meld GPU_HOST:~/projects/jit/
```

Build/install libcudf from this cuDF checkout using [CONTRIBUTING.md](../../../CONTRIBUTING.md), or provide a matching 26.12 development installation. The current checkout requires GCC 13.3+, CUDA toolkit 12.9+ and CMake 4.0+ for its build. Install LLVM/MLIR **21** with NVPTX support, Arrow and ArrowCompute C++ development packages, LLVM lit and FileCheck. Use compatible C++ runtime and CUDA installations across dependencies. The examples below use placeholder installation paths; replace them with the host's actual paths.

```bash
cmake -S cudf/cpp/examples/lazy_meld -B cudf/cpp/examples/lazy_meld/build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DMELD_SOURCE_DIR="$PWD/meld" \
  -DMLIR_DIR=/path/to/llvm-21/lib/cmake/mlir \
  -DCMAKE_PREFIX_PATH="/path/to/libcudf-install;/path/to/arrow-install" \
  -DLLVM_EXTERNAL_LIT=/path/to/lit
cmake --build cudf/cpp/examples/lazy_meld/build-gpu -j4
ctest --test-dir cudf/cpp/examples/lazy_meld/build-gpu --output-on-failure

compute-sanitizer --tool memcheck --error-exitcode 1 \
  cudf/cpp/examples/lazy_meld/build-gpu/lazy_gpu_test
compute-sanitizer --tool racecheck --error-exitcode 1 \
  cudf/cpp/examples/lazy_meld/build-gpu/lazy_gpu_test
```

`lazy_gpu_test` compares against a CPU reference and eager cuDF AST/JIT paths. Cases include block/bitmap boundaries, nulls in each input, all-null/empty/all-filtered input, valid NaN/infinity, sliced masks, two nondefault streams, repeated execution, retained module lifetime, and schema rejection. It uses an asynchronous RMM allocator. A passed test is required before interpreting timings.

The embedded Meld build is compiler-only. Validate the modified lowering in Meld's own GPU-enabled build as well, adjusting dependency/CUDA paths for the host:

```bash
cmake -S meld -B meld/build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/path/to/llvm-21/lib/cmake/mlir \
  -DCMAKE_PREFIX_PATH=/path/to/arrow-install \
  -DLLVM_EXTERNAL_LIT=/path/to/lit \
  -DMELD_ENABLE_NVPTX=ON -DMELD_ENABLE_CUDA=ON \
  -DMELD_ENABLE_CUDF=OFF -DMELD_BUILD_APPS=OFF
cmake --build meld/build-gpu --target check-meld -j4
```

Check that Meld's configure output actually reports GPU execution enabled; its CUDA option depends on discovery of driver headers/libraries. The changed regression checks cover no atomic partial-row claim, active-prefix calculation, empty input and uneven fragments. Full scalar-aggregate tests also exercise SUM, COUNT, MIN, MAX and AVG through the existing runner.

## Measure before expanding

```bash
cudf/cpp/examples/lazy_meld/build-gpu/lazy_gpu_test --benchmark 1000000 20
cudf/cpp/examples/lazy_meld/build-gpu/lazy_gpu_test --benchmark 10000000 20
```

The executable reports cold Meld compile/load time separately, then warmed medians and incremental peak device allocations for eager AST, eager JIT, and lazy Meld. It checks the benchmark data before timing and runs nullable and nonnullable variants. Input creation/upload is outside timing; compilation is outside warm timing; submission, execution, waiting and output/scratch destruction are included in wall time. CUDA event intervals include stream scheduling gaps and exclude final resource destruction, so they are not pure kernel-only timings. The eager paths retain their scalar literals/AST and compact only the downstream columns they need.

The CPU/GPU benchmark now executes all Q6 predicates on synthetic data, including dates inside and outside the shipping window. `q6_demo --lineitem` runs the same graph on a supplied lineitem file. Neither is a certified TPC-H benchmark or proof of peak performance. Sweep input sizes, predicate selectivity, null density, expression complexity and concurrency before drawing conclusions. Profile to confirm the absent full-column filter/projection intermediates, inspect allocation and launch overhead, and compare to a tuned fused kernel before calling the generated implementation competitive. No speedup is claimed by this change.
