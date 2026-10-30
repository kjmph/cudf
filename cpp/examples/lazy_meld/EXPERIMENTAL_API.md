# Experimental cuDF graph execution API

This extends the lazy/Meld proof of concept with a separate operation graph, physical plan, and
compiled program. It is an opt-in library in this example directory, **not an installed libcudf
API or interception of existing eager calls**. No further Meld changes were needed for this step.

CPU execution is runnable here. GPU planning and PTX compilation can be checked here, but the GPU
adapter and GPU tests have **not been built or executed** on this host.

[Full Q6](Q6.md) now exercises typed dates through the same graph, planner, and Meld backend.

## API shape

```cpp
#include <cudf/experimental/execution/cpu.hpp>
#include <iostream>

namespace ex = cudf::experimental::execution;
using ex::col;
using ex::lit;

auto graph = ex::input(3)                       // nullable FLOAT64 schema; no data access
  .filter(col(0) < lit(24))
  .group_by({col(0) + col(1)})                  // an expression, not a key column
  .sum(col(2) * lit(2))                        // output columns: key, sum
  .sort_by({0})
  .sum(col(1));

auto physical = ex::plan(graph, {.device = ex::target::cpu});
std::cout << physical.explain();

auto program = ex::compile_cpu(physical);       // explicit compilation boundary
auto pending = program.submit(host_columns);   // bind buffers; retain them until completion
auto result = pending.get();                   // owning host_table; one row for scalar SUM
```

The executable [graph_demo.cpp](tests/graph_demo.cpp) includes input generation, an independent
reference total, compilation timing, and warmed execution measurements. It runs this graph and
also prints the GPU plan without initializing CUDA. Sorting is retained to exercise a real
boundary; this experiment does not optimize away the redundant sort before the final SUM.

An engine can build the graph directly with these handles. Existing POC lazy chains use the same
nodes and expression semantics through `ex::from_lazy(lazy_frame_or_reduction)`. There is no new
Substrait importer, SQL parser, or requirement for an engine to construct Meld IR.

The public headers contain no LLVM/MLIR types. CMake aliases are
`cudf::experimental_execution`, `cudf::experimental_execution_cpu`, and
`cudf::experimental_execution_gpu`. The first supports graph construction and planning alone.

## What the planner actually chooses

For the example, automatic CPU planning produces:

```text
fragment 0 [meld]: filter -> group_by(expressions).sum(expression)
fragment 1 [host_reference]: stable_sort
fragment 2 [meld]: sum(expression)
```

The first fragment compiles the filter, key, and measure expressions together. Meld computes keys
per input row and feeds its hash table directly; it does not create per-input-row key or measure
columns. It stores group keys and aggregate state in the table and compacts the completed groups
at the fragment boundary. Sort consumes those groups. A second compiled fragment reduces them.

Automatic GPU planning produces:

```text
fragment 0 [cudf]: filter
fragment 1 [cudf]: group_by(expressions).sum(expression)
fragment 2 [cudf]: stable_sort
fragment 3 [meld]: sum(expression)
```

This GPU adapter supports Meld regions ending in scalar SUM. Other operations use ordinary cuDF
APIs. The groupby path evaluates keys and measures with cuDF AST kernels, materializes those
columns, then calls `cudf::groupby::groupby` with null keys included. **Generated expressions inside
cuDF's groupby/join kernels are not implemented.** `explain()` reports this gap explicitly. It does
not describe a materialized path as fused. Meld's own CPU grouping demonstrates the alternate
route; it is not evidence of cuDF GPU groupby fusion.

| Policy | CPU | GPU |
| --- | --- | --- |
| `automatic` | Meld row/aggregate regions; host reference sort | Meld filter/project/SUM regions; cuDF elsewhere |
| `kernels_only` | Untuned host reference, with operator outputs materialized | Ordinary cuDF AST/operator calls |
| `meld_only` | Require Meld for every region; reject sort | Require supported Meld scalar regions; reject groupby/sort/standalone row output |

`plan()` uses deterministic capability rules. It has no calibrated cost model, cardinality
statistics, compilation-amortization policy, adaptive recompilation, or guarantee of choosing the
fastest path. Forced policies provide comparison paths and reject unsupported requests. There is
no automatic CPU fallback. GPU plans can be inspected on a CPU-only machine; planning does not
mean the GPU program is available or tested.

Partitioning lives in [graph.cpp](src/graph.cpp); compilation and dispatch consume its decisions.
The shared expression/schema representation and Meld emitter are reused from the original POC,
instead of introducing a second expression language or separate GPU planner.

## GPU submission

```cpp
#include <cudf/experimental/execution/gpu.hpp>

auto physical = ex::plan(graph, {.device = ex::target::gpu});
auto program = ex::compile_gpu(physical);
auto pending = program.submit(device_table_view, stream, memory_resource);
// A consumer on this stream can use pending.result() in stream order.
// Another stream must first use pending.wait_on(consumer_stream).
pending.wait();
auto owned_table = pending.take_result();
```

Input buffers are borrowed and must remain unchanged and alive through producer completion. The
pending handle retains compiled modules, intermediates, expression literals, and outputs. Its
destructor waits before releasing resources; exception paths drain the submission stream before
releasing retained owners. `wait_on()` orders a consumer but does not extend the result's lifetime
through that consumer. Keep the CUDA context, stream and memory resource alive through cleanup.

All cuDF calls receive the supplied stream and RMM resource. Their internal cardinality/validity
queries can synchronize: lazy graph construction does not guarantee nonblocking host submission.
The adapter retains intermediates until completion and currently copies columns to assemble some
tables. It does not yet minimize peak memory or launch/allocation overhead. GPU failure-path and
concurrency validation remain part of the GPU-host work; source inspection is not a substitute.

## Supported semantics and limits

- One input table, one output, and linear operator chains. Handles may be reused to build separate
  graphs, but jointly compiled branches, joins, named input bindings, and multiple outputs are absent.
- Nullable FLOAT64 and DATE32 input and stored values. Use `input({data_type::float64,
  data_type::date32})` for a typed schema and `lit(date(1994, 1, 1))` for a date literal.
  Dates support comparisons, projection, grouping keys, and sorting; numeric expressions support
  arithmetic and SUM. GPU dates bind to cuDF TIMESTAMP_DAYS. No date arithmetic, decimal types,
  general integers, strings, arbitrary UDFs, or automatic type conversion.
- Filter, project, scalar SUM, expression-key groupby with one SUM, and stable ascending sort.
  Expressions bind to their immediate input schema; grouped output is keys followed by the SUM.
- Null filter predicates drop rows. SUM skips null values and is null for no valid contributions.
  Empty grouped input produces zero groups; null keys form groups. NaNs form one key group, and
  signed zeros share a group. Group order is unspecified until explicitly sorted. Sort places
  numbers before NaNs before nulls; equal keys retain input order. GPU parity is pending testing.
- Floating-point reduction order can change between implementations. CPU tests compare validity
  exactly, NaNs/infinities explicitly, and finite values with a relative tolerance of `1e-11`.
- CPU execution uses one thread per `run()`. `submit()` starts a `std::async` worker; it copies view
  descriptors and retains the program, but borrows the actual buffers until completion.
- The CPU grouped adapter allocates a power-of-two table with at least twice the input row count.
  It is a simple capacity bound, not a tuned memory policy or use of the full Meld scheduler.
- `execution_metrics::intermediate_bytes` counts logical output buffers between CPU fragments.
  It excludes final output, hash-table capacity, reference-internal expression buffers, bitmap
  normalization, and allocator overhead. It is **not peak allocated memory**. GPU dispatch counts
  report which implementation paths were submitted, not GPU timings.

## Validation and reproduction

The earlier [recorded CPU run and five-test result](results/graph-cpu-aarch64-2026-10-28.txt) include
fragment explanations and measurement conditions for the numeric graph API. The
[current six-test result](results/q6-cpu-aarch64-2026-10-29.txt) adds typed dates and full-Q6
validation against DuckDB on generated lineitem data. The graph-only build also passes both
of its tests without LLVM, Arrow, or CUDA.

Using the existing CPU SDK/build described in [README.md](README.md):

```bash
# From the jit workspace; the build is configured with LAZY_MELD_BUILD_GPU=OFF.
/tmp/jit-poc-sdk/bin/cmake --build /tmp/jit-poc-build -j2
/tmp/jit-poc-sdk/bin/ctest --test-dir /tmp/jit-poc-build --output-on-failure --timeout 60
/tmp/jit-poc-build/graph_demo 100000
/tmp/jit-poc-build/q6_demo 1000000
```

The six CPU-host tests cover the original POC and the new graph/planner/runtime. Tests include
independent hand-calculated groups, NaN payloads, signed zero, null keys/measures, empty and filtered
inputs, high cardinality, sliced validity masks, schema changes, forced-policy rejection, real mixed
dispatch, asynchronous reuse, and invalid bindings. The compiler test also emits PTX for the Meld
region of a mixed GPU plan and the full Q6 graph without requiring a CUDA device. Date tests check
the exact Q6 boundaries, a hand-calculated revenue of 37, null dates, negative day counts, typed
materialization/grouping, sliced masks, and asynchronous execution.

For graph construction/planning without LLVM, Arrow or CUDA:

```bash
cmake -S cudf/cpp/examples/lazy_meld -B /tmp/cudf-graph-frontend \
  -DLAZY_MELD_BUILD_COMPILER=OFF -DLAZY_MELD_BUILD_GPU=OFF
cmake --build /tmp/cudf-graph-frontend -j2
ctest --test-dir /tmp/cudf-graph-frontend --output-on-failure
```

On the GPU host, follow the matching cuDF/LLVM/CUDA setup in the original README and enable
`LAZY_MELD_BUILD_GPU=ON`, then:

```bash
cmake --build <gpu-build> --target graph_gpu_test lazy_gpu_test -j2
ctest --test-dir <gpu-build> -L GPU --output-on-failure --timeout 300
```

`graph_gpu_test` compares actual cuDF grouping to hand-calculated results, compares mixed execution
with forced cuDF execution, checks dispatch counters, and exercises result/program ownership.
Both GPU executables include the full Q6 date predicates and boundary fixture.
It is supplied for GPU-host validation, **not reported as passed**.
