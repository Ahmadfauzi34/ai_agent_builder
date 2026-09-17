# Python Graph Run / Output Decomposition Research

Status: proven research evidence for issue #213.

## Why this exists

PR #210 proved the supported Python host on a deterministic closed-loop control workload. PR #212 then showed that repeated `Graph.run(...) + Tensor.to_f32()` dominates that rollout while optimizer transport, candidate apply, and pure-Python environment work are comparatively small.

PR #212 deliberately stopped before inventing a new graph/tensor API because its dominant bucket still combined several costs:

```text
Graph.run(...)
  -> Python facade
  -> burn-research.ffi.v1
  -> handle validation
  -> CompiledGraph::run(...)
  -> output Tensor handle allocation
  -> Tensor.to_f32()
     -> tensor length query
     -> native tensor materialization/copy
     -> Python list materialization
```

This slice decomposes that existing boundary with the current installed wheel before any runtime or ABI change is considered.

## Workload

Keep the same micro-policy shape used by #210/#212:

```text
Linear(6 -> 2, bias=true)
parameter dim = 14
input = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0]
```

Raw ABI and typed-host graphs are built independently with the same structural layer id and canonical zero trainable state. Their program identity, binding identity, and output values must match exactly before timing is accepted.

## Measured paths

The installed-wheel research measures independently:

1. raw `br_v1_graph_run` through output-handle creation, excluding handle free from the timed interval;
2. typed `Graph.run` through typed output-handle creation, excluding close from the timed interval;
3. raw `br_v1_tensor_len` against a persistent output handle;
4. raw `br_v1_tensor_copy_f32` into a preallocated destination;
5. Python list materialization from that already-populated preallocated destination;
6. typed `Tensor.to_f32()` against a persistent output tensor;
7. raw run + length + destination allocation + copy + Python list + free;
8. typed `Graph.run + Tensor.to_f32 + close` end to end.

This gives a useful split without adding native instrumentation.

## Valid evidence

Python Graph Run Output Decomposition Research run #1 on head `44f9ea7731c010836adada72a9f6bf06a439b239` completed successfully and uploaded the report with:

```text
verdict  = PASS
decision = GRAPH_RUN_HANDLE_BOUNDARY_DOMINATES
```

Median timing:

```text
raw graph_run -> output handle          0.2416795 ms
typed Graph.run -> output handle        0.2870490 ms
raw tensor_len                           0.0057700 ms
raw tensor_copy_f32 preallocated         0.0060110 ms
Python list materialization              0.0005210 ms
typed Tensor.to_f32                      0.0142160 ms
raw run + copy + close                   0.2850410 ms
typed run + copy + close                 0.2956250 ms
```

Descriptive ratios:

```text
raw graph_run / typed full path          81.75%
typed output copy / typed full path       4.81%
raw copy / typed Tensor.to_f32            42.28%
Python materialization / typed to_f32      3.66%
```

Typed `Graph.run` adds about `0.04537 ms` over the raw `br_v1_graph_run` median, but the dominant bucket is already present in the raw ABI call.

Semantic proof passed simultaneously:

- installed-wheel import;
- host API `burn-research.python-host.v1` and ABI v1;
- exact raw/typed program identity equality;
- exact raw/typed binding identity equality;
- exact finite raw/typed output equality;
- stable program/binding identities through the timing loops;
- deterministic repeated execution;
- deterministic raw/typed handle cleanup;
- no private facade handles used in the raw proof.

## Decision

```text
GRAPH_RUN_HANDLE_BOUNDARY_DOMINATES
```

The evidence does **not** justify output-copy optimization, tensor zero-copy, NumPy/DLPack, graph batching, or ABI widening. On this workload, typed output copying is only about 4.8% of the full typed policy step, while the raw graph-run/output-handle call is about 81.8%.

The next justified evidence slice is therefore narrow native instrumentation inside the existing `br_v1_graph_run` path to separate:

```text
FFI/handle validation
CompiledGraph::run(...)
output-handle boxing/allocation
```

That instrumentation should remain research-only and must not change execution semantics or create a new public ABI primitive.

## What the measurements mean

`raw_graph_run_handle` still intentionally contains several operations:

```text
ABI crossing
+ handle validation
+ CompiledGraph::run(...)
+ output-handle allocation
```

Therefore this result selects native instrumentation as the next research step; it does not identify which operation inside that bucket dominates yet.

`raw_tensor_copy_preallocated` includes the current native `WasmTensor::to_array()` materialization performed by `br_v1_tensor_copy_f32` plus the copy into the supplied f32 destination.

`typed_tensor_to_f32` additionally includes the tensor length query, destination allocation, and Python list materialization.

## Decision rule

Timing is descriptive evidence only and is never a CI performance threshold.

Research classification:

```text
raw graph-run share >= 55% of typed full path
    -> GRAPH_RUN_HANDLE_BOUNDARY_DOMINATES

otherwise typed output-copy share >= 40% of typed full path
    -> OUTPUT_COPY_MATERIALIZATION_DOMINATES

otherwise
    -> MIXED_GRAPH_OUTPUT_BOUNDARY_COST
```

The classification only selects the next evidence slice.

### If graph-run/output-handle creation dominates

Investigate narrow native instrumentation that separates:

```text
validation
CompiledGraph::run
output-handle boxing
```

Do not add a batch execution primitive, persistent tensor pointer, or ABI symbol merely from this Python-level result.

### If output copy/materialization dominates

Investigate tensor-output transport narrowly while keeping ABI v1 semantics, ownership, and fail-closed behavior intact.

### If mixed

Keep the existing API unchanged until a larger/representative workload exposes a clearer pressure point.

## Correctness gates

The report is valid only if all of these hold:

- package and host import from the installed wheel outside the checkout;
- Python host schema remains `burn-research.python-host.v1`;
- ABI version remains v1;
- raw and typed program identities are exact matches;
- raw and typed binding identities are exact matches;
- raw and typed graph outputs are exact matches and finite;
- repeated execution is deterministic;
- program and binding identities remain stable through timing loops;
- all raw and typed output handles are deterministically freed/closed;
- the raw proof uses only public `ffi` / `lib`, not private facade handles.

## Scope

Research-only files:

- `.github/workflows/python-graph-run-output-decomposition-research.yml`;
- `scripts/research_python_graph_run_output_decomposition.py`;
- `docs/python-graph-run-output-decomposition-research.md`.

No changes to:

- `burn-research.ffi.v1` symbols/header;
- Python host/facade API;
- Rust graph/tensor runtime;
- `GraphParameterBinding` semantics;
- optimizer algorithms;
- ProgramBundle schema;
- Math Program v1-v9;
- Resolution/Authorization;
- support matrix.

No NumPy, DLPack, PyO3, persistent pointer cache, graph batching primitive, Go, or C++ work is introduced here.
