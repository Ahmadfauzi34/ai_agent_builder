# Python batch workload research

Status: **proven on installed-wheel research workflow**

Related: #184, #186, #188, #189, #190

## Question

The Python wheel and typed facade are already supported on the narrow verified matrix. The next question is not whether Python can call the runtime; it is whether the current copy-based tensor boundary creates enough practical pressure to justify widening the ABI or adding NumPy/zero-copy integration.

This research compares two host evaluation patterns over the same Linear graph and dataset:

```text
row-by-row
  N x tensor create -> graph.run -> output copy

batched
  1 x Tensor[N,F,1,1] -> graph.run -> output[N,1,1,1] copy
```

Linear already treats axis 0 as batch and axis 1 as feature width, so this comparison introduces no new execution semantics.

## Boundary

Preserve:

```text
Rust core semantics
    != language-neutral ABI v1
    != typed Python facade
    != host evaluation policy
    != performance evidence
```

The research does not add or alter `br_v1_*` symbols, Math Program semantics, graph semantics, optimizer semantics, ProgramBundle format, Resolution/Authorization, or the Python support matrix.

No NumPy, DLPack, buffer-protocol, zero-copy, PyO3, Go, or C++ work is introduced.

## Workload

The research script builds the current CFFI wheel, installs it into a fresh temporary venv outside the checkout, and runs only through the typed facade.

The deterministic workload uses:

- rank-4 f32 tensors;
- a feature-width-8 `Linear(8 -> 1, bias)` graph;
- 96 training rows and 32 held-out rows;
- `GraphParameterBinding.total_len` as the sole optimizer dimension source;
- strict ES with fixed seed, population 16, and 36 generations;
- 576 candidate evaluations;
- a Python-owned regression objective;
- one persistent batch tensor reused across candidate evaluations;
- stateful `ProgramBundle` export/import into a fresh registry after training.

## Evidence

`Python Workload Research` run #1 on PR #190 passed from the installed wheel.

Semantic evidence:

```text
parameter_dim                 = 9
max_row_batch_abs_diff        = 0.0
baseline_holdout_loss         = 0.1012418303
final_holdout_loss            = 0.01027935425
heldout_improvement_ratio     = 9.849x
checkpoint_bytes              = 480
replay_max_abs_diff           = 0.0
replay_loss                   = final_holdout_loss exactly
```

The workload therefore proves exact row/batch semantic agreement for this graph, material ES improvement, stable structural/binding identity, exact learned-vector replay, and exact held-out output/loss replay through a fresh ProgramBundle import.

## Timing evidence

Timing remains evidence only, not a CI threshold or portable performance claim.

Median held-out evaluation on the same GitHub runner/process:

```text
row mode total                = 9.970 ms
  tensor create               = 0.667 ms
  graph run                   = 8.213 ms
  output copy                 = 1.122 ms

batch mode total              = 0.392 ms
  tensor create               = 0.0369 ms
  graph run                   = 0.293 ms
  output copy                 = 0.0584 ms

row / batch total ratio       = 25.41x
```

The persistent training/holdout input tensors together required about `0.102 ms` to construct once.

Per candidate during batched training:

```text
canonical parameter apply     = 0.1315 ms
batch graph run               = 0.4063 ms
output copy                   = 0.0935 ms
Python objective arithmetic   = 0.0152 ms
```

Across all 576 candidates:

```text
ask                           = 1.558 ms total
candidate apply               = 75.735 ms total
graph run                     = 234.009 ms total
output copy                   = 53.869 ms total
Python objective              = 8.771 ms total
tell                          = 2.707 ms total
```

Checkpoint evidence remained small relative to the training workload:

```text
export                        = 0.159 ms
import + binding rebuild      = 0.285 ms
replay total                  = 0.399 ms
```

## Decision

Current decision:

```text
KEEP_COPY_BASED_BATCHED_TENSOR_BOUNDARY
DO_NOT_WIDEN_ABI_FOR_ZERO_COPY_YET
```

The key evidence is not merely that batching is faster. Batching reduced row-wise host crossing cost by roughly `25.4x`, and in the persistent batched candidate path the output copy (`~0.094 ms`) was materially smaller than graph execution (`~0.406 ms`) and was not the dominant per-candidate cost.

Therefore this workload does **not** justify adding NumPy-specific ABI semantics, DLPack, a buffer protocol, or zero-copy ownership complexity.

This decision is evidence-scoped, not permanent. A future workload with substantially larger outputs or device-memory pressure may reopen tensor interoperability, but it should demonstrate that batched copy cost is actually dominant first.

## Next performance question

The next measurable cost visible in this run is not output copy but the combination of graph execution and canonical parameter application. `GraphParameterBinding` apply remained about `0.131 ms/candidate` for only nine parameters.

That is not enough evidence by itself to optimize or cache binding state. If performance work continues, the safer next research slice is parameter-apply scaling across materially larger trainable graphs while keeping the current correctness boundary unchanged.

## Semantic gates retained

1. installed package comes from `site-packages`, not checkout;
2. row and batch outputs agree;
3. candidates, fitness, and outputs remain finite;
4. ES materially improves deterministic held-out loss;
5. program and binding identities remain stable while mutable parameters change;
6. best parameters read back exactly in canonical order;
7. ProgramBundle replay preserves program identity, binding identity, learned state, and held-out output/loss.
