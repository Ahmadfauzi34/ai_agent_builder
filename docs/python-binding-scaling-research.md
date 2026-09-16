# Python GraphParameterBinding scaling research

Status: **proven on installed-wheel scaling workflow**

Related: #165, #174, #176, #181, #185, #188, #190, #191, #192

## Why this research exists

The installed Python batch workload proved that output copy is not currently the dominant candidate-evaluation cost when the host batches work and reuses persistent input tensors.

On that nine-parameter workload, the next visible host-observed cost was canonical parameter application. That result was too small to justify changing `GraphParameterBinding` or adding cache state.

This slice therefore measured scaling before design.

## Boundary

Preserve:

```text
GraphParameterBinding correctness / identity
    != performance evidence
    != future cache/reuse design
```

This research does not change:

- canonical owner ordering;
- offset/length semantics;
- binding identity;
- finite/fail-closed atomic apply;
- graph/program identity;
- ABI v1 symbols;
- Python facade semantics;
- Math Program v1-v9;
- optimizer algorithms;
- ProgramBundle format;
- Resolution/Authorization.

No cached/reusable binding implementation is introduced here.

## Scaling families

The installed-wheel research uses only existing `Linear` layers and the typed Python facade.

### Single-owner parameter-length family

```text
Linear(8 -> 8)       72 params
Linear(32 -> 32)     1,056 params
Linear(64 -> 64)     4,160 params
Linear(128 -> 128)   16,512 params
```

### Fixed-width owner-count family

Chains of `Linear(64 -> 64)`:

```text
1 owner    4,160 params
4 owners   16,640 params
8 owners   33,280 params
16 owners  66,560 params
```

## Evidence

`Python Binding Scaling Research` run #1 on PR #192 passed from the installed wheel.

All semantic gates passed for every case:

- exact expected `binding.total_len`;
- finite initial state;
- deterministic f32 candidate apply and exact readback;
- stable program identity across mutable apply;
- stable binding identity across mutable apply;
- finite graph output;
- deterministic repeated apply/run.

The largest case used 16 canonical owners / 66,560 parameters. Its stateful `ProgramBundle` was 271,263 bytes and replayed program identity, binding identity, learned flat state, and output exactly.

## Host-observed timing

The research intentionally times through the installed typed Python facade. Therefore `apply_flat` includes the costs a real Python host sees:

```text
Python sequence normalization
  + CFFI buffer construction
  + ABI call
  + core binding validation
  + canonical owner setters
```

Timing is evidence only and is not a CI threshold, SLA, or portable benchmark claim.

### Single-owner parameter-length scaling

Median timings from the same workflow run:

```text
params    build ms   read ms   apply ms   run ms   copy ms   apply share
72        0.060      0.097     0.125      0.263    0.071     26.5%
1,056     0.066      0.201     0.192      0.311    0.181     27.4%
4,160     0.083      0.493     0.400      0.424    0.331     34.8%
16,512    0.183      1.697     1.173      0.751    0.649     45.5%
```

Median host-observed apply cost per parameter falls sharply as fixed overhead is amortized:

```text
72 params       1,742.7 ns/param
1,056 params      181.6 ns/param
4,160 params       96.1 ns/param
16,512 params      71.0 ns/param
```

This is inconsistent with a superlinear parameter-length pathology in the measured range.

### Fixed-width owner-count scaling

```text
owners  params    build ms   read ms   apply ms   run ms   copy ms   apply share
1       4,160     0.083      0.493     0.400      0.424    0.331     34.8%
4       16,640    0.258      1.797     1.445      1.582    0.332     42.6%
8       33,280    0.456      3.552     2.876      3.284    0.357     44.4%
16      66,560    0.924      7.173     5.519      6.740    0.376     43.9%
```

For the 4/8/16-owner cases, median apply cost was approximately:

```text
86.8 ns/param
86.4 ns/param
82.9 ns/param
```

The near-flat ns/parameter result indicates approximately linear scaling across the measured owner-count family. There is no evidence here of an owner-count explosion or superlinear canonical-binding failure mode.

`binding_build` also remained below 1 ms at 16 owners / 66,560 parameters. Since normal host code builds the binding once rather than per candidate, build cost is not currently a candidate-loop concern.

`read_flat` reached about 7.17 ms in the largest case. That operation is useful for checkpoint/reporting/inspection but should not be placed in a candidate-evaluation inner loop. This result does not justify changing apply semantics.

## Interpretation

The scaling evidence produces two simultaneous conclusions:

1. `apply_flat` is **material** in larger Python-host candidate loops. At 66,560 parameters it consumed about `5.52 ms` versus `6.74 ms` for graph execution, or roughly `43.9%` of the combined apply/run/copy median.
2. The measured apply path scales **reasonably and approximately linearly**. More owners did not introduce a disproportionate penalty beyond the expected increase in parameter/setter work.

Those facts are not enough to justify a core cache.

The current timing boundary includes Python sequence conversion and CFFI buffer construction. Therefore a core caching design based on these numbers would risk optimizing the wrong layer.

## Decision

Current decision:

```text
KEEP_CURRENT_BINDING_SEMANTICS
DO_NOT_ADD_BINDING_CACHE_YET
MEASURE_HOST_MARSHALLING_VS_ABI_CORE_NEXT
```

There is now evidence that apply cost is worth understanding, but not evidence that canonical validation/owner resolution is the dominant source of that cost.

Before any cache/reusable-validation design, the next research slice should decompose one candidate application into at least:

```text
Python sequence -> CFFI buffer preparation
raw br_v1_binding_apply_flat ABI call
facade GraphParameterBinding.apply_flat end-to-end
```

Use a preallocated raw CFFI `float[]` for the raw-call path so the difference between facade marshalling and the ABI/core call is observable on the same graph/candidate.

If raw ABI/core apply remains close to the facade end-to-end cost, then a narrowly scoped core optimization investigation may be justified.

If facade marshalling is a substantial share, optimize Python-side candidate transport first without changing binding identity or core semantics.

## Rule retained

Do not introduce reusable/cached binding state merely because `apply_flat` is measurable. Any future optimization must preserve canonical owner ordering, fail-closed atomicity, structural/binding identity checks, and exact ProgramBundle replay.
