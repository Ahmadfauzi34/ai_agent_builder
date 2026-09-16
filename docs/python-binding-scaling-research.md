# Python GraphParameterBinding scaling research

Status: **research candidate pending CI evidence**

Related: #165, #174, #176, #181, #185, #188, #190, #191

## Why this research exists

The installed Python batch workload proved that output copy is not currently the dominant candidate-evaluation cost when the host batches work and reuses persistent input tensors.

On that nine-parameter workload, the next visible host-observed cost was canonical parameter application. That result is too small to justify changing `GraphParameterBinding` or adding cache state.

This slice therefore measures scaling before design.

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

This primarily exposes flat-vector-length scaling with one trainable owner.

### Fixed-width owner-count family

Chains of `Linear(64 -> 64)`:

```text
1 owner    4,160 params
4 owners   16,640 params
8 owners   33,280 params
16 owners  66,560 params
```

This exposes repeated owner/setter cost while width remains fixed.

## Host-observed timing

The research intentionally times through the installed typed Python facade. Therefore `apply_flat` includes the costs a real Python host sees:

```text
Python sequence normalization
  + CFFI buffer construction
  + ABI call
  + core binding validation
  + canonical owner setters
```

The candidate vector is constructed before timing and explicitly rounded to f32 so `apply -> read_flat` can be asserted exactly.

For each case, after warm-up, report repeated-sample median/min/p90/max for:

- binding build;
- `read_flat`;
- `apply_flat`;
- graph run on a persistent batch tensor;
- output copy;
- combined apply + run + copy;
- apply share of combined median;
- median apply nanoseconds per parameter.

Timing is evidence only. It must not become a CI threshold or SLA.

## Semantic gates

Every case must prove:

1. installed package comes from `site-packages`, not checkout;
2. `binding.total_len` matches the expected parameter count;
3. initial flat state is finite and has the expected length;
4. deterministic finite candidate applies and reads back exactly;
5. program identity remains stable across mutation;
6. binding identity remains stable across mutation;
7. graph output is finite;
8. repeated same-candidate apply/run is deterministic.

The largest case additionally proves stateful `ProgramBundle` export/import into a fresh registry with exact program identity, binding identity, learned flat state, and output replay.

## Decision rule

Do not introduce reusable/cached binding state merely because `apply_flat` is measurable.

Only open an optimization design if the larger cases show a material and disproportionate binding cost relative to graph execution, especially if owner-count scaling suggests repeated validation/setter work that can be safely amortized while preserving identity validation.

If graph execution remains dominant and apply/build scale reasonably, keep the current correctness-first implementation unchanged.
