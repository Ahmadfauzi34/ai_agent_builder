# Python agent control-rollout research

Status: **installed-wheel agent workload proven; no product/API change**

Related: #184, #186, #188, #190, #192, #194, #196, #200, #203, #204, #205, #206, #207, #208, #209, #210

## Question

The Python host is now proven as an installed-wheel consumer and exposes the additive `EsOptimizer.ask_f32()` transport path. The next question is whether those pieces compose cleanly in a workload that behaves like an agent/controller rather than a static smoke test.

This research intentionally keeps policy, environment, optimization, and checkpoint boundaries separate:

```text
Python environment / rollout / reward / schedule
        |
        v
CompiledGraph policy execution
        |
        v
GraphParameterBinding canonical state
        ^
        |
EsOptimizer.ask_f32() / tell()
        |
        v
ProgramBundle checkpoint / replay
```

Python owns the environment and objective. The Rust reference machine does not gain a controller or rollout engine.

## Deterministic workload

The installed-wheel consumer builds a single proven graph primitive:

```text
Linear(6 -> 2, bias=true)
parameter dimension = 14
```

The six policy inputs are current tracking error, velocity, and previous control for a deterministic two-dimensional control system. The two outputs are clipped control actions.

Python advances the environment for several fixed initial conditions and targets. Each trajectory cost combines:

- tracking error;
- velocity penalty;
- control effort;
- a small deterministic expert-action shaping term;
- deterministic disturbances;
- terminal tracking error.

The expert term is only part of the Python-owned fitness function; it does not add another graph primitive or hidden optimizer path.

## Training path

The research uses only the supported first-class host API:

```text
host.EsOptimizer.strict(...)
  -> ask_f32()
  -> array('f') flat candidate batch
  -> memoryview candidate windows
  -> GraphParameterBinding.apply_flat(...)
  -> repeated Graph.run(...) rollouts
  -> Python reward
  -> tell(fitness)
```

OpenES is configured with deterministic seed, population 8, sigma 0.15, and learning rate 0.06 for 24 generations. The observed candidate count is still read from `optimizer.batch_size` after each ask rather than inferred from configuration.

## Proof boundary

A research run is semantically valid only when it proves:

- package and host imports come from the installed wheel outside the checkout;
- host API schema and ABI v1 identities are unchanged;
- `ask_f32()` returns contiguous native-f32 storage;
- candidate length equals `batch_size * binding.total_len`;
- each candidate window is exact-length and C-contiguous;
- every rollout reward is finite;
- `tell()` generation/evaluation cardinality is exact;
- program identity stays constant across all rollouts and generations;
- binding identity stays constant across all rollouts and generations;
- the final champion is reproducible before checkpoint export;
- stateful `ProgramBundle` import into a fresh registry preserves program identity, binding identity, flat parameter state, and deterministic rollout reward;
- handles are owned through context managers / `ExitStack`.

Timing is descriptive evidence only and never a CI threshold.

## Research decision

The report always uses `verdict = PASS` for proof/harness correctness. Separately it classifies the workload:

```text
PYTHON_HOST_AGENT_WORKLOAD_PROVEN
```

when the final champion reduces deterministic rollout cost by more than 5% versus a zero-policy baseline and more than 2% versus the first-generation champion.

Otherwise it emits:

```text
HOST_OR_POLICY_FRICTION_FOUND
```

That result would not justify an automatic primitive/API change. The next action would first classify the friction as transport, graph expressiveness, optimizer behavior, rollout overhead, or checkpoint/replay.

The percentages above are research classification heuristics, not support SLAs.

## Non-goals

This research does not change:

- ABI v1;
- `EsOptimizer` Rust algorithms;
- `GraphParameterBinding` semantics;
- graph execution semantics;
- Math Program v1-v9;
- ProgramBundle schema;
- Resolution/Authorization;
- Python support matrix.

It adds no NumPy, DLPack, PyO3, controller object, environment primitive, Go, or C++ support.

## Current result

Python Agent Control Rollout Research run #1 on head `64118cad1e5deeea0c0c46f1f7db35ff5634759a` completed successfully and uploaded `python-agent-control-rollout-report.json`.

```text
verdict  = PASS
decision = PYTHON_HOST_AGENT_WORKLOAD_PROVEN
```

Workload:

```text
policy        = Linear(6 -> 2, bias=true)
parameter dim = 14
population    = 8
generations   = 24
scenarios     = 4
horizon       = 16
```

Deterministic rollout reward improved from:

```text
zero policy              -2.4075891732
first-generation champion -4.6974486753
final champion           -0.5209996224
checkpoint replay        -0.5209996224
```

In cost terms, the final champion improved by about 78.36% versus the zero-policy baseline and about 88.91% versus the first-generation champion. The stateful checkpoint replay reproduced the final rollout reward within the required `1e-9` tolerance, and the exported bundle was 501 bytes.

Median timing evidence:

```text
ask_f32             0.0307 ms
apply candidate     0.0975 ms
rollout candidate  14.1544 ms
tell                0.0913 ms
```

The important architectural signal is that Python f32 transport is not the dominant cost on this agent-style workload. Repeated rollout execution dominates by more than two orders of magnitude over `ask_f32`, `apply_flat`, or `tell`. This does **not** justify a new batch/trajectory primitive by itself; it only identifies repeated policy execution as the next area to measure if a larger real workload later shows unacceptable wall-clock cost.

All proof conditions passed: native-f32 candidate storage, stable batch cardinality, finite fitness, stable program identity, stable binding identity, multi-generation optimizer lifecycle, and stateful ProgramBundle replay.
