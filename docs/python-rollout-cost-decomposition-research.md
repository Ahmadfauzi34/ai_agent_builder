# Python rollout cost decomposition research

Issue: #211

## Why this exists

PR #210 proved that the installed Python host can run a deterministic closed-loop control workload with `ask_f32()`, candidate application, repeated `Graph.run()`, Python-owned reward, `tell()`, and stateful `ProgramBundle` replay. The remaining dominant measured cost was the rollout itself, not optimizer or parameter transport.

This slice decomposes that rollout before any graph/tensor API change is considered.

## Boundary

No product surface changes are allowed here. The research uses the installed wheel and the existing Python host API only.

Measured components:

1. pure-Python environment/reward rollout with no graph calls;
2. `Tensor.vector(...)` construction + deterministic close;
3. `Graph.run(...) + Tensor.to_f32()` with a pre-built input tensor;
4. complete policy action (`Tensor.vector -> Graph.run -> output copy -> close`);
5. full graph-driven closed-loop rollout.

The policy remains `Linear(6 -> 2, bias=true)` with 14 trainable parameters and the same 4-scenario × 16-step environment shape used by the agent workload proof.

## Correctness gates

The script must prove all of the following before timing evidence is accepted:

- package and `burn_research_ffi.host` import from the installed wheel, not the checkout;
- host schema and ABI v1 identity are unchanged;
- graph output shape and values remain finite;
- graph rollout and Python-only rollout are deterministic;
- graph reward is unchanged before and after timing loops;
- program identity and binding identity remain stable;
- every owned tensor/graph/registry/layer/binding handle is deterministically closed.

## Classification

The report always separates semantic `verdict = PASS` from the evidence classification. The classification is based on median descriptive ratios only:

- `PYTHON_ENVIRONMENT_DOMINATES` when pure-Python environment rollout is at least 55% of full graph-rollout time;
- otherwise `GRAPH_RUN_OUTPUT_BOUNDARY_DOMINATES` when pre-built-input `Graph.run + output copy` is at least 55% of full policy-action time;
- otherwise `TENSOR_INPUT_MARSHALLING_DOMINATES` when tensor input construction is at least 55% of full policy-action time;
- otherwise `MIXED_ROLLOUT_COST`.

These percentages are research heuristics, not CI performance requirements or support SLAs.

`graph_run_output_copy` intentionally remains a combined bucket: it contains native graph execution, FFI crossing, output-handle creation, and output copy. This research does **not** pretend to separate those without native-side instrumentation.

## Valid evidence

Python Rollout Cost Decomposition Research run #1 on head `53ff25c211c0b2471ece0fcde444eef8217ad0a1` completed with `verdict = PASS` and uploaded `python-rollout-cost-decomposition-report`.

The evidence classification is:

```text
GRAPH_RUN_OUTPUT_BOUNDARY_DOMINATES
```

Median timing:

```text
pure-Python environment rollout   0.046751 ms
Tensor.vector construct + close   0.008683 ms
Graph.run + output copy           0.100742 ms
full policy action                0.112534 ms
full graph rollout                7.8154145 ms
```

Descriptive ratios:

```text
Graph.run + output / full policy action   89.52%
Tensor input / full policy action          7.72%
Python environment / full graph rollout    0.60%
```

The zero-policy graph reward remained exactly reproducible before and after the timing loops at `-2.407589173214531`. Program identity, binding identity, finite outputs, deterministic graph rollout, deterministic Python-only rollout, installed-wheel import, and deterministic handle cleanup all passed.

This rules out the Python environment loop and tensor input construction as the dominant cost for this workload. The remaining dominant bucket is specifically `Graph.run + output copy`, but that bucket still combines native graph execution, FFI crossing, output-handle creation, and output copying.

## Decision rule

No ABI, facade, graph runtime, tensor representation, batching API, or Math primitive should be added from intuition. The evidence now justifies one narrower next step only: instrument or otherwise prove the split inside `Graph.run + output copy` so native compute can be distinguished from boundary/handle/copy cost.

This result does **not** yet justify batching, a persistent tensor API, DLPack/NumPy, a new graph execution primitive, or ABI widening.
