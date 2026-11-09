# Burn Math Agent Pack

> Repo note: this file lives in the repo at `helper/AGENT.md` and is the
> contract for the agent toolset. The distributable pack is BUILT from this
> directory by `scripts/build_agent_pack.mjs` — the repo is the single source
> of truth, the pack is its output. Where this file says `engine-audit/`,
> read it as the engine bundle: in the repo that is `pkg/` (or wherever
> `BURN_WASM_PKG_DIR` points); in a built pack it is `engine-audit/`.


A deterministic math reference machine, packaged for any agent.
Rule zero: for any computation that matters, run it through this engine.
Do not compute it by reasoning, do not simulate it in your head, and do
not write your own float math and call it equivalent. Reasoning proposes;
this engine decides.

## Requirements

Node.js >= 18 (ES modules) and nothing else to install. No network, no
build step, no package install. Unzip, then run the commands below from
this directory. JSON (including error JSON) goes to stdout; plain
usage text goes to stderr.
Why >= 18: the wasm uses strict SIMD128 only — no Relaxed SIMD — so it
validates on V8 10.2 / Node 18 and up. Node 18 is covered by the CI
smoke test on this exact build; the pack tools were re-tested locally
on Node 24. On Node < 18, do not use this pack — upgrade first.

## Selftest first

Run this before using the pack for real work:

```
node bin/run.mjs examples/linear-relu.json
```

You must get `ok: true` and outputs `[4.5, 10.5]`. If you do not, stop:
the pack is broken or you are running it from the wrong directory. Do
not work around a failed selftest by adjusting the spec.

Two readings that look wrong but are by design: in `caller_declared`
mode the output field `ready` is `null` — there is no host signature
to gate on, so `ready: null` next to `ok: true` means "no host gate in
this mode", not "not ready"; the success signal is `ok: true`
(`execution_gate` explains the gate). And the first run takes ~5 s —
mostly session startup/teardown, not the 2-element computation; that
is normal.

## The workflow

"End-to-end" in this document means the full chain validate → run →
verify. The selftest above is the run step alone.

1. **validate** — check a plan before running it:
   `node bin/validate.mjs <spec.json>`
   Exit 0 means the plan passed three layers: (a) static — every
   constructor exists in the catalog and the static shape/dataflow
   check is clean; (b) arg types — layer args match the catalog
   parameters (required-arg counts, u32 = non-negative integer,
   bool = real boolean, f64 = finite number, null only for Option
   params, registry ids unique; weights/inputs are finite-number
   arrays; port shapes are 4-D); (c) engine — port count and unused
   ports. Exit 2 means it failed one of those layers, and the output
   carries a structured error with a remediation hint. Apply the
   hint; do not proceed to run.
   "Valid" does NOT guarantee run success: a spec can still fail
   at create (bad enum, weight count) or at run (runtime shape
   contract). It only guarantees the plan is not already
   known-bad before execution starts.
   `valid: true` + `code: "ok"` is a verdict about the static plan
   only, not a promise that `run` will succeed and not a verdict
   that the plan is executable. Read `static_warnings` as part of
   the verdict, not as decoration: a warning that names an
   execution rejection deterministically predicts a failed run,
   even though `code` stays `"ok"`. The concrete case is an int*
   spec that cannot run as written — a pure-int spec (output slot
   i64) or a dtype mismatch — which validates with a segment-plan
   warning naming the build failure (see the mixed-dtype paragraph
   below) and then fails at build in `run` (exit 2) with that same
   named error. The warning was the whole execution story; treat
   it as a planning error (add the missing cast, or route pure-int
   work through `bin/int.mjs`), not as a reason to re-run the same
   spec. A mixed spec that DOES end in f32 runs — its warning
   describes the segment plan, not a failure.
   `run` is the arbiter: only a run result proves executability.
   `code` stays `"ok"` here on purpose: it reports the static
   check, not executability, and warned-but-valid is still `ok`
   under that contract.
2. **run** — execute a validated spec:
   `node bin/run.mjs <spec.json>`
   Output: `{ok, outputs: {shape, values}, execution_gate,
   host_provenance}`. A spec is a JSON graph (see `spec-schema.md`):
   layers, steps, ports, optional weights and inputs.
   Reading the run output fields: `execution_gate` is a gate
   descriptor, not a pass/fail stamp — its `field` is `"ready"`,
   and its embedded `meaning` text is the definition to trust:
   `host_provenance.ready === true` is necessary and sufficient
   for the host to permit execution in signed modes.
   `host_provenance` is the ingress authority record. In the
   default (ingress) run its `mode` and `authority_mode` are
   `"caller_declared"`: inputs carry no signatures, so `ready` is
   `null`, `enforcement_effect` is `"none"` (the host checks
   structural readiness only), and the gate is deferred to the
   ingress `status.ready` field instead — which is why top-level
   `ready` is also `null`. `claim_verification` and
   `denial_reason` are `null`, and `host_authority_attested` is
   `false`, for the same reason: nothing was attested.
   `execution_gate` also notes it supersedes
   `execution_authorized`, which was removed because it was
   hardcoded false and never represented a gate decision. With
   `--direct` there is no ingress session at all, so
   `execution_gate`, `host_provenance`, and top-level `ready` are
   all `null` — and `mode` is `"direct"`. Those nulls are the
   honest record of "no ingress artifacts exist", not a fake
   gate: a direct run proves nothing about ingress authority
   because it never touches it. A third shape exists for
   mixed-dtype runs (session path only): `execution_gate`,
   `host_provenance`, and top-level `ready` are all `null` —
   exactly like `--direct` — because the run executes through
   direct registry primitives rather than the ingress session;
   but the output additionally carries `segments` (the segment
   receipt: index, kind, steps, layers, inputs, outputs per
   segment) and `boundary_casts`. Null gate/provenance PLUS a
   segment receipt is the mixed-run signature — do not read the
   nulls as "direct mode" (`mode` stays the session mode); the
   receipt is the provenance for a mixed run. When in doubt, read the `meaning`
   text embedded in the output itself; it states the gate rule
   in full.
3. **verify** — compare a candidate graph against a baseline graph:
   `node bin/verify.mjs <baseline.json> <candidate.json>`
   Exit 0 iff equivalent, 2 if not equivalent or the specs are
   invalid, 1 for tool-level failures. It reports `max_abs_error`
   and `max_rel_error`, plus `verdict: ACCEPT|REJECT`; a REJECT is
   a result, not an error to hide. `cases` is the case count;
   per-case detail lives under `case_details`.
   Worked example for the linear-relu chain:
   `node bin/verify.mjs examples/linear-relu.json examples/linear-relu-mutated.json`
   — the mutated file differs from the baseline in one weight only
   (first weight 1 → 2), so expect `verdict: REJECT`,
   `max_abs_error: 1`, exit 2. That REJECT is the correct answer
   for this pair, not a broken tool.
4. **checkpoint** — persist and restore a program bundle:
   `node bin/checkpoint.mjs export <spec.json>` gives `{bundle_base64, sha256}`.
   Extract `bundle_base64` into a `.b64` file first: `verify`/`restore` take a file,
   not inline JSON. `verify <bundle.b64> <expected-sha256>` checks integrity without
   running anything and reports `verdict: ACCEPT|REJECT`.
   `restore <bundle.b64> <spec.json> [expected-sha]` imports into a fresh registry
   and runs the spec's inputs; a wrong expected sha refuses the restore. Restore
   runs the bundle's program, not the spec's graph — the spec supplies only
   inputs/ports. Bundles must be canonical base64 (same rule as verify).

## Arithmetic without a spec

For one-off arithmetic and engineering math, do not hand-write a spec JSON:
- Binary vector ops: `node bin/arith.mjs <add|sub|mul|div|dot|dist|cos|xent|kl> <a1,a2,...> <b1,b2,...>`
- Unary & statistical ops: `node bin/arith.mjs <sqrt|exp|log|abs|norm|normalize|entropy|sum|mean|min|max|var|std> <a1,a2,...>`
- Clamp: `node bin/arith.mjs clamp <a1,a2,...> <min>,<max>`
- Matrix ops: `node bin/arith.mjs matmul <a11,a12;...> <b11,b12;...>` or `node bin/arith.mjs <sum_axis|mean_axis|min_axis|max_axis> <a11,a12;...> <0|1>`
- Batch mode (with `"$prev"` and `"$<idx>"` sequential register chaining): `node bin/arith.mjs --batch <ops.json>`
All ops execute directly through the native WASM kernels (`WasmNumericKernel`, `WasmLinearAlgebra`, `WasmStatistics`, `WasmReduction`, `WasmProbability`). A length-1 vector broadcasts to the other's length on elementwise ops (`add`, `sub`, `mul`, `div`); `div` runs natively via `WasmNumericKernel.div` (0 ULP drift across normal and subnormal ranges; division by zero and divisor underflow are clean errors, exit 2). Both input f32 overflow and output non-finite values (`Infinity`/`NaN`) are rejected fail-closed with exit 2 (`non-finite value at index ...`). Every arith output also carries `precision_warnings` (always present, possibly empty), naming inputs that lost mantissa precision on entry to f32 (e.g. 16777217 → 16777216); the `a`/`b` echo keeps the given values.

Exact integers are a separate tool, not an arith flag:
`node bin/int.mjs <add|sub|mul|div|rem> <a1,a2,...> <b1,b2,...> [--wrap]` (or `matmul` with `;`-separated rows, `to-f32 <a1,a2,...> [--allow-inexact]`, `from-f32 <f1,f2,...>`, or `--batch <ops.json>` with `"$prev"` / `"$<idx>"` chaining). Int runs through the registry (`LayerRegistry.forwardIntBinary`, `forwardIntCastToFloat`, `forwardIntCastFromFloat`), never the f32 graph. Inputs for i64 ops are integer strings (full i64 range); outputs are decimal strings. By default, arithmetic overflow beyond `[-2^63, 2^63 - 1]` (including intermediate `matmul` products/accumulations) is rejected fail-closed with exit 2 (`i64 overflow ...`); pass `--wrap` (or `"wrap": true` in batch) only when two's-complement wrapping is explicitly desired. `to-f32` enforces the exact integer-in-f32 boundary (`|x| <= 16777216`, `2^24`) unless `--allow-inexact` is passed. `div` truncates toward zero; division by zero (and `MIN/-1`) is an error for `div`/`rem`.

Mixed-dtype specs run (session path only, v1): `run.mjs` splits
the spec at dtype boundaries into f32 graph segments and int
registry-call segments over one shared `LayerRegistry`, and the
output carries a segment receipt (`segments`, `boundary_casts`)
alongside the null gate/provenance described above. The split is
static and fail-closed: dtype propagation rejects a mismatch
before any segment runs (named build error: step, slot, expected,
actual, missing cast); `intToFloat` rejects any element with
`|x| > 16777216` (`2^24`); `intFromFloat` rejects non-finite and
out-of-i64-range inputs. The output slot must be f32 in v1: a
pure-int spec (or a mixed spec ending in int) fails at build
with a named output-dtype error — end it with `intToFloat`, or
route the pure-int computation through `bin/int.mjs` above.
`validate` reports one segment-plan warning per spec: for a
runnable mixed spec, `mixed-dtype spec: N segments (kinds),
boundary casts at steps [...]...`; for a spec that fails at
build, `mixed-dtype build error (run will fail at build): <msg>`.
The old per-step "cannot execute inside MultiInputGraphPlan f32
graphs" text is gone: int steps no longer pretend to be graph
steps. `--direct` does not support mixed specs in v1 (named
refusal, exit 2); checkpoint export/restore refuse mixed specs
(no single mixed bundle format exists). `run.mjs` outputs are
JSON numbers, so i64 values beyond `2^53` cannot be exact there
— int intermediates stay exact inside the run and are only
exposed through the f32 output.

Elementwise transcendental and rounding kernels are a separate tool, not arith ops:
`node bin/kernel.mjs <sin|cos|tan|floor|ceil|round> <a1,a2,...>` (unary),
`node bin/kernel.mjs <atan2|powf|hypot> <a1,a2,...> <b1,b2,...>` (binary),
`node bin/kernel.mjs fma <a1,a2,...> <b1,b2,...> <c1,c2,...>` (a*b+c per element via f32::mul_add; fused execution is NOT claimed on wasm32 without a scalar fma instruction), or `node bin/kernel.mjs --batch <ops.json>` (with `"$prev"` / `"$<idx>"` sequential register chaining).
All 10 run through `WasmNumericKernel` with the engine's checked output validation, so domain failures are honest errors (exit 2): negative base with fractional exponent, 0 to a negative power, tan poles, and overflowing results fail instead of returning a value. Three traps, each pinned by a regression case: (1) kernel `cos` is the elementwise cosine in radians — arith's `cos` is the cosine similarity of two vectors, a different operation sharing a short name; (2) `round` is round-half-to-EVEN (2.5 -> 2, 3.5 -> 4), not JS `Math.round`; (3) `atan2` takes y first, then x. Unlike arith, kernel ops do NOT broadcast: every input vector must have exactly the same length, because the kernels themselves require same-shape inputs. Output shape matches arith (`ok`, `op`, echoes, `broadcast: null`, `engine`, `values`, `precision_warnings`). Shared f32 CLI parsing (vector/matrix parsing, quantization warnings, provenance, finite-output guard) lives in `lib/cli.mjs`, imported by both arith and kernel — never duplicated per tool.

## How to find things (read this, it is the whole trick)

Do NOT open anything under `engine-audit/`. That directory is the
verbatim CI build: the generated bindings alone are 11,320 lines, the
surface snapshot is 5,766 lines, the type file is 2,382 lines. Opening
them is how agents waste context and still miss the entry they need.
One exception: when a `reference/` page names a specific contract
file under `engine-audit/docs/`, reading exactly that file is allowed.
The ban still covers the code and generated bindings (`burn_research.js`,
`burn_research.d.ts`, `wasm-surface.actual.json`).

Instead, ask the query tools. They read the large files
programmatically and print exactly one entry:

- `node bin/find-op.mjs <constructor>` — one layer constructor,
  complete (signature, parameters, family). `--list` prints one line
  per constructor (52 total).
- `node bin/surface.mjs <name>` — one wasm function, class, or
  method. For a method, the answer pairs each signature with its
  class — parameter order differs between classes, so never assume
  one signature fits all. `--list` prints one line per entry
  (115 functions + 54 classes).
- `node bin/layout.mjs <constructor>` — that constructor's layout
  contract (input and output layout, with shapes). Ask here BEFORE
  writing a spec that uses a norm/conv/embedding layer: e.g.
  `layerNorm` requires `feature_last`, not the common
  `feature_axis1_singleton`. `--list` prints one line per constructor
  (52 total).

For a quick scan without any tool, read `contracts-index/`:
`catalog-index.jsonl` is one constructor per line (52 lines),
`surface-index.jsonl` is one surface entry per line (169 lines),
`layout-index.jsonl` is one layout contract per line (52 lines).
One entry is one line, so grep on these files is safe. Grep on the
raw files under `engine-audit/` is not.

For background, read `reference/` — one short page per topic, each
stating what it is for, when to use it, and when NOT to use it.
`INDEX.md` maps questions to the right tool or page. `MANIFEST.json`
pins the engine build (commit, sha256, entry counts).

## Spec rules that will bite you

- Shapes are 4-D `[N, C, H, W]`; a length-`n` vector uses `[1, n, 1, 1]`.
- At least 2 external ports, and every declared port must be consumed
  by a step, or validation fails (`plan_minimum_ports`,
  `plan_unused_ports`).
- Weights are column-major: for `linear`, `y_i = sum_j w[i + j*outDim]
  * x_j`. Length is `inDim*outDim`, plus `outDim` bias values when
  `bias=true`. Unset weights are deterministic zero, not an error.
- `args[0]` of every layer is its registry id (a non-negative integer),
  unique across layers. `weights` keys are spec.layers indices, not
  registry ids. A `bias` arg must be a real boolean — the string
  `"false"` is rejected, not read as false.
- `ok` means the tool ran to completion; for checks, the outcome is
  `verdict: ACCEPT|REJECT` plus the exit code. Never read `ok: true`
  alone as "the check passed": verify and checkpoint-verify return
  `ok: true` with `verdict: REJECT` and exit 2 when the check fails.
- Errors come in three shapes, all JSON on stdout — branch on what
  you actually got, not on one assumed shape:
  - Validate-layer failures: `{ok: false, valid: false, stage:
    "static"|"plan", code, remediation_hint}`, exit 2.
  - Tool-level failures (missing file, bad JSON, unreadable spec):
    `{ok: false, error}`, exit 1.
  - Run-phase engine failures: `{ok: false, error, error_envelope:
    {code, phase, remediation_hint, execution_started}}`, exit 1.
    Pre-execution failures carry `execution_started: false`; a
    runtime shape failure may carry `execution_started: true`.
  Run-phase failures also carry a `stage` field naming where the run
  stopped: `loadSpec` (the spec file could not be read or parsed),
  `argValues` (argument typing or input-length check failed — record
  carries `code: "invalid_argument"` and an `issues` list),
  `validatePlan` (the plan is invalid — record spreads the plan
  object), `build` (the graph could not be built: a static check
  failed, `splitSpec` threw, or the output slot is i64), or `run`
  (the engine threw during execution — record carries `error` and
  `error_envelope`, which may be null). The same five stages appear
  in single-spec, `--direct`, and batch records; the field is
  additive and changes no exit code. Do not confuse these with the
  validate-layer stages above (`"static"`|`"plan"`), which belong
  to `validate`/`shapecheck`, not to `run.mjs`.
  Either way, a failed run produces no result to use — never treat
  a partial output as an answer.

## What this pack is not

It is not a general math library, not a training framework, and not a
place to add features. The engine is frozen at the audited build in
`MANIFEST.json`. If you need an operation that is not in
`find-op --list`, that is a gap to report, not a reason to improvise
a replacement in your own code.

## Recursive & Agent-Harness Co-Processor (v3.8)

In addition to one-shot flat specs, this pack provides a 4-layer recursive co-processor and Stage-2/3/4 recursive upgrades across all existing domain features, designed to work alongside an LLM agent harness without mutating the frozen WASM binary (`MANIFEST.json` pins remain unchanged):

1. **Layer 1 — Structural & Compositional Recursion (`bin/compose.mjs` / `lib/composer.mjs`)**:
   - Author hierarchical specs (`schema: "burn-math.recursive-spec.v2"`, see `examples/recursive-gated-residual.v2.json`) with symbolic slot names, nested `subgraphs`, macros calling macros, and combinators:
     - `repeat` (`depth: N`, `tie_weights: true|false`) — unrolls a subgraph or macro $N$ times; when `tie_weights: true`, all iterations share the same `LayerRegistry` IDs (true weight-tied recurrence).
     - `residual` — wraps a subgraph/macro $F(x)$ with an automatic skip connection $F(x) + x$.
     - `parallel_concat` — runs multiple branches in parallel and joins them with a `concat` chain.
   - Guarded against cyclic subgraph definitions (`recursive_cycle_detected`, exit 2).
   - `--auto-repair` deterministically repairs missing `intFromFloat`/`intToFloat` boundary casts, drains unused ports, and shards oversized `linear` layers (`> 65535` weights), recording every edit in `meta.remediation_receipt`.
   - `--compact-slots` performs liveness-based slot register allocation (`compactSlotsByLiveness` in `lib/explain.mjs`), recycling dead intermediate slots so deep unrolled graphs stay well below the 64-slot `AgentWorkspace` limit.
   - `node bin/compose.mjs grad-spec <spec.json> [--wrt <slot>] [--order 1|2]` compiles reverse-mode VJP (order 1) or 2nd-order Hessian-Vector Product (order 2) into a standalone runnable Spec v1, supporting linear, bilinear (`mul`), rational (`div`), radical (`sqrt`), and non-linear (`sigmoid`, `tanh`) graphs.
2. **Layer 2 — Temporal & Execution-Time Recursion (`bin/rollout.mjs` / `lib/rollout.mjs`)**:
   - `node bin/rollout.mjs scan <spec.json> [--steps N] [--state-port <slot>]` runs recurrent state handoff ($s_{t+1} = \text{Graph}(s_t, x_t)$) inside one leak-free WASM scope with per-tick SHA-256 digests.
   - `node bin/rollout.mjs fixpoint <spec.json> [--tol 1e-6] [--max-steps 100]` converges only on a second step with strict contraction ($\|s_{t+1} - s_t\|_\infty \le \text{tol}$ AND the step shrank genuinely); a single step within tol never certifies. It reports `empirical_lipschitz` (computed independently of trajectory recording, so workflow stages that do not record trajectories still get it) and fails closed (exit 2) on explosive divergence; non-finite values surface as engine errors before the `DIVERGED_NON_FINITE` branch, which is a second defense, not the main path.
   - `node bin/rollout.mjs unroll <spec.json> [out.json] [--steps N] [--state-port <slot>] [--compact-slots]` temporally unrolls any 1-step recurrent spec across $N$ ticks into a weight-tied Spec v1 for end-to-end BPTT differentiation (`autodiff.mjs`, `compose.mjs grad-spec`) or training (`recurse.mjs solve-gd`).
   - `node bin/rollout.mjs bisect <base.json> <cand.json>` localizes the first divergent step via `runWithTrace` SHA-256 digests and extracts a runnable prefix sub-spec (prefix plus drain steps for newly-unused ports, honestly recorded in `meta.remediation_receipt` — "minimal" means prefix+drain, not prefix alone).
3. **Layer 3 — Recursive Search, Multi-Layer ES & Closed-Loop Reflective GD (`bin/recurse.mjs` / `lib/recurse.mjs`)**:
   - `node bin/recurse.mjs search <spec.json> [out.json] [--max-depth 3] [--quant-tol 1e-3]` searches a tree of proof-backed graph rewrites (`linear-branch-fusion`, `linear-chain-fusion` for rectangular $m \times k \times n$ matrices with bias, `relu-idempotent`, `transpose-involution`, `dead-step-elimination`) and int8 quantization plans over `CheckpointBranchSet`, gated by `MultiInputVerificationCases` (absolute OR relative tolerance — a node passes if either bound holds; `quant-tol` bounds absolute error over the case envelope, relative error is unbounded and not shown in search output) and profiled by the Fase A memory/time oracle. `quantized_saved_bytes` is a projection of engine state, not savings in the emitted artifact (the best spec still carries full f32 weights).
   - `node bin/recurse.mjs fit <spec.json> <targets.json> [gens] [--sensitivity-guided]` (and `bin/es_optimize.mjs`) optimizes weights across one or multiple weighted layers simultaneously.
   - `node bin/recurse.mjs solve-gd <spec.json> [targets.json] [--steps 20] [--lr 0.1] [--layer 0|all] [--multi-layer] [--precondition] [--out out.json]` runs closed-loop backtracking line-search gradient descent ($\eta \leftarrow \eta / 2$ on ascent) across a single layer or all active weighted layers jointly (`--multi-layer` / `--layer all`; without `--layer` on a multi-layer spec, ALL layers train) using analytic BPTT gradients (`autodiff.mjs`), optional diagonal Hessian preconditioning (`fit_hessian_diag.mjs --layer <idx>`), and `verifyGdSteps` trajectory certification (`ascent_count === 0, mismatch_count === 0`, where mismatch is now recomputed from recorded weights through fresh engine evaluations). Targets, when provided, must hold `{"cases":[{"inputs":[...],"expected":[...]}]}` (bare arrays are rejected); without targets, 1-output specs optimize the scalar output itself and multi-output specs minimize toward zero, and the mode taken is declared as `targets_mode` (`explicit` | `scalar_objective` | `zero_default`). `ok:true` means the trajectory is certified, not that the target was reached. `--precondition` rescales steps (~1/h), so `lr` needs retuning when it is on.
4. **Layer 4 — Agent Harness Integration (`bin/harness.mjs` / `lib/harness.mjs`)**:
   - `node bin/harness.mjs capabilities [compose|verify|differentiate|optimize|rollout|state]` maps all 47 CLIs across 6 intent domains.
   - `node bin/harness.mjs digest <spec.json> [--workspace .burn-harness]` runs a spec and returns a compact Semantic Digest (< 2 KB: shape, SHA-256, min/max/mean/std/norms, preview head, and `next_actions`) while storing full tensors in a content-addressed store (`burn://sha256:...`). Any spec can reference `"burn://sha256:..."` in `inputs` or `weights`; tampered artifacts fail closed (`artifact_integrity_failed`, exit 2).
   - `node bin/harness.mjs run-workflow <workflow.json | ->` executes multi-stage pipelines (`compile_v2` -> `validate` -> `rollout_unroll` -> `search_rewrites` -> `solve_gd` -> `grad_spec` -> `rollout_fixpoint` -> `checkpoint_promote`) in one warm WASM process and syncs `AgentWorkspace` state to `.burn-harness/workspace-state.json` (including partial state when a stage fails — the failure is reported as a stage report, not a lost trace). The `solve_gd` stage requires explicit `targets` (`{"cases":[...]}`); without them the stage fails closed instead of silently optimizing toward zero.
   - `node bin/harness.mjs branch-export` and `branch-merge` coordinate parallel subagent exploration gated by multi-case equivalence checks over `CheckpointBranchSet` (baseline inputs plus deterministic derived cases — zeros, ones, negated, scaled — kept only where the baseline evaluates cleanly; the surviving case names are reported as `equivalence_cases`).
5. **Stage-2, Stage-3 & Stage-4 Domain Feature Recursion (`v3.5`–`v3.7`)**:
   - **Autodiff (31 Rules), DAG Quantization (22 Ops) & Reference Oracle (34 Ops)**: `lib/autodiff.mjs` covers 31 proven VJP rules (`node bin/autodiff.mjs prove-rules` runs 32 dual hand + FD proofs, including `div`, `sqrt`, `mish`, `glu`, `rmsNorm`, `layerNorm`, `featureNorm`, `prelu`, `hardSigmoid`, `hardSwish`); `lib/quant_budget.mjs` computes analytical Lipschitz bounds across DAG topologies and 22 constructors (including gated `mul` and norms); `lib/ref.mjs` (`bin/diff_ref.mjs`) covers 34 constructors (including `mish`, `glu`, `rmsNorm`, `layerNorm`, `avgPool1d`, `maxPool1d`, `avgPool2d`, `maxPool2d`, `embedding`) in independent f64 reference math.
   - **Multi-Branch `CheckpointBranchSet` Competition (`bin/graph_mutate.mjs`)**: Supports both single-branch `GraphMutationTransaction` (`"edits": [...]`) and multi-branch `CheckpointBranchSet` competition (`"branches": [{"branch_id": ..., "edits": [...]}, ...]`), reporting byte-level `state_diff` (`changed_byte_count`, `ranges`) and promoting the winning branch via `commitBranchByReceipt`.
   - **Rectangular Weight Lab & Expression Trees (`bin/weight_lab.mjs`)**: Supports rectangular $m \times n$ matrices and `node bin/weight_lab.mjs eval <expr.json> [--out C.json]` to evaluate recursive `ref` / `literal` / `compose` / `blend` / `repeat` weight expression trees.
   - **Recursive 3D Transforms, Scene Trees & Inverse Kinematics (`bin/geom_build.mjs`, `bin/geom_project.mjs`, `bin/geom_batch.mjs`)**: `opToMat4` supports `{ "op": "repeat", "depth": N, "chain": [...] }`; `geom_build.mjs --scene-tree` and `--solve-ik` compile kinematic trees and solve IK; `geom_project.mjs` and `geom_batch.mjs` directly project hierarchical `nodes` scene trees to 2D screen coordinates.
   - **Universal `loadSpec`, Weight-Tied Diagnostics & Custom Tolerances**: `lib/spec.mjs` transparently compiles `burn-math.recursive-spec.v2` and hydrates `"burn://sha256:..."` across all 26+ CLIs; `layer_state.mjs inspect` reports `used_by_steps` and `is_weight_tied`; `trace_hash.mjs --isolate` extracts runnable prefix subspecs on divergence; `opt_compare.mjs --tol <tol>` supports approximate/quantized equivalence gates; and `fit_gd_verify.mjs --layer all` audits joint multi-layer GD trajectories.
