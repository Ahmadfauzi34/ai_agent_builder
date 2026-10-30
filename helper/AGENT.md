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

## The workflow

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
2. **run** — execute a validated spec:
   `node bin/run.mjs <spec.json>`
   Output: `{ok, outputs: {shape, values}, execution_gate,
   host_provenance}`. A spec is a JSON graph (see `spec-schema.md`):
   layers, steps, ports, optional weights and inputs.
3. **verify** — compare a candidate graph against a baseline graph:
   `node bin/verify.mjs <baseline.json> <candidate.json>`
   Exit 0 iff equivalent, 2 if not equivalent or the specs are
   invalid, 1 for tool-level failures. It reports `max_abs_error`
   and `max_rel_error`, plus `verdict: ACCEPT|REJECT`; a REJECT is
   a result, not an error to hide. `cases` is the case count;
   per-case detail lives under `case_details`.
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

For one-off arithmetic, do not hand-write a spec JSON:
`node bin/arith.mjs <add|sub|mul|div> <a1,a2,...> <b1,b2,...>`
prints engine output (`{ok, op, a, b, broadcast, engine, values}`).
A length-1 vector broadcasts to the other's length; anything else must
match exactly. `div` is mul by the host-computed reciprocal (division
by zero and subnormal-divisor underflow are clean errors, exit 2);
precision: div stays within 1 ulp of f32(true division) for normal
values (measured 2026-10-31; pinned by the regression div probe).
`matmul` takes `;`-separated rows (a bare comma list is one row),
single-batch `[1,1,M,K] @ [1,1,K,N]`, output flat row-major with
`shape: [M,N]`; `--batch ops.json` runs N ops
(add/sub/mul/div/matmul) in one engine load from a `{"ops":[...]}`
file; a failing op is recorded with its index and the rest still run;
exit 0 iff every op succeeds. `engine.wasm_sha256_short`
identifies the build that computed the result.

Exact integers are a separate tool, not an arith flag:
`node bin/int.mjs <add|sub|mul> <a1,a2,...> <b1,b2,...>` (or `matmul`
with `;`-separated rows, single-batch). Int runs through the registry
(`LayerRegistry.forwardIntBinary`), never the f32 graph — graph slots
are f32 and addBinary rejects int specs by design (CP5 correction
2026-10-31). Inputs are integer strings (full i64 range); outputs are
decimal strings (BigInt is not JSON-serializable and Number() would
round beyond 2^53). Overflow of the arithmetic itself wraps
two's-complement silently in release builds (Rust native `+`/`*`);
"never wraps" is the caller's invariant. No div (no intDiv
constructor), no casts, no batch in v1.

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
  per constructor (48 total).
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
  (48 total).

For a quick scan without any tool, read `contracts-index/`:
`catalog-index.jsonl` is one constructor per line (48 lines),
`surface-index.jsonl` is one surface entry per line (169 lines),
`layout-index.jsonl` is one layout contract per line (48 lines).
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
  Either way, a failed run produces no result to use — never treat
  a partial output as an answer.

## What this pack is not

It is not a general math library, not a training framework, and not a
place to add features. The engine is frozen at the audited build in
`MANIFEST.json`. If you need an operation that is not in
`find-op --list`, that is a gap to report, not a reason to improvise
a replacement in your own code.
