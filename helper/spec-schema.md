# Spec schema

All `bin/` CLIs consume one JSON spec. The canonical example is
`examples/linear-relu.json` — copy it, do not retype the example below
from memory. The example below IS that file, verbatim in structure.

```json
{
  "numSlots": 5,
  "layers": [
    {"constructor": "linear", "args": [0, 2, 2, false]},
    {"constructor": "relu",   "args": [1]},
    {"constructor": "add",    "args": [2]}
  ],
  "steps": [
    {"kind": "unary",  "layer": 0, "slots": [0, 2]},
    {"kind": "unary",  "layer": 1, "slots": [2, 3]},
    {"kind": "binary", "layer": 2, "slots": [3, 1, 4]}
  ],
  "outputSlot": 4,
  "ports": [
    {"slot": 0, "role": "observation", "shape": [1, 2, 1, 1],
     "layout": "feature_axis1_singleton"},
    {"slot": 1, "role": "observation", "shape": [1, 2, 1, 1],
     "layout": "feature_axis1_singleton"}
  ],
  "weights": {
    "0": [1, 0.5, -0.25, 2]
  },
  "inputs": {
    "0": [1.0, -2.0],
    "1": [3.0, 10.5]
  }
}
```

Run it: `node bin/run.mjs examples/linear-relu.json` prints
`outputs: {shape: [1, 2, 1, 1], values: [4.5, 10.5]}`.

Hand-check (column-major): `linear([1,-2]) = [1.5,-3.5]` (weights
`[1, 0.5, -0.25, 2]`, `y_0 = 1*1 + (-0.25)*(-2)`, `y_1 = 0.5*1 + 2*(-2)`),
`relu` gives `[1.5, 0]`, `add` with `[3, 10.5]` gives `[4.5, 10.5]`.

Notes:

- **Constructors**: only the 48 in the catalog. Check with
  `node bin/find-op.mjs --list`; check one with
  `node bin/find-op.mjs <name>`. Examples in this pack: `linear`,
  `relu`, `add`, `layerNorm` (camelCase — `layer_norm` does not exist),
  `sigmoid`, `tanh`, `mul`, `concat`, `transpose`, `matmul`, `softmax`,
  `embedding`, `conv2d`. There is NO `reshape` constructor. The full
  catalog is `engine-audit/docs/agent-layer-catalog.v1.json` — do not
  open it directly; use `find-op`.
  `args[0]` is always the non-negative registry layer id.
- **Roles** (7 canonical): `observation`, `state`, `feature`, `candidate`,
  `parameter`, `reward`, `context`. Extension roles: `x-` prefix, e.g. `x-my-role`.
- **Layouts**: `feature_axis1_singleton` is the common one for
  `[1, C, 1, 1]` vectors. Some constructors require a different layout:
  `layerNorm` requires `feature_last` `[B, *, *, F]`. Check a
  constructor's layout with `node bin/layout.mjs <name>` — do not
  guess it from the vector example.
- **Shapes** are 4-D `[N, C, H, W]`; a length-`n` vector input uses `[1, n, 1, 1]`.
- **Weights are column-major**: for `linear`, `y_i = sum_j w[i + j*outDim] * x_j`
  (equivalently `y_i = w[i]*x0 + w[i+2]*x1` for 2→2). Length is `inDim*outDim`,
  plus `outDim` bias values appended when `bias=true`.
  See `examples/linear-relu-bias.json` for a biased linear.
  Weights keys are spec.layers INDICES (0..layers.length-1), not registry
  ids — and registry ids (`args[0]`) must be unique across layers, or
  one layer's weights silently overwrite another's.
- **Unset weights are deterministic zero** (audited) — a spec without
  `weights` runs reproducibly; add `weights` for real parameters.
- **What `validate` checks** (three layers): (1) static — every
  constructor exists in the catalog, and the static shape/dataflow
  check reports no mismatches and no dataflow issues; (2) arg types —
  layer args match the catalog parameters (required-arg counts, u32 =
  non-negative integer, bool = real boolean, f64 = finite number, null
  only for Option params, registry ids unique); (3) engine —
  at least 2 declared ports (`plan_minimum_ports`) and every declared
  port consumed (`plan_unused_ports`). "Valid" does NOT guarantee run
  success: a spec can still fail at create (bad enum, weight count)
  or at run (runtime shape contract), with a structured envelope.
- `verify.mjs` compares two specs graph-vs-graph over `cases.json`
  (`[{"0":[...],"1":[...]}, ...]`, slot → values per case); default is one
  case from the baseline spec's `inputs`, applied to both.

## Integer constructors (int*)

`intAdd`, `intSub`, `intMul`, `intMatmul`, `intToFloat`, `intFromFloat`
operate on `WasmIntTensor` (i64) via `forwardIntBinary`/`forwardIntCast`.
They CANNOT execute inside a `MultiInputGraphPlan` f32 graph: a spec
that uses them validates with a `static_warnings` entry per step (not an
error — the warning is the signal), but the engine rejects the graph at
create. Do not mix int* layers into f32 graph plans; treat them as a
separate integer execution path.

## Weight-count formulas (validate cross-check)

For each layer index present in `weights`, `validate` warns (in
`static_warnings`, not an error) when the array length differs from the
exact count below. A spec WITHOUT weights is still valid — the engine
supplies defaults — so a missing entry warns nothing. The function
`expectedWeightCount` in `lib/spec.mjs` is the single implementation of
this table; `validate` calls it, and the regression suite
(`bin/regression.mjs`) pins it.

| Constructor | Formula (args after id) |
|---|---|
| `linear(in_dim, out_dim, bias)` | `in_dim*out_dim + (bias ? out_dim : 0)` |
| `batchNorm(num_features, eps?)` | `2*num_features` |
| `groupNorm(num_groups, num_channels, eps?)` | `2*num_channels` |
| `instanceNorm(num_channels, eps?)` | `2*num_channels` |
| `layerNorm(size, eps?)` | `2*size` |
| `rmsNorm(size, eps?)` | `size` |
| `conv1d(in_ch, out_ch, kernel, ...)` | `in_ch*out_ch*kernel + out_ch` |
| `conv2d(in_ch, out_ch, kh, kw, ...)` | `in_ch*out_ch*kh*kw + out_ch` |
| `convTranspose2d(in_ch, out_ch, kh, kw, ...)` | `in_ch*out_ch*kh*kw + out_ch` |
| `embedding(vocab_size, d_model)` | `vocab_size*d_model` |

Conv bias is unconditional: the engine's conv constructors take no bias
flag, and the trailing `out_ch` values are always a per-output-channel
bias appended after the kernel weights (probed 2026-10-30: `conv1d`
`[w=2, b=10]` on `x=3` gives `16`, and a bias-less vector is rejected
at bind; the 2-channel case `[2, 3, 10, 20]` on `x=5` gives `[20, 35]`,
i.e. kernel-first-then-bias, not interleaved).

`validate` also warns when an `inputs` array length differs from the
product of its port shape, and when a `weights`/`inputs` key has no
matching layer/port. These are cross-checks against typos and truncated
arrays, not hard errors: the engine itself rejects a true weight-count
mismatch at create (`weight count mismatch`).

## Verify tolerances

`verify.mjs <base> <cand> [cases.json] [absTol] [relTol]`: both
tolerances default to `1e-6`, must be finite numbers `>= 0` (0 means
exact equality), and an invalid value (non-numeric, NaN, Infinity,
negative) is rejected with exit 2 before any graph is built. The same
rule applies per pair in `--batch` mode (that pair reports `ok:false`,
the batch continues). `cases.json` may be omitted, or given as `""`,
which means the same as omitting it: one case from the baseline spec's
`inputs`, applied to both graphs.

## When the engine says no

Errors come in three shapes, all JSON on stdout (see `AGENT.md` for the
full list). Validate-layer failures carry a top-level `code`/`stage`/
`remediation_hint` (exit 2); tool-level failures (missing file, bad
JSON) are a bare `{ok:false, error}` (exit 1); run-phase engine
failures carry `error_envelope: {code, phase, message,
remediation_hint, ...}` (exit 1) — branch on `code` there and apply
`remediation_hint`. Common codes: `plan_minimum_ports`,
`plan_unused_ports`, `unknown_layer_constructor`,
`static_check_failed`, `invalid_enum` (with `allowed_values`),
`invalid_argument`, `weight count mismatch`. Pre-execution failures
carry `execution_started:false`; a runtime shape failure may carry
`execution_started:true` — either way, do not treat a partial output
as a result.
