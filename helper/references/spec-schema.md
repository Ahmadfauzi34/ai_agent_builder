# Spec schema

All `bin/` CLIs consume one JSON spec. Fields:

```jsonc
{
  "numSlots": 6,                       // total graph slots (inputs + intermediates + output)
  "layers": [                           // registry layers, in order
    {"constructor": "linear", "args": [0, 2, 2, false]},
                                        // args: [registryId, inDim, outDim, bias]
    {"constructor": "relu",   "args": [1]},
    {"constructor": "add",    "args": [2]}
  ],
  "steps": [                            // dataflow; layer = index into layers[]
    {"kind": "unary",  "layer": 0, "slots": [0, 2]},
    {"kind": "unary",  "layer": 1, "slots": [1, 3]},
    {"kind": "binary", "layer": 2, "slots": [2, 3, 4]},
    {"kind": "unary",  "layer": 3, "slots": [4, 5]}
  ],
  "outputSlot": 5,
  "ports": [                            // external input ports (min 2, all must be consumed)
    {"slot": 0, "role": "observation", "shape": [1, 2, 1, 1],
     "layout": "feature_axis1_singleton"},
    {"slot": 1, "role": "observation", "shape": [1, 2, 1, 1],
     "layout": "feature_axis1_singleton"}
  ],
  "weights": {                          // optional; key = index into layers[]
    "0": [1, 0.5, -0.25, 2],            // column-major, length inDim*outDim (+outDim if bias)
    "1": [2, 1, 0.5, -1]
  },
  "inputs": {                           // optional (run/verify); key = port slot
    "0": [1.0, -2.0],
    "1": [0.5, 0.25]
  }
}
```

Notes:

- **Constructors**: `linear`, `relu`, `add` (also `sigmoid`, `tanh`, `mul`,
  `concat`, `reshape`, `transpose`, `matmul`, `layer_norm`, `softmax`,
  `embedding`, `conv2d` — see `vendor/burn-research-pkg/docs/agent-layer-catalog.v1.json`).
  `args[0]` is always the non-negative registry layer id.
- **Roles** (7 canonical): `observation`, `state`, `feature`, `candidate`,
  `parameter`, `reward`, `context`. Extension roles: `x-` prefix, e.g. `x-my-role`.
- **Layouts** (8 canonical): `feature_axis1_singleton` is the common one for
  `[1, C, 1, 1]` vectors; see `capabilities` vocabulary in the vendor docs.
- **Shapes** are 4-D `[N, C, H, W]`; a length-`n` vector input uses `[1, n, 1, 1]`.
- **Weights are column-major**: for `linear`, `y_i = sum_j w[i + j*outDim] * x_j`
  (equivalently `y_i = w[i]*x0 + w[i+2]*x1` for 2→2). Length is `inDim*outDim`,
  plus `outDim` bias values appended when `bias=true`.
- **Unset weights are deterministic zero** (audited) — a spec without
  `weights` runs reproducibly; add `weights` for real parameters.
- **Plan invariants** (checked by `validate`, fail fast with codes):
  at least 2 declared ports (`plan_minimum_ports`); every declared port must
  be consumed by a step (`plan_unused_ports`).
- `verify.mjs` compares two specs graph-vs-graph over `cases.json`
  (`[{"0":[...],"1":[...]}, ...]`, slot → values per case); default is one
  case from the baseline spec's `inputs`, applied to both.

## Minimal example

`run.mjs` on the example above prints (values abbreviated):

```json
{
  "ok": true,
  "outputs": {"shape": [1, 2, 1, 1], "values": [2.625, 0]},
  "execution_gate": {"field": "ready", "...": "..."},
  "host_provenance": {"mode": "caller_declared", "...": "..."}
}
```

Hand-check (column-major): `lin0([1,-2]) = [1.5,-3.5]`,
`lin1([0.5,0.25]) = [1.125,0.25]`, `add = [2.625,-3.25]`,
`relu = [2.625,0]`. The engine output matches exactly.

## When the engine says no

Errors come back as `{ok:false, error, error_envelope:
{code, phase, message, remediation_hint, ...}}`. Branch on `code`, apply
`remediation_hint`. Common ones: `plan_minimum_ports`, `plan_unused_ports`,
`invalid_enum` (with `allowed_values`), `invalid_argument`,
`weight count mismatch`. Nothing starts executing unless validation passes
(`execution_started:false`).
