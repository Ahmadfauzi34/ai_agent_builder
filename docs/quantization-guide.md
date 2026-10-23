# Quantized State Guide (int8 PTQ)

The engine's layers live in f32. Their *checkpoints* don't have to.
This guide covers the quantized state API: int8 weight storage with
analytical error bounds, using Burn 0.20.1's own quantization machinery.

## 1. What it is (and isn't)

**Is:** a compressed *representation* of layer weights for storage and
transport. Weights are stored as int8 + f32 scales; checkpoints get ~4x
smaller.

**Isn't:** faster compute. Burn has no integer compute — every op on a
quantized tensor dequantizes to float first. There is also no QAT
(quantization-aware training); this is PTQ (post-training quantization)
only. Live layers stay f32; only the serialized bytes are quantized.

**Rule of thumb:** use quantized state when you ship, store, or cache
checkpoints. Keep f32 state when you train (ES) or need exact
reproducibility of every bit.

## 2. The API

Two methods on every layer type with state (`WasmLinear`, `WasmConv`,
`WasmNorm`, `WasmEmbedding`, `WasmActivation`, `WasmGhostModule`,
`WasmSeBlock`):

| Method | Args | Returns |
|---|---|---|
| `get_quantized_state(scheme)` | `"q8-per-tensor"` \| `"q8-per-block"` | `Uint8Array` (enveloped bytes) |
| `quantize_state_report(scheme)` | same | JSON string (see §4) |

`scheme` is a closed string enum — unknown values are a loud `Err`, not a
silent default.

`load_state` accepts **both** encodings: it auto-detects the `BQZ1`
envelope, dequantizes to f32, and runs the exact existing validation +
load path. No separate load function, no caller branching.

## 3. The schemes

- **`q8-per-tensor`**: one f32 scale per parameter. Simplest; one clean
  bound (`scale/2`). Vulnerable to outliers — a single large weight
  inflates the scale for the whole tensor.
- **`q8-per-block`** (default recommendation): one scale per canonical
  block. Outliers are confined to their block.

Canonical block geometry (deterministic, no agent choice):
- 1-D params (bias, norm) → per-tensor (per-element blocks would be
  degenerate).
- D-dim params (D ≥ 2) → blocks `[1, b1, …, b_{D-1}]` where `b_i` is the
  largest divisor of `d_i` fitting in u8 (Burn's `BlockSize` limit).
  For dims ≤ 255 this is one dim-0 slice per block (e.g. per-output-channel
  for conv, per-row for linear).

## 4. The report (for agents)

`quantize_state_report` returns:

```jsonc
{
  "scheme": "q8-per-block",
  "params": [
    {"index": 0, "shape": [8, 64], "num_scales": 8,
     "scale_max": 0.0077, "bound": 0.00385}
  ],
  "bytes_f32": 2304,
  "bytes_q8": 640,
  "ratio": 0.28
}
```

`bound` is the analytical worst case: `max|w − dequantize(w)| ≤ bound`
per parameter. It is computable *before* any experiment, from the weights
alone. Use it to derive tolerances (e.g. for a linear layer,
`|Δyᵢ| ≤ bound_W · Σ|x| + bound_b`) instead of inventing a tolerance.

## 5. Guarantees and limits

- **Deterministic**: quantization is a pure function
  (min-max → scale → round). Same weights → byte-identical output, every
  time. No randomness anywhere in the path.
- **Backend limit**: on the NdArray backend (what the WASM engine uses),
  only int8 symmetric (`Q8S`/`Q8F`) is implemented. int4/int2/FP8/FP4 exist
  as types but panic `unimplemented!` — the scheme enum deliberately
  doesn't offer them.
- **Overhead honesty**: dequantize-then-compute costs a full pass over the
  weights per use (measured ~3.5x on 256² matmuls, ~1.8x on 1024² with
  Burn's reference implementation). Quantized *inference* is slower, not
  faster — the win is checkpoint size and load time.
- **Scope**: weights only in effect (all float params are quantized,
  including biases). Activations stay f32 — matching Burn's `Quantizer`
  semantics.
