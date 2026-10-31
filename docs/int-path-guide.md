# Integer Path Guide (i64)

The engine has two numeric worlds. Most layers live in the **f32 world**
(`WasmTensor`): fast, differentiable-friendly, but a float. Boundary #5
adds the **integer world** (`WasmIntTensor`): i64 tensors with exact
integer arithmetic. This guide tells you which world to reach for, and
how to use the integer one without getting stuck.

## 1. Why the integer world exists

An f32 has 24 bits of mantissa. That means integers with |x| > 2^24
(16,777,216) **cannot all be represented** — 16,777,217 rounds to
16,777,216. If you accumulate counts, token ids, or any integer
quantity through f32, large values silently lose exactness. The integer
world stores i64 (Burn `Int` on NdArray), exact up to 2^63 − 1
(9.2 × 10^18).

**Rule of thumb:** if the quantity you are computing must be *exactly*
right and it is an integer, use the integer world. If it needs
fractions, gradients, or neural-network semantics, stay in f32.

## 2. The constructors

| Constructor | Layer type | Arity | What it does |
|---|---|---|---|
| `intAdd(id)` | `int_binary` (0x16) | 2 | elementwise `a + b`, shapes must match |
| `intSub(id)` | `int_binary` (0x16) | 2 | elementwise `a - b`, shapes must match |
| `intMul(id)` | `int_binary` (0x16) | 2 | elementwise `a * b`, shapes must match |
| `intMatmul(id)` | `int_binary` (0x16) | 2 | batched matmul over last two dims: `[*,*,m,k] @ [*,*,k,n]` |
| `intToFloat(id)` | `int_cast` (0x17) | 1 | i64 → f32 cast |
| `intFromFloat(id)` | `int_cast` (0x17) | 1 | f32 → i64 cast, truncating toward zero |

All six are **stateless and parameter-free** (`num_params() == 0`,
empty state). Register with `LayerRegistry.initAgentLayer(spec)`,
then run with the int forward entry points:

- `registry.forwardIntBinary(id, a, b)` → `WasmIntTensor`
- `registry.forwardIntCastToFloat(id, x)` → `WasmTensor` (f32)
- `registry.forwardIntCastFromFloat(id, x)` → `WasmIntTensor`

Do **not** call `forwardLayer` with an int layer id: it returns a clean
error telling you to use the int entry points. Int layers also cannot
appear in graph plans — graph slots are f32; a plan referencing an int
layer fails validation with a descriptive message instead of silently
misbehaving.

## 3. Use cases

### Case A — Exact accumulation ("the cashier")

You are summing integer quantities that can exceed 2^24: event counts,
inventory, token frequencies.

```js
// WRONG (f32 world): 16777217 + 1 === 16777216, silently.
const spec = AgentLayerSpec.intAdd(1);
registry.initAgentLayer(spec);

const a = WasmIntTensor.fromNumbers([16777217, 100], [1, 2, 1, 1]);
const b = WasmIntTensor.fromNumbers([1, -30], [1, 2, 1, 1]);
const out = registry.forwardIntBinary(1, a, b);
console.log(out.to_array()); // BigInt64Array [ 16777218n, 70n ] — exact
```

`fromNumbers` accepts plain JS numbers and **rejects** anything that is
not a safe integer (non-finite, fractional, or |v| > 2^53 − 1) with an
error naming the offending index. For magnitudes beyond 2^53, construct
directly from a `BigInt64Array`:

```js
const big = new WasmIntTensor(new BigInt64Array([9007199254740993n]), [1, 1, 1, 1]);
```

### Case B — Exact integer matmul

Counting co-occurrences, exact integer linear transforms, combinatorics:

```js
const spec = AgentLayerSpec.intMatmul(2);
registry.initAgentLayer(spec);
// [1,1,2,2] @ [1,1,2,2]
const A = WasmIntTensor.fromNumbers([1, 2, 3, 4], [1, 1, 2, 2]);
const B = WasmIntTensor.fromNumbers([5, 6, 7, 8], [1, 1, 2, 2]);
const C = registry.forwardIntBinary(2, A, B);
console.log(C.to_array()); // BigInt64Array [ 19n, 22n, 43n, 50n ]
```

Shape contract mirrors the f32 `matmul`: batch dims must match and the
contracted dims must agree (`a[..,m,k] @ b[..,k,n]`); violations are
`Err`, never a trap.

### Case C — Bridge between worlds

Feed integer results into an f32 pipeline, or pull f32 values into exact
integer land:

```js
registry.initAgentLayer(AgentLayerSpec.intToFloat(3));
registry.initAgentLayer(AgentLayerSpec.intFromFloat(4));

const ints = WasmIntTensor.fromNumbers([42, -7], [1, 2, 1, 1]);
const floats = registry.forwardIntCastToFloat(3, ints); // WasmTensor [42.0, -7.0]

// f32 -> i64 truncates toward zero: 42.9 -> 42, -7.9 -> -7
const back = WasmTensor.fromNumbers // (f32 tensor)
const ints2 = registry.forwardIntCastFromFloat(4, back);
```

**Cast semantics, honestly:**
- `intToFloat`: exact only for |x| ≤ 2^24. Larger magnitudes round —
  this is precisely the loss the integer world exists to avoid, so
  cast *out* only when you no longer need exactness.
- `intFromFloat`: truncates toward zero (Rust `as i64`). NaN → 0;
  out-of-range magnitudes saturate (Rust `as` semantics). It does not
  error — document your assumption if you rely on inputs being integral.

Calling a cast layer in the wrong direction is a clean `Err`
(`"IntCast layer is not a to_float cast"`), not a silent wrong answer.

### Case D — What NOT to do

- **Neural-network weights, activations, losses:** these need fractions
  and gradients. The integer world has no autodiff and no activations.
  Stay in f32.
- **Inside graphs:** `compileGraph` / multi-input plans only carry f32.
  An int layer id in a plan fails validation — compose integer steps
  imperatively through the registry instead.
- **Huge i64 near 2^63:** see overflow below.

## 4. BigInt handling (the #1 hang point)

`WasmIntTensor.to_array()` returns a **`BigInt64Array`**, not `number[]`.
Two things bite agents here:

1. `bigints[0] + 1` throws — you cannot mix `BigInt` and `Number`.
   Convert explicitly: `Number(bigints[0])` (only safe when you know
   the value fits; otherwise keep it a BigInt).
2. `JSON.stringify(bigints)` throws — BigInts are not JSON-serializable.
   Map to strings or numbers first: `Array.from(bigints, String)`.

`fromNumbers` exists so the *input* side stays ergonomic: pass plain
numbers, get a loud error if any is not a safe integer.

## 5. Overflow policy

Integer ops are **unconditionally checked**: add/sub/mul/matmul/div/rem
run on plain i64 data through Rust `checked_*` semantics, so overflow,
division by zero, and `MIN / -1` are all clean `Err` values — in debug
AND release builds alike. The engine has no wrapping mode and no flag
to enable one. (Burn's own NdArray int ops wrap in release builds, which
is why this layer never calls them for arithmetic.)

Two's-complement wraparound exists only as an explicit caller-side
emulation: `int.mjs --wrap` computes the wrapped result in exact BigInt
(per-op wrap for matmul) and never calls the engine for the wrapped op.
Division by zero still errors under `--wrap` — Rust `wrapping_div`
panics on a zero divisor, so there is no wrapped value for it; `MIN /
-1` under `--wrap` gives `MIN` (div) or `0` (rem).

Range-checking inputs before `intAdd`/`intMul` is no longer required
for correctness — the engine rejects overflow itself — but the JS tools
still pre-verify and report the exact overflowing result first, so the
error names the offending index and value.

## 6. Quick reference

```js
// 1. spec + register
registry.initAgentLayer(AgentLayerSpec.intAdd(1));
// 2. build i64 tensors (plain numbers OK if safe integers)
const a = WasmIntTensor.fromNumbers([16777217], [1, 1, 1, 1]);
const b = new WasmIntTensor(new BigInt64Array([1n]), [1, 1, 1, 1]);
// 3. run
const out = registry.forwardIntBinary(1, a, b);
// 4. read back (BigInt64Array!)
const [total] = out.to_array(); // 16777218n
```
