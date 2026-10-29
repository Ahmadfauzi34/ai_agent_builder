# Optimasi & fitting domain pack (fit)

Verifier pack for optimization and fitting work. The stance is the same as
gradcheck's: these tools MEASURE and REPORT — they do not optimize, and
they do not judge beyond their hard checks. The caller decides.

## Why "fit" and not "opt"

`opt_*` in this helper is already taken by low-level WASM optimization
(wasm-opt, profiling, the memory oracle — see `lib/opt.mjs`). Naming this
pack `opt` would collide. `fit` = fitting a model to data, verifying an
optimizer's claims.

## No engine change

All tools here are forward-run-based (finite differences over spec runs).
The engine exposes no graph backward pass and no gradient-based optimizer
(only the gradient-free `EsOptimizer`, used by `es_optimize`), and this
pack does not need either. Core WASM is enough.

## The tools

Shared pure helpers live in `lib/fit.mjs` (no WASM imports, byte-identical
in repo and skill). Bin adapters are thin: repo imports `lib/engine.mjs`,
skill imports `lib/wasm_direct.mjs` — the only difference.

### Existing (reused, not rewritten)

- `bin/gradcheck.mjs` — numeric Jacobian by central differences, optionally
  verified against an agent-supplied analytic Jacobian (exit 0/2).
- `bin/es_optimize.mjs` — gradient-free evolution-strategy optimization
  via the engine's `EsOptimizer` (ask/tell over generations).

### New in this pack

- `bin/fit_hessian_diag.mjs <spec.json> [slot=0] [h=1e-2] [tolAbs=0.05] [tolRel=0.1]`
  Hessian DIAGONAL only: `H_ii^(o) = (f_o(x+h e_i) - 2 f_o(x) + f_o(x-h e_i)) / h^2`,
  measured at h and h/2. Cross terms are not measured. Cost, known before
  running: 4n + 1 forward runs for n inputs. Each entry carries a trust
  status: `trusted` (the two step sizes agree), `zero_within_noise` (both
  estimates negligible — the honest answer for linear and piecewise-linear
  specs, whose true diagonal is 0), or `untrusted` (disagreement — do not
  use the value). Exit 0 when the measurement succeeded; untrusted entries
  are findings, not failures. Exit 2 on bad input or a crashed run.
  Defaults calibrated on the fit battery (f32 second differences carry
  ~1e-2 noise at h=1e-2; the battery asserts quad2 diagonals within 0.15
  of the true values 2 and 0).

- `bin/fit_sensitivity.mjs <spec.json> [topK=10] [threshold=0]`
  Ranked sensitivity REPORT on top of gradcheck. It runs the sibling
  `gradcheck.mjs` (same directory, same Node) and ranks its Jacobian:
  per-output top-K, a global top-K, threshold flags, and a summary (max /
  mean |J|, count above threshold). The Jacobian has exactly one producer;
  this tool only ranks it. Exit 0 when the report was produced, 2 when
  gradcheck failed. A low-sensitivity spec is a finding, not an error.

- `bin/fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3]`
  Verifies a CLAIMED gradient-descent trajectory
  (`{"slot"?: "0", "steps": [{"weights", "loss", "grad"?, "lr"?}]}`).
  The weights fill one input slot (default "0"). Loss scalarization rule
  (single rule, documented): a single-element output IS the loss; a vector
  output is the sum of squares. If your GD optimizes a different loss,
  express it as a spec whose output IS that loss. Checks:
  1. Recomputation (HARD): recomputed loss must match claimed loss within
     tol (abs and rel). A mismatch exits 2 — the trajectory does not
     describe this spec.
  2. Descent (reported): recomputed loss vs the previous step. An ascent
     is counted and listed but does not change the exit code — GD with too
     large a step legitimately ascends.
  3. First-order prediction (flagged): when the previous step's grad and lr
     are supplied, actual decrease is compared to `lr * |grad|^2` of the
     step that was taken (departure gradient — the step from s-1 to s was
     computed from the gradient AT s-1). Ratio outside [0, 2] is flagged
     `prediction_off` (heuristic, not a verdict). For a quadratic loss the
     ratio is exactly `1 - lr` — a useful sanity check (battery: 0.9 at
     lr=0.1).
  4. Gradient norm trend: first vs last supplied norm.
  Cost: one forward run per step.

## Worked values (fit battery, all verified)

- Spec `x*x + y` at x=[2,3], y=[10,20]: outputs [14,29]; Jacobian diag [4,6] (slot 0); Hessian
  diagonal [[2.01, 0], [0, 1.98]] — both 2-entries `trusted`, both 0-entries
  `zero_within_noise`, zero `untrusted`.
- Pure linear spec: Hessian diagonal exactly 0, all `zero_within_noise`.
- GD on f(x)=x^2+5 from x=2, lr=0.1: losses 9 → 7.56 → 6.6384 recomputed
  exactly; prediction ratios 0.9, 0.9 (= 1 - lr); no flags.
- Same trajectory with one tampered loss: exit 2, mismatch counted.
- Same trajectory with lr=1.5 (overshoot to x=-4, loss 21): exit 0,
  ascent counted (1), prediction ratio -0.5 flagged — the ascent is
  reported, not hidden, and not a tool failure.

## Known behavior, not bugs

- ReLU at the kink: a second difference across a kink of a piecewise-linear
  function is NOT zero — it scales like 1/h (e.g. relu at exactly 0 gives
  ~1/h). That is the measurement correctly reporting "not twice
  differentiable here", not a tool error. The audit fixture specA hits this
  (its linear layer outputs exactly 0 into a relu), which is why the fit
  battery uses a kink-free linear fixture instead.
- Discrete index inputs (e.g. embedding ids) are not differentiable; all
  three tools are meaningful only for continuous-valued inputs.
- `fit_sensitivity` inherits gradcheck's subgradient-average behavior at
  relu kinks.
