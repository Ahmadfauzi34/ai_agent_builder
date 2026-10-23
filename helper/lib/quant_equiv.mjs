// burn-math #13 Fase 2: equivalence harness for the engine's REAL quantized
// state path (getQuantizedState -> loadState -> forward).
//
// This is deliberately NOT a simulation: it exercises the engine's own
// int8 quantization (burn 0.20.1, storage-level: live stays f32), the BQZ1
// envelope, and load_state's auto-detection. Compare with
// bin/quant_verify.mjs, which simulates int8 host-side in JS.
//
// Verdicts:
//   ACCEPT   - every case's maxAbsDiff <= tolerance
//   REJECT   - some case exceeds tolerance (with per-case detail)
//   MEASURED - no tolerance supplied: numbers reported, no judgment
//
// The harness is layer-agnostic: makeLayer() is a factory returning a fresh
// layer instance (any of the 7 stateful types). Weights are cloned via
// get_state/load_state, so the factory's init randomness does not matter -
// only its STRUCTURE must be deterministic (same dims, same bias flag).
//
// Tolerance is the caller's explicit judgment: a number, or a function
// (report, inputCase) => number. linearQuantTolerance(report) derives the
// conservative analytical bound for linear layers:
//   |Δy_i| <= bound_W * sum|x| + bound_b
// where bound_W/bound_b are the per-param analytical bounds from the
// engine's own report (max over blocks for q8-per-block: conservative,
// documented, never underestimated).
// Find the weight-like (highest-rank) and bias-like (rank-1 or scalar)
// params in a quant report. Structural, not name-based.
function splitWeightBias(report) {
  const params = [...(report.params ?? [])].sort((a, b) => b.shape.length - a.shape.length);
  const weight = params[0];
  const bias = params.find((p) => p !== weight && p.shape.length <= 1);
  return { weight, bias };
}

// Conservative analytical tolerance for a LINEAR layer:
// |Δy_i| <= bound_W * Σ|x_j| + bound_b  (bound = max over blocks).
// Returns a function (inputValues: number[]) => number.
export function linearQuantTolerance(report) {
  const { weight, bias } = splitWeightBias(report);
  if (!weight) throw new Error('linearQuantTolerance: report has no params');
  const boundW = weight.bound;
  const boundB = bias ? bias.bound : 0;
  return (inputValues) => {
    let sumAbs = 0;
    for (const v of inputValues) sumAbs += Math.abs(v);
    return boundW * sumAbs + boundB;
  };
}

// One-line human summary of a quant report.
export function summarizeQuantReport(report) {
  const ps = (report.params ?? [])
    .map((p) => `[${p.shape.join('x')}] bound=${p.bound}`)
    .join(' ');
  return `${report.scheme}: ${ps} bytes ${report.bytes_f32}->${report.bytes_q8} (ratio ${report.ratio})`;
}

// The harness.
//
// rt         - wasm runtime from loadWasm()
// makeLayer  - () => fresh layer instance, deterministic STRUCTURE
// scheme     - "q8-per-tensor" | "q8-per-block"
// inputCases - [{ values: number[], shape: number[] }]
// opts       - { tolerance?: number | ((report, inputCase) => number) }
//
// Unknown scheme strings are a loud engine Err (not a silent default).
export async function quantizeLayerEquivalence(rt, makeLayer, scheme, inputCases, opts = {}) {
  if (!Array.isArray(inputCases) || inputCases.length === 0)
    throw new Error('quantizeLayerEquivalence: inputCases must be a non-empty array');
  const tolOpt = opts.tolerance;

  // Reference (f32) and bit-identical f32 clone, then the quantized twin.
  const ref = makeLayer();
  const refBytes = ref.get_state();
  const cloneCheck = makeLayer();
  try {
    cloneCheck.load_state(refBytes); // f32 path must keep working (regression)
    // Prove the f32 round-trip is bit-identical, not just non-throwing.
    {
      const c0 = inputCases[0];
      const t = new rt.WasmTensor(Float32Array.from(c0.values), Uint32Array.from(c0.shape));
      try {
        const oa = ref.forward(t);
        const ob = cloneCheck.forward(t);
        try {
          const ya = oa.to_array();
          const yb = ob.to_array();
          for (let i = 0; i < ya.length; i++) {
            if (ya[i] !== yb[i])
              throw new Error(`f32 load_state round-trip not bit-identical at output ${i}`);
          }
        } finally {
          oa.free();
          ob.free();
        }
      } finally {
        t.free();
      }
    }
    const report = JSON.parse(ref.quantize_state_report(scheme));
    const qbytes = ref.get_quantized_state(scheme);
    const twin = makeLayer();
    try {
      twin.load_state(qbytes); // BQZ1 envelope + auto-detect path

      const tolFn =
        typeof tolOpt === 'function' ? tolOpt :
        typeof tolOpt === 'number' ? () => tolOpt :
        null;

      const cases = [];
      let maxAbsDiff = 0;
      for (const c of inputCases) {
        const t = new rt.WasmTensor(Float32Array.from(c.values), Uint32Array.from(c.shape));
        let ya, yb;
        try {
          const oa = ref.forward(t);
          const ob = twin.forward(t);
          try {
            ya = oa.to_array();
            yb = ob.to_array();
          } finally {
            oa.free();
            ob.free();
          }
        } finally {
          t.free();
        }
        let d = 0;
        if (ya.length !== yb.length)
          throw new Error(`output length mismatch: ${ya.length} vs ${yb.length}`);
        for (let i = 0; i < ya.length; i++) {
          const ad = Math.abs(ya[i] - yb[i]);
          if (ad > d) d = ad;
        }
        if (d > maxAbsDiff) maxAbsDiff = d;
        const tolerance = tolFn ? tolFn(report, c) : null;
        cases.push({
          maxAbsDiff: d,
          tolerance,
          within: tolerance === null ? null : d <= tolerance,
          outputLen: ya.length,
        });
      }

      const bytesF32 = report.bytes_f32;
      const bytesQ8 = report.bytes_q8;
      // Vacuous test detection: all-zero weights quantize exactly (bound 0),
      // so ACCEPT would prove nothing about the quantization path.
      const vacuous = (report.params ?? []).every((p) => p.bound === 0);
      let verdict, reason;
      if (tolFn === null) {
        verdict = 'MEASURED';
        reason = `no tolerance supplied: maxAbsDiff ${maxAbsDiff} reported without judgment` +
          (vacuous ? ' (vacuous: all-zero weights)' : '');
      } else {
        const bad = cases.findIndex((c) => !c.within);
        if (bad === -1) {
          verdict = 'ACCEPT';
          reason = `all ${cases.length} case(s) within tolerance (maxAbsDiff ${maxAbsDiff})` +
            (vacuous ? ' (vacuous: all-zero weights)' : '');
        } else {
          verdict = 'REJECT';
          reason = `case ${bad}: maxAbsDiff ${cases[bad].maxAbsDiff} > tolerance ${cases[bad].tolerance}`;
        }
      }
      return {
        verdict,
        reason,
        vacuous,
        scheme: report.scheme,
        report,
        summary: summarizeQuantReport(report),
        bytesF32,
        bytesQ8,
        ratio: report.ratio,
        maxAbsDiff,
        perCase: cases,
      };
    } finally {
      twin.free();
    }
  } finally {
    cloneCheck.free();
    ref.free();
  }
}
