// Proof harness for the autodiff map: every rule proven TWICE.
//   (1) hand-computed exact case (catches semantic errors),
//   (2) central finite differences of the engine's forward pass (independent oracle).
// A rule passing only (1) could share the author's error; a rule passing
// only (2) could be vacuous. Both are required.
import { RULES, inferShapes, backwardWithSaved, numel } from '../lib/autodiff.mjs';

const FEAT = 'feature_axis1_singleton';
const SEQ = 'sequence_feature_axis2_singleton_width';
const ANY = 'any_rank4'; // accepts any rank-4 shape (for test tensors without a semantic layout)
const P = (slot, shape, layout) => ({ slot, role: 'observation', shape, layout });
const U = (layer, a, b) => ({ kind: 'unary', layer, slots: [a, b] });
const B = (layer, a, b, c) => ({ kind: 'binary', layer, slots: [a, b, c] });

// Build a minimal spec exercising one op. inPorts: [{shape, layout, values}].
// A dummy port is drained so the engine's 2-port minimum is satisfied.
function oneOpSpec(decl, inPorts, weights, outSlot) {
  const layers = [{ constructor: decl.name, args: decl.args }];
  const steps = [];
  const ports = [];
  const inputs = {};
  const inSlots = [];
  inPorts.forEach((p, i) => {
    ports.push(P(i, p.shape, p.layout));
    inputs[i] = p.values;
    inSlots.push(i);
  });
  const isBin = inPorts.length === 2;
  const oSlot = inPorts.length;
  steps.push(isBin ? B(0, inSlots[0], inSlots[1], oSlot) : U(0, inSlots[0], oSlot));
  // dummy drain
  const ds = inPorts.length + 1;
  ports.push(P(ds, [1, 1, 1, 1], FEAT));
  inputs[ds] = [0];
  layers.push({ constructor: 'linear', args: [1, 1, 1, false] });
  steps.push(U(1, ds, ds + 1));
  return {
    spec: {
      layers, steps, numSlots: ds + 2, outputSlot: outSlot ?? oSlot,
      ports, inputs, weights: { ...(weights ?? {}), 1: [1] },
    },
    outSlot: oSlot,
    inSlots,
  };
}

const approx = (a, b, tol) => Math.abs(a - b) <= tol;

// Each proof: {name, build() -> {spec, outSlot, inSlots, weightLayers},
//   hand: {dout, inGrads, paramGrads}, fdTol}
const PROOFS = [
  {
    name: 'linear', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'linear', args: [0, 2, 2, false] },
        [{ shape: [1, 2, 1, 1], layout: FEAT, values: [3, 4] }],
        { 0: [1, 0, 0, 1] });
      return { spec, outSlot, inSlots, dout: [1, 1] };
    },
    hand: { inGrads: [[[1, 1]]], paramGrads: { 0: { w: [3, 3, 4, 4], b: null } } },
  },
  {
    name: 'matmul', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'matmul', args: [0] },
        [
          { shape: [1, 1, 2, 2], layout: ANY, values: [1, 2, 3, 4] },
          { shape: [1, 1, 2, 2], layout: ANY, values: [1, 0, 0, 1] },
        ]);
      return { spec, outSlot, inSlots, dout: [1, 1, 1, 1] };
    },
    hand: { inGrads: [[[1, 1, 1, 1]], [[4, 4, 6, 6]]] },
  },
  {
    name: 'transpose', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'transpose', args: [0, 1, 2] },
        [{ shape: [1, 2, 2, 1], layout: SEQ, values: [1, 2, 3, 4] }]);
      return { spec, outSlot, inSlots, dout: [1, 2, 3, 4] };
    },
    hand: { inGrads: [[[1, 3, 2, 4]]] },
  },
  {
    name: 'add', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'add', args: [0] },
        [
          { shape: [1, 2, 1, 1], layout: FEAT, values: [1, 2] },
          { shape: [1, 2, 1, 1], layout: FEAT, values: [3, 4] },
        ]);
      return { spec, outSlot, inSlots, dout: [1, 1] };
    },
    hand: { inGrads: [[[1, 1]], [[1, 1]]] },
  },
  {
    name: 'sub', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'sub', args: [0] },
        [
          { shape: [1, 2, 1, 1], layout: FEAT, values: [1, 2] },
          { shape: [1, 2, 1, 1], layout: FEAT, values: [3, 4] },
        ]);
      return { spec, outSlot, inSlots, dout: [2, 3] };
    },
    hand: { inGrads: [[[2, 3]], [[-2, -3]]] },
  },
  {
    name: 'mul', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'mul', args: [0] },
        [
          { shape: [1, 2, 1, 1], layout: FEAT, values: [2, 3] },
          { shape: [1, 2, 1, 1], layout: FEAT, values: [4, 5] },
        ]);
      return { spec, outSlot, inSlots, dout: [1, 1] };
    },
    hand: { inGrads: [[[4, 5]], [[2, 3]]] },
  },
  {
    name: 'relu', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'relu', args: [0] },
        [{ shape: [1, 2, 1, 1], layout: FEAT, values: [-1, 2] }]);
      return { spec, outSlot, inSlots, dout: [1, 1] };
    },
    hand: { inGrads: [[[0, 1]]] },
  },
  {
    name: 'leakyRelu', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'leakyRelu', args: [0, 0.1] },
        [{ shape: [1, 2, 1, 1], layout: FEAT, values: [-2, 3] }]);
      return { spec, outSlot, inSlots, dout: [1, 1] };
    },
    hand: { inGrads: [[[0.1, 1]]] },
  },
  {
    name: 'sigmoid', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'sigmoid', args: [0] },
        [{ shape: [1, 1, 1, 1], layout: FEAT, values: [0] }]);
      return { spec, outSlot, inSlots, dout: [1] };
    },
    hand: { inGrads: [[[0.25]]] },
  },
  {
    name: 'tanh', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'tanh', args: [0] },
        [{ shape: [1, 1, 1, 1], layout: FEAT, values: [0] }]);
      return { spec, outSlot, inSlots, dout: [1] };
    },
    hand: { inGrads: [[[1]]] },
  },
  {
    name: 'gelu', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'gelu', args: [0] },
        [{ shape: [1, 1, 1, 1], layout: FEAT, values: [1] }]);
      return { spec, outSlot, inSlots, dout: [1] };
    },
    // Phi(1)+1*phi(1) = 0.8413447461+0.2419707245 = 1.0833154706
    hand: { inGrads: [[[1.08331547]]] },
  },
  {
    name: 'softplus', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'softplus', args: [0, 1.0] },
        [{ shape: [1, 1, 1, 1], layout: FEAT, values: [0] }]);
      return { spec, outSlot, inSlots, dout: [1] };
    },
    hand: { inGrads: [[[0.5]]] },
  },
  {
    name: 'softmax', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'softmax', args: [0, 3] },
        [{ shape: [1, 1, 1, 2], layout: ANY, values: [1, 2] }]);
      return { spec, outSlot, inSlots, dout: [1, 0] };
    },
    // s=[0.268941,0.731059]; dx=s*(G-sum(G*s)), sum=0.268941
    hand: { inGrads: [[[0.19661193, -0.19661193]]] },
  },
  {
    name: 'logSoftmax', fdTol: 1e-3,
    build() {
      // logSoftmax needs a valid layout: use [1,1,2,1] with dim=2
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'logSoftmax', args: [0, 2] },
        [{ shape: [1, 1, 2, 1], layout: ANY, values: [1, 2] }]);
      return { spec, outSlot, inSlots, dout: [1, 0] };
    },
    hand: { inGrads: [[[0.73105858, -0.73105858]]] },
  },
  {
    name: 'concat', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'concat', args: [0, 2] },
        [
          { shape: [1, 1, 2, 1], layout: ANY, values: [1, 2] },
          { shape: [1, 1, 1, 1], layout: ANY, values: [3] },
        ]);
      return { spec, outSlot, inSlots, dout: [1, 1, 1] };
    },
    hand: { inGrads: [[[1, 1]], [[1]]] },
  },
  {
    // y[c]=x[c+1] (rows move up, bottom zero-padded): L=y0+y2+y4=x2+x4.
    name: 'shiftUp', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'shiftUp', args: [0, 1] },
        [{ shape: [1, 1, 3, 2], layout: ANY, values: [1, 2, 3, 4, 5, 6] }]);
      return { spec, outSlot, inSlots, dout: [1, 0, 1, 0, 1, 0] };
    },
    hand: { inGrads: [[[0, 0, 1, 0, 1, 0]]] },
  },
  {
    // y=[0,0,x0,x1,x2,x3]: L=y2+y5=x0+x3.
    name: 'shiftDown', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'shiftDown', args: [0, 1] },
        [{ shape: [1, 1, 3, 2], layout: ANY, values: [1, 2, 3, 4, 5, 6] }]);
      return { spec, outSlot, inSlots, dout: [0, 0, 1, 0, 0, 1] };
    },
    hand: { inGrads: [[[1, 0, 0, 1, 0, 0]]] },
  },
  {
    // y=[x1,x2,0,x4,x5,0]: L=y0+y4=x1+x5.
    name: 'shiftLeft', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'shiftLeft', args: [0, 1] },
        [{ shape: [1, 1, 2, 3], layout: ANY, values: [1, 2, 3, 4, 5, 6] }]);
      return { spec, outSlot, inSlots, dout: [1, 0, 0, 0, 1, 0] };
    },
    hand: { inGrads: [[[0, 1, 0, 0, 0, 1]]] },
  },
  {
    // y=[0,x0,x1, 0,x3,x4]: L=y1+y3=x0+0 (y3 is zero-pad, not x3).
    name: 'shiftRight', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'shiftRight', args: [0, 1] },
        [{ shape: [1, 1, 2, 3], layout: ANY, values: [1, 2, 3, 4, 5, 6] }]);
      return { spec, outSlot, inSlots, dout: [0, 1, 0, 1, 0, 0] };
    },
    hand: { inGrads: [[[1, 0, 0, 0, 0, 0]]] },
  },
  {
    // Edge: s >= dim => forward is all zeros => gradient is all zeros.
    name: 'shiftUp-saturated', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'shiftUp', args: [0, 9] },
        [{ shape: [1, 1, 3, 2], layout: ANY, values: [1, 2, 3, 4, 5, 6] }]);
      return { spec, outSlot, inSlots, dout: [1, 1, 1, 1, 1, 1] };
    },
    hand: { inGrads: [[[0, 0, 0, 0, 0, 0]]] },
  },
  {
    // Added 2026-10-28: prove-rules previously reported 20/20 while
    // covering NEITHER pooling rule (PROOFS and RULES were never linked).
    // These two entries close that gap for real.
    name: 'avgPool2d', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        // args are [id, kh, kw, sh, sw, ph, pw]: a 2x2 kernel, stride 1.
        // (An earlier draft wrote [0,2,1,...] = a 2x1 kernel, whose
        // output has 2 elements — the 1-element dout then produced NaN
        // grads that JSON printed as nulls. The dout-length guard below
        // now catches that class of typo loudly.)
        { name: 'avgPool2d', args: [0, 2, 2, 1, 1, 0, 0] },
        [{ shape: [1, 1, 2, 2], layout: ANY, values: [1, 2, 3, 4] }]);
      // dout 1 (not 8): central FD noise scales as dout*f32_eps/(2*eps),
      // so dout 8 pushed maxErr to ~1.1e-3, just over the 1e-3 tol, while
      // the rule itself was right. dout 1 proves the same 1/4 divisor.
      return { spec, outSlot, inSlots, dout: [1] };
    },
    hand: { inGrads: [[[0.25, 0.25, 0.25, 0.25]]] },
  },
  {
    name: 'avgPool1d', fdTol: 1e-3,
    build() {
      const { spec, outSlot, inSlots } = oneOpSpec(
        { name: 'avgPool1d', args: [0, 2, 1] },
        [{ shape: [1, 1, 5, 1], layout: ANY, values: [1, 2, 3, 4, 5] }]);
      return { spec, outSlot, inSlots, dout: [1, 1, 1, 1] };
    },
    hand: { inGrads: [[[0.5, 1, 1, 1, 0.5]]] },
  },
];

export async function proveAll(runForward) {
  const results = [];
  for (const p of PROOFS) {
    const built = p.build();
    const { spec, outSlot, dout, inSlots } = built;
    // Guard the proof itself: a dout whose length mismatches the output
    // numel silently breeds NaN grads (seen 2026-10-28 as [4,null,4,null]
    // after JSON turned NaN into null). Fail loudly instead.
    try {
      const _shapes = inferShapes(spec);
      const _n = numel(_shapes.get(outSlot));
      if (dout.length !== _n) {
        results.push({ name: p.name, hand: false, finiteDiff: false, pass: false,
          detail: `PROOF BUG: dout length ${dout.length} != output numel ${_n}` });
        continue;
      }
    } catch (e) {
      results.push({ name: p.name, hand: false, finiteDiff: false, pass: false,
        detail: 'PROOF BUG: inferShapes threw: ' + String(e).slice(0, 120) });
      continue;
    }
    const backward = (sp, seed) => {
      const shapes = inferShapes(sp);
      const saved = new Map();
      const need = new Set();
      for (const step of sp.steps) {
        const rule = RULES[sp.layers[step.layer].constructor];
        const ins = step.slots.slice(0, step.kind === 'binary' ? 2 : 1);
        const os = step.slots[step.kind === 'binary' ? 2 : 1];
        ins.forEach((x) => need.add(x));
        for (const w of rule.saved) if (w === 'output') need.add(os);
      }
      for (const sl of need) saved.set(sl, { values: runForward({ ...sp, outputSlot: sl }, sl), shape: shapes.get(sl) });
      const weightOf = (li, args) => {
        const vals = sp.weights?.[String(li)];
        const [, inD, outD, bias] = args, nw = inD * outD;
        return { w: vals.slice(0, nw), b: bias ? vals.slice(nw, nw + outD) : null };
      };
      return backwardWithSaved(sp, shapes, saved, seed, weightOf);
    };
    const r = { name: p.name, hand: false, finiteDiff: false, detail: '' };
    try {
      // (1) hand-computed
      const b = backward(spec, { [outSlot]: dout });
      let ok = true, msg = [];
      p.hand.inGrads.forEach((exp, k) => {
        const got = b.inputGrads[String(inSlots[k])];
        const good = got && got.length === exp[0].length && got.every((v, i) => approx(v, exp[0][i], 1e-6));
        if (!good) { ok = false; msg.push(`inGrad[${k}] got=${JSON.stringify(got?.map((v) => +v.toFixed(6)))} exp=${JSON.stringify(exp[0])}`); }
      });
      if (p.hand.paramGrads) {
        for (const [li, pg] of Object.entries(p.hand.paramGrads)) {
          const got = b.paramGrads[li];
          if (pg.w && !got?.w.every((v, i) => approx(v, pg.w[i], 1e-6))) { ok = false; msg.push(`paramGrad w mismatch`); }
        }
      }
      r.hand = ok; r.detail = msg.join('; ');
      // (2) finite differences vs engine forward, same dout weighting
      const eps = 1e-3;
      const fwd = (inputs) => runForward({ ...spec, inputs, outputSlot: outSlot }, outSlot);
      let fdOk = true, maxErr = 0;
      for (let pi = 0; pi < inSlots.length; pi++) {
        const slot = inSlots[pi], base = [...spec.inputs[slot]];
        const analytic = b.inputGrads[String(slot)];
        for (let i = 0; i < base.length; i++) {
          const xp = { ...spec.inputs, [slot]: base.map((v, j) => (j === i ? v + eps : v)) };
          const xm = { ...spec.inputs, [slot]: base.map((v, j) => (j === i ? v - eps : v)) };
          const op = fwd(xp), om = fwd(xm);
          const nd = (op.reduce((a, v, j) => a + dout[j] * v, 0) - om.reduce((a, v, j) => a + dout[j] * v, 0)) / (2 * eps);
          const err = Math.abs(nd - analytic[i]);
          maxErr = Math.max(maxErr, err);
          if (err > p.fdTol) fdOk = false;
        }
      }
      r.finiteDiff = fdOk; r.maxErr = maxErr;
      r.pass = r.hand && r.finiteDiff;
    } catch (e) {
      r.pass = false; r.detail = 'EXCEPTION: ' + String(e).slice(0, 160);
    }
    results.push(r);
  }
  const passed = results.filter((r) => r.pass).length;
  return { ok: passed === results.length, passed, total: results.length, results };
}
