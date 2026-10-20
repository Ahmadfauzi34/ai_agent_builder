// Core static shape + dataflow check (no engine). Shared by bin/shapecheck.mjs
// (CLI) and bin/opt_compare.mjs (pre-pass gate).
import { ruleFor, arityOf, ShapeError } from './shapes.mjs';

export function checkSpec(spec) {
  const steps = [];
  const mismatches = [];
  const warnings = [];
  const dataflowIssues = [];

  if (!Number.isInteger(spec.numSlots) || spec.numSlots <= 0)
    dataflowIssues.push('numSlots must be a positive integer');
  if (!(spec.outputSlot >= 0 && spec.outputSlot < spec.numSlots))
    dataflowIssues.push(`outputSlot ${spec.outputSlot} out of range 0..${spec.numSlots - 1}`);

  const slotShapes = new Map();
  for (const p of spec.ports) {
    if (!(p.slot >= 0 && p.slot < spec.numSlots)) {
      dataflowIssues.push(`port slot ${p.slot} out of range`);
      continue;
    }
    slotShapes.set(p.slot, [...p.shape]);
  }

  spec.steps.forEach((st, i) => {
    const layerDef = spec.layers[st.layer];
    const rec = { index: i, layer: layerDef?.constructor ?? `!bad-layer-index:${st.layer}`, kind: st.kind, in_shapes: [], out_shape: null, rule: null };
    if (!layerDef) {
      mismatches.push({ step: i, message: `step references missing layer index ${st.layer}` });
      steps.push(rec);
      return;
    }
    const ctor = layerDef.constructor;
    const arity = arityOf(ctor);
    if (arity === null) {
      warnings.push(`step ${i}: unknown constructor '${ctor}' — arity and shape unchecked`);
    } else {
      const wantKind = arity === 1 ? 'unary' : 'binary';
      const wantSlots = arity === 1 ? 2 : 3;
      if (st.kind !== wantKind || st.slots.length !== wantSlots) {
        mismatches.push({
          step: i,
          message: `constructor '${ctor}' has arity ${arity} (needs ${wantKind} step with ${wantSlots} slots), got kind '${st.kind}' with ${st.slots.length} slots`,
        });
        steps.push(rec);
        return;
      }
    }

    const inSlots = st.kind === 'unary' ? [st.slots[0]] : [st.slots[0], st.slots[1]];
    const outSlot = st.slots[st.slots.length - 1];
    let blocked = false;
    for (const s of [...inSlots, outSlot]) {
      if (!(s >= 0 && s < spec.numSlots)) {
        dataflowIssues.push(`step ${i}: slot ${s} out of range 0..${spec.numSlots - 1}`);
        blocked = true;
      }
    }
    for (const s of inSlots) {
      if (s >= 0 && s < spec.numSlots && !slotShapes.has(s)) {
        dataflowIssues.push(`step ${i}: slot ${s} read before written (no port, no earlier step output)`);
        blocked = true;
      }
    }
    if (blocked) { steps.push(rec); return; }

    const inShapes = inSlots.map((s) => slotShapes.get(s));
    rec.in_shapes = inShapes.map((s) => [...s]);
    const rule = ruleFor(ctor);
    if (!rule) {
      warnings.push(`step ${i}: no reliable shape rule for '${ctor}' — output shape unchecked`);
      rec.rule = 'unchecked';
    } else {
      try {
        const outShape = rule(inShapes, layerDef.args);
        rec.out_shape = outShape;
        rec.rule = 'checked';
        slotShapes.set(outSlot, outShape);
      } catch (e) {
        if (e instanceof ShapeError) {
          mismatches.push({ step: i, layer: ctor, message: e.message, in_shapes: rec.in_shapes });
          rec.rule = 'mismatch';
        } else throw e;
      }
    }
    steps.push(rec);
  });

  if (!slotShapes.has(spec.outputSlot))
    dataflowIssues.push(`outputSlot ${spec.outputSlot} is never written`);

  const ok = mismatches.length === 0 && dataflowIssues.length === 0;
  return { ok, steps, mismatches, warnings, dataflow: { ok: dataflowIssues.length === 0, issues: dataflowIssues } };
}
