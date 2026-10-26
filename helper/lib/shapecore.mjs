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

  // Written-ness and shape-known-ness are SEPARATE. slotShapes used to
  // serve both jobs, so a step whose constructor has no shape rule
  // (ruleFor === null) never marked its outSlot written — every later
  // reader was falsely accused of read-before-written and valid specs
  // (matmul, transpose, ...) failed shapecheck while the engine ran
  // them fine. writtenSlots tracks "a value exists here" (ports + every
  // step that runs, rule or not); slotShapes tracks only shapes we
  // actually know. Malformed ports/steps are reported, never thrown.
  const slotShapes = new Map();
  const writtenSlots = new Set();
  const seenPortSlots = new Set();
  for (const p of spec.ports) {
    if (!p || typeof p !== 'object') {
      dataflowIssues.push(`port entry is null or not an object`);
      continue;
    }
    if (seenPortSlots.has(p.slot)) dataflowIssues.push(`duplicate port slot ${p.slot}`);
    seenPortSlots.add(p.slot);
    if (!(p.slot >= 0 && p.slot < spec.numSlots)) {
      dataflowIssues.push(`port slot ${p.slot} out of range`);
      continue;
    }
    // A declared in-range port counts as written even when its shape is
    // malformed — the shape problem is reported separately, it must not
    // cascade into false read-before-written issues downstream.
    writtenSlots.add(p.slot);
    if (!Array.isArray(p.shape)) {
      dataflowIssues.push(`port slot ${p.slot} has no shape array`);
      continue;
    }
    if (p.shape.length !== 4) {
      dataflowIssues.push(`port slot ${p.slot}: port.shape must contain four dimensions, got ${JSON.stringify(p.shape)}`);
      continue;
    }
    slotShapes.set(p.slot, [...p.shape]);
  }

  spec.steps.forEach((st, i) => {
    if (!st || typeof st !== 'object') {
      mismatches.push({ step: i, message: `step ${i} is null or not an object` });
      steps.push({ index: i, layer: `!bad-step:${i}`, kind: null, in_shapes: [], out_shape: null, rule: null });
      return;
    }
    const layerDef = spec.layers[st.layer];
    const rec = { index: i, layer: layerDef?.constructor ?? `!bad-layer-index:${st.layer}`, kind: st.kind, in_shapes: [], out_shape: null, rule: null };
    if (!layerDef) {
      mismatches.push({ step: i, message: `step references missing layer index ${st.layer}` });
      steps.push(rec);
      return;
    }
    if (!Array.isArray(st.slots)) {
      mismatches.push({ step: i, message: `step ${i} has no slots array` });
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
      if (s >= 0 && s < spec.numSlots && !writtenSlots.has(s)) {
        dataflowIssues.push(`step ${i}: slot ${s} read before written (no port, no earlier step output)`);
        blocked = true;
      }
    }
    if (blocked) { steps.push(rec); return; }

    const inShapes = inSlots.map((s) => slotShapes.get(s));
    rec.in_shapes = inShapes.map((s) => (s ? [...s] : null));
    const rule = ruleFor(ctor);
    // A step that runs writes its outSlot even when its output shape is
    // unknown (no rule, unknown input shape, or a rule mismatch — the
    // mismatch itself already fails the check, it must not also cascade
    // into false read-before-written issues downstream).
    if (outSlot >= 0 && outSlot < spec.numSlots) writtenSlots.add(outSlot);
    if (!rule) {
      warnings.push(`step ${i}: no reliable shape rule for '${ctor}' — output shape unchecked`);
      rec.rule = 'unchecked';
    } else if (inShapes.some((s) => s === undefined)) {
      warnings.push(`step ${i}: input shape unknown — output shape unchecked`);
      rec.rule = 'unchecked';
    } else {
      try {
        const outShape = rule(inShapes, layerDef.args);
        rec.out_shape = outShape;
        rec.rule = 'checked';
        slotShapes.set(outSlot, outShape);
      } catch (e) {
        // A static checker must not throw on malformed input. Shape
        // errors are mismatches; anything else (e.g. a rule tripping
        // over missing args) is reported as a mismatch too, never a
        // crash — the engine remains the final judge at run time.
        mismatches.push({ step: i, layer: ctor, message: e instanceof Error ? e.message : String(e), in_shapes: rec.in_shapes });
        rec.rule = 'mismatch';
      }
    }
    steps.push(rec);
  });

  if (!writtenSlots.has(spec.outputSlot))
    dataflowIssues.push(`outputSlot ${spec.outputSlot} is never written`);

  const ok = mismatches.length === 0 && dataflowIssues.length === 0;
  return { ok, steps, mismatches, warnings, dataflow: { ok: dataflowIssues.length === 0, issues: dataflowIssues } };
}
