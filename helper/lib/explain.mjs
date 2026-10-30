// burn-math explain: the slot-by-slot map behind `validate.mjs --explain`.
// Pure static analysis — no session, no engine run. Shapes come only
// from checkSpec (shapecore, the single shape authority); parameter
// counts only from expectedWeightCount; layouts only from the layout
// contracts doc. This module never invents a shape, a count, or a layout.
//
// Layout propagation (shared with validate.mjs, so the two can never
// drift): ports keep their declared layout; a step's output layout is
// its constructor's contract output, except 'preserve_input', which
// inherits the first input slot's layout. A constructor with no
// contract entry leaves its output layout null — null means "no layout
// known", which is distinct from the literal 'unknown' layout value.
// Propagation is forward-only (inputs are ports or earlier outputs,
// guaranteed by the dataflow check), so the final map is also the map
// at every step — validate.mjs's per-step compatibility warnings can
// read it directly.
import { expectedWeightCount } from './spec.mjs';

export function propagateLayouts(spec, layoutCtors) {
  const map = new Map();
  for (const p of (spec.ports ?? [])) {
    if (p && typeof p === 'object' && Number.isInteger(p.slot)) map.set(p.slot, p.layout ?? null);
  }
  for (const st of (spec.steps ?? [])) {
    if (!st || !Array.isArray(st.slots)) continue;
    const decl = spec.layers?.[st.layer];
    if (!decl) continue;
    const lc = layoutCtors && Object.hasOwn(layoutCtors, decl.constructor) ? layoutCtors[decl.constructor] : null;
    if (!lc) continue;
    const inSlots = st.kind === 'binary' ? st.slots.slice(0, 2) : st.slots.slice(0, 1);
    const outSlot = st.slots[st.slots.length - 1];
    // 'unknown' fallback mirrors validate.mjs's original behavior for a
    // preserve_input step whose input layout is missing; a constructor
    // with no contract entry leaves the slot unset (explain: null).
    if (lc.output === 'preserve_input') map.set(outSlot, map.get(inSlots[0]) ?? 'unknown');
    else map.set(outSlot, lc.output);
  }
  return map;
}

const numelOf = (shape) => (Array.isArray(shape) ? shape.reduce((a, b) => a * b, 1) : null);

export function explainSpec(spec, checked, layoutCtors) {
  const layouts = propagateLayouts(spec, layoutCtors);
  const steps = checked?.steps ?? [];

  // Slot -> {shape, producer} from ports, then step outputs.
  const slotInfo = new Map();
  for (const p of (spec.ports ?? [])) {
    if (!p || typeof p !== 'object' || !Number.isInteger(p.slot)) continue;
    slotInfo.set(p.slot, { shape: Array.isArray(p.shape) && p.shape.length === 4 ? [...p.shape] : null, producer: null });
  }
  for (const s of steps) {
    if (!s || typeof s !== 'object') continue;
    const st = spec.steps?.[s.index];
    if (!st || !Array.isArray(st.slots)) continue;
    const outSlot = st.slots[st.slots.length - 1];
    // checkSpec's step.layer is the constructor NAME; the layer INDEX is
    // spec.steps[s.index].layer. Keep both, never conflate them.
    slotInfo.set(outSlot, {
      shape: Array.isArray(s.out_shape) ? [...s.out_shape] : null,
      producer: { step: s.index, layer: st.layer, constructor: s.layer },
    });
  }

  const slots = [...slotInfo.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([slot, info]) => {
      const numel = numelOf(info.shape);
      return {
        slot,
        shape: info.shape,
        layout: layouts.has(slot) ? layouts.get(slot) : null,
        kind: slot === spec.outputSlot ? 'output' : (info.producer ? 'intermediate' : 'port'),
        producer: info.producer,
        numel,
        bytes: numel === null ? null : numel * 4,
      };
    });

  const layers = (spec.layers ?? []).map((decl, index) => {
    const used_by_steps = [];
    (spec.steps ?? []).forEach((st, si) => { if (st && st.layer === index) used_by_steps.push(si); });
    const w = (spec.weights ?? {})[String(index)];
    return {
      index,
      constructor: decl?.constructor ?? null,
      params: decl ? expectedWeightCount(decl) : null,
      weights_declared: Array.isArray(w) ? w.length : null,
      used_by_steps,
    };
  });

  const stepsOut = steps.map((s) => {
    const st = spec.steps?.[s.index];
    const inSlots = st ? (st.kind === 'binary' ? st.slots.slice(0, 2) : st.slots.slice(0, 1)) : [];
    const outSlot = st && Array.isArray(st.slots) ? st.slots[st.slots.length - 1] : null;
    return {
      index: s.index,
      layer: st ? st.layer : null,
      constructor: s.layer,
      kind: s.kind,
      input_slots: inSlots,
      output_slot: outSlot,
      in_shapes: s.in_shapes ?? [],
      out_shape: s.out_shape ?? null,
      rule: s.rule ?? null,
    };
  });

  const mapped = new Set(slots.map((s) => s.slot));
  const unmapped_slots = [];
  for (let s = 0; s < (spec.numSlots ?? 0); s++) if (!mapped.has(s)) unmapped_slots.push(s);

  return {
    ok: true,
    valid: true,
    mode: 'explain',
    slots,
    layers,
    steps: stepsOut,
    totals: {
      params: layers.reduce((a, l) => a + (l.params ?? 0), 0),
      slots: slots.length,
      tensor_bytes: slots.reduce((a, s) => a + (s.bytes ?? 0), 0),
    },
    output_slot: spec.outputSlot,
    unmapped_slots,
    notes: [
      'bytes assume f32 tensors (4 bytes/element), the plan element type',
      'tensor_bytes is the static footprint (sum over slots), not peak live memory; liveness analysis is a separate follow-up',
      'shape null means no static shape rule for that step (the slot is still written)',
    ],
  };
}
