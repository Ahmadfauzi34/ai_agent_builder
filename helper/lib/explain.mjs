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

// Liveness analysis (pure, static — no session, no engine run).
// live_before(k), k = 0..n, is the set of slots that must be in memory
// before step k runs: def < k AND last_use >= k. def is -1 for ports
// (the buffer exists from the start, even if a later step rewrites the
// slot) and the producing step index otherwise. last_use is the last
// reading step, n (past the last step) for the output slot, and def
// itself for a slot never read — so a dead slot (written, never read,
// not the output) is live at no step boundary and contributes 0 to the
// peak: this models a runtime that frees dead outputs right after the
// producing step. Bytes come from the slot map (numel * 4, f32); a slot
// with unknown bytes that is ever live makes the peak a lower bound.
export function analyzeLiveness(spec, slots) {
  const bytesBySlot = new Map(slots.map((s) => [s.slot, s.bytes]));
  const steps = spec.steps ?? [];
  const n = steps.length;
  const def = new Map();
  const lastUse = new Map();
  for (const p of (spec.ports ?? [])) {
    if (!p || typeof p !== 'object' || !Number.isInteger(p.slot)) continue;
    def.set(p.slot, -1);
    lastUse.set(p.slot, -1);
  }
  steps.forEach((st, si) => {
    if (!st || !Array.isArray(st.slots)) return;
    const inSlots = st.kind === 'binary' ? st.slots.slice(0, 2) : st.slots.slice(0, 1);
    for (const s of inSlots) {
      if (!def.has(s)) def.set(s, -1);
      lastUse.set(s, Math.max(lastUse.get(s) ?? -1, si));
    }
    const outSlot = st.slots[st.slots.length - 1];
    if (!def.has(outSlot)) def.set(outSlot, si);
    if (!lastUse.has(outSlot)) lastUse.set(outSlot, si);
  });
  if (Number.isInteger(spec.outputSlot)) lastUse.set(spec.outputSlot, n);

  const slotRows = [...bytesBySlot.keys()].sort((a, b) => a - b).map((slot) => ({
    slot,
    def_step: def.has(slot) ? def.get(slot) : null,
    last_use_step: lastUse.has(slot) ? lastUse.get(slot) : null,
    bytes: bytesBySlot.get(slot),
  }));
  const liveBefore = [];
  let everLiveUnknown = false;
  for (let k = 0; k <= n; k++) {
    const live = slotRows.filter((r) => r.def_step !== null && r.def_step < k && r.last_use_step !== null && r.last_use_step >= k);
    const unknown = live.filter((r) => r.bytes === null);
    if (unknown.length) everLiveUnknown = true;
    liveBefore.push({
      step: k,
      bytes: unknown.length ? null : live.reduce((a, r) => a + (r.bytes ?? 0), 0),
      slots: live.map((r) => r.slot),
    });
  }
  const unknownBytesSlots = slotRows.filter((r) => r.bytes === null && liveBefore.some((lb) => lb.slots.includes(r.slot))).map((r) => r.slot);
  let peakStep = 0;
  let peakBytes = liveBefore.length ? (liveBefore[0].bytes ?? 0) : 0;
  for (let k = 1; k < liveBefore.length; k++) {
    const b = liveBefore[k].bytes ?? 0;
    if (b > peakBytes) { peakBytes = b; peakStep = k; }
  }
  return {
    peak_bytes: peakBytes,
    peak_step: peakStep,
    peak_slots: liveBefore[peakStep]?.slots ?? [],
    peak_is_lower_bound: everLiveUnknown,
    unknown_bytes_slots: unknownBytesSlots,
    live_before: liveBefore,
    slots: slotRows,
  };
}

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
    liveness: analyzeLiveness(spec, slots),
    output_slot: spec.outputSlot,
    unmapped_slots,
    notes: [
      'bytes assume f32 tensors (4 bytes/element), the plan element type',
      'tensor_bytes is the static footprint (sum over slots); liveness.peak_bytes is the peak live set across step boundaries, usually smaller — a runtime that frees dead outputs promptly needs peak_bytes, not the footprint',
      'shape null means no static shape rule for that step (the slot is still written)',
      'peak_bytes is a lower bound (liveness.peak_is_lower_bound) when a slot with unknown bytes is ever live',
    ],
  };
}
