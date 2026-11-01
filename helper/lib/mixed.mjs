// Mixed-dtype orchestration (Option B, design-mixed-dtype.md §3).
import { WEIGHTED_CONSTRUCTORS, LAYER_TYPE } from './wasm_direct.mjs';
//
// A spec with int steps runs as a sequence of SEGMENTS against ONE
// shared LayerRegistry: f32 segments compile to their own graph
// (direct primitives, not ingress sessions), int segments execute as
// direct registry calls. Values cross segments as Float32Array /
// BigInt64Array — never JSON numbers for i64.
//
// Invariants (all fail-closed, all checked before any segment runs
// unless noted as value-dependent):
// - Static dtype propagation (§3.1): ports are f32; int binary family
//   -> i64; intToFloat -> f32; intFromFloat -> i64; everything else
//   -> f32. A step fed by a wrong-dtype slot is a BUILD error naming
//   step, slot, expected, actual, and the missing cast.
// - Single-write per slot (v1 rule): a slot written twice is a BUILD
//   error. (Specs are authored this way; the rule keeps the producer
//   map total so fan-out analysis stays exact.)
// - Output slot must be f32 in v1 (§6.4, fork F1): an i64 output slot
//   is a BUILD error ("end with intToFloat or use int.mjs").
// - Boundary exactness (§6.1, value-dependent, checked at the cast
//   step before the engine cast runs): intToFloat rejects any element
//   |x| > 2^24; intFromFloat rejects non-finite and out-of-i64-range
//   inputs. In-range fractional truncation stays allowed (defined
//   semantics). No allow-flag in v1.
// - Weights are registry-keyed, not graph-keyed: all f32 weights are
//   set once on the shared registry (same validation as buildGraph),
//   so they are visible to every segment graph of their layer. The
//   partition rule (§3.1) is therefore satisfied by construction —
//   a weight can never land on a segment that does not own its layer,
//   because there is exactly one registry. partitionWeights() still
//   exists: it validates keys and groups them per segment for the
//   receipt/battery assertions.
// - Init-once: all layers are init'd once up front and the SAME spec
//   objects are reused across segment builders. The engine silently
//   accepts re-init (even with a different constructor, probed
//   2026-11-01), so re-init per segment would be a silent-collision
//   hazard; init-once removes it structurally. checkArgValues
//   already rejects duplicate registry ids across the whole spec.
// - Segment plans bypass buildGraph's host minimum-ports rule: the
//   whole-spec plan check is the gate; segment plans are internal
//   artifacts, and the engine accepts 1-port plans directly (probed
//   2026-11-01). Synthetic segment ports use role 'x-segment-input'
//   and layout 'any_rank4' (accepted by the engine, probed).
// - Multi-output segments compile ONE graph per output slot (same
//   steps, different setOutput). Values are deterministic, so
//   recompute is bit-identical. A segment with no downstream output
//   still runs (effective output = its last step's output slot,
//   result discarded) so a throwing dead segment stays loud.
// - Materialization (§3.2): each consuming segment gets its own
//   tensor object (a copy of the values) — never a tensor shared
//   across segments. The receipt lists inputs per segment, so the
//   per-consumer materializations are explicit by construction.

export const INT_BINARY_CONSTRUCTORS = new Set([
  'intAdd', 'intSub', 'intMul', 'intMatmul', 'intDiv', 'intRem',
]);
export const INT_CAST_CONSTRUCTORS = new Set(['intToFloat', 'intFromFloat']);
export const INT_CONSTRUCTORS = new Set([
  ...INT_BINARY_CONSTRUCTORS, ...INT_CAST_CONSTRUCTORS,
]);

export const F32_EXACT_INT_MAX = 16777216n; // 2^24
export const I64_MIN = -(2n ** 63n);
export const I64_MAX = 2n ** 63n - 1n;

// Synthetic port vocabulary for prior-run outputs feeding a segment.
export const SEGMENT_INPUT_ROLE = 'x-segment-input';
export const SEGMENT_INPUT_LAYOUT = 'any_rank4';

export class MixedBuildError extends Error {
  constructor(message) {
    super(message);
    this.name = 'MixedBuildError';
    this.code = 'mixed_build';
  }
}

function buildError(message) {
  return new MixedBuildError(message);
}

// True when any STEP uses an int constructor (declared-but-unused int
// layers do not count: today's direct path runs them fine, and B must
// not newly refuse specs that already work).
export function specUsesInt(spec) {
  for (const st of (spec.steps ?? [])) {
    const decl = spec.layers?.[st.layer];
    if (decl && INT_CONSTRUCTORS.has(decl.constructor)) return true;
  }
  return false;
}

export function stepKind(spec, stepIndex) {
  const decl = spec.layers[spec.steps[stepIndex].layer];
  return decl && INT_CONSTRUCTORS.has(decl.constructor) ? 'int' : 'f32';
}

function stepInputSlots(step) {
  return step.kind === 'binary' ? step.slots.slice(0, 2) : step.slots.slice(0, 1);
}

function stepOutputSlot(step) {
  return step.kind === 'binary' ? step.slots[2] : step.slots[1];
}

function expectedInputDtype(ctor) {
  if (INT_BINARY_CONSTRUCTORS.has(ctor)) return 'i64';
  if (ctor === 'intToFloat') return 'i64';
  if (ctor === 'intFromFloat') return 'f32';
  return 'f32';
}

function outputDtype(ctor) {
  if (INT_BINARY_CONSTRUCTORS.has(ctor)) return 'i64';
  if (ctor === 'intToFloat') return 'f32';
  if (ctor === 'intFromFloat') return 'i64';
  return 'f32';
}

// Forward dtype pass over steps in order. Returns { slotDtype: Map }.
// Throws MixedBuildError on the first mismatch / double-write /
// unknown slot. Pure analysis: no engine, no values.
export function propagateDtype(spec) {
  const slotDtype = new Map();
  for (const p of (spec.ports ?? [])) slotDtype.set(p.slot, 'f32');
  const written = new Set();
  for (let si = 0; si < (spec.steps ?? []).length; si++) {
    const st = spec.steps[si];
    const decl = spec.layers?.[st.layer];
    if (!decl) throw buildError(`step ${si}: references missing layer index ${st.layer}`);
    const ctor = decl.constructor;
    const expected = expectedInputDtype(ctor);
    for (const s of stepInputSlots(st)) {
      const actual = slotDtype.get(s);
      if (actual === undefined) {
        throw buildError(`step ${si} (${ctor}): slot ${s} has no dtype (not a port and not produced by an earlier step)`);
      }
      if (actual !== expected) {
        const missing = expected === 'i64' ? 'intFromFloat' : 'intToFloat';
        throw buildError(`step ${si} (${ctor}): slot ${s} holds ${actual} but ${ctor} expects ${expected} — insert ${missing}`);
      }
    }
    const out = stepOutputSlot(st);
    if (written.has(out)) {
      throw buildError(`step ${si} (${ctor}): slot ${out} is written twice (single-write per slot in v1)`);
    }
    written.add(out);
    slotDtype.set(out, outputDtype(ctor));
  }
  return { slotDtype };
}

// Maximal-run cut (§3.2): consecutive steps of the same kind merge.
// Each segment gains inputs/outputs in ORIGINAL slot ids:
// - inputs: slots consumed from outside the segment (original ports
//   or prior-segment outputs), in first-use order.
// - outputs: slots produced by this segment and consumed by a later
//   segment, plus the spec output slot when this segment produces it.
// Cast steps are listed for the receipt (elements filled at run time).
export function splitSpec(spec) {
  const { slotDtype } = propagateDtype(spec);
  const segments = [];
  for (let si = 0; si < (spec.steps ?? []).length; si++) {
    const kind = stepKind(spec, si);
    const last = segments[segments.length - 1];
    if (last && last.kind === kind) {
      last.steps[1] = si;
      last.stepIndices.push(si);
      last.layers.push(spec.steps[si].layer);
    } else {
      segments.push({
        index: segments.length, kind, steps: [si, si],
        stepIndices: [si], layers: [spec.steps[si].layer],
      });
    }
  }
  // Producer map: original slot -> segment index (single-write rule
  // makes this total over written slots).
  const producer = new Map();
  for (const seg of segments) {
    for (const si of seg.stepIndices) producer.set(stepOutputSlot(spec.steps[si]), seg.index);
  }
  // Consumer map: original slot -> set of consuming segment indices.
  const consumers = new Map();
  for (const seg of segments) {
    for (const si of seg.stepIndices) {
      for (const s of stepInputSlots(spec.steps[si])) {
        if (!consumers.has(s)) consumers.set(s, new Set());
        consumers.get(s).add(seg.index);
      }
    }
  }
  for (const seg of segments) {
    const inputs = [];
    const seenInputs = new Set();
    for (const si of seg.stepIndices) {
      for (const s of stepInputSlots(spec.steps[si])) {
        if (producer.get(s) === seg.index) continue; // internal use
        if (!seenInputs.has(s)) { seenInputs.add(s); inputs.push(s); }
      }
    }
    seg.inputs = inputs;
    const outputs = [];
    for (const si of seg.stepIndices) {
      const out = stepOutputSlot(spec.steps[si]);
      const laterConsumers = [...(consumers.get(out) ?? [])].some((ci) => ci > seg.index);
      if (laterConsumers || out === spec.outputSlot) outputs.push(out);
    }
    seg.outputs = outputs;
    seg.casts = seg.stepIndices
      .filter((si) => INT_CAST_CONSTRUCTORS.has(spec.layers[spec.steps[si].layer].constructor))
      .map((si) => ({
        step: si,
        direction: spec.layers[spec.steps[si].layer].constructor === 'intToFloat' ? 'i64->f32' : 'f32->i64',
      }));
  }
  return { segments, slotDtype, producer };
}

// Weights grouped by owning segment (validation + structure). Keys
// naming no layer at all are BUILD errors; keys for layers with no
// steps are grouped under segment -1 (they are still set on the
// shared registry exactly like buildGraph sets them — the grouping
// is descriptive, the registry is authoritative).
export function partitionWeights(spec, segments) {
  const layerSegment = new Map(); // layer index -> segment index
  for (const seg of segments) {
    for (const li of new Set(seg.layers)) layerSegment.set(li, seg.index);
  }
  const grouped = new Map(); // segment index -> { layerIndex: values }
  for (const [indexStr, values] of Object.entries(spec.weights ?? {})) {
    const li = Number(indexStr);
    if (!spec.layers[li]) {
      throw buildError(`weights key '${indexStr}': layer index ${li} does not exist (0..${spec.layers.length - 1})`);
    }
    const segIdx = layerSegment.has(li) ? layerSegment.get(li) : -1;
    if (!grouped.has(segIdx)) grouped.set(segIdx, {});
    grouped.get(segIdx)[indexStr] = values;
  }
  return grouped;
}

// Human-readable segment plan for validate's replacement warning
// (§6.3): number of segments, kinds, boundary steps, policy note.
export function segmentPlanWarning(spec, segments) {
  const kinds = segments.map((s) => s.kind).join(',');
  const casts = segments.flatMap((s) => s.casts.map((c) => c.step));
  const hasF32 = segments.some((s) => s.kind === 'f32');
  const hasInt = segments.some((s) => s.kind === 'int');
  if (hasInt && !hasF32) {
    return `int-only spec: single int run; output slot dtype is determined by the final cast (run.mjs cannot emit i64 outputs — end with intToFloat or use int.mjs)`;
  }
  return `mixed-dtype spec: ${segments.length} segments (${kinds}), boundary casts at steps [${casts.join(',')}]; int steps execute as registry calls (not in f32 graphs); boundaries are fail-closed (intToFloat rejects |x| > 2^24; intFromFloat rejects non-finite/out-of-i64-range inputs)`;
}

// ---- Execution (direct primitives, shared registry) ----

function freeQuiet(obj) {
  if (obj && typeof obj.free === 'function') {
    try { obj.free(); } catch { /* already freed: teardown never fails */ }
  }
}

function checkIntToFloatBoundary(stepIndex, values) {
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < -F32_EXACT_INT_MAX || v > F32_EXACT_INT_MAX) {
      throw new Error(`step ${stepIndex} (intToFloat): precision loss at index ${i}: ${v} exceeds exact f32 integer range [-16777216, 16777216]`);
    }
  }
}

function checkIntFromFloatBoundary(stepIndex, values) {
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      throw new Error(`step ${stepIndex} (intFromFloat): non-finite input at index ${i}: ${v}`);
    }
    if (v < -9223372036854775808 || v >= 9223372036854775808) {
      throw new Error(`step ${stepIndex} (intFromFloat): input out of i64 range at index ${i}: ${v}`);
    }
  }
}

// Remap one f32 segment's steps to sub-slots. Inputs (external slots)
// take the lowest sub-ids in first-use order; internal slots follow.
// Returns { slotMap: Map orig->sub, subSteps, inputOrder: [orig slots] }.
export function remapF32Segment(spec, segment) {
  const slotMap = new Map();
  const inputOrder = [];
  let next = 0;
  const producedHere = new Set();
  for (const si of segment.stepIndices) {
    const st = spec.steps[si];
    for (const s of stepInputSlots(st)) {
      if (producedHere.has(s)) continue; // internal: already mapped
      if (!slotMap.has(s)) { slotMap.set(s, next++); inputOrder.push(s); }
    }
    const out = stepOutputSlot(st);
    slotMap.set(out, next++);
    producedHere.add(out);
  }
  const subSteps = segment.stepIndices.map((si) => {
    const st = spec.steps[si];
    const slots = st.kind === 'binary'
      ? [slotMap.get(st.slots[0]), slotMap.get(st.slots[1]), slotMap.get(st.slots[2])]
      : [slotMap.get(st.slots[0]), slotMap.get(st.slots[1])];
    return { layer: st.layer, kind: st.kind, slots };
  });
  return { slotMap, subSteps, inputOrder, subNumSlots: next };
}

function runIntSegment(rt, reg, spec, segment, valueStore, boundaryCasts) {
  for (const si of segment.stepIndices) {
    const st = spec.steps[si];
    const decl = spec.layers[st.layer];
    const ctor = decl.constructor;
    const regId = decl.args[0];
    if (ctor === 'intToFloat') {
      const [inSlot, outSlot] = st.slots;
      const v = valueStore.get(inSlot);
      if (!v) throw new Error(`step ${si} (intToFloat): input slot ${inSlot} has no value`);
      checkIntToFloatBoundary(si, v.values);
      const ti = new rt.WasmIntTensor(new BigInt64Array(v.values), new Uint32Array(v.shape));
      let tf = null;
      try {
        tf = reg.forwardIntCastToFloat(regId, ti);
        const values = Float32Array.from(tf.to_array());
        const shape = Array.from(tf.shape());
        valueStore.set(outSlot, { dtype: 'f32', values, shape });
        boundaryCasts.push({ step: si, direction: 'i64->f32', elements: values.length, inexact_rejected: [] });
      } finally { freeQuiet(tf); freeQuiet(ti); }
    } else if (ctor === 'intFromFloat') {
      const [inSlot, outSlot] = st.slots;
      const v = valueStore.get(inSlot);
      if (!v) throw new Error(`step ${si} (intFromFloat): input slot ${inSlot} has no value`);
      checkIntFromFloatBoundary(si, v.values);
      const tf = new rt.WasmTensor(Float32Array.from(v.values), new Uint32Array(v.shape));
      let ti = null;
      try {
        ti = reg.forwardIntCastFromFloat(regId, tf);
        const values = BigInt64Array.from(ti.to_array());
        const shape = Array.from(ti.shape());
        valueStore.set(outSlot, { dtype: 'i64', values, shape });
        boundaryCasts.push({ step: si, direction: 'f32->i64', elements: values.length, inexact_rejected: [] });
      } finally { freeQuiet(ti); freeQuiet(tf); }
    } else {
      // int binary family: same-shape elementwise, or batched matmul
      // (engine enforces; this host code only transports exact values).
      const [aSlot, bSlot, outSlot] = st.slots;
      const a = valueStore.get(aSlot);
      const b = valueStore.get(bSlot);
      if (!a) throw new Error(`step ${si} (${ctor}): input slot ${aSlot} has no value`);
      if (!b) throw new Error(`step ${si} (${ctor}): input slot ${bSlot} has no value`);
      const ta = new rt.WasmIntTensor(new BigInt64Array(a.values), new Uint32Array(a.shape));
      const tb = new rt.WasmIntTensor(new BigInt64Array(b.values), new Uint32Array(b.shape));
      let out = null;
      try {
        out = reg.forwardIntBinary(regId, ta, tb);
        const values = BigInt64Array.from(out.to_array());
        const shape = Array.from(out.shape());
        valueStore.set(outSlot, { dtype: 'i64', values, shape });
      } finally { freeQuiet(out); freeQuiet(tb); freeQuiet(ta); }
    }
  }
}

function runF32Segment(rt, reg, specs, spec, segment, portMap, valueStore) {
  const { slotMap, subSteps, subNumSlots } = remapF32Segment(spec, segment);
  // Effective outputs: downstream outputs, or (dead segment) the last
  // step's output so the segment still executes and stays loud.
  const effOutputs = segment.outputs.length
    ? segment.outputs
    : [stepOutputSlot(spec.steps[segment.stepIndices[segment.stepIndices.length - 1]])];
  const results = new Map(); // orig slot -> { values, shape }
  for (const outOrig of effOutputs) {
    const outSub = slotMap.get(outOrig);
    let b = null, plan = null, g = null, bundle = null;
    try {
      b = new rt.AgentGraphBuilder(subNumSlots);
      for (const sst of subSteps) {
        const s = specs[sst.layer];
        if (!s) throw new Error(`segment ${segment.index}: step references missing layer index ${sst.layer}`);
        if (sst.kind === 'unary' && sst.slots.length === 2) b.addUnary(s, sst.slots[0], sst.slots[1]);
        else if (sst.kind === 'binary' && sst.slots.length === 3) b.addBinary(s, sst.slots[0], sst.slots[1], sst.slots[2]);
        else throw new Error(`segment ${segment.index}: bad step (need unary:[a,b] or binary:[a,b,c]): ${JSON.stringify(sst)}`);
      }
      b.setOutput(outSub);
      plan = new rt.MultiInputGraphPlan(b);
      // Declare every external input as a port: original ports keep
      // their declaration; prior-run outputs get synthetic ports.
      const subPorts = [];
      for (const origSlot of segment.inputs) {
        const sub = slotMap.get(origSlot);
        const port = portMap.get(origSlot);
        if (port) subPorts.push({ slot: sub, role: port.role, shape: port.shape, layout: port.layout });
        else {
          const v = valueStore.get(origSlot);
          if (!v) throw new Error(`segment ${segment.index}: input slot ${origSlot} has no value (port without inputs, or producer did not run)`);
          subPorts.push({ slot: sub, role: SEGMENT_INPUT_ROLE, shape: v.shape, layout: SEGMENT_INPUT_LAYOUT });
        }
        plan.addInputPort(sub, subPorts[subPorts.length - 1].role, ...subPorts[subPorts.length - 1].shape, subPorts[subPorts.length - 1].layout, false, 0n);
      }
      g = reg.compileMultiInputGraph(plan);
      bundle = new rt.MultiInputInputBundle(plan);
      for (let pi = 0; pi < subPorts.length; pi++) {
        const p = subPorts[pi];
        const origSlot = segment.inputs[pi];
        const v = valueStore.get(origSlot);
        if (!v) throw new Error(`segment ${segment.index}: input slot ${origSlot} has no value`);
        // Fresh tensor per consumer: the materialization rule (§3.2).
        const t = new rt.WasmTensor(Float32Array.from(v.values), new Uint32Array(v.shape));
        const ok = bundle.bindInput(p.slot, t, p.role, p.layout, 'burn-math-skill', 1n, '');
        t.free();
        if (!ok) throw new Error(`segment ${segment.index}: bindInput failed for slot ${origSlot}`);
      }
      const out = g.run(reg, bundle);
      try {
        results.set(outOrig, { values: Float32Array.from(out.to_array()), shape: Array.from(out.shape()) });
      } finally { freeQuiet(out); }
    } finally {
      freeQuiet(bundle); freeQuiet(g); freeQuiet(plan); freeQuiet(b);
    }
  }
  for (const [origSlot, val] of results) {
    valueStore.set(origSlot, { dtype: 'f32', values: val.values, shape: val.shape });
  }
}

// Run a spec containing int steps (mixed or int-only-with-f32-output).
// Caller must have run propagateDtype + the output-dtype check first:
// this function executes segments in order and never returns partial
// output (it throws; the caller converts to a {ok:false} record).
// Returns { outputs: { shape, values }, segments: [receipt], boundary_casts: [...] }.
// Lifetime: reg/specs/graphs/plans/builders/bundles freed even on
// throw, mirroring buildGraph's exception safety.
export function runMixed(rt, spec, segments, portMap) {
  const reg = new rt.LayerRegistry();
  const specs = [];
  const valueStore = new Map();
  const boundaryCasts = [];
  try {
    // Init once (all layers), reuse spec objects across builders.
    for (const [li, decl] of spec.layers.entries()) {
      if (typeof decl.constructor !== 'string' || !Object.hasOwn(rt.AgentLayerSpec, decl.constructor) || typeof rt.AgentLayerSpec[decl.constructor] !== 'function') {
        throw new Error(`unknown layer constructor: ${decl.constructor}`);
      }
      const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
      if (typeof s !== 'object' || s === null) {
        throw new Error(`unknown layer constructor: ${decl.constructor}`);
      }
      reg.initAgentLayer(s);
      specs.push(s);
    }
    // Weights once on the shared registry (same validation as
    // buildGraph: weighted layer or error).
    for (const [indexStr, values] of Object.entries(spec.weights ?? {})) {
      const li = Number(indexStr);
      const decl = spec.layers[li];
      const wt = decl ? WEIGHTED_CONSTRUCTORS[decl.constructor] : null;
      if (!wt) throw new Error(`weights for layer index ${indexStr}: not a weighted layer`);
      reg.setWeightsFlat(decl.args[0], LAYER_TYPE[wt], Float32Array.from(values));
    }
    // Seed the store with port inputs (f32 by construction).
    for (const [slotStr, values] of Object.entries(spec.inputs ?? {})) {
      const slot = Number(slotStr);
      const port = portMap.get(slot);
      if (!port) throw new Error(`input for slot ${slot} has no declared port`);
      valueStore.set(slot, { dtype: 'f32', values: Float32Array.from(values), shape: [...port.shape] });
    }
    for (const seg of segments) {
      if (seg.kind === 'int') runIntSegment(rt, reg, spec, seg, valueStore, boundaryCasts);
      else runF32Segment(rt, reg, specs, spec, seg, portMap, valueStore);
    }
    const final = valueStore.get(spec.outputSlot);
    if (!final) throw new Error(`output slot ${spec.outputSlot} was never produced`);
    if (final.dtype !== 'f32') throw new Error(`output slot ${spec.outputSlot} dtype ${final.dtype}; end with intToFloat or use int.mjs`);
    return {
      outputs: { shape: final.shape, values: Array.from(final.values) },
      segments: segments.map((s) => ({
        index: s.index, kind: s.kind, steps: s.steps,
        layers: [...new Set(s.layers)], inputs: s.inputs, outputs: s.outputs,
      })),
      boundary_casts: boundaryCasts,
    };
  } finally {
    for (let i = specs.length - 1; i >= 0; i--) freeQuiet(specs[i]);
    freeQuiet(reg);
  }
}
