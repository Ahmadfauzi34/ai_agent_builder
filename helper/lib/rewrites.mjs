// Graph rewrite transforms for burn-math (used by bin/rewrite.mjs and lib/recurse.mjs).
// Each entry: { name, description, preconditions, apply(spec) -> newSpec }.
// apply() throws RewriteError when preconditions are not met — the caller
// reports the reason instead of producing a bad graph.
//
// Design rules:
// - Transforms are pure: deep-clone in, new spec out; input never mutated.
// - Slot/layer surgery is explicit and minimal; numSlots/ports are preserved.
// - Every transform in the shipped catalog has a recorded proof run
//   (see ../references/rewrites.v1.json); a transform without proof is not
//   added here.
import { createRequire } from 'node:module';
import { catalogPath } from './engine.mjs';
import { composeAffineGeneral } from './linear_equiv.mjs';

const require = createRequire(import.meta.url);
const CATALOG = require(catalogPath()).constructors;

export class RewriteError extends Error {}

const clone = (o) => JSON.parse(JSON.stringify(o));
const isLinear = (l) => l?.constructor === 'linear';
const isRelu = (l) => l?.constructor === 'relu';
const isAdd = (l) => l?.constructor === 'add';
const isTranspose = (l) => l?.constructor === 'transpose';

// Input slots of a step (all but the last entry).
function inputsOf(st) {
  return st.kind === 'unary' ? [st.slots[0]] : [st.slots[0], st.slots[1]];
}

// All steps (except `exceptIdx`) that read `slot` as an input, or write it.
function usesOf(spec, slot, exceptIdx = -1) {
  const reads = [], writes = [];
  spec.steps.forEach((st, i) => {
    if (i === exceptIdx) return;
    if (inputsOf(st).includes(slot)) reads.push(i);
    if (st.slots[st.slots.length - 1] === slot) writes.push(i);
  });
  return { reads, writes };
}

function maxRegistryId(spec) {
  return Math.max(0, ...spec.layers.map((l) => l.args[0] ?? 0));
}

function flatToAffine(flat, inDim, outDim, bias) {
  const W = [];
  for (let i = 0; i < outDim; i++) {
    const row = [];
    for (let k = 0; k < inDim; k++) row.push(flat[k * outDim + i]);
    W.push(row);
  }
  const b = bias ? flat.slice(inDim * outDim, inDim * outDim + outDim) : new Array(outDim).fill(0);
  return { W, b };
}

function affineToFlat(W, b, inDim, outDim, hasBias) {
  const flat = [];
  for (let k = 0; k < inDim; k++) {
    for (let i = 0; i < outDim; i++) flat.push(W[i][k]);
  }
  if (hasBias) flat.push(...b);
  return flat;
}

function eliminateDeadStepsRewrite(spec) {
  const s = clone(spec);
  const liveSlots = new Set([s.outputSlot]);
  const liveSteps = new Set();
  for (let si = s.steps.length - 1; si >= 0; si--) {
    const st = s.steps[si];
    const outSlot = st.slots[st.slots.length - 1];
    if (liveSlots.has(outSlot)) {
      liveSteps.add(si);
      const inCount = st.kind === 'binary' ? 2 : 1;
      for (let i = 0; i < inCount; i++) liveSlots.add(st.slots[i]);
    }
  }
  const portSlots = new Set((s.ports ?? []).map((p) => p.slot));
  const consumedPorts = new Set();
  for (const si of liveSteps) {
    const st = s.steps[si];
    const inCount = st.kind === 'binary' ? 2 : 1;
    for (let i = 0; i < inCount; i++) {
      if (portSlots.has(st.slots[i])) consumedPorts.add(st.slots[i]);
    }
  }
  for (let si = 0; si < s.steps.length; si++) {
    if (liveSteps.has(si)) continue;
    const st = s.steps[si];
    const inCount = st.kind === 'binary' ? 2 : 1;
    for (let i = 0; i < inCount; i++) {
      if (portSlots.has(st.slots[i]) && !consumedPorts.has(st.slots[i])) {
        liveSteps.add(si);
        consumedPorts.add(st.slots[i]);
      }
    }
  }
  if (liveSteps.size === s.steps.length) {
    throw new RewriteError('dead-step-elimination: no dead steps found (all steps contribute to outputSlot or required port consumption)');
  }
  s.steps = s.steps.filter((_, idx) => liveSteps.has(idx));
  return s;
}

const REWRITES = [
  {
    name: 'linear-branch-fusion',
    description: 'linear(a)(x) + linear(b)(x) -> linear(a+b)(x): two bias-free linears on the same input whose outputs are added fuse into one linear with summed weights.',
    preconditions: 'two unary linear layers (bias=false, same in/out dims, weights present) on the same input slot, consumed only by one binary add; intermediate slots not used elsewhere and not the output slot',
    apply(spec) {
      const s = clone(spec);
      const steps = s.steps;
      for (let i = 0; i < steps.length; i++) {
        for (let j = i + 1; j < steps.length; j++) {
          const si = steps[i], sj = steps[j];
          if (si.kind !== 'unary' || sj.kind !== 'unary') continue;
          const la = s.layers[si.layer], lb = s.layers[sj.layer];
          if (!isLinear(la) || !isLinear(lb)) continue;
          const [, inA, outA, biasA] = la.args;
          const [, inB, outB, biasB] = lb.args;
          if (biasA || biasB || inA !== inB || outA !== outB) continue;
          const [inSlot, t1] = si.slots;
          const [inSlot2, t2] = sj.slots;
          if (inSlot !== inSlot2) continue;
          const wa = s.weights?.[String(si.layer)], wb = s.weights?.[String(sj.layer)];
          if (!wa || !wb) continue;
          for (let k = j + 1; k < steps.length; k++) {
            const sk = steps[k];
            if (sk.kind !== 'binary' || !isAdd(s.layers[sk.layer])) continue;
            const [a, b, t3] = sk.slots;
            if (!((a === t1 && b === t2) || (a === t2 && b === t1))) continue;
            // intermediates must be private to this pattern
            const u1 = usesOf(s, t1), u2 = usesOf(s, t2);
            const onlyK = (u) => u.reads.length === 1 && u.reads[0] === k && u.writes.length === 1;
            if (!onlyK(u1) || !onlyK(u2)) continue;
            if (t1 === s.outputSlot || t2 === s.outputSlot) continue;
            // --- apply ---
            const newId = maxRegistryId(s) + 1;
            const fusedW = wa.map((v, n) => v + wb[n]);
            const newLayerIdx = s.layers.length;
            s.layers.push({ constructor: 'linear', args: [newId, inA, outA, false] });
            s.weights = { ...(s.weights ?? {}), [String(newLayerIdx)]: fusedW };
            const fused = { kind: 'unary', layer: newLayerIdx, slots: [inSlot, t3] };
            s.steps = steps.filter((_, n) => n !== i && n !== j && n !== k);
            s.steps.splice(i, 0, fused);
            return s;
          }
        }
      }
      throw new RewriteError('linear-branch-fusion: no matching pattern (two bias-free linears on one input, added once, intermediates private)');
    },
  },
  {
    name: 'linear-chain-fusion',
    description: 'linear2(linear1(x)) -> linear_fused(x): two sequential unary linear layers (general rectangular dims, with or without bias) fuse into one linear layer via exact affine composition W = W2*W1, b = W2*b1 + b2.',
    preconditions: 'two chained unary linear layers with explicit weights where the intermediate slot is private to the chain and not the output slot',
    apply(spec) {
      const s = clone(spec);
      const steps = s.steps;
      for (let i = 0; i < steps.length; i++) {
        for (let j = i + 1; j < steps.length; j++) {
          const si = steps[i], sj = steps[j];
          if (si.kind !== 'unary' || sj.kind !== 'unary') continue;
          const l1 = s.layers[si.layer], l2 = s.layers[sj.layer];
          if (!isLinear(l1) || !isLinear(l2)) continue;
          const [inSlot, t1] = si.slots;
          const [midIn, t2] = sj.slots;
          if (t1 !== midIn || t1 === s.outputSlot) continue;
          const u = usesOf(s, t1);
          if (!(u.reads.length === 1 && u.reads[0] === j && u.writes.length === 1 && u.writes[0] === i)) continue;
          const [, in1, out1, bias1] = l1.args;
          const [, in2, out2, bias2] = l2.args;
          if (out1 !== in2) continue;
          const w1 = s.weights?.[String(si.layer)], w2 = s.weights?.[String(sj.layer)];
          if (!w1 || !w2) continue;

          const aff1 = flatToAffine(w1, in1, out1, Boolean(bias1));
          const aff2 = flatToAffine(w2, in2, out2, Boolean(bias2));
          const fusedAff = composeAffineGeneral(aff1, aff2);
          const hasBias = Boolean(bias1 || bias2);
          const fusedFlat = affineToFlat(fusedAff.W, fusedAff.b, in1, out2, hasBias);

          const newId = maxRegistryId(s) + 1;
          const newLayerIdx = s.layers.length;
          s.layers.push({ constructor: 'linear', args: [newId, in1, out2, hasBias] });
          s.weights = { ...(s.weights ?? {}), [String(newLayerIdx)]: fusedFlat };
          const fusedStep = { kind: 'unary', layer: newLayerIdx, slots: [inSlot, t2] };
          s.steps = steps.filter((_, n) => n !== i && n !== j);
          s.steps.splice(i, 0, fusedStep);
          return s;
        }
      }
      throw new RewriteError('linear-chain-fusion: no chained unary linear pair with explicit weights and a private intermediate slot');
    },
  },
  {
    name: 'relu-idempotent',
    description: 'relu(relu(x)) -> relu(x): the second relu is a no-op.',
    preconditions: 'two chained unary relus where the intermediate slot is private to the chain',
    apply(spec) {
      const s = clone(spec);
      const steps = s.steps;
      for (let i = 0; i < steps.length; i++) {
        for (let j = i + 1; j < steps.length; j++) {
          const si = steps[i], sj = steps[j];
          if (si.kind !== 'unary' || sj.kind !== 'unary') continue;
          if (!isRelu(s.layers[si.layer]) || !isRelu(s.layers[sj.layer])) continue;
          const t1 = si.slots[1];
          if (sj.slots[0] !== t1) continue;
          const t2 = sj.slots[1];
          const u = usesOf(s, t1);
          if (!(u.reads.length === 1 && u.reads[0] === j && u.writes.length === 1 && u.writes[0] === i)) continue;
          // --- apply: drop step j, rename t2 -> t1 everywhere ---
          if (t1 === s.outputSlot) continue;
          s.steps = steps.filter((_, n) => n !== j);
          for (const st of s.steps) st.slots = st.slots.map((sl) => (sl === t2 ? t1 : sl));
          if (s.outputSlot === t2) s.outputSlot = t1;
          return s;
        }
      }
      throw new RewriteError('relu-idempotent: no chained relu pair with a private intermediate slot');
    },
  },
  {
    name: 'transpose-involution',
    description: 'transpose(d0,d1)(transpose(d0,d1)(x)) -> x: two back-to-back transposes swapping the same pair of axes cancel out.',
    preconditions: 'two chained unary transpose steps on the same axis pair {d0, d1} with a private intermediate slot and at least one remaining step in the graph',
    apply(spec) {
      const s = clone(spec);
      const steps = s.steps;
      if (steps.length <= 2) {
        throw new RewriteError('transpose-involution: requires at least one remaining step after cancelling the transpose pair');
      }
      for (let i = 0; i < steps.length; i++) {
        for (let j = i + 1; j < steps.length; j++) {
          const si = steps[i], sj = steps[j];
          if (si.kind !== 'unary' || sj.kind !== 'unary') continue;
          const la = s.layers[si.layer], lb = s.layers[sj.layer];
          if (!isTranspose(la) || !isTranspose(lb)) continue;
          const [, a0, a1] = la.args;
          const [, b0, b1] = lb.args;
          const sameAxes = (a0 === b0 && a1 === b1) || (a0 === b1 && a1 === b0);
          if (!sameAxes) continue;
          const [inSlot, t1] = si.slots;
          const [midIn, t2] = sj.slots;
          if (t1 !== midIn || t1 === s.outputSlot) continue;
          const u = usesOf(s, t1);
          if (!(u.reads.length === 1 && u.reads[0] === j && u.writes.length === 1 && u.writes[0] === i)) continue;
          s.steps = steps.filter((_, n) => n !== i && n !== j);
          for (const st of s.steps) st.slots = st.slots.map((sl) => (sl === t2 ? inSlot : sl));
          if (s.outputSlot === t2) s.outputSlot = inSlot;
          return s;
        }
      }
      throw new RewriteError('transpose-involution: no cancelling transpose pair found');
    },
  },
  {
    name: 'dead-step-elimination',
    description: 'Remove steps whose outputs never reach outputSlot (while preserving required port consumption).',
    preconditions: 'at least one dead step that does not contribute to outputSlot and is not the sole consumer of a declared port',
    apply: eliminateDeadStepsRewrite,
  },
  {
    name: 'add-commute',
    description: 'add(a,b) -> add(b,a): canonicalizes operand order (no numeric change).',
    preconditions: 'at least one binary add step with distinct operands',
    apply(spec) {
      const s = clone(spec);
      for (let k = 0; k < s.steps.length; k++) {
        const st = s.steps[k];
        if (st.kind === 'binary' && isAdd(s.layers[st.layer]) && st.slots[0] !== st.slots[1]) {
          s.steps[k] = { ...st, slots: [st.slots[1], st.slots[0], st.slots[2]] };
          return s;
        }
      }
      throw new RewriteError('add-commute: no binary add step with distinct operands');
    },
  },
];

export const SIMPLIFY_REWRITE_ORDER = [
  'linear-branch-fusion',
  'linear-chain-fusion',
  'relu-idempotent',
  'transpose-involution',
  'dead-step-elimination',
];

export function listRewrites() {
  return REWRITES.map((r) => ({ name: r.name, description: r.description, preconditions: r.preconditions }));
}

export function applyRewrite(name, spec) {
  const r = REWRITES.find((x) => x.name === name);
  if (!r) throw new RewriteError(`unknown rewrite '${name}' (available: ${REWRITES.map((x) => x.name).join(', ')})`);
  return r.apply(spec);
}

// Recursively apply rewrite rules until no rule fires (fixpoint).
export function applyRewritesRecursive(nameOrAll, spec, { maxPasses = 32 } = {}) {
  const rules = (nameOrAll === 'all' || nameOrAll === 'fixpoint')
    ? SIMPLIFY_REWRITE_ORDER
    : [nameOrAll, ...(nameOrAll !== 'dead-step-elimination' ? ['dead-step-elimination'] : [])];
  let cur = clone(spec);
  const applied = [];
  for (let pass = 0; pass < maxPasses; pass++) {
    let changed = false;
    for (const rName of rules) {
      try {
        cur = applyRewrite(rName, cur);
        applied.push(rName);
        changed = true;
        break;
      } catch (e) {
        if (!(e instanceof RewriteError)) throw e;
      }
    }
    if (!changed) break;
  }
  if (applied.length === 0) {
    throw new RewriteError(`${nameOrAll}: no matching rewrite pattern found`);
  }
  return { spec: cur, applied_rewrites: applied };
}

export { CATALOG };
