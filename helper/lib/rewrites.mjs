// Graph rewrite transforms for burn-math (used by bin/rewrite.mjs).
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

const require = createRequire(import.meta.url);
const CATALOG = require(catalogPath()).constructors;

export class RewriteError extends Error {}

const clone = (o) => JSON.parse(JSON.stringify(o));
const isLinear = (l) => l?.constructor === 'linear';
const isRelu = (l) => l?.constructor === 'relu';
const isAdd = (l) => l?.constructor === 'add';

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
  return Math.max(...spec.layers.map((l) => l.args[0] ?? 0));
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
          const [ida, inA, outA, biasA] = la.args;
          const [idb, inB, outB, biasB] = lb.args;
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
          if (t1 === s.outputSlot) continue;
          // --- apply: drop step j, rename t2 -> t1 everywhere ---
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

export function listRewrites() {
  return REWRITES.map((r) => ({ name: r.name, description: r.description, preconditions: r.preconditions }));
}

export function applyRewrite(name, spec) {
  const r = REWRITES.find((x) => x.name === name);
  if (!r) throw new RewriteError(`unknown rewrite '${name}' (available: ${REWRITES.map((x) => x.name).join(', ')})`);
  return r.apply(spec);
}

export { CATALOG };
