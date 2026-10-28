#!/usr/bin/env node
// burn-math shard_spec: rewrite a spec so no layer exceeds the runner
// protocol's per-layer weight cap.
//
// Where the cap lives (measured 2026-10-21): the session runner's
// setWeights framing carries the weight count in a u16 field, so any
// layer with > 65535 weight values fails every runner-based CLI
// (run/verify/profile/compare) with an out-of-range error at phase
// setWeights. The WASM-direct path has no such cap (1e6 weights run
// fine) — this tool exists for the runner path.
//
// Method: an oversized `linear` layer is split by OUTPUT rows into block
// linears (each <= cap), whose outputs are rejoined with a concat chain
// (concat dim=1). Per output element the dot product is unchanged, so the
// sharded graph is the same function — and this CLI proves it before
// printing: (a) the block weight slices reassemble into the original
// flat array EXACTLY, and (b) original vs sharded are both run through
// the engine on the spec's inputs and compared.
//
// Usage: node shard_spec.mjs <spec.json> [maxWeights=65535]
// Prints {ok, sharded, max_weights, shards, reassembly_exact, verified,
// max_abs_error, spec}. Exit 0 iff nothing needed sharding, or the
// sharded spec reassembles exactly and verifies within 1e-4.
// Scope: linear layers only; a layer whose single output row already
// exceeds the cap (in_dim too wide) cannot be output-sharded and is
// reported as unshardable.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';

const [specFile, capArg] = process.argv.slice(2);
if (!specFile) { console.error('usage: node shard_spec.mjs <spec.json> [maxWeights]'); process.exit(2); }
const cap = capArg !== undefined ? Number(capArg) : 65535;
if (!(cap >= 1)) failCli('maxWeights must be positive');

let spec;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }

// printJson + process.exit truncates large outputs still queued in the
// stdout pipe; spec-carrying prints go through a flush-awaited write.
async function printBig(obj, code) {
  await new Promise((res) => process.stdout.write(JSON.stringify(obj, null, 2) + '\n', res));
  process.exit(code);
}

const weightsOf = (layerIdx) => spec.weights?.[String(layerIdx)] ?? null;

// Which layers need sharding?
const shardPlan = new Map(); // layerIndex -> {in, out, bias, blocks:[[r0,r1),...]}
const unshardable = [];
for (let li = 0; li < spec.layers.length; li++) {
  const decl = spec.layers[li];
  if (decl.constructor !== 'linear') continue;
  const w = weightsOf(li);
  if (!w || w.length <= cap) continue;
  const [, inDim, outDim, bias] = decl.args;
  const perRow = inDim + (bias ? 1 : 0);
  const blockOut = Math.floor(cap / perRow);
  if (blockOut < 1) { unshardable.push({ layer_index: li, in_dim: inDim, out_dim: outDim, weights: w.length }); continue; }
  const blocks = [];
  for (let r = 0; r < outDim; r += blockOut) blocks.push([r, Math.min(outDim, r + blockOut)]);
  shardPlan.set(li, { inDim, outDim, bias: !!bias, blocks });
}

if (unshardable.length > 0) {
  printJsonAndExit({ ok: false, sharded: false, error: 'input dim too wide for output sharding', unshardable }, 2);
}
if (shardPlan.size === 0) {
  await printBig({ ok: true, sharded: false, max_weights: cap, spec }, 0);
}

// Slice a block's flat weights out of the column-major original:
// kernel flat[k*out + i] (k = input, i = output row), bias appended last.
function blockWeights(w, inDim, outDim, bias, r0, r1) {
  const out = [];
  for (let k = 0; k < inDim; k++) {
    for (let i = r0; i < r1; i++) out.push(w[k * outDim + i]);
  }
  if (bias) for (let i = r0; i < r1; i++) out.push(w[inDim * outDim + i]);
  return out;
}

// Rebuild the spec.
let nextRegistryId = Math.max(...spec.layers.map(l => l.args[0])) + 1;
let nextSlot = spec.numSlots;
const newLayers = [];
const newWeights = {};
const layerRemap = new Map(); // old index -> new index (unsharded only)
for (let li = 0; li < spec.layers.length; li++) {
  if (shardPlan.has(li)) continue;
  layerRemap.set(li, newLayers.length);
  newLayers.push(spec.layers[li]);
  const w = weightsOf(li);
  if (w) newWeights[String(newLayers.length - 1)] = w;
}
let concatLayerIdx = null;
const shards = [];
const newSteps = [];
for (const step of spec.steps) {
  const plan = shardPlan.get(step.layer);
  if (!plan) {
    newSteps.push({ ...step, layer: layerRemap.get(step.layer) });
    continue;
  }
  const [inSlot, outSlot] = step.slots;
  const w = weightsOf(step.layer);
  const blockIdxs = [];
  const blockSlots = [];
  for (const [r0, r1] of plan.blocks) {
    const idx = newLayers.length;
    newLayers.push({ constructor: 'linear', args: [nextRegistryId++, plan.inDim, r1 - r0, plan.bias] });
    newWeights[String(idx)] = blockWeights(w, plan.inDim, plan.outDim, plan.bias, r0, r1);
    blockIdxs.push(idx);
    blockSlots.push(nextSlot++);
  }
  blockIdxs.forEach((idx, b) => {
    const isLastBlock = plan.blocks.length === 1;
    newSteps.push({ kind: 'unary', layer: idx, slots: [inSlot, isLastBlock ? outSlot : blockSlots[b]] });
  });
  if (plan.blocks.length > 1) {
    if (concatLayerIdx === null) {
      concatLayerIdx = newLayers.length;
      newLayers.push({ constructor: 'concat', args: [nextRegistryId++, 1] });
    }
    let acc = blockSlots[0];
    for (let b = 1; b < plan.blocks.length; b++) {
      const isLast = b === plan.blocks.length - 1;
      const target = isLast ? outSlot : nextSlot++;
      newSteps.push({ kind: 'binary', layer: concatLayerIdx, slots: [acc, blockSlots[b], target] });
      acc = target;
    }
  }
  shards.push({
    layer_index: step.layer, in_dim: plan.inDim, out_dim: plan.outDim,
    blocks: plan.blocks.map(([a, b]) => b - a),
  });
}
const sharded = {
  ...spec,
  numSlots: nextSlot,
  layers: newLayers,
  steps: newSteps,
  weights: newWeights,
};

// (a) Reassembly: block slices must rebuild each original flat array.
let reassemblyExact = true;
for (const [li, plan] of shardPlan) {
  const w = weightsOf(li);
  // Re-derive the block slices from the plan and unslice them back into
  // the original column-major layout; every element must match exactly.
  const parts = plan.blocks.map(([r0, r1]) => blockWeights(w, plan.inDim, plan.outDim, plan.bias, r0, r1));
  const kernel = new Array(plan.inDim * plan.outDim);
  for (let k = 0; k < plan.inDim; k++) {
    for (let b = 0; b < plan.blocks.length; b++) {
      const [r0, r1] = plan.blocks[b];
      const seg = parts[b].slice(k * (r1 - r0), (k + 1) * (r1 - r0));
      for (let i = 0; i < seg.length; i++) kernel[k * plan.outDim + r0 + i] = seg[i];
    }
  }
  for (let i = 0; i < kernel.length; i++) if (kernel[i] !== w[i]) { reassemblyExact = false; break; }
  if (reassemblyExact && plan.bias) {
    for (let i = 0; i < plan.outDim; i++) {
      const b = plan.blocks.findIndex(([r0, r1]) => i >= r0 && i < r1);
      const [r0] = plan.blocks[b];
      const biasVal = parts[b][plan.inDim * (plan.blocks[b][1] - r0) + (i - r0)];
      if (biasVal !== w[plan.inDim * plan.outDim + i]) { reassemblyExact = false; break; }
    }
  }
  if (!reassemblyExact) break;
}

// (b) Engine cross-check on the spec's inputs.
let maxAbs = null;
try {
  const rt = await loadWasm();
  const runSpec = (s) => {
    const { reg, plan, g } = buildGraph(rt, s);
    const bundle = makeBundle(rt, plan, s.inputs, portBySlot(s));
    const out = g.run(reg, bundle);
    const v = Array.from(out.to_array());
    out.free();
    return v;
  };
  const a = runSpec(spec);
  const b = runSpec(sharded);
  if (a.length !== b.length) throw new Error(`output length ${a.length} != sharded ${b.length}`);
  maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
} catch (e) {
  await printBig({ ok: false, sharded: true, reassembly_exact: reassemblyExact, verified: false, error: e.message, shards, spec: sharded }, 2);
}

const verified = reassemblyExact && maxAbs !== null && maxAbs <= 1e-4;
await printBig({
  ok: verified,
  sharded: true,
  max_weights: cap,
  shards,
  reassembly_exact: reassemblyExact,
  verified,
  max_abs_error: maxAbs,
  spec: sharded,
}, verified ? 0 : 2);
