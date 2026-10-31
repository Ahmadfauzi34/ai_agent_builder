#!/usr/bin/env node
// burn-math geom_batch: batched 3D model -> 2D screen projection.
//
// Same pipeline as bin/geom_project.mjs (engine does the linear MVP part as
// f32, host does perspective divide + viewport), but vertices are processed
// BATCHED: up to BATCH_MAX vertices per engine run instead of one.
//
// Batching construction: one linear(4N,4N) layer whose weight matrix is
// block-diagonal with N copies of the 4x4 MVP matrix (column-major).
// Engine limit discovered empirically: setWeights rejects >65535 weight
// values -> D*D <= 65535 -> D <= 255 -> N <= 63 vertices per run.
// Larger meshes are processed in chunks of 63 (bit-exact vs per-vertex).
//
// Usage: node geom_batch.mjs <model.json> [out.json] [--selftest]
// model.json: same format as geom_project.mjs
//   {"vertices": [[x,y,z], ...],
//    "model": [chain ops, geom_build format],
//    "camera": {"eye","target","up","fov_deg","aspect","near","far"},
//    "viewport": {"w":800,"h":600}}
import { startRunner, unwrap, printJson } from '../lib/runner.mjs';
import {
  rotX4, rotY4, rotZ4, rotAxisAngle4, euler4, quatToMat4,
  translate4, scale4, mat4Mul, composeChain, lookAt4, perspective4,
  homogenize,
} from '../lib/geom3.mjs';
import fs from 'node:fs';

export const BATCH_MAX = 63; // setWeights cap: weight count <= 65535 -> (4N)^2 <= 65535

function opToMat4(op) {
  switch (op.op) {
    case 'rotate_x': return rotX4(op.degrees ?? 0);
    case 'rotate_y': return rotY4(op.degrees ?? 0);
    case 'rotate_z': return rotZ4(op.degrees ?? 0);
    case 'rotate_axis': return rotAxisAngle4(...op.axis, op.degrees ?? 0);
    case 'euler': return euler4(op.roll ?? 0, op.pitch ?? 0, op.yaw ?? 0, op.order ?? 'ZYX');
    case 'quaternion': return quatToMat4(op.w ?? 1, op.x ?? 0, op.y ?? 0, op.z ?? 0);
    case 'translate': return translate4(...op.by);
    case 'scale': return scale4(...op.by);
    case 'matrix': {
      if (!Array.isArray(op.m) || op.m.length !== 16) throw new Error('"matrix" op needs "m": 16 numbers (row-major)');
      return op.m.slice();
    }
    default: throw new Error(`unknown op '${op.op}'`);
  }
}

function buildMVP(model) {
  const M = composeChain((model.model ?? []).map(opToMat4));
  const c = model.camera;
  const V = lookAt4(c.eye, c.target, c.up);
  const P = perspective4(c.fov_deg, c.aspect ?? 1, c.near, c.far);
  return mat4Mul(P, mat4Mul(V, M)); // row-major 4x4
}

// Block-diagonal column-major weights: N copies of row-major 4x4 mvp.
function blockDiagWeights(mvp, n) {
  const D = 4 * n;
  const w = new Array(D * D).fill(0);
  for (let b = 0; b < n; b++)
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        w[(4 * b + j) * D + (4 * b + i)] = mvp[i * 4 + j];
  return w;
}

function specForBatch(mvp, n) {
  const D = 4 * n;
  const port = (slot) => ({ slot, role: 'observation', shape: [1, D, 1, 1], layout: 'feature_axis1_singleton' });
  return {
    layers: [{ constructor: 'linear', args: [100, D, D, false] }, { constructor: 'add', args: [200] }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    numSlots: 4,
    outputSlot: 3,
    ports: [port(0), port(1)],
    weights: { 0: blockDiagWeights(mvp, n) },
    inputs: { 1: new Array(D).fill(0) },
  };
}

const BIND = { role: 'observation', layout: 'feature_axis1_singleton', source: 'burn-math-skill', revision: 1, fingerprint: '' };

// One engine run for a chunk of exactly BATCH_MAX vertices -> flat clip coords.
// The chunk is PADDED to BATCH_MAX so every chunk reuses ONE setup
// (validatePlan + create + setWeights); only slot 0 is re-bound per chunk.
// Padding vertices are [0,0,0] and their outputs are discarded by the caller.
async function runChunkPadded(send, verts63) {
  unwrap(await send({ op: 'clear', slot: 0 }));
  unwrap(await send({ op: 'bind', slot: 0, values: verts63.flatMap(homogenize), shape: [1, 4 * BATCH_MAX, 1, 1], ...BIND }));
  const ran = unwrap(await send({ op: 'run' }));
  return Array.from(ran.values).slice(0, 4 * BATCH_MAX);
}

// Batched clip coords for all vertices: ONE setup, then one bind+run per
// 63-vertex chunk (last chunk padded). Returns clips in vertex order.
// If `timings` is given, fills {setupMs, chunkMs[]} with measured values.
export async function runClipsBatched(mvp, vertices, timings = null) {
  const { send, stop } = startRunner();
  try {
    const tSetup = Date.now();
    const spec = specForBatch(mvp, BATCH_MAX);
    const { planFragment } = await import('../lib/spec.mjs');
    const plan = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
    if (!plan.valid) throw new Error('validatePlan failed: ' + JSON.stringify(plan));
    const logicalPorts = spec.ports.map((p, i) => ({ id: `in${i}`, slot: p.slot, source: 'burn-math-skill' }));
    unwrap(await send({ op: 'create', layers: spec.layers, steps: spec.steps, numSlots: spec.numSlots, outputSlot: spec.outputSlot, ports: spec.ports, logicalPorts }));
    unwrap(await send({ op: 'setWeights', layer: 0, values: spec.weights[0] }));
    unwrap(await send({ op: 'bind', slot: 1, values: spec.inputs[1], shape: spec.ports[1].shape, ...BIND }));
    if (timings) { timings.setupMs = Date.now() - tSetup; timings.chunkMs = []; }
    const clips = [];
    const pad = [0, 0, 0];
    for (let s = 0; s < vertices.length; s += BATCH_MAX) {
      const chunk = vertices.slice(s, s + BATCH_MAX);
      const padded = chunk.concat(new Array(BATCH_MAX - chunk.length).fill(pad));
      const tC = Date.now();
      const flat = await runChunkPadded(send, padded);
      if (timings) timings.chunkMs.push(Date.now() - tC);
      for (let b = 0; b < chunk.length; b++) clips.push(flat.slice(4 * b, 4 * b + 4));
    }
    return clips;
  } finally {
    try { await stop(); } catch { /* ignore */ }
  }
}

// Host-side tail: perspective divide + viewport (same as geom_project.mjs).
export function projectClip(clip, viewport) {
  const [x, y, z, w] = clip;
  if (!(w > 0)) return { behind: true, clip };
  const nx = x / w, ny = y / w, nz = z / w;
  const sx = (nx * 0.5 + 0.5) * viewport.w;
  const sy = (1 - (ny * 0.5 + 0.5)) * viewport.h;
  return { behind: false, clip, ndc: [nx, ny, nz], screen: [sx, sy] };
}

export async function projectModel(model, timings = null) {
  const mvp = buildMVP(model);
  const clips = await runClipsBatched(mvp, model.vertices, timings);
  return clips.map((c) => projectClip(c, model.viewport));
}

// ---------- independent f64 reference (separate scalar code, no geom3 imports)
const R = {
  norm: (v) => { const n = Math.hypot(...v); return v.map((c) => c / n); },
  sub: (a, b) => a.map((c, i) => c - b[i]),
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  dot: (a, b) => a.reduce((s, c, i) => s + c * b[i], 0),
  mul: (A, B) => Array.from({ length: 16 }, (_, i) => {
    const r = (i / 4) | 0, c = i % 4;
    let s = 0; for (let k = 0; k < 4; k++) s += A[r * 4 + k] * B[k * 4 + c]; return s;
  }),
  clip: (M, v) => [0, 1, 2, 3].map((i) => M[i * 4] * v[0] + M[i * 4 + 1] * v[1] + M[i * 4 + 2] * v[2] + M[i * 4 + 3]),
  lookAt(eye, t, up) {
    const z = this.norm(this.sub(eye, t)), x = this.norm(this.cross(up, z)), y = this.cross(z, x);
    return [x[0], x[1], x[2], -this.dot(x, eye), y[0], y[1], y[2], -this.dot(y, eye), z[0], z[1], z[2], -this.dot(z, eye), 0, 0, 0, 1];
  },
  persp(fov, a, n, f) {
    const g = 1 / Math.tan((fov * Math.PI) / 360);
    return [g / a, 0, 0, 0, 0, g, 0, 0, 0, 0, (f + n) / (n - f), (2 * f * n) / (n - f), 0, 0, -1, 0];
  },
  rotY(d) { const a = (d * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1]; },
  transl(tx, ty, tz) { return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1]; },
};

function refProject(model, vertices) {
  const M = R.mul(R.transl(0, 0, -8), R.rotY(30));
  const c = model.camera;
  const MVP = R.mul(R.persp(c.fov_deg, c.aspect ?? 1, c.near, c.far), R.mul(R.lookAt(c.eye, c.target, c.up), M));
  return vertices.map((v) => {
    const clip = R.clip(MVP, [...v, 1]);
    const [x, y, z, w] = clip;
    if (!(w > 0)) return { behind: true, clip };
    const nx = x / w, ny = y / w;
    return { behind: false, clip, screen: [(nx * 0.5 + 0.5) * model.viewport.w, (1 - (ny * 0.5 + 0.5)) * model.viewport.h] };
  });
}

const rawArgs = process.argv.slice(2);
if (rawArgs.includes('--selftest')) {
  // 70 vertices forces 2 chunks (63 + 7) -> exercises the chunk boundary.
  const verts = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) verts.push([sx, sy, sz]);
  for (let i = 8; i < 69; i++) verts.push([Math.sin(i) * 1.5, Math.cos(i * 1.7) * 1.5, -2 - (i % 4)]);
  verts.push([0, 0, 12]); // behind the camera
  const model = {
    vertices: verts,
    model: [{ op: 'rotate_y', degrees: 30 }, { op: 'translate', by: [0, 0, -8] }],
    camera: { eye: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov_deg: 60, aspect: 4 / 3, near: 0.1, far: 100 },
    viewport: { w: 800, h: 600 },
  };
  const timings = {};
  const t0 = Date.now();
  const got = await projectModel(model, timings);
  const dtBatched = Date.now() - t0;
  const ref = refProject(model, verts);
  let maxDiff = 0, behindOk = true;
  for (let i = 0; i < verts.length; i++) {
    if (got[i].behind !== ref[i].behind) behindOk = false;
    if (!got[i].behind) maxDiff = Math.max(maxDiff, ...got[i].screen.map((c, j) => Math.abs(c - ref[i].screen[j])));
  }
  // per-vertex loop timing for comparison (same runner protocol as geom_project.mjs)
  const mvp = buildMVP(model);
  const w4 = (() => { const w = new Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) w[i + j * 4] = mvp[i * 4 + j]; return w; })();
  const { send, stop } = startRunner();
  const { planFragment } = await import('../lib/spec.mjs');
  const mkSpec4 = () => ({
    layers: [{ constructor: 'linear', args: [100, 4, 4, false] }, { constructor: 'add', args: [200] }],
    steps: [{ kind: 'unary', layer: 0, slots: [0, 2] }, { kind: 'binary', layer: 1, slots: [2, 1, 3] }],
    numSlots: 4, outputSlot: 3,
    ports: [{ slot: 0, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' }, { slot: 1, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' }],
  });
  const spec4 = mkSpec4();
  unwrap(await send({ op: 'validatePlan', ...planFragment(spec4) }));
  unwrap(await send({ op: 'create', layers: spec4.layers, steps: spec4.steps, numSlots: spec4.numSlots, outputSlot: spec4.outputSlot, ports: spec4.ports, logicalPorts: [{ id: 'in0', slot: 0, source: 'burn-math-skill' }, { id: 'in1', slot: 1, source: 'burn-math-skill' }] }));
  unwrap(await send({ op: 'setWeights', layer: 0, values: w4 }));
  unwrap(await send({ op: 'bind', slot: 1, values: [0, 0, 0, 0], shape: [1, 4, 1, 1], ...BIND }));
  const t1 = Date.now();
  const loopClips = [];
  for (const v of verts) {
    unwrap(await send({ op: 'clear', slot: 0 }));
    unwrap(await send({ op: 'bind', slot: 0, values: homogenize(v), shape: [1, 4, 1, 1], ...BIND }));
    loopClips.push(Array.from(unwrap(await send({ op: 'run' })).values).slice(0, 4));
  }
  const dtLoop = Date.now() - t1;
  await stop();
  let maxLoopDiff = 0;
  for (let i = 0; i < verts.length; i++)
    for (let k = 0; k < 4; k++) maxLoopDiff = Math.max(maxLoopDiff, Math.abs(got[i].clip[k] - loopClips[i][k]));

  const tol = 1e-4;
  const ok = maxDiff <= tol && behindOk && maxLoopDiff === 0 && got[verts.length - 1].behind === true;
  const meanChunkMs = timings.chunkMs.reduce((a, b) => a + b, 0) / timings.chunkMs.length;
  const runs2000 = Math.ceil(2000 / BATCH_MAX);
  printJson({
    ok,
    vertices: verts.length, chunks: Math.ceil(verts.length / BATCH_MAX), batch_max_per_run: BATCH_MAX,
    batch_limit_cause: 'setWeights rejects weight count > 65535 (u16); (4N)^2 <= 65535 -> N <= 63',
    batched_vs_pervertex_loop_max_abs_diff: maxLoopDiff,
    batched_vs_f64ref_max_abs_screen_px: maxDiff, tol,
    behind_consistent: behindOk, behind_vertex_flagged: got[verts.length - 1].behind === true,
    timing_ms: {
      per_vertex_loop: dtLoop,
      batched_total: dtBatched,
      batched_setup_once: timings.setupMs,
      batched_mean_per_chunk: +meanChunkMs.toFixed(1),
      speedup: +(dtLoop / dtBatched).toFixed(2),
    },
    extrapolated_2000v_measured_basis: {
      runs: runs2000,
      per_vertex_loop_s: +((2000 / verts.length) * dtLoop / 1000).toFixed(1),
      batched_s: +((timings.setupMs + runs2000 * meanChunkMs) / 1000).toFixed(2),
      note: 'batched = one setup (measured) + 32 chunk bind+run (measured mean); loop scales linearly from measured 70v',
    },
    note: ok ? 'batched clip coords are BIT-IDENTICAL to the per-vertex loop; f32 vs f64 ref within tol' : 'MISMATCH — see fields',
  });
  process.exit(ok ? 0 : 2);
}

const modelFile = rawArgs[0];
if (!modelFile || rawArgs.includes('--help') || rawArgs.includes('-h')) { console.error('usage: node geom_batch.mjs <model.json> [out.json] [--selftest]'); process.exit(2); }
const outFile = rawArgs.find((a) => a !== modelFile && a.endsWith('.json') && !a.startsWith('--'));
try {
  const model = JSON.parse(fs.readFileSync(modelFile, 'utf8'));
  if (!Array.isArray(model.vertices) || model.vertices.length === 0) throw new Error('model.json needs "vertices": [[x,y,z], ...]');
  if (!model.camera || !model.viewport) throw new Error('model.json needs "camera" and "viewport"');
  const t0 = Date.now();
  const results = await projectModel(model);
  const out = {
    ok: true,
    vertices: model.vertices.length,
    chunks: Math.ceil(model.vertices.length / BATCH_MAX),
    projected: results.filter((r) => !r.behind).length,
    behind_camera: results.filter((r) => r.behind).length,
    engine_ms: Date.now() - t0,
    results,
  };
  if (outFile) { fs.writeFileSync(outFile, JSON.stringify(out, null, 1)); out.out_file = outFile; }
  printJson(out);
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
