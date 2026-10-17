#!/usr/bin/env node
// burn-math geom_project: 3D model -> 2D screen projection, end to end.
//
// The engine does the LINEAR part (model/view/projection matrices fused into
// one linear(4,4) layer, f32) and produces clip coordinates [x,y,z,w].
// The PERSPECTIVE DIVIDE (x/w, y/w, z/w) and viewport mapping stay host-side
// in plain JS — the engine has no div op, so this is a documented boundary,
// not a silent gap. Vertices behind the camera (w <= 0) are flagged, never
// projected (no NaN/Inf).
//
// Usage: node geom_project.mjs <model.json> [out.json] [--selftest]
//
// model.json:
//   {"vertices": [[x,y,z], ...],
//    "triangles": [[a,b,c], ...],      // optional: indices into vertices;
//                                      // enables near-plane clipping stage
//    "model": [chain ops, geom_build format],      // model -> world
//    "camera": {"eye":[x,y,z], "target":[x,y,z], "up":[x,y,z],
//               "fov_deg":60, "aspect":1.333, "near":0.1, "far":100},
//    "viewport": {"w":800, "h":600}}
//
// --selftest: cube demo. Engine (f32, one MVP linear layer) vs a fully
// independent f64 reference written with separate scalar code (no imports
// from lib/geom3.mjs builders), plus a negative control (moved camera must
// change the picture) and a behind-camera vertex (must be flagged, not NaN).
import { startRunner, unwrap, printJson } from '../lib/runner.mjs';
import {
  rotX4, rotY4, rotZ4, rotAxisAngle4, euler4, quatFromEuler, quatToMat4,
  translate4, scale4, mat4Mul, mat4ToColumnMajor, homogenize,
  composeChain, applyMat4Clip, lookAt4, perspective4, clipTriangleNear,
} from '../lib/geom3.mjs';
import fs from 'node:fs';

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
    case 'matrix': { // raw row-major 4x4, for pre-composed or external matrices
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
  return mat4Mul(P, mat4Mul(V, M));
}

function specFor(mvpColMajor) {
  const port = (slot) => ({ slot, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' });
  return {
    layers: [{ constructor: 'linear', args: [100, 4, 4, false] }, { constructor: 'add', args: [200] }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    numSlots: 4,
    outputSlot: 3,
    ports: [port(0), port(1)],
    weights: { 0: mvpColMajor },
    inputs: { 1: [0, 0, 0, 0] },
  };
}

async function runClips(spec, vertices) {
  const { send, stop } = startRunner();
  try {
    const { planFragment } = await import('../lib/spec.mjs');
    const plan = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
    if (!plan.valid) throw new Error('validatePlan failed: ' + JSON.stringify(plan));
    const logicalPorts = spec.ports.map((p, i) => ({ id: `in${i}`, slot: p.slot, source: 'burn-math-skill' }));
    unwrap(await send({ op: 'create', layers: spec.layers, steps: spec.steps, numSlots: spec.numSlots, outputSlot: spec.outputSlot, ports: spec.ports, logicalPorts }));
    unwrap(await send({ op: 'setWeights', layer: 0, values: spec.weights[0] }));
    const port0 = spec.ports[0], port1 = spec.ports[1];
    unwrap(await send({ op: 'bind', slot: 1, values: spec.inputs[1], shape: port1.shape, role: port1.role, layout: port1.layout, source: 'burn-math-skill', revision: 1, fingerprint: '' }));
    const clips = [];
    for (const v of vertices) {
      unwrap(await send({ op: 'clear', slot: 0 }));
      unwrap(await send({ op: 'bind', slot: 0, values: homogenize(v), shape: port0.shape, role: port0.role, layout: port0.layout, source: 'burn-math-skill', revision: 1, fingerprint: '' }));
      const ran = unwrap(await send({ op: 'run' }));
      clips.push(Array.from(ran.values).slice(0, 4));
    }
    return clips;
  } finally {
    try { await stop(); } catch { /* ignore */ }
  }
}

// Host-side tail of the pipeline: perspective divide + viewport.
function projectClip(clip, viewport) {
  const [x, y, z, w] = clip;
  if (!(w > 0)) return { behind: true, clip };
  const nx = x / w, ny = y / w, nz = z / w;
  const sx = (nx * 0.5 + 0.5) * viewport.w;
  const sy = (1 - (ny * 0.5 + 0.5)) * viewport.h;
  return { behind: false, clip, ndc: [nx, ny, nz], screen: [sx, sy] };
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
  return vertices.map((v) => projectClip(R.clip(MVP, v), model.viewport));
}

const rawArgs = process.argv.slice(2);
if (rawArgs.includes('--selftest')) {
  const verts = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) verts.push([sx, sy, sz]);
  verts.push([0, 0, 12]); // behind the camera after the model transform (eye at origin, looking down -z)
  const model = {
    vertices: verts,
    model: [{ op: 'rotate_y', degrees: 30 }, { op: 'translate', by: [0, 0, -8] }],
    camera: { eye: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov_deg: 60, aspect: 4 / 3, near: 0.1, far: 100 },
    viewport: { w: 800, h: 600 },
  };
  const mvp = buildMVP(model);
  const clips = await runClips(specFor(mat4ToColumnMajor(mvp)), verts);
  const got = clips.map((c) => projectClip(c, model.viewport));
  const ref = refProject(model, verts);

  let maxDiff = 0, behindOk = true;
  for (let i = 0; i < verts.length; i++) {
    if (got[i].behind !== ref[i].behind) behindOk = false;
    if (!got[i].behind) maxDiff = Math.max(maxDiff, ...got[i].screen.map((c, j) => Math.abs(c - ref[i].screen[j])));
  }
  // negative control: moved camera must change the picture materially
  const moved = JSON.parse(JSON.stringify(model));
  moved.camera.eye = [2.5, 0, 0];
  const refMoved = refProject(moved, verts);
  let ctrlDiff = 0;
  for (let i = 0; i < verts.length; i++) {
    if (!ref[i].behind && !refMoved[i].behind)
      ctrlDiff = Math.max(ctrlDiff, ...ref[i].screen.map((c, j) => Math.abs(c - refMoved[i].screen[j])));
  }
  const tol = 1e-4;
  // ---- near-plane clipping tests (pure clip-space, Sutherland–Hodgman) ----
  // inside(p) <=> z + w >= 0
  const clipTests = [];
  { // 1. fully inside -> unchanged, 1 triangle
    const r = clipTriangleNear([0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]);
    clipTests.push(['all-inside -> 1 triangle', r.triangles.length === 1 && !r.spansCamera]);
  }
  { // 2. fully outside (z < -w, w > 0) -> 0 triangles
    const r = clipTriangleNear([0, 0, -2, 1], [1, 0, -2, 1], [0, 1, -2, 1]);
    clipTests.push(['all-outside -> 0 triangles', r.triangles.length === 0 && !r.spansCamera]);
  }
  { // 3. straddling (2 in, 1 out) -> 2 triangles, new verts ON the plane
    const r = clipTriangleNear([0, 0, 0.5, 1], [0, 0, -2, 1], [1, 0, 0.5, 1]);
    const onPlane = r.triangles.flat().every((v) => v[2] + v[3] >= -1e-9);
    clipTests.push(['straddle -> 2 triangles on/inside plane', r.triangles.length === 2 && onPlane && !r.spansCamera]);
  }
  { // 4. spans camera (w <= 0) -> flagged, never silently projected
    const r = clipTriangleNear([0, 0, 0, -1], [1, 0, 0, 1], [0, 1, 0, 1]);
    clipTests.push(['w<=0 -> spansCamera flag', r.spansCamera === true && r.triangles.length === 0]);
  }
  const clipOk = clipTests.every(([, ok]) => ok);
  for (const [name, ok] of clipTests) if (!ok) console.error('CLIP FAIL:', name);
  const ok = maxDiff <= tol && behindOk && ctrlDiff > 1 && got[8].behind === true && clipOk;
  printJson({
    ok,
    engine: 'f32 MVP linear(4,4) via burn-math runner',
    reference: 'independent f64 scalar implementation (no geom3 imports)',
    vertices: verts.length, projected: got.filter((g) => !g.behind).length,
    behind_flagged: got.filter((g) => g.behind).length,
    max_abs_screen_px: maxDiff, tol,
    behind_consistent: behindOk,
    negative_control_moved_camera_px: ctrlDiff,
    clip_tests: clipTests.map(([name, ok]) => ({ name, ok })),
    note: ok ? 'engine clip coords (f32) + host-side perspective divide match the independent f64 reference' : 'MISMATCH — see fields',
  });
  process.exit(ok ? 0 : 2);
}

const modelFile = rawArgs[0];
if (!modelFile) { console.error('usage: node geom_project.mjs <model.json> [out.json] [--selftest]'); process.exit(2); }
const outFile = rawArgs.find((a) => a !== modelFile && a.endsWith('.json') && !a.startsWith('--'));
try {
  const model = JSON.parse(fs.readFileSync(modelFile, 'utf8'));
  if (!Array.isArray(model.vertices) || model.vertices.length === 0) throw new Error('model.json needs "vertices": [[x,y,z], ...]');
  if (!model.camera || !model.viewport) throw new Error('model.json needs "camera" and "viewport"');
  const mvp = buildMVP(model);
  const clips = await runClips(specFor(mat4ToColumnMajor(mvp)), model.vertices);
  const results = clips.map((c) => projectClip(c, model.viewport));
  const out = {
    ok: true,
    vertices: model.vertices.length,
    projected: results.filter((r) => !r.behind).length,
    behind_camera: results.filter((r) => r.behind).length,
    results,
  };
  // Optional triangle stage: near-plane clip in clip space, then project.
  // "triangles": [[a,b,c], ...] index into vertices. Backward compatible:
  // absent -> vertex soup only, exactly as before.
  if (Array.isArray(model.triangles)) {
    out.triangles = model.triangles.map(([a, b, c], ti) => {
      for (const idx of [a, b, c])
        if (!(idx >= 0 && idx < clips.length)) throw new Error(`triangle ${ti}: vertex index ${idx} out of range`);
      const { triangles: clipped, spansCamera } = clipTriangleNear(clips[a], clips[b], clips[c]);
      return {
        input: [a, b, c],
        spans_camera: spansCamera,
        clipped_count: clipped.length,
        clipped: clipped.map((tri) => tri.map((v) => projectClip(v, model.viewport))),
      };
    });
    out.triangles_clipped_away = out.triangles.filter((t) => !t.spans_camera && t.clipped_count === 0).length;
    out.triangles_spans_camera = out.triangles.filter((t) => t.spans_camera).length;
  }
  if (outFile) { fs.writeFileSync(outFile, JSON.stringify(out, null, 1)); out.out_file = outFile; }
  printJson(out);
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
