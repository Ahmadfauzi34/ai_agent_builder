#!/usr/bin/env node
// burn-math geom_build: 3D transform chain -> engine spec.
//
// Usage: node geom_build.mjs <chain.json> [out.json] [--compose] [--selftest]
//
// chain.json: {"input": [x,y,z], "chain": [ops...]}
//   ops: {"op":"rotate_x"|"rotate_y"|"rotate_z", "degrees":d}
//        {"op":"rotate_axis", "axis":[x,y,z], "degrees":d}
//        {"op":"euler", "roll":r, "pitch":p, "yaw":y, "order":"ZYX"}
//        {"op":"quaternion", "w":w, "x":x, "y":y, "z":z}
//        {"op":"translate", "by":[tx,ty,tz]}
//        {"op":"scale", "by":[sx,sy,sz]}
//
// Each op becomes one linear(4,4) layer holding its homogeneous 4x4 matrix
// (input homogenized to [x,y,z,1]); a trailing zero-add consumes the second
// required port. With --compose the whole chain fuses into ONE linear layer
// holding the product matrix (prove with: verify.mjs chain.json-spec
// composed-spec).
//
// --selftest checks the builders against known rotations (no engine).
// Prints {ok, spec?|out_file, chain_matrices, composed_matrix}.
import { loadSpec } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import {
  rotX4, rotY4, rotZ4, rotAxisAngle4, euler4, quatFromEuler, quatToMat4,
  translate4, scale4, mat4Mul, mat4ToColumnMajor, homogenize, composeChain,
  applyMat4, mat4Identity, normalMat3, applyMat3,
} from '../lib/geom3.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
if (rawArgs.includes('--selftest')) {
  const close = (a, b, tol = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < tol);
  const tests = [
    ['Rz(90)·[1,0,0] = [0,1,0]', applyMat4(rotZ4(90), [1, 0, 0]), [0, 1, 0]],
    ['Rx(90)·[0,1,0] = [0,0,1]', applyMat4(rotX4(90), [0, 1, 0]), [0, 0, 1]],
    ['Ry(90)·[0,0,1] = [1,0,0]', applyMat4(rotY4(90), [0, 0, 1]), [1, 0, 0]],
    ['Rz(45)²·[1,0,0] = Rz(90)·[1,0,0]',
      applyMat4(mat4Mul(rotZ4(45), rotZ4(45)), [1, 0, 0]), applyMat4(rotZ4(90), [1, 0, 0])],
    ['translate·[1,2,3] + [10,0,0]', applyMat4(translate4(10, 0, 0), [1, 2, 3]), [11, 2, 3]],
    ['scale·[1,2,3]', applyMat4(scale4(2, 3, 4), [1, 2, 3]), [2, 6, 12]],
    ['quat(Rz90)·[1,0,0] = [0,1,0]',
      applyMat4(quatToMat4(...quatFromEuler(0, 0, 90)), [1, 0, 0]), [0, 1, 0]],
    ['euler(10,20,30) = quat(euler(10,20,30))',
      applyMat4(euler4(10, 20, 30), [1, 2, 3]),
      applyMat4(quatToMat4(...quatFromEuler(10, 20, 30)), [1, 2, 3])],
    ['rotAxisAngle z 90 = rotZ 90',
      applyMat4(rotAxisAngle4(0, 0, 5, 90), [1, 0, 0]), [0, 1, 0]],
  ];
  // Normal-matrix invariant: under non-uniform scale, inverse-transpose
  // preserves perpendicularity while the naive model-matrix does not.
  {
    const M = mat4Mul(translate4(5, 0, 0), mat4Mul(rotZ4(30), scale4(2, 1, 1)));
    const dir = (m4, [x, y, z]) => [ // direction transform (w=0, no divide)
      m4[0] * x + m4[1] * y + m4[2] * z,
      m4[4] * x + m4[5] * y + m4[6] * z,
      m4[8] * x + m4[9] * y + m4[10] * z,
    ];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const t = [Math.SQRT1_2, Math.SQRT1_2, 0], n = [-Math.SQRT1_2, Math.SQRT1_2, 0]; // perpendicular, diagonal to scale axes
    const Mt = dir(M, t);
    const Nn = applyMat3(normalMat3(M), n);
    const naive = dir(M, n);
    tests.push(
      ['normalMat3 preserves perpendicularity', [dot(Mt, Nn)], [0]],
      ['naive model-matrix does NOT (control)', [Math.abs(dot(Mt, naive)) > 0.1 ? 1 : 0], [1]],
    );
  }
  let pass = 0;
  for (const [name, got, want] of tests) {
    const ok = close(got, want);
    if (ok) pass++;
    else console.error('FAIL:', name, 'got', got, 'want', want);
  }
  printJson({ ok: pass === tests.length, passed: pass, total: tests.length });
  process.exit(pass === tests.length ? 0 : 2);
}

const chainFile = rawArgs[0];
if (!chainFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node geom_build.mjs <chain.json> [out.json] [--compose] [--selftest]');
  process.exit(2);
}
const compose = rawArgs.includes('--compose');
const outFile = rawArgs.find((a) => a.endsWith('.json') && a !== chainFile);

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
    default: throw new Error(`unknown op '${op.op}'`);
  }
}

try {
  const { input, chain } = JSON.parse(fs.readFileSync(chainFile, 'utf8'));
  if (!Array.isArray(input) || input.length !== 3) throw new Error('chain.json needs "input": [x,y,z]');
  if (!Array.isArray(chain) || chain.length === 0) throw new Error('chain.json needs a non-empty "chain" array');
  const mats = chain.map(opToMat4);
  const layers = [], steps = [], weights = {};
  const port = (slot) => ({ slot, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' });

  const built = compose
    ? [{ mat: composeChain(mats), name: 'composed' }]
    : mats.map((mat, i) => ({ mat, name: `chain[${i}]` }));

  built.forEach(({ mat }, i) => {
    const id = 100 + i; // registry ids clear of typical small-id specs
    layers.push({ constructor: 'linear', args: [id, 4, 4, false] });
    weights[String(i)] = mat4ToColumnMajor(mat);
  });
  // chain the linears: slot0 -> 2 -> 3 -> ...
  let prev = 0, next = 2;
  built.forEach((_, i) => {
    steps.push({ kind: 'unary', layer: i, slots: [prev, next] });
    prev = next; next++;
  });
  // trailing zero-add consumes the second required port (adds nothing)
  const addIdx = built.length;
  layers.push({ constructor: 'add', args: [200] });
  steps.push({ kind: 'binary', layer: addIdx, slots: [prev, 1, next] });

  const spec = {
    numSlots: next + 1,
    layers, steps,
    outputSlot: next,
    ports: [port(0), port(1)],
    weights,
    inputs: { 0: homogenize(input), 1: [0, 0, 0, 0] },
  };
  const result = {
    ok: true,
    composed: compose,
    steps: built.length,
    out_shape: [1, 4, 1, 1],
    note: 'output is homogeneous [x,y,z,1]; dehomogenize host-side. Engine has no perspective divide.',
    ...(outFile ? {} : { spec }),
  };
  if (outFile) {
    fs.writeFileSync(outFile, JSON.stringify(spec, null, 1));
    result.out_file = outFile;
  }
  // also report the composed matrix (row-major) for inspection
  result.composed_matrix_rowmajor = composeChain(mats);
  printJson(result);
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
