#!/usr/bin/env node
// burn-math geom_build: 3D transform chain, hierarchical scene tree, and inverse kinematics -> engine spec.
//
// Usage:
//   node geom_build.mjs <chain.json> [out.json] [--compose] [--selftest]
//   node geom_build.mjs --scene-tree <tree.json> [out.json]
//   node geom_build.mjs --solve-ik <ik.json> [out.json]
//
// chain.json: {"input": [x,y,z], "chain": [ops...]}
//   ops: {"op":"rotate_x"|"rotate_y"|"rotate_z", "degrees":d}
//        {"op":"rotate_axis", "axis":[x,y,z], "degrees":d}
//        {"op":"euler", "roll":r, "pitch":p, "yaw":y, "order":"ZYX"}
//        {"op":"quaternion", "w":w, "x":x, "y":y, "z":z}
//        {"op":"translate", "by":[tx,ty,tz]}
//        {"op":"scale", "by":[sx,sy,sz]}
//        (plus "matrix" and "repeat" ops via lib/geom3.mjs opToMat4)
//
// Each op becomes one linear(4,4) layer holding its homogeneous 4x4 matrix
// (input homogenized to [x,y,z,1]); a trailing zero-add consumes the second
// required port. With --compose the whole chain fuses into ONE linear layer
// holding the product matrix (prove with: verify.mjs chain.json-spec
// composed-spec).
//
// --selftest checks the builders against known rotations (no engine).
// Prints {ok, spec?|out_file, composed, steps, out_shape,
// reference_point_f64, composed_matrix_rowmajor}.
// Note: output is homogeneous [x,y,z,1]; dehomogenize host-side. Engine has no perspective divide.
import { printJson, printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';
import {
  rotX4, rotY4, rotZ4, rotAxisAngle4, euler4, quatFromEuler, quatToMat4,
  translate4, scale4, mat4Mul, mat4ToColumnMajor, homogenize, composeChain,
  applyMat4, normalMat3, applyMat3, opToMat4, compileSceneTree, solveIK,
} from '../lib/geom3.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node geom_build.mjs <chain.json> [out.json] [--compose] [--selftest] | --scene-tree <tree.json> [out.json] | --solve-ik <ik.json> [out.json]');
  process.exit(2);
}
assertKnownFlags(rawArgs, { valued: ['--scene-tree', '--solve-ik'], boolean: ['--compose', '--selftest'], prog: 'geom_build' });

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
  {
    const M = mat4Mul(translate4(5, 0, 0), mat4Mul(rotZ4(30), scale4(2, 1, 1)));
    // Normal-matrix invariant: under non-uniform scale, inverse-transpose
    // preserves perpendicularity while the naive model-matrix does not.
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
  printJsonAndExit({ ok: pass === tests.length, passed: pass, total: tests.length }, pass === tests.length ? 0 : 2);
}

try {
  if (rawArgs.includes('--scene-tree')) {
    const idx = rawArgs.indexOf('--scene-tree');
    const treeFile = rawArgs[idx + 1];
    if (!treeFile) throw new Error('usage: node geom_build.mjs --scene-tree <tree.json> [out.json]');
    const outFile = rawArgs.find((a) => a.endsWith('.json') && a !== treeFile);
    const tree = JSON.parse(fs.readFileSync(treeFile, 'utf8'));
    const res = compileSceneTree(tree);
    if (outFile) {
      fs.writeFileSync(outFile, JSON.stringify(res.spec, null, 2) + '\n');
      res.out_file = outFile;
    }
    printJsonAndExit(res, 0);
  }

  if (rawArgs.includes('--solve-ik')) {
    const idx = rawArgs.indexOf('--solve-ik');
    const ikFile = rawArgs[idx + 1];
    if (!ikFile) throw new Error('usage: node geom_build.mjs --solve-ik <ik.json> [out.json]');
    const outFile = rawArgs.find((a) => a.endsWith('.json') && a !== ikFile);
    const ikSpec = JSON.parse(fs.readFileSync(ikFile, 'utf8'));
    const res = solveIK(ikSpec);
    if (outFile && res.ok && res.scene_tree?.spec) {
      fs.writeFileSync(outFile, JSON.stringify(res.scene_tree.spec, null, 2) + '\n');
      res.out_file = outFile;
    }
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  const chainFile = rawArgs[0];
  const compose = rawArgs.includes('--compose');
  const outFile = rawArgs.find((a) => a.endsWith('.json') && a !== chainFile);

  const rawDoc = JSON.parse(fs.readFileSync(chainFile, 'utf8'));
  const input = rawDoc.input ?? rawDoc.point;
  const chain = rawDoc.chain;
  if (!Array.isArray(input) || input.length !== 3) throw new Error('chain.json needs "input" (or "point"): [x,y,z]');
  if (!Array.isArray(chain) || !chain.length) throw new Error('chain.json needs a non-empty "chain" array');
  const mats = chain.map(opToMat4);
  const composedMat = composeChain(mats);
  const layers = [], steps = [], weights = {};
  const port = (slot) => ({ slot, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' });

  const built = compose
    ? [{ mat: composedMat, name: 'composed' }]
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
    reference_point_f64: applyMat4(composedMat, input),
    note: 'output is homogeneous [x,y,z,1]; dehomogenize host-side. Engine has no perspective divide.',
    ...(outFile ? {} : { spec }),
  };
  if (outFile) {
    fs.writeFileSync(outFile, JSON.stringify(spec, null, 1));
    result.out_file = outFile;
  }
  // also report the composed matrix (row-major) for inspection
  result.composed_matrix_rowmajor = composedMat;
  printJson(result);
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
