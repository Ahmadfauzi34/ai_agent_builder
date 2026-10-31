#!/usr/bin/env node
// burn-math weight_lab: the manual-weighting workbench.
//
//   run <weights.json> --in 80,80
//       Execute v_out = W·v + b through the engine (f32); compare vs JS f64.
//   gate <current.json> <candidate.json> [--jump 1.0]
//       Adaptation gate: ACCEPT iff (1) the candidate passes its kind's
//       property checks (stability / orthonormality / ...) AND (2) the
//       behavior jump on probes stays within --jump (absolute output units).
//       Exit 0 = ACCEPT, 2 = REJECT. This is the "valid transition" check
//       an adaptive system runs BEFORE installing new weights.
//   certify <weights.json> [--repertoire <path>]
//       Full battery (properties + engine probes + fingerprint) then append
//       to the repertoire. Certified sets are the only ones an adaptive
//       policy may install live.
//   compose <A.json> <B.json> [--out C.json]
//       B-after-A affine fusion (W_C = W_B W_A, b_C = W_B b_A + b_B) with a
//       PROPERTY certificate: contraction composes via the Frobenius
//       norm bound (||W_C||_F <= ||W_A||_F||W_B||_F < 1, proven without
//       measuring the product); orthonormality/det compose by the product
//       identity; otherwise the product is measured directly (a refused
//       bound is not a rejection). Exit 0 = COMPOSED, 2 = REJECTED.
//   --selftest
//       Engine execution vs the HTML thesis example's elastic preset,
//       property checks incl. a deliberately unstable W (must FAIL),
//       gate ACCEPT (identical set) and gate REJECT (perturbed candidate).
//
// weights.json:
//   {"name":"elastic","kind":"map|dynamics|calibration",
//    "W":[[1.2,-0.4],[0.5,0.8]],"b":[20,-10],
//    "probes":[{"in":[80,80],"want":[84,94]}],
//    "requires":{"spectral_radius_lt":1}}
import { startRunner, unwrap, printJson, failCli } from '../lib/runner.mjs';
import { planFragment } from '../lib/spec.mjs';
import {
  matVecMul, affineSpec, spectralRadius, checkProperties, weightFingerprint,
  composeProperties,
} from '../lib/weights.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_REPERTOIRE = path.join(HERE, '..', 'references', 'weight_repertoire.v1.json');
const BIND = { role: 'observation', layout: 'feature_axis1_singleton', source: 'burn-math-skill', revision: 1, fingerprint: '' };

function loadSet(f) {
  const s = JSON.parse(fs.readFileSync(f, 'utf8'));
  if (!Array.isArray(s.W) || !Array.isArray(s.b)) throw new Error(`${f}: need "W" (row-major) and "b"`);
  s.kind ??= 'map';
  s.probes ??= [];
  return s;
}

// One runner session; install W/b once, run all probe inputs. Returns
// engine outputs in probe order.
async function engineProbe(W, b, inputs) {
  const n = W.length;
  const spec = affineSpec(W, b, inputs[0]);
  const { send, stop } = startRunner();
  try {
    const plan = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
    if (!plan.valid) throw new Error('validatePlan failed: ' + JSON.stringify(plan));
    const logicalPorts = spec.ports.map((p, i) => ({ id: `in${i}`, slot: p.slot, source: 'burn-math-skill' }));
    unwrap(await send({ op: 'create', layers: spec.layers, steps: spec.steps, numSlots: spec.numSlots, outputSlot: spec.outputSlot, ports: spec.ports, logicalPorts }));
    unwrap(await send({ op: 'setWeights', layer: 0, values: spec.weights[0] }));
    unwrap(await send({ op: 'bind', slot: 1, values: b, shape: [1, n, 1, 1], ...BIND }));
    const outs = [];
    for (const v of inputs) {
      unwrap(await send({ op: 'clear', slot: 0 }));
      unwrap(await send({ op: 'bind', slot: 0, values: v, shape: [1, n, 1, 1], ...BIND }));
      outs.push(Array.from(unwrap(await send({ op: 'run' })).values).slice(0, n));
    }
    return outs;
  } finally {
    try { await stop(); } catch { /* ignore */ }
  }
}

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node weight_lab.mjs (run|gate|certify|compose) ... | --selftest');
  process.exit(2);
}
const cmd = rawArgs[0];

if (cmd === '--selftest' || rawArgs.includes('--selftest')) {
  const results = [];
  const t = (name, ok, extra = {}) => { results.push({ name, ok, ...extra }); if (!ok) console.error('FAIL:', name, extra); };
  // 1. engine executes the HTML thesis example's elastic preset exactly
  const elastic = { W: [[1.2, -0.4], [0.5, 0.8]], b: [20, -10] };
  const [eng] = await engineProbe(elastic.W, elastic.b, [[80, 80]]);
  const js = matVecMul(elastic.W, [80, 80]).map((x, i) => x + elastic.b[i]);
  t('engine elastic [80,80] = [84,94]', Math.abs(eng[0] - 84) < 1e-6 && Math.abs(eng[1] - 94) < 1e-6, { engine: eng, js });
  // 2. property checks: stable contraction passes, unstable W is caught
  const inv = { kind: 'dynamics', W: [[-0.8, 0], [0, -0.8]], requires: {} };
  const invProps = checkProperties(inv);
  t('inverted preset: spectral radius 0.8 < 1', invProps.length === 1 && invProps[0].ok, { rho: invProps[0]?.value });
  const unstable = { kind: 'dynamics', W: [[2, 0], [0, 0.5]], requires: {} };
  const unProps = checkProperties(unstable);
  t('unstable W (rho=2) is flagged', unProps.length === 1 && !unProps[0].ok, { rho: unProps[0]?.value });
  // 3. gate ACCEPT: identical set
  const cur = { name: 'elastic', kind: 'map', W: elastic.W, b: elastic.b, probes: [{ in: [80, 80], want: [84, 94] }] };
  const gateAccept = await gateSets(cur, cur, 1.0);
  t('gate identical set -> ACCEPT', gateAccept.verdict === 'ACCEPT', gateAccept);
  // 4. gate REJECT: perturbed candidate jumps 400 on the probe
  const pert = JSON.parse(JSON.stringify(cur));
  pert.W[0][0] += 5;
  const gateReject = await gateSets(cur, pert, 1.0);
  t('gate perturbed set -> REJECT', gateReject.verdict === 'REJECT', gateReject);
  // 5. compose norm-bound: 0.5I then 0.6I — contraction certified by the
  //    Frobenius bound, product never power-iterated for the verdict
  const dyn = (s, b) => ({ name: `d${s}`, kind: 'dynamics', W: [[s, 0], [0, s]], b, requires: {} });
  const cNorm = composeProperties(dyn(0.5, [1, 2]), dyn(0.6, [3, 4]));
  const wOk = Math.abs(cNorm.W[0][0] - 0.3) < 1e-12 && Math.abs(cNorm.W[1][1] - 0.3) < 1e-12
    && Math.abs(cNorm.W[0][1]) < 1e-12 && Math.abs(cNorm.W[1][0]) < 1e-12;
  const bOk = Math.abs(cNorm.b[0] - 3.6) < 1e-12 && Math.abs(cNorm.b[1] - 5.2) < 1e-12;
  t('compose 0.5I+0.6I -> norm_bound COMPOSED, W=0.3I, b=[3.6,5.2]',
    cNorm.verdict === 'COMPOSED' && cNorm.method === 'norm_bound' && wOk && bOk, cNorm.certificate);
  // 6. compose orthonormal: Rz(30) then Rz(15) = Rz(45); engine fidelity —
  //    one fused run equals two sequential runs (meaning preserved)
  const d2r = Math.PI / 180;
  const rz = (deg, nm) => {
    const c = Math.cos(deg * d2r), s = Math.sin(deg * d2r);
    return { name: nm, kind: 'calibration', W: [[c, -s], [s, c]], b: [0, 0], requires: { det_positive: true } };
  };
  const cRot = composeProperties(rz(30, 'r30'), rz(15, 'r15'));
  const angOk = Math.abs(cRot.W[0][0] - Math.SQRT1_2) < 1e-12 && Math.abs(cRot.W[0][1] + Math.SQRT1_2) < 1e-12;
  const [fused] = await engineProbe(cRot.W, cRot.b, [[1, 0]]);
  const [mid] = await engineProbe(rz(30).W, [0, 0], [[1, 0]]);
  const [seq] = await engineProbe(rz(15).W, [0, 0], [mid]);
  const fuseDiff = Math.max(Math.abs(fused[0] - seq[0]), Math.abs(fused[1] - seq[1]));
  t('compose Rz30+Rz15 -> orthonormal COMPOSED = Rz45, fused==sequential',
    cRot.verdict === 'COMPOSED' && cRot.method === 'orthonormal' && angOk && fuseDiff <= 1e-4,
    { det: cRot.certificate.det_product, fuseDiff });
  // 7. honest fallback: inverted(-0.8I) twice — rho=0.8<1 each, but the norm
  //    bound (1.28 >= 1) refuses; direct measurement accepts (rho=0.64).
  //    Bound refusal is not a rejection.
  const cInv = composeProperties(
    { name: 'inv1', kind: 'dynamics', W: [[-0.8, 0], [0, -0.8]], b: [0, 0], requires: {} },
    { name: 'inv2', kind: 'dynamics', W: [[-0.8, 0], [0, -0.8]], b: [0, 0], requires: {} });
  t('compose inverted^2 -> bound refuses, measured ACCEPTS (rho=0.64)',
    cInv.verdict === 'COMPOSED' && cInv.method === 'measured', cInv.certificate);
  // 8. negative: unstable factor (2I) — bound refuses AND measured rho=1.0
  //    is not < 1 -> REJECTED, with numbers.
  const cBad = composeProperties(dyn(0.5, [0, 0]), { name: 'unst', kind: 'dynamics', W: [[2, 0], [0, 2]], b: [0, 0], requires: {} });
  t('compose 0.5I+2I -> REJECTED (rho_product=1.0 not < 1)',
    cBad.verdict === 'REJECTED' && cBad.method === 'measured', cBad.certificate);
  const ok = results.every((r) => r.ok);
  printJson({ ok, tests: results });
  process.exit(ok ? 0 : 2);
}

async function gateSets(cur, cand, jumpTol) {
  const propFails = checkProperties(cand).filter((p) => !p.ok);
  const probeIns = [...new Map([...(cur.probes ?? []), ...(cand.probes ?? [])].map((p) => [p.in.join(','), p.in])).values()];
  const inputs = probeIns.length ? probeIns : [[...cand.b.map(() => 0)]];
  const [outCur, outCand] = await Promise.all([
    engineProbe(cur.W, cur.b, inputs),
    engineProbe(cand.W, cand.b, inputs),
  ]);
  let maxJump = 0;
  for (let i = 0; i < inputs.length; i++)
    for (let k = 0; k < outCur[i].length; k++)
      maxJump = Math.max(maxJump, Math.abs(outCand[i][k] - outCur[i][k]));
  const jumpOk = maxJump <= jumpTol;
  const verdict = propFails.length === 0 && jumpOk ? 'ACCEPT' : 'REJECT';
  return {
    verdict,
    properties_ok: propFails.length === 0,
    property_failures: propFails,
    max_jump: maxJump, jump_tol: jumpTol, jump_ok: jumpOk,
    probes: inputs.length,
  };
}

try {
  if (cmd === 'run') {
    const f = rawArgs[1];
    const inFlag = rawArgs.indexOf('--in');
    if (!f || inFlag < 0) throw new Error('usage: weight_lab.mjs run <weights.json> --in 80,80');
    const set = loadSet(f);
    const v = rawArgs[inFlag + 1].split(',').map(Number);
    const [eng] = await engineProbe(set.W, set.b, [v]);
    const js = matVecMul(set.W, v).map((x, i) => x + set.b[i]);
    const diff = Math.max(...eng.map((x, i) => Math.abs(x - js[i])));
    printJson({ ok: true, name: set.name ?? null, kind: set.kind, engine_f32: eng, js_f64: js, max_abs_diff: diff });
    process.exit(0);
  }
  if (cmd === 'gate') {
    const [curF, candF] = [rawArgs[1], rawArgs[2]];
    const jFlag = rawArgs.indexOf('--jump');
    if (!curF || !candF) throw new Error('usage: weight_lab.mjs gate <current.json> <candidate.json> [--jump 1.0]');
    const res = await gateSets(loadSet(curF), loadSet(candF), jFlag >= 0 ? Number(rawArgs[jFlag + 1]) : 1.0);
    printJson({ ok: true, ...res });
    process.exit(res.verdict === 'ACCEPT' ? 0 : 2);
  }
  if (cmd === 'certify') {
    const f = rawArgs[1];
    const rFlag = rawArgs.indexOf('--repertoire');
    if (!f) throw new Error('usage: weight_lab.mjs certify <weights.json> [--repertoire <path>]');
    const set = loadSet(f);
    const props = checkProperties(set);
    if (props.some((p) => !p.ok)) {
      printJson({ ok: false, verdict: 'REFUSED', reason: 'property checks failed', properties: props });
      process.exit(2);
    }
    const probeIns = (set.probes ?? []).map((p) => p.in);
    let probeMaxErr = 0;
    if (probeIns.length) {
      const outs = await engineProbe(set.W, set.b, probeIns);
      set.probes.forEach((p, i) => {
        const js = matVecMul(set.W, p.in).map((x, k) => x + set.b[k]);
        probeMaxErr = Math.max(probeMaxErr, ...outs[i].map((x, k) => Math.abs(x - (p.want?.[k] ?? js[k]))));
      });
    }
    const entry = {
      name: set.name ?? path.basename(f, '.json'),
      kind: set.kind,
      W: set.W, b: set.b,
      fingerprint: weightFingerprint(set.W, set.b),
      properties: props,
      probe_max_abs_err: probeMaxErr,
      probes: (set.probes ?? []).length,
      engine: 'f32 burn-math (audited 2d8f42d)',
      certified_at: new Date().toISOString(),
    };
    const repPath = rFlag >= 0 ? rawArgs[rFlag + 1] : DEFAULT_REPERTOIRE;
    let rep = { version: 1, sets: [] };
    if (fs.existsSync(repPath)) rep = JSON.parse(fs.readFileSync(repPath, 'utf8'));
    rep.sets = [...rep.sets.filter((e) => e.name !== entry.name), entry];
    fs.writeFileSync(repPath, JSON.stringify(rep, null, 1) + '\n');
    printJson({ ok: true, verdict: 'CERTIFIED', entry: { ...entry, W: '…', b: '…' }, repertoire: repPath, sets: rep.sets.length });
    process.exit(0);
  }
  if (cmd === 'compose') {
    const [aF, bF] = [rawArgs[1], rawArgs[2]];
    const oFlag = rawArgs.indexOf('--out');
    if (!aF || !bF) throw new Error('usage: weight_lab.mjs compose <A.json> <B.json> [--out C.json]  (B after A)');
    const res = composeProperties(loadSet(aF), loadSet(bF));
    if (oFlag >= 0 && res.verdict === 'COMPOSED') {
      fs.writeFileSync(rawArgs[oFlag + 1], JSON.stringify({
        name: `${path.basename(aF, '.json')}_then_${path.basename(bF, '.json')}`,
        kind: res.kind, W: res.W, b: res.b,
        composed_from: [path.basename(aF), path.basename(bF)],
        composition_method: res.method,
      }, null, 1) + '\n');
    }
    printJson({ ok: true, ...res, out: oFlag >= 0 ? rawArgs[oFlag + 1] : null });
    process.exit(res.verdict === 'COMPOSED' ? 0 : 2);
  }
  throw new Error('usage: weight_lab.mjs (run|gate|certify|compose) ... | --selftest');
} catch (e) {
  failCli(e.message, e.envelope ?? null, 2);
}
