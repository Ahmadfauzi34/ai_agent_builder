#!/usr/bin/env node
// burn-math compose: Recursive Spec v2 compiler, verifier, and symbolic backward graph generator.
//
// Usage:
//   node bin/compose.mjs compile <spec-v2.json> [out-v1.json] [--auto-repair] [--compact-slots] [--workspace]
//   node bin/compose.mjs prove <spec-v2.json> [reference-v1.json] [--auto-repair] [--compact-slots]
//   node bin/compose.mjs grad-spec <spec.json> [out-grad.json] [--wrt <slot>] [--order 1|2] [--dout <v1,v2,...>] [--hvp-vec <v1,v2,...>]
import fs from 'node:fs';
import path from 'node:path';
import { compileRecursiveSpec, compileBackwardSpec } from '../lib/composer.mjs';
import { loadWasm, withGraphScope } from '../lib/wasm_direct.mjs';
import { loadSpec, portBySlot, hydrateSpecArtifactsInPlace, isRecursiveSpecV2 } from '../lib/spec.mjs';
import { printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node compose.mjs compile <spec-v2.json> [out.json] [--auto-repair] [--compact-slots] [--workspace] | prove <spec-v2.json> [ref.json] | grad-spec <spec.json> [out.json] [--wrt <slot>] [--order 1|2] [--dout <v1,v2,...>] [--hvp-vec <v1,v2,...>]');
  process.exit(2);
}

const cmd = rawArgs[0];
const autoRepair = rawArgs.includes('--auto-repair');
const compactSlots = rawArgs.includes('--compact-slots');
const useWorkspace = rawArgs.includes('--workspace');

function flagVal(flag, defVal) {
  const idx = rawArgs.indexOf(flag);
  if (idx >= 0 && idx + 1 < rawArgs.length) return rawArgs[idx + 1];
  return defVal;
}

const VALUE_FLAGS = new Set(['--wrt', '--order', '--dout', '--hvp-vec']);
const positional = rawArgs.slice(1).filter((a, i, arr) => {
  if (a.startsWith('--')) return false;
  if (i > 0 && VALUE_FLAGS.has(arr[i - 1])) return false;
  return true;
});
assertKnownFlags(rawArgs.slice(1), { valued: [...VALUE_FLAGS], boolean: ['--auto-repair', '--compact-slots', '--workspace'], prog: `compose ${cmd}` });

try {
  if (cmd === 'compile') {
    const inFile = positional[0];
    const outFile = positional[1] ?? null;
    if (!inFile) {
      console.error('usage: node compose.mjs compile <spec-v2.json> [out.json] [--auto-repair] [--compact-slots] [--workspace]');
      process.exit(2);
    }
    const baseDir = path.dirname(path.resolve(inFile));
    const raw = JSON.parse(fs.readFileSync(inFile, 'utf8'));
    hydrateSpecArtifactsInPlace(raw, baseDir);
    const rt = useWorkspace ? await loadWasm() : null;
    const compiled = compileRecursiveSpec(raw, {
      baseDir,
      rt,
      useWorkspace,
      autoRepair,
      compactSlots,
      validate: true,
    });
    if (outFile) {
      const { meta: _m, ...v1Only } = compiled;
      fs.writeFileSync(outFile, JSON.stringify(v1Only, null, 2) + '\n');
    }
    printJsonAndExit({
      ok: true,
      mode: 'compile',
      out_file: outFile,
      layers_count: compiled.layers.length,
      steps_count: compiled.steps.length,
      num_slots: compiled.numSlots,
      output_slot: compiled.outputSlot,
      meta: compiled.meta,
      spec: outFile ? undefined : compiled,
    }, 0);
  }

  if (cmd === 'prove') {
    const inFile = positional[0];
    const refFile = positional[1] ?? null;
    if (!inFile) {
      console.error('usage: node compose.mjs prove <spec-v2.json> [ref.json] [--auto-repair] [--compact-slots]');
      process.exit(2);
    }
    const baseDir = path.dirname(path.resolve(inFile));
    const raw = JSON.parse(fs.readFileSync(inFile, 'utf8'));
    hydrateSpecArtifactsInPlace(raw, baseDir);
    const rt = await loadWasm();
    const compiled = compileRecursiveSpec(raw, {
      baseDir,
      rt,
      useWorkspace: true,
      autoRepair,
      compactSlots,
      validate: true,
    });

    const { values: outValues, shape: outShape } = await withGraphScope(rt, compiled, async ({ g, reg, bundle }) => {
      const b = bundle(compiled.inputs, portBySlot(compiled));
      const out = g.run(reg, b);
      const vals = Array.from(out.to_array());
      const sh = Array.from(out.shape());
      out.free();
      return { values: vals, shape: sh };
    });

    let refComparison = null;
    let verdict = outValues.every((v) => Number.isFinite(v)) ? 'ACCEPT' : 'REJECT';
    if (refFile) {
      const refSpec = loadSpec(refFile);
      const { values: refValues, shape: refShape } = await withGraphScope(rt, refSpec, async ({ g, reg, bundle }) => {
        const b = bundle(refSpec.inputs, portBySlot(refSpec));
        const out = g.run(reg, b);
        const vals = Array.from(out.to_array());
        const sh = Array.from(out.shape());
        out.free();
        return { values: vals, shape: sh };
      });
      const shapeEqual = JSON.stringify(outShape) === JSON.stringify(refShape);
      let maxAbs = 0;
      if (!shapeEqual || outValues.length !== refValues.length) {
        verdict = 'REJECT';
        maxAbs = Infinity;
      } else {
        for (let i = 0; i < outValues.length; i++) {
          maxAbs = Math.max(maxAbs, Math.abs(outValues[i] - refValues[i]));
        }
        if (maxAbs > 1e-6) verdict = 'REJECT';
      }
      refComparison = {
        equivalent: verdict === 'ACCEPT',
        shape_equal: shapeEqual,
        compiled_shape: outShape,
        reference_shape: refShape,
        max_abs_error: maxAbs,
        reference_outputs: refValues,
      };
    }

    printJsonAndExit({
      ok: verdict === 'ACCEPT',
      verdict,
      proof_mode: refFile ? 'reference_equivalence' : 'compile_and_execute_smoke',
      outputs: { shape: outShape, values: outValues },
      reference_comparison: refComparison,
      meta: compiled.meta,
      spec: compiled,
    }, verdict === 'ACCEPT' ? 0 : 2);
  }

  if (cmd === 'grad-spec') {
    const inFile = positional[0];
    const outFile = positional[1] ?? null;
    if (!inFile) {
      console.error('usage: node compose.mjs grad-spec <spec.json> [out.json] [--wrt <slot>] [--order 1|2] [--dout <v1,v2,...>] [--hvp-vec <v1,v2,...>]');
      process.exit(2);
    }
    const rawWrt = flagVal('--wrt', '0');
    const wrtSlot = /^(?:0|[1-9]\d*)$/.test(String(rawWrt)) ? Number(rawWrt) : String(rawWrt);
    const order = Number(flagVal('--order', 1));
    if (order !== 1 && order !== 2) {
      printJsonAndExit({ ok: false, error: '--order must be 1 or 2' }, 2);
    }
    const doutRaw = flagVal('--dout', null);
    const hvpRaw = flagVal('--hvp-vec', null);
    const parseVecOpt = (s, flagName) => {
      if (s === null) return null;
      const arr = String(s).split(',').map((x) => Number(x.trim()));
      if (arr.length === 0 || arr.some((v) => !Number.isFinite(v))) {
        throw new Error(`${flagName} must be a comma-separated list of finite numbers`);
      }
      return arr;
    };
    const doutValues = parseVecOpt(doutRaw, '--dout');
    const hvpVecValues = parseVecOpt(hvpRaw, '--hvp-vec');

    const baseDir = path.dirname(path.resolve(inFile));
    const raw = JSON.parse(fs.readFileSync(inFile, 'utf8'));
    hydrateSpecArtifactsInPlace(raw, baseDir);
    const baseSpec = isRecursiveSpecV2(raw)
      ? compileRecursiveSpec(raw, { baseDir, autoRepair: true })
      : loadSpec(inFile);

    let gradSpec = compileBackwardSpec(baseSpec, { wrtSlot, doutValues });
    const resolvedWrtSlot = gradSpec.meta.wrt_slot;
    if (order === 2) {
      gradSpec = compileBackwardSpec(gradSpec, { wrtSlot: resolvedWrtSlot, doutValues: hvpVecValues });
      gradSpec.meta.second_order_semantics = hvpVecValues
        ? 'hessian_vector_product_custom_v'
        : 'hessian_vector_product_ones (H * 1; equals Hessian diagonal iff Hessian is diagonal/elementwise)';
    }
    if (outFile) {
      const { meta: _m, ...v1Only } = gradSpec;
      fs.writeFileSync(outFile, JSON.stringify(v1Only, null, 2) + '\n');
    }
    const rt = await loadWasm();
    const gradValues = await withGraphScope(rt, gradSpec, async ({ g, reg, bundle }) => {
      const b = bundle(gradSpec.inputs, portBySlot(gradSpec));
      const out = g.run(reg, b);
      const vals = Array.from(out.to_array());
      out.free();
      return vals;
    });
    printJsonAndExit({
      ok: true,
      order,
      wrt_slot: resolvedWrtSlot,
      ...(order === 2 ? { second_order_semantics: gradSpec.meta.second_order_semantics } : {}),
      grad_values: gradValues,
      out_file: outFile,
      meta: gradSpec.meta,
      spec: outFile ? undefined : gradSpec,
    }, 0);
  }

  printJsonAndExit({ ok: false, error: `unknown subcommand '${cmd}' (use: compile | prove | grad-spec)` }, 2);
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  const code = e?.code ? 2 : 1;
  printJsonAndExit({
    ok: false,
    code: e?.code ?? 'compose_error',
    error: msg,
    remediation_hint: e?.remediation_hint ?? null,
  }, code);
}
