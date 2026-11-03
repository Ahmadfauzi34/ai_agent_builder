#!/usr/bin/env node
// burn-math harness: Unified Agent-Harness Co-Processor Interface.
//
// Usage:
//   node bin/harness.mjs capabilities [compose|verify|differentiate|optimize|rollout|state]
//   node bin/harness.mjs digest <spec.json> [--workspace .burn-harness]
//   node bin/harness.mjs resolve-artifact <burn://sha256:...> [--workspace .burn-harness]
//   node bin/harness.mjs run-workflow <workflow.json | -> [--workspace .burn-harness]
//   node bin/harness.mjs branch-export <spec.json> [--workspace .burn-harness]
//   node bin/harness.mjs branch-merge <baseline.json> <cand1.json> [cand2.json ...] [--tol 1e-4]
//   node bin/harness.mjs workspace-status [--workspace .burn-harness]
import fs from 'node:fs';
import path from 'node:path';
import { loadWasm, withGraphScope } from '../lib/wasm_direct.mjs';
import { portBySlot, hydrateSpecArtifactsInPlace, isRecursiveSpecV2 } from '../lib/spec.mjs';
import { compileRecursiveSpec } from '../lib/composer.mjs';
import {
  getCapabilityCatalog,
  computeSemanticDigest,
  resolveArtifactUri,
  executeHarnessWorkflow,
  exportSubagentBranches,
  mergeSubagentBranches,
  ensureWorkspaceDir,
} from '../lib/harness.mjs';
import { printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node harness.mjs capabilities [domain] | digest <spec.json> | resolve-artifact <uri> | run-workflow <workflow.json|-> | branch-export <spec.json> | branch-merge <base.json> <cand1.json...> | workspace-status');
  process.exit(2);
}

const cmd = rawArgs[0];
function flagVal(flag, defVal) {
  const idx = rawArgs.indexOf(flag);
  if (idx >= 0 && idx + 1 < rawArgs.length) return rawArgs[idx + 1];
  return defVal;
}

const valuedFlags = new Set(['--workspace', '--tol']);
const positional = rawArgs.slice(1).filter((a, i, arr) => {
  if (a.startsWith('--')) return false;
  if (i > 0 && valuedFlags.has(arr[i - 1])) return false;
  return true;
});
assertKnownFlags(rawArgs.slice(1), { valued: [...valuedFlags], prog: `harness ${cmd}` });

const workspaceDir = flagVal('--workspace', '.burn-harness');

function loadV1OrV2(file, wsDir) {
  const baseDir = path.dirname(path.resolve(file));
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  hydrateSpecArtifactsInPlace(raw, baseDir, wsDir);
  if (isRecursiveSpecV2(raw)) {
    const compiled = compileRecursiveSpec(raw, { baseDir, autoRepair: true });
    hydrateSpecArtifactsInPlace(compiled, baseDir, wsDir);
    return compiled;
  }
  return raw;
}

try {
  if (cmd === 'capabilities') {
    const domain = positional[0] ?? null;
    const cat = getCapabilityCatalog(domain);
    printJsonAndExit(cat, cat.ok ? 0 : 2);
  }

  if (cmd === 'workspace-status') {
    const { stateFile } = ensureWorkspaceDir(workspaceDir);
    const st = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
    printJsonAndExit({ ok: true, workspace: path.resolve(workspaceDir), state: st }, 0);
  }

  if (cmd === 'resolve-artifact') {
    const uri = positional[0];
    if (!uri) {
      console.error('usage: node harness.mjs resolve-artifact <burn://sha256:...> [--workspace .burn-harness]');
      process.exit(2);
    }
    const rec = resolveArtifactUri(uri, workspaceDir);
    printJsonAndExit({ ok: true, uri, ...rec }, 0);
  }

  if (cmd === 'digest') {
    const specFile = positional[0];
    if (!specFile) {
      console.error('usage: node harness.mjs digest <spec.json> [--workspace .burn-harness]');
      process.exit(2);
    }
    const rt = await loadWasm();
    const spec = loadV1OrV2(specFile, workspaceDir);
    const out = await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
      const b = bundle(spec.inputs, portBySlot(spec));
      const t = g.run(reg, b);
      const values = Array.from(t.to_array());
      const shape = Array.from(t.shape());
      t.free();
      return { shape, values };
    });
    const digest = computeSemanticDigest(out.values, out.shape, {
      workspaceDir,
      persistThreshold: 1,
      includeFullValues: false,
    });
    printJsonAndExit({
      ok: true,
      mode: 'semantic_digest',
      digest,
      next_actions: [
        `Fetch full values if needed: node bin/harness.mjs resolve-artifact ${digest.artifact_ref}`,
        `Search graph optimizations: node bin/recurse.mjs search ${specFile}`,
      ],
    }, 0);
  }

  if (cmd === 'run-workflow') {
    const wfSource = positional[0] ?? '-';
    const rawWf = wfSource === '-'
      ? fs.readFileSync(0, 'utf8')
      : fs.readFileSync(wfSource, 'utf8');
    const wf = JSON.parse(rawWf);
    if (!wf.workspace && rawArgs.includes('--workspace')) wf.workspace = workspaceDir;
    const baseDir = wfSource === '-' ? process.cwd() : path.dirname(path.resolve(wfSource));
    const rt = await loadWasm();
    const res = await executeHarnessWorkflow(rt, wf, { baseDir });
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  if (cmd === 'branch-export') {
    const specFile = positional[0];
    if (!specFile) {
      console.error('usage: node harness.mjs branch-export <spec.json> [--workspace .burn-harness]');
      process.exit(2);
    }
    const rt = await loadWasm();
    const spec = loadV1OrV2(specFile, workspaceDir);
    const res = exportSubagentBranches(rt, spec, { workspaceDir });
    printJsonAndExit(res, 0);
  }

  if (cmd === 'branch-merge') {
    const [baseFile, ...candFiles] = positional;
    if (!baseFile || candFiles.length === 0) {
      console.error('usage: node harness.mjs branch-merge <baseline.json> <cand1.json> [cand2.json ...] [--tol 1e-4]');
      process.exit(2);
    }
    const tol = Number(flagVal('--tol', 1e-4));
    const rt = await loadWasm();
    const baseSpec = loadV1OrV2(baseFile, workspaceDir);
    const res = mergeSubagentBranches(rt, baseSpec, candFiles, { absTol: tol, relTol: tol });
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  printJsonAndExit({
    ok: false,
    error: `unknown harness command '${cmd}' (use: capabilities | digest | resolve-artifact | run-workflow | branch-export | branch-merge | workspace-status)`,
  }, 2);
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  printJsonAndExit({ ok: false, code: e?.code ?? 'harness_error', error: msg }, 2);
}
