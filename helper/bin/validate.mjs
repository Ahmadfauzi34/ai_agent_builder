#!/usr/bin/env node
// burn-math validate: session-free plan check against the audited engine.
// Usage: node validate.mjs <spec.json>
// Exit 0 + {"ok":true,"valid":true,...} when the plan is sound;
// exit 2 + the structured analysis (code plan_minimum_ports / plan_unused_ports)
// when it is not. No session is created.
import { startRunner, unwrap, printJson, failCli } from '../lib/runner.mjs';
import { loadSpec, planFragment } from '../lib/spec.mjs';

const [specFile] = process.argv.slice(2);
if (!specFile) { console.error('usage: node validate.mjs <spec.json>'); process.exit(2); }

let spec;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }

const { send, stop } = startRunner();
try {
  const result = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
  await stop();
  printJson({ ok: true, ...result });
  process.exit(result.valid ? 0 : 2);
} catch (e) {
  try { await stop(); } catch { /* ignore */ }
  failCli(e.message, e.envelope ?? null);
}
