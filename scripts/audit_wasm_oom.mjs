// WASM OOM boundary audit: map the failure boundary and verify it fails
// gracefully (catchable error, not a hang).
//
// Findings (2026-10-10, postmerge pkg):
// - steady-state tensor cost ~= 1x byte size (100M f32 elems = 400MB data
//   ~= 391MB WASM; first alloc carries ~380MB one-time runtime overhead)
// - practical capacity: 8 x 400MB tensors (3.2GB data) before the 4GB
//   linear-memory ceiling; the 9th allocation throws
// - at exhaustion, allocation throws RuntimeError: unreachable in ~13s for
//   the full fill loop — catchable by the host, never a hang. (The message
//   is a generic WASM trap, not a friendly "out of memory"; improving it
//   is a separate task.)
//
// Note: linear memory never shrinks, so the exhaustive test runs LAST in a
// single runtime instance.
//
// Usage: node scripts/audit_wasm_oom.mjs [packageDir]
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');

let capturedMemory = null;
const OrigInstance = WebAssembly.Instance;
WebAssembly.Instance = function (module, imports) {
  const inst = new OrigInstance(module, imports);
  if (inst.exports && inst.exports.memory instanceof WebAssembly.Memory) {
    capturedMemory = inst.exports.memory;
  }
  return inst;
};
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();
WebAssembly.Instance = OrigInstance;
if (!capturedMemory) throw new Error('wasm memory not captured');
const memory = capturedMemory;

const pages = () => memory.buffer.byteLength / 65536;

function check(cond, msg) {
  if (!cond) throw new Error(`oom audit failed: ${msg}`);
  console.log(`ok - ${msg}`);
}

// ---- B1: large tensor within capacity succeeds ----
{
  const t = new m.WasmTensor(new Float32Array(100_000_000).fill(1),
                             new Uint32Array([100_000_000]));
  check(pages() > 22, `B1 400MB tensor allocated (${pages()} pages)`);
  t.free();
}

// ---- B2: small allocs still fine near (not at) the boundary ----
{
  const t = new m.WasmTensor(new Float32Array(1000).fill(1),
                             new Uint32Array([1000]));
  const v = Array.from(t.to_array());
  check(v.length === 1000 && v[0] === 1, 'B2 runtime healthy before exhaustion');
  t.free();
}

// ---- B3: true exhaustion -> catchable error, not a hang (LAST) ----
// NB: memory.grow() only ADDS address space; exhaustion means actually
// allocating it. Hold every tensor so the allocator cannot reuse.
{
  const held = [];
  let threw = null;
  const t0 = Date.now();
  for (let i = 0; i < 10; i++) {
    try {
      // 100M elems ~= 860MB WASM each; ~4-5 fill the 4GB address space
      held.push(new m.WasmTensor(new Float32Array(100_000_000).fill(7),
                                 new Uint32Array([100_000_000])));
    } catch (e) {
      threw = e;
      break;
    }
  }
  const dt = Date.now() - t0;
  check(threw !== null, `B3 exhaustion throws after ${held.length} x 400MB tensors`);
  check(threw instanceof Error, `B3 error is catchable (${threw?.constructor?.name})`);
  check(dt < 30000, `B3 fails fast, no hang (${dt}ms)`);
  console.log(`info - B3 practical capacity: ${held.length} x 400MB tensors before OOM`);
}

console.log('\noom audit: all checks passed');
