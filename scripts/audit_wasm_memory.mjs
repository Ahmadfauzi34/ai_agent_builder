// WASM memory oracle: deterministic leak detection via linear memory.
//
// WASM linear memory grows one way (never shrinks back to the host), so
// `memory.size()` in 64KiB pages is an exact, noise-free allocator signal —
// unlike native RSS with its ±50MB allocator sawtooth. Identical operation
// cycles must leave the page count unchanged; any monotonic growth is a
// real leak, not noise.
//
// Usage: node scripts/audit_wasm_memory.mjs [packageDir]
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Capture the wasm instance's exported memory: patch the constructor
// before the generated JS instantiates the module.
let capturedMemory = null;
const OrigInstance = WebAssembly.Instance;
WebAssembly.Instance = function (module, imports) {
  const inst = new OrigInstance(module, imports);
  if (inst.exports && inst.exports.memory instanceof WebAssembly.Memory) {
    capturedMemory = inst.exports.memory;
  }
  return inst;
};

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();
if (!capturedMemory) throw new Error('wasm memory not captured');

const pages = () => capturedMemory.buffer.byteLength / 65536;

function check(cond, msg) {
  if (!cond) throw new Error(`memory oracle failed: ${msg}`);
  console.log(`ok - ${msg}`);
}

function tensor(n, fill) {
  return new m.WasmTensor(new Float32Array(n).fill(fill), new Uint32Array([n]));
}

// ---- T1: tensor alloc/free cycles leave pages unchanged ----
{
  for (let i = 0; i < 5; i++) tensor(100000, 1).free(); // warmup
  const base = pages();
  for (let i = 0; i < 50; i++) tensor(100000, 1).free();
  check(pages() === base, `T1 tensor cycles stable (${base} pages)`);
}

// ---- T2: negative control — the oracle has teeth ----
// (allocating without free MUST grow pages; if it didn't, the oracle
// would be blind)
{
  const base = pages();
  const held = [];
  for (let i = 0; i < 10; i++) held.push(tensor(100000, 2));
  const grown = pages() > base;
  for (const t of held) t.free();
  check(grown, `T2 unfreed allocs grow pages (${base} -> ${pages()} pages)`);
  // after freeing, a fresh cycle must not grow further (allocator reuse)
  const base2 = pages();
  for (let i = 0; i < 10; i++) tensor(100000, 3).free();
  check(pages() === base2, `T2 allocator reuses freed blocks (${base2} pages)`);
}

// ---- T3: math program build/compile/run/free cycles ----
{
  const caps = JSON.parse(m.mathProgramCapabilities());
  const OP = caps.opcodes;
  const runOnce = () => {
    const b = new m.WasmMathProgramBuilder(1, 2);
    b.addUnary(OP.abs, 0, 1);
    b.setOutput(1);
    const p = b.compile();
    const t = tensor(2, -3);
    const out = p.run1(t);
    const v = Array.from(out.to_array());
    if (v[0] !== 3 || v[1] !== 3) throw new Error('wrong result');
    t.free(); out.free(); p.free();
    if (b.free) b.free();
  };
  for (let i = 0; i < 5; i++) runOnce(); // warmup
  const base = pages();
  for (let i = 0; i < 50; i++) runOnce();
  check(pages() === base, `T3 program cycles stable (${base} pages)`);
}

// ---- T4: mixed workload (tensors + programs interleaved) ----
{
  const caps = JSON.parse(m.mathProgramCapabilities());
  const OP = caps.opcodes;
  for (let i = 0; i < 5; i++) tensor(50000, 1).free();
  const base = pages();
  for (let i = 0; i < 30; i++) {
    const t = tensor(50000, i);
    const b = new m.WasmMathProgramBuilder(1, 2);
    b.addUnary(OP.abs, 0, 1);
    b.setOutput(1);
    const p = b.compile();
    const tt = tensor(2, -1);
    const out = p.run1(tt);
    out.free(); tt.free(); p.free();
    if (b.free) b.free();
    t.free();
  }
  check(pages() === base, `T4 mixed cycles stable (${base} pages)`);
}

// ---- T5: error paths must not leak ----
// Entry-validation throws are a classic leak vector: intermediates
// allocated before the throw must still be released.
{
  const caps = JSON.parse(m.mathProgramCapabilities());
  const OP = caps.opcodes;
  for (let i = 0; i < 5; i++) {
    try {
      const b = new m.WasmMathProgramBuilder(1, 2);
      b.addUnary(OP.abs, 0, 1);
      b.setOutput(1);
      const p = b.compile();
      p.run1(tensor(2, NaN)); // throws: entry validation
    } catch { /* expected */ }
  }
  const base = pages();
  for (let i = 0; i < 50; i++) {
    try {
      const b = new m.WasmMathProgramBuilder(1, 2);
      b.addUnary(OP.abs, 0, 1);
      b.setOutput(1);
      const p = b.compile();
      const t = tensor(2, NaN);
      p.run1(t); // throws
      t.free();
    } catch { /* expected */ }
  }
  check(pages() === base, `T5 error-path cycles stable (${base} pages)`);
}

// ---- T6: to_array() round-trips (host copies) ----
{
  for (let i = 0; i < 5; i++) Array.from(tensor(100000, 1).to_array());
  const base = pages();
  for (let i = 0; i < 50; i++) {
    const t = tensor(100000, i);
    const arr = t.to_array();
    if (arr.length !== 100000) throw new Error('wrong length');
    t.free();
  }
  check(pages() === base, `T6 to_array cycles stable (${base} pages)`);
}

// ---- T7: layer state cycles (differential pair with tests/mem_differential.rs) ----
{
  for (let i = 0; i < 5; i++) {
    const reg = new m.LayerRegistry();
    reg.initAgentLayer(m.AgentLayerSpec.linear(1, 8, 4));
    const st = reg.getLayerState(1, 0x01);
    reg.loadLayerState(1, 0x01, st);
  }
  const base = pages();
  for (let i = 0; i < 50; i++) {
    const reg = new m.LayerRegistry();
    reg.initAgentLayer(m.AgentLayerSpec.linear(1, 8, 4));
    const st = reg.getLayerState(1, 0x01);
    reg.loadLayerState(1, 0x01, st);
  }
  check(pages() === base, `T7 layer state cycles stable (${base} pages)`);
}

console.log('\nmemory oracle: all checks passed');
