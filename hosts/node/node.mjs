import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { webcrypto } from 'node:crypto';

// Node 18 ESM quirk (lane #2 family, 2026-10-30): in Node 18,
// globalThis.crypto is present in CJS contexts but UNDEFINED inside ESM
// modules — and the wasm glue calls globalThis.crypto.getRandomValues for
// thread_rng (ParamId generation on weighted layer creation). Without this
// line, every session create containing a weighted layer (linear,
// embedding, layerNorm, ...) fails under Node 18 — surfacing misleadingly
// as "attempted to take ownership of Rust value while it was borrowed"
// (phase create), a downstream artifact of the failed RNG call, not an
// ownership bug. Weight-free creates (e.g. the smoke test's add-only
// session) never touch the RNG, which is why CI never caught this.
// engine.mjs and wasm_direct.mjs already had this polyfill; this loader
// is the third one that needs it. Node 24+ has crypto in ESM, so this
// is a no-op there.
globalThis.crypto ??= webcrypto;

const DEFAULT_PACKAGE_DIR = path.dirname(fileURLToPath(import.meta.url));

function resolvePackageDir(packageDir) {
  if (packageDir === undefined || packageDir === null) {
    return DEFAULT_PACKAGE_DIR;
  }
  if (packageDir instanceof URL) {
    if (packageDir.protocol !== 'file:') {
      throw new TypeError(`Node host adapter requires a file: URL, got ${packageDir.protocol}`);
    }
    return fileURLToPath(packageDir);
  }
  return path.resolve(String(packageDir));
}

/**
 * Load the packaged Burn/WASM runtime from a local filesystem directory.
 *
 * This is the supported Node path for the wasm-pack `web` artifact. It avoids
 * default async `file://` fetch semantics by reading the sibling WASM bytes and
 * passing them to wasm-bindgen's synchronous initializer.
 */
export async function loadBurnRuntime(packageDir = DEFAULT_PACKAGE_DIR) {
  const dir = resolvePackageDir(packageDir);
  const jsPath = path.join(dir, 'burn_research.js');
  const wasmPath = path.join(dir, 'burn_research_bg.wasm');

  const [runtime, wasmBytes] = await Promise.all([
    import(pathToFileURL(jsPath).href),
    fs.promises.readFile(wasmPath),
  ]);

  runtime.initSync({ module: wasmBytes });
  return runtime;
}

export function burnRuntimePackageDir() {
  return DEFAULT_PACKAGE_DIR;
}
