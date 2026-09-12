import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

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
