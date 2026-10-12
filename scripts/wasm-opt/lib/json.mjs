// Minimal JSON-envelope printing for the wasm-opt CLIs.
export function printJson(obj) {
  process.stdout.write(JSON.stringify(obj, null, 2) + '\n');
}
