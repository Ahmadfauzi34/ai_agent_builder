import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_PKG_DIR = path.resolve(scriptDir, '..', 'pkg');
const ACTUAL_FILE = 'wasm-surface.actual.json';

function fail(message) {
  throw new Error(`WASM runtime surface: ${message}`);
}

export function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const entries = Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`);
  return `{${entries.join(',')}}`;
}

function sha256(bytes) {
  return `sha256:${crypto.createHash('sha256').update(bytes).digest('hex')}`;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function readContract(pkgDir, file) {
  const filePath = path.join(pkgDir, file);
  if (!fs.existsSync(filePath)) fail(`required packaged contract is missing: ${file}`);
  const bytes = fs.readFileSync(filePath);
  const contract = JSON.parse(bytes.toString('utf8'));
  const schema = contract.schema ?? contract.schema_id ?? contract.schema_version;
  return {file, schema, sha256: sha256(bytes), value: contract};
}

function trailingVersion(value, label) {
  if (typeof value !== 'string') fail(`${label} must be a versioned schema string`);
  const match = value.match(/\.v([1-9][0-9]*)$/);
  if (!match) fail(`${label} has no terminal .vN version: ${value}`);
  return Number(match[1]);
}

function normalizedExportSurface(surface) {
  if (!surface || !surface.classes || !Array.isArray(surface.functions)) fail('WASM export projection is malformed');
  return {
    default_init: surface.default_init,
    functions: [...surface.functions].sort(),
    classes: Object.fromEntries(Object.keys(surface.classes).sort().map((name) => {
      const members = surface.classes[name];
      if (!Array.isArray(members)) fail(`WASM export members for ${name} are malformed`);
      return [name, [...members].sort()];
    })),
  };
}

function compareDeclaredSurface(pkgDir, contract, projection) {
  const declared = readContract(pkgDir, contract.wasm_export_contract).value;
  if (canonicalJson(normalizedExportSurface(projection)) !== canonicalJson(normalizedExportSurface(declared))) {
    fail('generated wasm-bindgen declaration projection differs from the pinned WASM export contract');
  }
}

async function inspectRawModule(pkgDir) {
  const wasmPath = path.join(pkgDir, 'burn_research_bg.wasm');
  let module;
  try {
    module = await WebAssembly.compile(fs.readFileSync(wasmPath));
  } catch (error) {
    fail(`cannot compile packaged WASM binary for module inspection: ${error.message}`);
  }
  const stableRecords = (records) => records
    .map(({module: importModule, name, kind}) => ({
      ...(importModule === undefined ? {} : {module: importModule}),
      name,
      kind,
    }))
    .sort((left, right) => {
      const leftKey = canonicalJson(left);
      const rightKey = canonicalJson(right);
      return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
    });
  return {
    imports: stableRecords(WebAssembly.Module.imports(module)),
    exports: stableRecords(WebAssembly.Module.exports(module)),
  };
}

async function loadRuntime(pkgDir) {
  const adapterPath = path.join(pkgDir, 'node.mjs');
  if (!fs.existsSync(adapterPath)) fail(`packaged Node adapter is missing: ${adapterPath}`);
  const adapter = await import(pathToFileURL(adapterPath).href);
  if (typeof adapter.loadBurnRuntime !== 'function') fail('Node adapter does not export loadBurnRuntime()');
  return adapter.loadBurnRuntime();
}

export async function buildWasmSurfaceActual(pkgDir = DEFAULT_PKG_DIR, suppliedRuntime) {
  pkgDir = path.resolve(pkgDir);
  const runtimeContract = readContract(pkgDir, 'runtime-surface.v1.json').value;
  if (runtimeContract.schema !== 'burn-research.runtime-surface.v1' || runtimeContract.version !== 1) {
    fail('unsupported runtime-surface contract');
  }

  const bindingProjectionFile = runtimeContract.wasm_declaration_projection?.generated_projection;
  if (!bindingProjectionFile) fail('runtime-surface contract has no declaration projection');
  const bindingProjection = readContract(pkgDir, bindingProjectionFile).value;
  if (bindingProjection.schema !== 'burn-research.wasm-bindgen-surface.actual.v1') {
    fail(`unexpected declaration projection schema: ${bindingProjection.schema}`);
  }
  compareDeclaredSurface(pkgDir, runtimeContract, bindingProjection);

  const artifactDigests = {};
  for (const file of runtimeContract.artifact_identity?.files ?? []) {
    const filePath = path.join(pkgDir, file);
    if (!fs.existsSync(filePath)) fail(`artifact file is missing: ${file}`);
    const bytes = fs.readFileSync(filePath);
    artifactDigests[file] = {sha256: sha256(bytes), byte_length: bytes.byteLength};
  }
  const declaredDtsHash = artifactDigests[bindingProjection.artifact]?.sha256;
  if (!declaredDtsHash || bindingProjection.artifact_sha256 !== declaredDtsHash) {
    fail('declaration projection does not match the packaged TypeScript declaration bytes');
  }

  const contractFiles = {
    wasm_export_contract: runtimeContract.wasm_export_contract,
    runtime_surface_contract: 'runtime-surface.v1.json',
    ...runtimeContract.host_capability_contracts.contracts,
  };
  const contractDigests = {};
  const hostContractDescriptors = {};
  for (const [key, file] of Object.entries(contractFiles)) {
    const contract = readContract(pkgDir, file);
    contractDigests[file] = {schema: contract.schema, sha256: contract.sha256};
    if (key !== 'wasm_export_contract' && key !== 'runtime_surface_contract') {
      hostContractDescriptors[key] = {
        owner: runtimeContract.host_capability_contracts.owner,
        ...contractDigests[file],
        file,
        definition: contract.value,
      };
    }
  }

  const runtime = suppliedRuntime ?? await loadRuntime(pkgDir);
  const capabilitySnapshots = {};
  for (const [groupId, group] of Object.entries(runtimeContract.wasm_capability_groups ?? {})) {
    const snapshots = {};
    for (const [capabilityId, entrypoint] of Object.entries(group.capabilities ?? {})) {
      if (typeof runtime[entrypoint] !== 'function') fail(`runtime capability entrypoint is missing: ${entrypoint}`);
      let payload;
      try {
        payload = JSON.parse(runtime[entrypoint]());
      } catch (error) {
        fail(`${entrypoint} did not return valid JSON: ${error.message}`);
      }
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        fail(`${entrypoint} must return a JSON object`);
      }
      snapshots[capabilityId] = {
        entrypoint,
        schema: payload.schema ?? payload.schema_id ?? null,
        payload,
      };
    }
    capabilitySnapshots[groupId] = {owner: group.owner, capabilities: snapshots};
  }

  const versionChannels = {};
  for (const [channel, source] of Object.entries(
    runtimeContract.wasm_capability_groups.math_interaction.version_channels ?? {},
  )) {
    const capability = capabilitySnapshots.math_interaction?.capabilities[source.capability];
    if (!capability) fail(`${channel} refers to missing capability ${source.capability}`);
    const schema = capability.payload[source.field];
    versionChannels[channel] = {
      capability: capability.entrypoint,
      field: source.field,
      schema,
      version: trailingVersion(schema, channel),
    };
  }
  if (!versionChannels.math_program_surface_version || !versionChannels.math_interaction_protocol_version) {
    fail('MathProgram generation and interaction protocol must be separate version channels');
  }

  const body = {
    schema: 'burn-research.wasm-surface.actual.v1',
    surface_version: 1,
    contract: {
      schema: runtimeContract.schema,
      version: runtimeContract.version,
      file: 'runtime-surface.v1.json',
      sha256: contractDigests['runtime-surface.v1.json'].sha256,
    },
    artifacts: artifactDigests,
    wasm_exports: bindingProjection,
    wasm_binary_surface: await inspectRawModule(pkgDir),
    capability_groups: capabilitySnapshots,
    math_version_channels: versionChannels,
    host_capabilities: {
      owner: runtimeContract.host_capability_contracts.owner,
      contracts: hostContractDescriptors,
    },
    authority_boundaries: runtimeContract.authority_boundaries,
  };
  const digest = sha256(Buffer.from(canonicalJson(body), 'utf8'));
  return {
    ...body,
    fingerprint: {
      algorithm: 'sha256',
      canonicalization: runtimeContract.fingerprint.canonicalization,
      authority: runtimeContract.fingerprint.authority,
      value: digest,
    },
  };
}

export async function verifyWasmSurfaceActual(pkgDir = DEFAULT_PKG_DIR, suppliedRuntime) {
  pkgDir = path.resolve(pkgDir);
  const actualPath = path.join(pkgDir, ACTUAL_FILE);
  if (!fs.existsSync(actualPath)) fail(`generated artifact is missing: ${actualPath}`);
  const actual = readJson(actualPath);
  const expected = await buildWasmSurfaceActual(pkgDir, suppliedRuntime);
  if (canonicalJson(actual) !== canonicalJson(expected)) {
    fail('generated JSON, capability snapshot, contract digest, or artifact fingerprint differs from runtime');
  }
  return actual;
}

async function main() {
  const args = process.argv.slice(2);
  const mode = args.find((arg) => arg === '--write' || arg === '--check');
  if (!mode || args.filter((arg) => arg === '--write' || arg === '--check').length !== 1) {
    fail('usage: node scripts/generate_wasm_surface_actual.mjs [pkg-dir] (--write | --check)');
  }
  const pkgArg = args.find((arg) => !arg.startsWith('--'));
  const pkgDir = path.resolve(pkgArg ?? DEFAULT_PKG_DIR);
  const actualPath = path.join(pkgDir, ACTUAL_FILE);

  if (mode === '--write') {
    const expected = await buildWasmSurfaceActual(pkgDir);
    fs.writeFileSync(actualPath, `${JSON.stringify(expected, null, 2)}\n`);
    if (canonicalJson(readJson(actualPath)) !== canonicalJson(expected)) {
      fail('generated JSON could not be read back exactly');
    }
    console.log(JSON.stringify({
      verdict: 'GENERATED_AND_VERIFIED',
      artifact: actualPath,
      fingerprint: expected.fingerprint.value,
      wasm_sha256: expected.artifacts['burn_research_bg.wasm'].sha256,
      capability_groups: Object.keys(expected.capability_groups),
      host_contracts: Object.keys(expected.host_capabilities.contracts),
    }, null, 2));
  } else {
    const verified = await verifyWasmSurfaceActual(pkgDir);
    console.log(JSON.stringify({
      verdict: 'PASS',
      artifact: actualPath,
      fingerprint: verified.fingerprint.value,
      wasm_sha256: verified.artifacts['burn_research_bg.wasm'].sha256,
    }, null, 2));
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.stack ?? String(error));
    process.exitCode = 1;
  });
}
