// Spec format for the helper CLIs (see ../spec-schema.md).
// Loads a JSON spec file and normalizes the optional sections.
// Upgraded in v3.8: transparently hydrates content-addressed `burn://sha256:...`
// artifact URIs (including inside Recursive Spec v2 step/subgraph weights) and
// compiles `burn-math.recursive-spec.v2` specs across all spec-consuming CLIs.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { catalogPath } from './engine.mjs';
import { compileRecursiveSpec } from './composer.mjs';

export function loadCatalog() {
  return JSON.parse(fs.readFileSync(catalogPath(), 'utf8'));
}

// Port v2.5: strict machine typing. u32 is a REAL 32-bit range
// (0..4294967295) — the old "non-negative integer" check let id 4294967296
// through, and the engine then coerced it silently. f32 is finite AFTER
// Math.fround — 1e39 is a finite JS number but overflows f32 to Infinity
// (weights 1e39 used to pass as "finite"). Keys are canonical: "0", "1",
// never "", "01", or " 1".
const MAX_U32 = 0xFFFFFFFF;
const CANONICAL_UINT_KEY = /^(?:0|[1-9]\d*)$/;
const ARTIFACT_URI_PREFIX = 'burn://sha256:';

export function isFiniteF32(v) {
  return typeof v === 'number' && Number.isFinite(v) && Number.isFinite(Math.fround(v));
}

export function isU32(v) {
  return Number.isInteger(v) && v >= 0 && v <= MAX_U32;
}

// One catalog parameter's type, checked strictly. The catalog's
// `parameters` array (name, type, optional) is authoritative.
function typeIssue(layerLabel, param, value) {
  const type = param.type ?? '';
  const isOption = type.startsWith('Option<');
  const base = isOption ? type.slice(7, -1) : type;
  if (value === null) {
    if (isOption) return null; // JSON has no undefined: null IS how an
    // optional param is skipped mid-list (e.g. conv2d strides).
    return `${layerLabel}: arg ${param.name} must not be null (not an Option param)`;
  }
  if (base === 'u32') {
    if (!isU32(value)) return `${layerLabel}: arg ${param.name} must be a u32 integer (0..${MAX_U32}), got ${JSON.stringify(value)}`;
  } else if (base === 'f32') {
    if (!isFiniteF32(value)) return `${layerLabel}: arg ${param.name} must be a finite f32 number, got ${JSON.stringify(value)}`;
  } else if (base === 'f64') {
    if (typeof value !== 'number' || !Number.isFinite(value)) return `${layerLabel}: arg ${param.name} must be a finite number (f64), got ${JSON.stringify(value)}`;
  } else if (base === 'bool') {
    if (typeof value !== 'boolean') return `${layerLabel}: arg ${param.name} must be a boolean, got ${JSON.stringify(value)}`;
  } else if (typeof value !== 'number' || !Number.isFinite(value)) {
    return `${layerLabel}: arg ${param.name} must be a finite number, got ${JSON.stringify(value)}`;
  }
  return null;
}

// Ported from pack v2.4 (Audit_7 E4): strict argument/value typing for
// specs. The engine (via wasm-bindgen and the runner protocol) silently
// coerces mistyped values — a string 'false' for linear's bias flag was
// treated as no-bias and returned ok:true, and string weights became NaN
// through Float32Array.from. This check runs BEFORE any execution
// (shapecheck's pre-check, run's pre-flight, wasm_direct buildGraph) and
// fails closed with issue strings. Rules: layer args must be an array;
// required-arg counts come from the catalog (optional:false); u32 args
// must be integers in 0..4294967295 (no silent truncation of 1.9 to 1,
// no silent wrap of 2^32); f32 args and weights/inputs values must be
// finite after fround (1e39 overflows f32); bool args must be real
// booleans; null is allowed only for Option params; registry ids
// (args[0]) must be unique across layers — duplicates made run return
// wrong numbers with ok:true, because weights are keyed by layer
// index, not by id. Weights/inputs must be plain objects of
// finite-f32-number arrays keyed by canonical u32 slots; port shapes
// must be 4-element positive-u32 arrays; port slots must be unique
// u32 integers; step entries must be objects with u32 layer/slots.
export function checkArgValues(spec, catalog) {
  const issues = [];
  const ctors = (catalog ?? loadCatalog()).constructors ?? {};
  const seenIds = new Map(); // registry id -> first layer index
  for (const [li, decl] of (spec.layers ?? []).entries()) {
    if (!decl || typeof decl !== 'object') { issues.push(`layer ${li}: not an object`); continue; }
    if (!Array.isArray(decl.args)) { issues.push(`layer ${li} (${decl.constructor}): args must be an array`); continue; }
    // hasOwn: without it, ctors['toString'] returns the inherited
    // Object.prototype function and a bogus constructor looks known.
    const entry = Object.hasOwn(ctors, decl.constructor) ? ctors[decl.constructor] : undefined;
    if (!entry) continue; // unknown constructors are rejected elsewhere (shapecheck arity / engine)
    const params = (entry.parameters ?? []).length
      ? entry.parameters
      : (() => {
        const inner = (entry.signature ?? '').replace(/^[^\(]*\(/, '').replace(/\)[^\)]*$/, '');
        if (!inner.trim()) return [];
        return inner.split(',').map((t) => {
          const token = t.trim();
          const name = token.replace(/\?$/, '');
          return { name, type: name === 'bias' ? 'bool' : (name === 'id' ? 'u32' : 'f64'), optional: token.endsWith('?') };
        });
      })();
    const required = params.filter((p) => !p.optional).length;
    if (decl.args.length < required) {
      issues.push(`layer ${li} (${decl.constructor}): expects at least ${required} args (${params.filter((p) => !p.optional).map((p) => p.name).join(', ')}), got ${decl.args.length}`);
    }
    if (decl.args.length > params.length) {
      issues.push(`layer ${li} (${decl.constructor}): expects at most ${params.length} args, got ${decl.args.length}`);
    }
    decl.args.forEach((a, i) => {
      const param = params[i];
      if (!param) return; // extra args already reported by the count check
      const issue = typeIssue(`layer ${li} (${decl.constructor})`, param, a);
      if (issue) issues.push(issue);
    });
    const id = decl.args[0];
    if (isU32(id)) {
      if (seenIds.has(id)) {
        issues.push(`duplicate registry id ${id} (layers ${seenIds.get(id)} and ${li}) — registry ids (args[0]) must be unique; weights are keyed by layer index, not by id`);
      } else seenIds.set(id, li);
    }
  }
  for (const [si, st] of (spec.steps ?? []).entries()) {
    if (!st || typeof st !== 'object') { issues.push(`step ${si}: not an object`); continue; }
    if (!isU32(st.layer)) issues.push(`step ${si}: layer must be a u32 layer index, got ${JSON.stringify(st.layer)}`);
    if (Array.isArray(st.slots) && st.slots.some((s) => !isU32(s))) {
      issues.push(`step ${si}: slots must be u32 integers, got ${JSON.stringify(st.slots)}`);
    }
  }
  // Port v2.5: weights/inputs values must be finite f32 (not merely finite
  // JS numbers — 1e39 overflows f32), and keys must be canonical u32
  // ("0", "1" — never "" or "01", which coerce or vanish downstream).
  for (const [section, obj] of [['weights', spec.weights], ['inputs', spec.inputs]]) {
    if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
      issues.push(`${section} must be an object mapping slot -> values array`);
      continue;
    }
    for (const [slotKey, values] of Object.entries(obj)) {
      if (!CANONICAL_UINT_KEY.test(slotKey) || !isU32(Number(slotKey))) {
        issues.push(`${section} key ${JSON.stringify(slotKey)}: must be a canonical non-negative integer key (e.g. "0", "1")`);
      }
      if (!Array.isArray(values) || values.some((v) => !isFiniteF32(v))) {
        issues.push(`${section} slot ${slotKey}: values must be an array of finite f32 numbers`);
      }
    }
  }
  const seenPortSlots = new Set();
  for (const p of (spec.ports ?? [])) {
    if (!p || typeof p !== 'object') { issues.push(`port entry is null or not an object`); continue; }
    if (!isU32(p.slot)) {
      issues.push(`port slot must be a u32 integer, got ${JSON.stringify(p.slot)}`);
    } else {
      if (seenPortSlots.has(p.slot)) issues.push(`duplicate port slot ${p.slot}`);
      seenPortSlots.add(p.slot);
    }
    if (!Array.isArray(p.shape) || p.shape.length !== 4 ||
        p.shape.some((d) => !isU32(d) || d <= 0)) {
      issues.push(`port slot ${p.slot}: port.shape must contain four positive u32 integer dimensions, got ${JSON.stringify(p.shape)}`);
    }
  }
  return issues;
}

// Input-length gate (gap-d, 2026-11-05): every inputs array must hold
// exactly numel(port.shape) values. Same comparison validate.mjs
// makes as a WARNING (F4 / gap-5 model: validate warns, exit 0) —
// but the run path has no warning channel, so it fails here with
// the same slot-naming message instead of the engine's bare
// WasmTensor::new error (which names only shape and counts, and is
// ambiguous when ports share a shape). Only canonical keys matching
// a declared port with a valid 4-D shape are checked; a key with no
// declared port is left alone here (validate already warns on it,
// and the engine silently ignores it — no run failure to attribute).
export function checkInputLengths(spec) {
  const issues = [];
  const portShapeBySlot = new Map();
  for (const p of (spec.ports ?? [])) {
    if (!p || typeof p !== 'object' || !isU32(p.slot)) continue;
    if (portShapeBySlot.has(p.slot)) continue; // duplicate already reported
    if (!Array.isArray(p.shape) || p.shape.length !== 4 ||
        p.shape.some((d) => !isU32(d) || d <= 0)) continue; // shape issue already reported
    portShapeBySlot.set(p.slot, p.shape);
  }
  const inputs = spec.inputs;
  if (inputs === null || typeof inputs !== 'object' || Array.isArray(inputs)) return issues;
  for (const [slotKey, values] of Object.entries(inputs)) {
    if (!Array.isArray(values)) continue; // type issue already reported
    if (!CANONICAL_UINT_KEY.test(slotKey) || !isU32(Number(slotKey))) continue;
    const shape = portShapeBySlot.get(Number(slotKey));
    if (!shape) continue;
    const numel = shape[0] * shape[1] * shape[2] * shape[3];
    if (values.length !== numel) {
      issues.push(`inputs slot ${slotKey}: expected ${numel} values for shape ${JSON.stringify(shape)}, got ${values.length}`);
    }
  }
  return issues;
}

// Port v2.5 (F4): exact f32 weight-element count for the 10 weighted
// constructors, from the catalog signatures. Returns null for
// non-weighted constructors or for args that fail basic typing
// (callers run checkArgValues first; null here means "no expectation").
// Formulas: linear in_dim*out_dim + (bias ? out_dim : 0);
// batchNorm/groupNorm/instanceNorm/layerNorm 2 * features; rmsNorm size
// (gamma only); conv1d in_ch*out_ch*kernel + out_ch;
// conv2d/convTranspose2d in_ch*out_ch*kh*kw + out_ch; embedding
// vocab_size*d_model. Note: the conv formulas add out_ch unconditionally
// (the doc table's convention); a wrong formula only warns, never fails.
export function expectedWeightCount(decl) {
  if (!decl || typeof decl !== 'object' || !Array.isArray(decl.args)) return null;
  const a = decl.args;
  switch (decl.constructor) {
    case 'linear': {
      const [, inDim, outDim, bias] = a;
      if (!isU32(inDim) || !isU32(outDim) || typeof bias !== 'boolean') return null;
      return inDim * outDim + (bias ? outDim : 0);
    }
    case 'batchNorm': return isU32(a[1]) ? 2 * a[1] : null;
    case 'groupNorm': return isU32(a[2]) ? 2 * a[2] : null;
    case 'instanceNorm': return isU32(a[1]) ? 2 * a[1] : null;
    case 'layerNorm': return isU32(a[1]) ? 2 * a[1] : null;
    case 'rmsNorm': return isU32(a[1]) ? a[1] : null;
    case 'conv1d':
      return (isU32(a[1]) && isU32(a[2]) && isU32(a[3])) ? a[1] * a[2] * a[3] + a[2] : null;
    case 'conv2d':
    case 'convTranspose2d':
      return (isU32(a[1]) && isU32(a[2]) && isU32(a[3]) && isU32(a[4]))
        ? a[1] * a[2] * a[3] * a[4] + a[2] : null;
    case 'embedding':
      return (isU32(a[1]) && isU32(a[2])) ? a[1] * a[2] : null;
    default: return null;
  }
}

function resolveArtifactUriFromCandidates(uri, candidateWorkspaces) {
  if (typeof uri !== 'string' || !uri.startsWith(ARTIFACT_URI_PREFIX)) {
    throw new Error(`invalid artifact URI: ${JSON.stringify(uri)} (expected ${ARTIFACT_URI_PREFIX}<64-hex>)`);
  }
  const expectedHex = uri.slice(ARTIFACT_URI_PREFIX.length);
  if (!/^[0-9a-f]{64}$/.test(expectedHex)) {
    throw new Error(`malformed sha256 in artifact URI: ${uri}`);
  }
  for (const ws of candidateWorkspaces) {
    if (!ws) continue;
    const file = path.join(path.resolve(ws), 'artifacts', `${expectedHex}.json`);
    if (fs.existsSync(file)) {
      const raw = fs.readFileSync(file, 'utf8').trim();
      const actualHex = createHash('sha256').update(raw).digest('hex');
      if (actualHex !== expectedHex) {
        const err = new Error(`artifact integrity check failed for ${uri}: expected ${expectedHex}, got ${actualHex}`);
        err.code = 'artifact_integrity_failed';
        throw err;
      }
      return JSON.parse(raw);
    }
  }
  throw new Error(`artifact not found in workspace: ${uri}`);
}

function hydrateValueOrUri(v, candidateWorkspaces) {
  if (typeof v === 'string' && v.startsWith(ARTIFACT_URI_PREFIX)) {
    return resolveArtifactUriFromCandidates(v, candidateWorkspaces).values;
  }
  if (v && typeof v === 'object' && !Array.isArray(v) && typeof v.$artifact === 'string') {
    return resolveArtifactUriFromCandidates(v.$artifact, candidateWorkspaces).values;
  }
  return v;
}

function weightsContainUri(w) {
  if (typeof w === 'string') return w.startsWith(ARTIFACT_URI_PREFIX);
  if (Array.isArray(w)) return w.some(weightsContainUri);
  if (w && typeof w === 'object') {
    if (typeof w.$artifact === 'string') return true;
    return Object.values(w).some(weightsContainUri);
  }
  return false;
}

function hydrateStepListArtifacts(steps, candidateWorkspaces, isV2 = true) {
  if (!Array.isArray(steps)) return;
  for (const st of steps) {
    if (!st || typeof st !== 'object') continue;
    if (st.weights !== undefined) {
      // Minor fail-closed fix: a v1 step carries no weights (the
      // engine reads top-level spec.weights), so a URI in v1 step
      // weights used to be "resolved" into a field nothing reads —
      // silently misleading. v2 step weights are real (the composer
      // consumes them), so only the v1 case is rejected.
      if (!isV2 && weightsContainUri(st.weights)) {
        throw new Error('hydrate: artifact URI in v1 step weights is ignored by the engine (v1 steps carry no weights); move artifact URIs to top-level "weights" or "inputs"');
      }
      if (typeof st.weights === 'string' || (st.weights && typeof st.weights === 'object' && !Array.isArray(st.weights) && typeof st.weights.$artifact === 'string')) {
        st.weights = hydrateValueOrUri(st.weights, candidateWorkspaces);
      } else if (st.weights && typeof st.weights === 'object' && !Array.isArray(st.weights)) {
        for (const [k, v] of Object.entries(st.weights)) {
          st.weights[k] = hydrateValueOrUri(v, candidateWorkspaces);
        }
      }
    }
    if (Array.isArray(st.branches)) {
      for (const b of st.branches) {
        if (Array.isArray(b)) hydrateStepListArtifacts(b, candidateWorkspaces);
        else if (b && typeof b === 'object') hydrateStepListArtifacts([b], candidateWorkspaces);
      }
    }
  }
}

export function hydrateSpecArtifactsInPlace(spec, baseDir = process.cwd(), explicitWorkspace = null) {
  if (!spec || typeof spec !== 'object') return spec;
  const candidateWorkspaces = [
    explicitWorkspace,
    spec.workspace ? path.resolve(baseDir, spec.workspace) : null,
    process.env.BURN_HARNESS_WORKSPACE,
    path.join(baseDir, '.burn-harness'),
    path.resolve('.burn-harness'),
  ].filter(Boolean);

  for (const section of ['inputs', 'weights']) {
    const obj = spec[section];
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) continue;
    for (const [k, v] of Object.entries(obj)) {
      obj[k] = hydrateValueOrUri(v, candidateWorkspaces);
    }
  }
  hydrateStepListArtifacts(spec.steps, candidateWorkspaces, isRecursiveSpecV2(spec));
  if (spec.subgraphs && typeof spec.subgraphs === 'object') {
    for (const sg of Object.values(spec.subgraphs)) {
      if (sg && typeof sg === 'object') hydrateStepListArtifacts(sg.steps, candidateWorkspaces, true);
    }
  }
  return spec;
}

export function isRecursiveSpecV2(spec) {
  if (!spec || typeof spec !== 'object') return false;
  if (spec.schema === 'burn-math.recursive-spec.v2') return true;
  if (spec.subgraphs && typeof spec.subgraphs === 'object') return true;
  if (spec.output !== undefined && spec.numSlots === undefined) return true;
  if (Array.isArray(spec.steps) && spec.steps.some((s) => s && typeof s === 'object' && typeof s.op === 'string')) return true;
  return false;
}

export function loadSpec(file, opts = {}) {
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); }
  catch (e) { throw new Error(`cannot read spec file ${file}: ${e.message}`); }
  let spec;
  try { spec = JSON.parse(raw); }
  catch (e) { throw new Error(`spec file ${file} is not valid JSON: ${e.message}`); }

  const baseDir = path.dirname(path.resolve(file));
  hydrateSpecArtifactsInPlace(spec, baseDir, opts.workspaceDir ?? null);

  if (isRecursiveSpecV2(spec)) {
    const compiled = compileRecursiveSpec(spec, {
      baseDir,
      autoRepair: Boolean(opts.autoRepair ?? spec.auto_repair ?? false),
      compactSlots: Boolean(opts.compactSlots ?? spec.compact_slots ?? false),
      validate: false,
    });
    hydrateSpecArtifactsInPlace(compiled, baseDir, opts.workspaceDir ?? null);
    return compiled;
  }

  for (const key of ['numSlots', 'layers', 'steps', 'outputSlot', 'ports']) {
    if (spec[key] === undefined) throw new Error(`spec missing required key: ${key}`);
  }
  if (!Array.isArray(spec.layers) || !Array.isArray(spec.steps) || !Array.isArray(spec.ports)) {
    throw new Error('spec.layers / spec.steps / spec.ports must be arrays');
  }
  spec.weights = spec.weights ?? {};
  spec.inputs = spec.inputs ?? {};
  return spec;
}

// Port lookup by slot, for binding inputs with the declared shape/role/layout.
// Port v2.5: fail fast on non-u32 or duplicate slots (the old version
// silently kept the last duplicate and let fractional slots through).
export function portBySlot(spec) {
  const m = new Map();
  for (const p of spec.ports) {
    if (!p || typeof p !== 'object' || !isU32(p.slot)) {
      throw new Error(`invalid port slot ${JSON.stringify(p?.slot)}`);
    }
    if (m.has(p.slot)) throw new Error(`duplicate port slot ${p.slot}`);
    m.set(p.slot, p);
  }
  return m;
}

// Sub-spec for the session-free validatePlan op.
export function planFragment(spec) {
  return {
    numSlots: spec.numSlots,
    steps: spec.steps,
    ports: spec.ports,
    outputSlot: spec.outputSlot,
  };
}

// Ported from the agent pack (was pack-only; the skill never needed it
// because skill validate went through the session engine). Shared by
// lib/wasm_direct.mjs and bin/validate.mjs: the in-process plan-port
// check (minimum 2 ports, every declared port consumed).
export function analyzePlanPorts(spec) {
  const minimum_ports = 2;
  const declared = (spec.ports ?? []).map((p) => p?.slot).filter((s) => isU32(s));
  const declaredSet = new Set(declared);
  const written = new Set();
  const consumed = new Set();
  for (const step of (spec.steps ?? [])) {
    if (!step || !Array.isArray(step.slots)) continue;
    const slots = step.slots;
    const inputs = step.kind === 'binary' ? slots.slice(0, 2) : slots.slice(0, 1);
    const out = step.kind === 'binary' ? slots[2] : slots[1];
    for (const slot of inputs) {
      if (!written.has(slot) && declaredSet.has(slot)) consumed.add(slot);
    }
    if (out !== undefined) written.add(out);
  }
  if (!written.has(spec.outputSlot) && declaredSet.has(spec.outputSlot)) {
    consumed.add(spec.outputSlot);
  }
  const bySlot = (a, b) => a - b;
  const declared_ports = [...declared].sort(bySlot);
  const consumed_ports = [...consumed].sort(bySlot);
  const unused_ports = [...declared].filter((s) => !consumed.has(s)).sort(bySlot);
  const valid = declared_ports.length >= minimum_ports && unused_ports.length === 0;
  const code = valid
    ? 'ok'
    : (declared_ports.length < minimum_ports ? 'plan_minimum_ports' : 'plan_unused_ports');
  return {
    minimum_ports,
    declared_ports,
    consumed_ports,
    unused_ports,
    valid,
    code,
  };
}
