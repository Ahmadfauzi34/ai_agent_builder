// burn-math Recursive Structural & Compositional Compiler (Spec v2 -> Canonical Flat Spec v1)
// + Symbolic Autodiff Graph Compiler (Forward Spec -> Runnable Backward Spec v1, incl. div/sqrt/sigmoid/tanh HVP & intermediate/symbolic --wrt)
// + Liveness Slot Register Compaction & Deterministic Auto-Repair with Remediation Receipts.

import fs from 'node:fs';
import path from 'node:path';
import { MACROS } from './macros.mjs';
import { checkArgValues, analyzePlanPorts } from './spec.mjs';
import { checkSpec } from './shapecore.mjs';
import { inferShapes, ssaNormalizeSpec } from './autodiff.mjs';
import { compactSlotsByLiveness } from './explain.mjs';
import { INT_BINARY_CONSTRUCTORS, INT_CONSTRUCTORS } from './mixed.mjs';

export const RECURSIVE_SPEC_SCHEMA = 'burn-math.recursive-spec.v2';
const MAX_RECURSION_DEPTH = 32;
const WORKSPACE_MAX_SLOTS = 64;

export class CompositionError extends Error {
  constructor(code, message, hint = null) {
    super(message);
    this.name = 'CompositionError';
    this.code = code;
    this.remediation_hint = hint;
  }
}

const clone = (o) => JSON.parse(JSON.stringify(o));
const copyWeightVal = (w) => (Array.isArray(w) ? [...w] : clone(w));

function createAllocator({ idBase = 0, slotBase = 0, workspace = null, registry = null } = {}) {
  let nextLayerId = idBase;
  let nextSlotId = slotBase;
  const slotNames = new Map();
  const slotMeta = new Map();

  return {
    bindFixedSlot(name, slot, meta = {}) {
      const k = String(name);
      slotNames.set(k, slot);
      slotMeta.set(slot, { name: k, ...meta });
      if (slot >= nextSlotId) nextSlotId = slot + 1;
      return slot;
    },
    resolveSlot(ref, owner = 'composer', createIfMissing = false) {
      if (typeof ref === 'number' && Number.isInteger(ref) && ref >= 0) {
        if (ref >= nextSlotId) nextSlotId = ref + 1;
        return ref;
      }
      const k = String(ref);
      if (/^(?:0|[1-9]\d*)$/.test(k) && !slotNames.has(k) && !createIfMissing) {
        const num = Number(k);
        if (num >= nextSlotId) nextSlotId = num + 1;
        return num;
      }
      if (slotNames.has(k)) return slotNames.get(k);
      if (!createIfMissing) {
        throw new CompositionError(
          'unresolved_symbolic_slot',
          `symbolic slot '${k}' is read before it is declared in ports or produced by an earlier step`,
          `declare '${k}' in spec.ports or ensure an earlier step writes out: '${k}'`,
        );
      }
      let slot;
      if (workspace) {
        try {
          slot = workspace.reserveSlot(owner || k);
          if (slot < nextSlotId) slot = nextSlotId++;
          else nextSlotId = Math.max(nextSlotId, slot + 1);
        } catch {
          slot = nextSlotId++;
        }
      } else {
        slot = nextSlotId++;
      }
      slotNames.set(k, slot);
      slotMeta.set(slot, { name: k, owner });
      return slot;
    },
    freshSlot(label = 'tmp') {
      let slot;
      if (workspace) {
        try {
          slot = workspace.reserveSlot(label);
          if (slot < nextSlotId) slot = nextSlotId++;
          else nextSlotId = Math.max(nextSlotId, slot + 1);
        } catch {
          slot = nextSlotId++;
        }
      } else {
        slot = nextSlotId++;
      }
      slotMeta.set(slot, { name: `${label}#${slot}`, owner: label });
      return slot;
    },
    freshLayerId(label = 'layer') {
      if (workspace && registry) {
        try {
          const id = workspace.reserveLayerId(registry, label);
          nextLayerId = Math.max(nextLayerId, id + 1);
          return id;
        } catch {
          return nextLayerId++;
        }
      }
      return nextLayerId++;
    },
    get numSlots() {
      return nextSlotId;
    },
    get nextLayerId() {
      return nextLayerId;
    },
    get symbolTable() {
      return Object.fromEntries(slotNames.entries());
    },
  };
}

function normalizePorts(rawPorts, alloc) {
  const ports = [];
  if (Array.isArray(rawPorts)) {
    for (const p of rawPorts) {
      const slot = typeof p.slot === 'number' ? p.slot : alloc.freshSlot(p.id ?? p.name ?? 'port');
      const name = p.id ?? p.name ?? String(slot);
      alloc.bindFixedSlot(name, slot, { role: p.role, shape: p.shape, layout: p.layout });
      ports.push({
        slot,
        role: p.role ?? 'observation',
        shape: p.shape,
        layout: p.layout ?? 'feature_axis1_singleton',
      });
    }
  } else if (rawPorts && typeof rawPorts === 'object') {
    let idx = 0;
    for (const [name, p] of Object.entries(rawPorts)) {
      const slot = typeof p.slot === 'number' ? p.slot : idx;
      idx = Math.max(idx, slot + 1);
      alloc.bindFixedSlot(name, slot, { role: p.role, shape: p.shape, layout: p.layout });
      ports.push({
        slot,
        role: p.role ?? 'observation',
        shape: p.shape,
        layout: p.layout ?? 'feature_axis1_singleton',
      });
    }
  } else {
    throw new CompositionError('invalid_ports', 'spec.ports must be an array or an object mapping portName -> portSpec');
  }
  return ports;
}

function inlineBuiltinMacro(macroName, params, stepWeights, inSlots, outSlot, state, prefix) {
  const def = MACROS[macroName];
  if (!def) {
    throw new CompositionError(
      'unknown_macro',
      `unknown built-in macro '${macroName}' (available: ${Object.keys(MACROS).join(', ')})`,
      `use one of ${Object.keys(MACROS).join(', ')} or define '${macroName}' under spec.subgraphs`,
    );
  }
  const p = { ...params };
  if (macroName === 'attention_head') {
    if (inSlots.length >= 3) {
      p.q_slot = inSlots[0];
      p.k_slot = inSlots[1];
      p.v_slot = inSlots[2];
    }
    p.out_slot = outSlot;
  } else if (macroName === 'recurrent_cell') {
    if (inSlots.length >= 2) {
      p.state_slot = inSlots[0];
      p.in_slot = inSlots[1];
    } else if (inSlots.length === 1) {
      p.state_slot = inSlots[0];
    }
    p.out_slot = outSlot;
  } else {
    if (inSlots.length >= 1) p.in_slot = inSlots[0];
    p.out_slot = outSlot;
  }
  const localAlloc = {
    layerId: () => state.alloc.freshLayerId(`${prefix}:${macroName}`),
    slot: () => state.alloc.freshSlot(`${prefix}:${macroName}:tmp`),
  };
  const frag = def.expand(p, localAlloc);
  const layerOffset = state.layers.length;
  for (const l of frag.layers) state.layers.push(l);
  for (const s of frag.steps) {
    state.steps.push({
      kind: s.kind,
      layer: s.layer + layerOffset,
      slots: [...s.slots],
    });
  }
  if (stepWeights && typeof stepWeights === 'object') {
    for (const [k, w] of Object.entries(stepWeights)) {
      const globalIdx = Number(k) + layerOffset;
      state.weights[String(globalIdx)] = copyWeightVal(w);
    }
  }
  state.expansionLog.push({
    kind: 'builtin_macro',
    name: macroName,
    prefix,
    layers_added: frag.layers.length,
    steps_added: frag.steps.length,
  });
}

function expandStepList(steps, scopeEnv, state, callStack, prefix, baseDir, tiedCache = null) {
  if (callStack.length > MAX_RECURSION_DEPTH) {
    throw new CompositionError(
      'recursion_depth_exceeded',
      `maximum recursion depth (${MAX_RECURSION_DEPTH}) exceeded at ${callStack.join(' -> ')}`,
      'reduce repeat depth or check for unbounded subgraph recursion',
    );
  }

  const resolveInSlot = (sym) => {
    const k = String(sym);
    if (scopeEnv.has(k)) return scopeEnv.get(k);
    return state.alloc.resolveSlot(k, prefix, false);
  };

  const resolveOutSlot = (sym) => {
    if (sym === undefined || sym === null) return state.alloc.freshSlot(`${prefix}:out`);
    if (typeof sym === 'number' && Number.isInteger(sym) && sym >= 0) {
      return state.alloc.resolveSlot(sym, prefix, false);
    }
    const k = String(sym);
    if (scopeEnv.has(k)) return scopeEnv.get(k);
    const scopedKey = prefix === 'root' ? k : `${prefix}/${k}`;
    const slot = state.alloc.resolveSlot(scopedKey, prefix, true);
    scopeEnv.set(k, slot);
    return slot;
  };

  for (let si = 0; si < steps.length; si++) {
    const st = steps[si];
    const stepPrefix = `${prefix}.s${si}`;

    if ((st.kind === 'unary' || st.kind === 'binary') && typeof st.layer === 'number' && Array.isArray(st.slots)) {
      const mappedSlots = st.slots.map((s, idx) => {
        const isOut = idx === st.slots.length - 1;
        return isOut ? resolveOutSlot(s) : resolveInSlot(s);
      });
      const layerIdx = (state.v1LayerOffset ?? 0) + st.layer;
      state.steps.push({ kind: st.kind, layer: layerIdx, slots: mappedSlots });
      continue;
    }

    const op = st.op ?? st.constructor ?? st.kind;
    if (!op || typeof op !== 'string') {
      throw new CompositionError('invalid_step_op', `step ${stepPrefix} is missing 'op' or 'constructor'`);
    }

    const rawIn = st.in ?? st.inputs ?? (st.slots ? st.slots.slice(0, -1) : []);
    const inList = Array.isArray(rawIn) ? rawIn : [rawIn];
    const rawOut = st.out ?? st.output ?? (st.slots ? st.slots[st.slots.length - 1] : undefined);

    if (op === 'repeat') {
      const depth = Number(st.depth ?? st.count ?? 1);
      if (!Number.isInteger(depth) || depth < 1 || depth > MAX_RECURSION_DEPTH) {
        throw new CompositionError(
          'invalid_repeat_depth',
          `repeat depth must be an integer in 1..${MAX_RECURSION_DEPTH}, got ${JSON.stringify(st.depth ?? st.count)}`,
        );
      }
      const targetSub = st.subgraph ?? st.macro ?? st.target;
      if (!targetSub) {
        throw new CompositionError('missing_repeat_target', `repeat step at ${stepPrefix} requires 'subgraph' or 'macro'`);
      }
      if (inList.length < 1) {
        throw new CompositionError('missing_repeat_input', `repeat step at ${stepPrefix} requires at least 1 input`);
      }
      const finalOutSlot = resolveOutSlot(rawOut);
      const tieWeights = Boolean(st.tie_weights);
      const sharedTiedMap = tieWeights ? new Map() : null;
      // B3 fix: tied + per-depth weights used to silently discard the
      // per-depth weights (the tied branch reads only st.weights) and
      // run the layer as a zero matrix with ok:true. Now it fails
      // closed with a named error.
      if (tieWeights && Array.isArray(st.weights_per_depth)) {
        throw new CompositionError('tied_weights_per_depth_conflict', `repeat step at ${stepPrefix} combines tie_weights:true with weights_per_depth; per-depth weights are discarded when tied — use tied st.weights, or untied repeat with weights_per_depth`);
      }

      let curInSlot = resolveInSlot(inList[0]);
      const extraInSlots = inList.slice(1).map(resolveInSlot);
      for (let d = 0; d < depth; d++) {
        const iterOutSlot = d === depth - 1 ? finalOutSlot : state.alloc.freshSlot(`${stepPrefix}:rep${d}`);
        const subStep = {
          op: targetSub.startsWith('macro:') || MACROS[targetSub] ? `macro:${targetSub.replace(/^macro:/, '')}` : `subgraph:${targetSub.replace(/^subgraph:/, '')}`,
          params: st.params ?? {},
          weights: tieWeights ? (st.weights ?? {}) : (Array.isArray(st.weights_per_depth) ? st.weights_per_depth[d] : st.weights),
          in: [curInSlot, ...extraInSlots],
          out: iterOutSlot,
        };
        expandStepList([subStep], new Map(scopeEnv), state, [...callStack, `repeat(${targetSub}#${d})`], `${stepPrefix}.r${d}`, baseDir, sharedTiedMap);
        curInSlot = iterOutSlot;
      }
      state.expansionLog.push({ kind: 'repeat', target: targetSub, depth, tie_weights: tieWeights, prefix: stepPrefix });
      continue;
    }

    if (op === 'residual') {
      const targetSub = st.subgraph ?? st.macro ?? st.target;
      if (!targetSub || inList.length < 1) {
        throw new CompositionError('invalid_residual', `residual step at ${stepPrefix} requires 'subgraph'/'macro' and at least 1 input`);
      }
      const xSlot = resolveInSlot(inList[0]);
      const fxSlot = state.alloc.freshSlot(`${stepPrefix}:fx`);
      const outSlot = resolveOutSlot(rawOut);
      const innerOp = targetSub.startsWith('macro:') || MACROS[targetSub]
        ? `macro:${targetSub.replace(/^macro:/, '')}`
        : `subgraph:${targetSub.replace(/^subgraph:/, '')}`;
      expandStepList(
        [{ op: innerOp, params: st.params ?? {}, weights: st.weights, in: inList.map(resolveInSlot), out: fxSlot }],
        new Map(scopeEnv),
        state,
        [...callStack, `residual(${targetSub})`],
        `${stepPrefix}.inner`,
        baseDir,
        tiedCache,
      );
      const addLayerIdx = state.layers.length;
      state.layers.push({ constructor: 'add', args: [state.alloc.freshLayerId(`${stepPrefix}:res_add`)] });
      state.steps.push({ kind: 'binary', layer: addLayerIdx, slots: [fxSlot, xSlot, outSlot] });
      state.expansionLog.push({ kind: 'residual', target: targetSub, prefix: stepPrefix });
      continue;
    }

    if (op === 'parallel_concat') {
      const branches = st.branches;
      const concatDim = st.dim ?? 1;
      if (!Array.isArray(branches) || branches.length < 2) {
        throw new CompositionError('invalid_parallel_concat', `parallel_concat at ${stepPrefix} requires at least 2 branches`);
      }
      const outSlot = resolveOutSlot(rawOut);
      const branchOutSlots = [];
      for (let bi = 0; bi < branches.length; bi++) {
        const bOut = state.alloc.freshSlot(`${stepPrefix}:b${bi}`);
        branchOutSlots.push(bOut);
        const bSpec = branches[bi];
        const bSteps = Array.isArray(bSpec) ? clone(bSpec) : [{ ...bSpec, in: bSpec.in ?? inList, out: bOut }];
        const bEnv = new Map(scopeEnv);
        if (Array.isArray(bSpec) && bSteps.length > 0) {
          const lastIdx = bSteps.length - 1;
          bSteps[lastIdx] = { ...bSteps[lastIdx], out: bOut };
        }
        expandStepList(bSteps, bEnv, state, [...callStack, `branch#${bi}`], `${stepPrefix}.b${bi}`, baseDir, tiedCache);
      }
      const concatLayerIdx = state.layers.length;
      state.layers.push({ constructor: 'concat', args: [state.alloc.freshLayerId(`${stepPrefix}:concat`), concatDim] });
      let accSlot = branchOutSlots[0];
      for (let bi = 1; bi < branchOutSlots.length; bi++) {
        const isLast = bi === branchOutSlots.length - 1;
        const target = isLast ? outSlot : state.alloc.freshSlot(`${stepPrefix}:cat${bi}`);
        state.steps.push({ kind: 'binary', layer: concatLayerIdx, slots: [accSlot, branchOutSlots[bi], target] });
        accSlot = target;
      }
      state.expansionLog.push({ kind: 'parallel_concat', branches: branches.length, dim: concatDim, prefix: stepPrefix });
      continue;
    }

    if (op === 'ref' || st.$ref) {
      const refPath = st.$ref ?? st.file;
      if (!refPath) throw new CompositionError('missing_ref_path', `ref step at ${stepPrefix} requires '$ref' or 'file'`);
      const absPath = path.resolve(baseDir, refPath);
      if (callStack.includes(`file:${absPath}`)) {
        throw new CompositionError(
          'recursive_cycle_detected',
          `cyclic spec reference detected: ${[...callStack, `file:${absPath}`].join(' -> ')}`,
          'break the cycle between referenced spec files',
        );
      }
      const childSpec = JSON.parse(fs.readFileSync(absPath, 'utf8'));
      const inSlots = inList.map(resolveInSlot);
      const outSlot = resolveOutSlot(rawOut);
      inlineChildSpec(childSpec, inSlots, outSlot, state, [...callStack, `file:${absPath}`], stepPrefix, path.dirname(absPath));
      continue;
    }

    if (op.startsWith('subgraph:') || op.startsWith('macro:') || state.subgraphs[op] || MACROS[op]) {
      const cleanName = op.replace(/^(?:subgraph|macro):/, '');
      const inSlots = inList.map(resolveInSlot);
      const outSlot = resolveOutSlot(rawOut);

      if (!op.startsWith('macro:') && state.subgraphs[cleanName]) {
        const cycleToken = `subgraph:${cleanName}`;
        if (callStack.includes(cycleToken)) {
          throw new CompositionError(
            'recursive_cycle_detected',
            `recursive subgraph cycle detected: ${[...callStack, cycleToken].join(' -> ')}`,
            `use 'repeat' with an explicit finite 'depth' instead of unguarded self-recursion on '${cleanName}'`,
          );
        }
        const sg = state.subgraphs[cleanName];
        const sgInNames = sg.in ?? sg.inputs ?? [];
        if (sgInNames.length !== inSlots.length) {
          throw new CompositionError(
            'subgraph_arity_mismatch',
            `subgraph '${cleanName}' expects ${sgInNames.length} inputs (${sgInNames.join(', ')}), got ${inSlots.length} at ${stepPrefix}`,
          );
        }
        const childEnv = new Map();
        sgInNames.forEach((name, idx) => childEnv.set(String(name), inSlots[idx]));
        const sgOutName = String(sg.out ?? sg.output);
        childEnv.set(sgOutName, outSlot);

        const paramMap = { ...(sg.defaults ?? {}), ...(st.params ?? {}) };
        const instantiatedSteps = clone(sg.steps ?? []).map((cs) => substituteParams(cs, paramMap, st.weights));
        expandStepList(
          instantiatedSteps,
          childEnv,
          state,
          [...callStack, cycleToken],
          `${stepPrefix}:${cleanName}`,
          baseDir,
          tiedCache,
        );
        state.expansionLog.push({ kind: 'subgraph', name: cleanName, prefix: stepPrefix });
        continue;
      }

      inlineBuiltinMacro(cleanName, st.params ?? {}, st.weights, inSlots, outSlot, state, stepPrefix);
      continue;
    }

    const inSlots = inList.map(resolveInSlot);
    const outSlot = resolveOutSlot(rawOut);
    const arity = inSlots.length;
    if (arity !== 1 && arity !== 2) {
      throw new CompositionError(
        'invalid_primitive_arity',
        `primitive op '${op}' at ${stepPrefix} requires 1 or 2 inputs, got ${arity}`,
      );
    }

    const tieKey = tiedCache ? `${op}:${JSON.stringify(st.args ?? st.params ?? [])}:${st.tie_id ?? si}` : null;
    let layerIdx;
    if (tiedCache && tiedCache.has(tieKey)) {
      layerIdx = tiedCache.get(tieKey);
    } else {
      const regId = state.alloc.freshLayerId(`${stepPrefix}:${op}`);
      const rawArgs = st.args !== undefined
        ? (st.include_id === true ? st.args : [regId, ...st.args])
        : buildArgsFromParams(op, regId, st.params ?? {});
      layerIdx = state.layers.length;
      state.layers.push({ constructor: op, args: rawArgs });
      if (st.weights !== undefined) {
        state.weights[String(layerIdx)] = copyWeightVal(st.weights);
      }
      if (tiedCache) tiedCache.set(tieKey, layerIdx);
    }

    state.steps.push({
      kind: arity === 2 ? 'binary' : 'unary',
      layer: layerIdx,
      slots: [...inSlots, outSlot],
    });
  }
}

function substituteParams(stepObj, paramMap, callerWeights) {
  const s = clone(stepObj);
  const replaceVal = (v) => {
    if (typeof v === 'string' && v.startsWith('$')) {
      const k = v.slice(1);
      if (Object.hasOwn(paramMap, k)) return paramMap[k];
    }
    return v;
  };
  if (Array.isArray(s.args)) s.args = s.args.map(replaceVal);
  if (s.params && typeof s.params === 'object') {
    for (const [k, v] of Object.entries(s.params)) s.params[k] = replaceVal(v);
  }
  if (s.weights === undefined && callerWeights && s.weight_key !== undefined && callerWeights[s.weight_key]) {
    s.weights = callerWeights[s.weight_key];
  }
  return s;
}

function buildArgsFromParams(op, regId, params) {
  switch (op) {
    case 'linear':
      return [regId, params.in_dim, params.out_dim, params.bias ?? false];
    case 'layerNorm':
    case 'batchNorm':
    case 'instanceNorm':
    case 'rmsNorm':
      return params.eps !== undefined ? [regId, params.size ?? params.num_features ?? params.num_channels, params.eps] : [regId, params.size ?? params.num_features ?? params.num_channels];
    case 'groupNorm':
      return params.eps !== undefined ? [regId, params.num_groups, params.num_channels, params.eps] : [regId, params.num_groups, params.num_channels];
    case 'softmax':
    case 'logSoftmax':
    case 'concat':
    case 'glu':
      return [regId, params.dim ?? 1];
    case 'transpose':
      return [regId, params.dim0 ?? 2, params.dim1 ?? 3];
    case 'leakyRelu':
      return [regId, params.negative_slope ?? 0.01];
    case 'prelu':
      return [regId, params.num_parameters ?? 1, params.alpha ?? 0.25];
    case 'hardSigmoid':
      return [regId, params.alpha ?? 0.2, params.beta ?? 0.5];
    case 'softplus':
      return [regId, params.beta ?? 1.0];
    case 'featureNorm':
      return params.eps !== undefined ? [regId, params.eps] : [regId];
    case 'embedding':
      return [regId, params.vocab_size, params.d_model ?? params.dim];
    case 'avgPool1d':
    case 'maxPool1d':
      return [regId, params.kernel_size ?? params.k ?? 2, params.stride ?? null, params.padding ?? null];
    case 'avgPool2d':
    case 'maxPool2d':
      return [regId, params.kh ?? 2, params.kw ?? 2, params.sh ?? null, params.sw ?? null, params.ph ?? null, params.pw ?? null];
    case 'shiftUp':
    case 'shiftDown':
    case 'shiftLeft':
    case 'shiftRight':
      return [regId, params.shift_size ?? params.shift ?? 1];
    default:
      if (Array.isArray(params.args)) return [regId, ...params.args];
      return [regId];
  }
}

function inlineChildSpec(childSpec, callerInSlots, callerOutSlot, state, callStack, prefix, baseDir) {
  // B2 fix: thread the caller's file stack into the child compile
  // (callStack already holds 'root' + ancestor file:* tokens).
  const compiledChild = compileRecursiveSpec(childSpec, { baseDir, validate: false, parentFileStack: callStack });
  if (compiledChild.ports.length !== callerInSlots.length) {
    throw new CompositionError(
      'ref_port_arity_mismatch',
      `referenced spec at ${prefix} declares ${compiledChild.ports.length} ports, but caller supplied ${callerInSlots.length} inputs`,
    );
  }
  const slotRemap = new Map();
  compiledChild.ports.forEach((p, idx) => slotRemap.set(p.slot, callerInSlots[idx]));
  slotRemap.set(compiledChild.outputSlot, callerOutSlot);
  const mapSlot = (s) => {
    if (!slotRemap.has(s)) slotRemap.set(s, state.alloc.freshSlot(`${prefix}:s${s}`));
    return slotRemap.get(s);
  };
  const layerOffset = state.layers.length;
  for (const l of compiledChild.layers) {
    const newId = state.alloc.freshLayerId(`${prefix}:${l.constructor}`);
    state.layers.push({ constructor: l.constructor, args: [newId, ...l.args.slice(1)] });
  }
  for (const [k, w] of Object.entries(compiledChild.weights ?? {})) {
    state.weights[String(Number(k) + layerOffset)] = copyWeightVal(w);
  }
  for (const st of compiledChild.steps) {
    state.steps.push({
      kind: st.kind,
      layer: st.layer + layerOffset,
      slots: st.slots.map(mapSlot),
    });
  }
  state.expansionLog.push({ kind: 'ref_spec', prefix, layers_added: compiledChild.layers.length, steps_added: compiledChild.steps.length });
}

export function autoRepairSpec(spec, { maxWeightsPerLayer = 65535 } = {}) {
  let current = clone(spec);
  const receipts = [];

  if (current.steps.some((st) => INT_CONSTRUCTORS.has(current.layers[st.layer]?.constructor))) {
    const slotDtype = new Map();
    for (const p of current.ports) slotDtype.set(p.slot, 'f32');
    let nextRegId = Math.max(0, ...current.layers.map((l) => l.args[0] ?? 0)) + 1;
    let nextSlot = current.numSlots;
    const newSteps = [];

    for (let si = 0; si < current.steps.length; si++) {
      const st = current.steps[si];
      const decl = current.layers[st.layer];
      const ctor = decl?.constructor;
      const expectedIn = (INT_BINARY_CONSTRUCTORS.has(ctor) || ctor === 'intToFloat') ? 'i64' : 'f32';
      const inCount = st.kind === 'binary' ? 2 : 1;
      const fixedSlots = [...st.slots];

      for (let ii = 0; ii < inCount; ii++) {
        const s = fixedSlots[ii];
        const actual = slotDtype.get(s);
        if (actual && actual !== expectedIn) {
          const castCtor = expectedIn === 'i64' ? 'intFromFloat' : 'intToFloat';
          const castLayerIdx = current.layers.length;
          const castRegId = nextRegId++;
          const castSlot = nextSlot++;
          current.layers.push({ constructor: castCtor, args: [castRegId] });
          newSteps.push({ kind: 'unary', layer: castLayerIdx, slots: [s, castSlot] });
          slotDtype.set(castSlot, expectedIn);
          fixedSlots[ii] = castSlot;
          receipts.push({
            repair: 'insert_dtype_boundary_cast',
            at_step: si,
            input_position: ii,
            from_slot: s,
            to_slot: castSlot,
            cast_constructor: castCtor,
            reason: `step ${si} (${ctor}) expects ${expectedIn} but slot ${s} holds ${actual}`,
          });
        }
      }
      const outSlot = fixedSlots[inCount];
      const outType = (INT_BINARY_CONSTRUCTORS.has(ctor) || ctor === 'intFromFloat') ? 'i64' : 'f32';
      slotDtype.set(outSlot, outType);
      newSteps.push({ ...st, slots: fixedSlots });
    }

    if (slotDtype.get(current.outputSlot) === 'i64') {
      const castLayerIdx = current.layers.length;
      const castRegId = nextRegId++;
      const f32OutSlot = nextSlot++;
      current.layers.push({ constructor: 'intToFloat', args: [castRegId] });
      newSteps.push({ kind: 'unary', layer: castLayerIdx, slots: [current.outputSlot, f32OutSlot] });
      receipts.push({
        repair: 'insert_terminal_intToFloat',
        from_slot: current.outputSlot,
        to_slot: f32OutSlot,
        reason: 'mixed-dtype v1 requires f32 outputSlot',
      });
      current.outputSlot = f32OutSlot;
      slotDtype.set(f32OutSlot, 'f32');
    }

    current.steps = newSteps;
    current.numSlots = Math.max(current.numSlots, nextSlot);
  }

  const planCheck = analyzePlanPorts(current);
  if (!planCheck.valid && planCheck.code === 'plan_unused_ports') {
    let nextRegId = Math.max(0, ...current.layers.map((l) => l.args[0] ?? 0)) + 1;
    let nextSlot = current.numSlots;
    for (const unusedSlot of planCheck.unused_ports) {
      const port = current.ports.find((p) => p.slot === unusedSlot);
      const dim = port?.shape?.[1] ?? 1;
      const drainLayerIdx = current.layers.length;
      const drainRegId = nextRegId++;
      const drainOutSlot = nextSlot++;
      current.layers.push({ constructor: 'relu', args: [drainRegId] });
      current.steps.push({ kind: 'unary', layer: drainLayerIdx, slots: [unusedSlot, drainOutSlot] });
      receipts.push({
        repair: 'drain_unused_port',
        port_slot: unusedSlot,
        drain_slot: drainOutSlot,
        dim,
        reason: `port slot ${unusedSlot} was declared but not read by any step (plan_unused_ports)`,
      });
    }
    current.numSlots = nextSlot;
  }

  const oversized = [];
  for (let li = 0; li < current.layers.length; li++) {
    const decl = current.layers[li];
    const w = current.weights?.[String(li)];
    if (decl?.constructor === 'linear' && Array.isArray(w) && w.length > maxWeightsPerLayer) {
      oversized.push(li);
    }
  }
  if (oversized.length > 0) {
    let nextRegId = Math.max(0, ...current.layers.map((l) => l.args[0] ?? 0)) + 1;
    let nextSlot = current.numSlots;
    const shardSet = new Set(oversized);
    const newLayers = [];
    const newWeights = {};
    const layerRemap = new Map();

    for (let li = 0; li < current.layers.length; li++) {
      if (shardSet.has(li)) continue;
      layerRemap.set(li, newLayers.length);
      newLayers.push(current.layers[li]);
      if (current.weights?.[String(li)]) newWeights[String(newLayers.length - 1)] = current.weights[String(li)];
    }

    let concatLayerIdx = null;
    const newSteps = [];
    for (const st of current.steps) {
      if (!shardSet.has(st.layer)) {
        newSteps.push({ ...st, layer: layerRemap.get(st.layer) });
        continue;
      }
      const decl = current.layers[st.layer];
      const [, inDim, outDim, bias] = decl.args;
      const w = current.weights[String(st.layer)];
      const perRow = inDim + (bias ? 1 : 0);
      const blockOut = Math.max(1, Math.floor(maxWeightsPerLayer / perRow));
      const blocks = [];
      for (let r = 0; r < outDim; r += blockOut) blocks.push([r, Math.min(outDim, r + blockOut)]);

      const [inSlot, outSlot] = st.slots;
      const blockSlots = [];
      for (const [r0, r1] of blocks) {
        const blkIdx = newLayers.length;
        newLayers.push({ constructor: 'linear', args: [nextRegId++, inDim, r1 - r0, Boolean(bias)] });
        const slice = [];
        for (let k = 0; k < inDim; k++) {
          for (let i = r0; i < r1; i++) slice.push(w[k * outDim + i]);
        }
        if (bias) {
          for (let i = r0; i < r1; i++) slice.push(w[inDim * outDim + i]);
        }
        newWeights[String(blkIdx)] = slice;
        const bSlot = blocks.length === 1 ? outSlot : nextSlot++;
        blockSlots.push(bSlot);
        newSteps.push({ kind: 'unary', layer: blkIdx, slots: [inSlot, bSlot] });
      }
      if (blocks.length > 1) {
        if (concatLayerIdx === null) {
          concatLayerIdx = newLayers.length;
          newLayers.push({ constructor: 'concat', args: [nextRegId++, 1] });
        }
        let acc = blockSlots[0];
        for (let b = 1; b < blockSlots.length; b++) {
          const target = b === blockSlots.length - 1 ? outSlot : nextSlot++;
          newSteps.push({ kind: 'binary', layer: concatLayerIdx, slots: [acc, blockSlots[b], target] });
          acc = target;
        }
      }
      receipts.push({
        repair: 'recursive_shard_linear',
        layer_index: st.layer,
        in_dim: inDim,
        out_dim: outDim,
        blocks: blocks.map(([a, b]) => b - a),
        max_weights: maxWeightsPerLayer,
      });
    }
    current.layers = newLayers;
    current.steps = newSteps;
    current.weights = newWeights;
    current.numSlots = nextSlot;
  }

  return { spec: current, repaired: receipts.length > 0, remediation_receipt: receipts };
}

export function compileRecursiveSpec(rawSpec, {
  baseDir = process.cwd(),
  rt = null,
  useWorkspace = false,
  autoRepair = false,
  compactSlots = false,
  validate = true,
  // B2 fix: file-reference ancestry threaded through nested compiles.
  // Without this, inlineChildSpec compiled each child with a fresh
  // stack, so the file-cycle guard was unreachable and the depth
  // limit never accumulated across files. Null = top-level compile
  // (stack starts at ['root'], as before); nested compiles receive
  // the caller's stack verbatim (it already holds 'root' + file:*
  // tokens), so depth grows by exactly 1 per file level.
  parentFileStack = null,
} = {}) {
  const inputSpec = clone(rawSpec);

  let workspace = null;
  let registry = null;
  if (useWorkspace && rt) {
    workspace = new rt.AgentWorkspace(WORKSPACE_MAX_SLOTS);
    registry = new rt.LayerRegistry();
  }

  try {
    const preDeclaredLayers = Array.isArray(inputSpec.layers) ? clone(inputSpec.layers) : [];
    const maxPreId = preDeclaredLayers.length
      ? Math.max(...preDeclaredLayers.map((l) => l.args?.[0] ?? 0)) + 1
      : 0;
    const alloc = createAllocator({ idBase: maxPreId, slotBase: 0, workspace, registry });
    const ports = normalizePorts(inputSpec.ports, alloc);

    if (typeof inputSpec.numSlots === 'number' && inputSpec.numSlots > alloc.numSlots) {
      alloc.bindFixedSlot('__v1_cap__', inputSpec.numSlots - 1);
    }

    const state = {
      alloc,
      layers: preDeclaredLayers,
      v1LayerOffset: 0,
      steps: [],
      weights: clone(inputSpec.weights ?? {}),
      subgraphs: inputSpec.subgraphs ?? {},
      expansionLog: [],
    };

    const rootEnv = new Map();
    for (const [sym, slot] of Object.entries(alloc.symbolTable)) {
      if (sym !== '__v1_cap__') rootEnv.set(sym, slot);
    }

    expandStepList(inputSpec.steps ?? [], rootEnv, state, parentFileStack ? [...parentFileStack] : ['root'], 'root', baseDir, null);

    const rawOutRef = inputSpec.output ?? inputSpec.outputSlot;
    if (rawOutRef === undefined) {
      throw new CompositionError('missing_output_slot', 'spec must declare output (symbolic name) or outputSlot (number)');
    }
    const outputSlot = rootEnv.has(String(rawOutRef))
      ? rootEnv.get(String(rawOutRef))
      : alloc.resolveSlot(rawOutRef, 'output', false);

    const normalizedInputs = {};
    for (const [k, vals] of Object.entries(inputSpec.inputs ?? {})) {
      const slot = rootEnv.has(String(k))
        ? rootEnv.get(String(k))
        : alloc.resolveSlot(k, 'inputs', false);
      normalizedInputs[String(slot)] = vals;
    }

    let flatSpec = {
      numSlots: Math.max(alloc.numSlots, outputSlot + 1),
      layers: state.layers,
      steps: state.steps,
      outputSlot,
      ports,
      weights: state.weights,
      inputs: normalizedInputs,
    };

    let remediationReceipt = [];
    if (autoRepair) {
      const rep = autoRepairSpec(flatSpec);
      flatSpec = rep.spec;
      remediationReceipt = rep.remediation_receipt;
    }

    let compactionInfo = null;
    if (compactSlots) {
      const comp = compactSlotsByLiveness(flatSpec);
      flatSpec = comp.spec;
      compactionInfo = {
        slots_before: comp.slots_before,
        slots_after: comp.slots_after,
        slots_saved: comp.slots_saved,
      };
    }

    if (validate) {
      const argIssues = checkArgValues(flatSpec);
      if (argIssues.length > 0) {
        throw new CompositionError('invalid_argument', `compiled spec failed argument check: ${argIssues.join('; ')}`);
      }
      const shapeCheck = checkSpec(flatSpec);
      if (!shapeCheck.ok) {
        throw new CompositionError(
          'static_check_failed',
          `compiled spec failed static shape/dataflow check: ${JSON.stringify(shapeCheck.mismatches.concat(shapeCheck.dataflow.issues))}`,
        );
      }
    }

    let workspaceSnapshot = null;
    if (workspace) {
      workspace.recordEvent('compileRecursiveSpec', String(flatSpec.outputSlot), `layers=${flatSpec.layers.length};steps=${flatSpec.steps.length}`);
      workspaceSnapshot = JSON.parse(workspace.snapshot());
    }

    return {
      ...flatSpec,
      meta: {
        schema_compiled_from: inputSpec.schema ?? 'v1-compatible',
        symbol_table: alloc.symbolTable,
        expansions: state.expansionLog,
        remediation_receipt: remediationReceipt,
        slot_compaction: compactionInfo,
        workspace_snapshot: workspaceSnapshot,
      },
    };
  } finally {
    if (workspace && typeof workspace.free === 'function') {
      try { workspace.free(); } catch { /* ignore */ }
    }
    if (registry && typeof registry.free === 'function') {
      try { registry.free(); } catch { /* ignore */ }
    }
  }
}

function transposeColumnMajorWeights(weights, inDim, outDim) {
  const out = new Array(inDim * outDim);
  for (let j = 0; j < inDim; j++) {
    for (let i = 0; i < outDim; i++) {
      out[i * inDim + j] = weights[j * outDim + i];
    }
  }
  return out;
}

// Build column-major weights for `1 - z = (-I) z + 1` on a [B, D, 1, 1] tensor.
function oneMinusWeights(dim) {
  const w = [];
  for (let j = 0; j < dim; j++) {
    for (let i = 0; i < dim; i++) w.push(i === j ? -1 : 0);
  }
  for (let i = 0; i < dim; i++) w.push(1);
  return w;
}

function resolveWrtSlot(forwardSpec, wrtSlot) {
  if (typeof wrtSlot === 'number' && Number.isInteger(wrtSlot) && wrtSlot >= 0) return wrtSlot;
  const k = String(wrtSlot);
  if (/^(?:0|[1-9]\d*)$/.test(k)) return Number(k);
  const symTable = forwardSpec.meta?.symbol_table ?? {};
  if (Object.hasOwn(symTable, k)) return symTable[k];
  throw new CompositionError(
    'unresolved_wrt_slot',
    `cannot resolve --wrt '${k}' to a slot index (not a non-negative integer or a symbolic slot in spec.meta.symbol_table)`,
  );
}

export function compileBackwardSpec(forwardSpec, { wrtSlot = 0, doutValues = null } = {}) {
  const resolvedWrt = resolveWrtSlot(forwardSpec, wrtSlot);
  // SSA-normalize forward steps first so slot reuse (--compact-slots or in-place updates)
  // never overwrites saved forward activations needed by backward steps!
  const { ssaSpec: s, finalSlotMap } = ssaNormalizeSpec(forwardSpec);
  const shapeInfo = checkSpec(s);
  if (!shapeInfo.ok) {
    throw new CompositionError('static_check_failed', `cannot differentiate invalid forward spec: ${JSON.stringify(shapeInfo.mismatches)}`);
  }

  const slotShapes = inferShapes(clone(s));
  const outShape = slotShapes.get(s.outputSlot);
  if (!outShape) {
    throw new CompositionError('unknown_output_shape', `cannot infer shape for outputSlot ${s.outputSlot}`);
  }
  const outNumel = outShape.reduce((a, b) => a * b, 1);

  let nextRegId = Math.max(0, ...s.layers.map((l) => l.args[0] ?? 0)) + 1;
  let nextSlot = s.numSlots;

  const layers = clone(s.layers);
  const steps = s.steps.map((st) => ({ kind: st.kind, layer: st.layer, slots: [...st.slots] }));
  const weights = clone(s.weights ?? {});
  const ports = clone(s.ports);
  const inputs = clone(s.inputs ?? {});

  const doutSlot = nextSlot++;
  const doutLayout = (outShape[2] === 1 && outShape[3] === 1)
    ? 'feature_axis1_singleton'
    : 'any_rank4';
  ports.push({
    slot: doutSlot,
    role: 'context',
    shape: [...outShape],
    layout: doutLayout,
  });
  inputs[String(doutSlot)] = doutValues ?? new Array(outNumel).fill(1);

  const gradSlotOf = new Map();
  const savedGradSlotOf = new Map();
  gradSlotOf.set(s.outputSlot, doutSlot);
  savedGradSlotOf.set(s.outputSlot, doutSlot);

  const accumulateGrad = (targetForwardSlot, contribGradSlot) => {
    if (!gradSlotOf.has(targetForwardSlot)) {
      gradSlotOf.set(targetForwardSlot, contribGradSlot);
      savedGradSlotOf.set(targetForwardSlot, contribGradSlot);
      return;
    }
    const prevSlot = gradSlotOf.get(targetForwardSlot);
    const sumSlot = nextSlot++;
    const addIdx = layers.length;
    layers.push({ constructor: 'add', args: [nextRegId++] });
    steps.push({ kind: 'binary', layer: addIdx, slots: [prevSlot, contribGradSlot, sumSlot] });
    gradSlotOf.set(targetForwardSlot, sumSlot);
    savedGradSlotOf.set(targetForwardSlot, sumSlot);
  };

  // Emit `1 - z` via a biased linear(-I, +1) step for [B, D, 1, 1] tensors
  const emitOneMinus = (inSlot, sh) => {
    if (sh[2] !== 1 || sh[3] !== 1) {
      throw new CompositionError('unsupported_shape_for_polynomial_vjp', `in-graph sigmoid/tanh VJP requires [B, D, 1, 1] shape, got [${sh}]`);
    }
    const dim = sh[1];
    const linIdx = layers.length;
    layers.push({ constructor: 'linear', args: [nextRegId++, dim, dim, true] });
    weights[String(linIdx)] = oneMinusWeights(dim);
    const resSlot = nextSlot++;
    steps.push({ kind: 'unary', layer: linIdx, slots: [inSlot, resSlot] });
    return resSlot;
  };

  const SHIFT_INV = {
    shiftUp: 'shiftDown',
    shiftDown: 'shiftUp',
    shiftLeft: 'shiftRight',
    shiftRight: 'shiftLeft',
  };

  for (let si = s.steps.length - 1; si >= 0; si--) {
    const st = s.steps[si];
    const decl = s.layers[st.layer];
    const ctor = decl.constructor;
    const outSlot = st.slots[st.slots.length - 1];
    const upGradSlot = gradSlotOf.get(outSlot);
    if (upGradSlot === undefined) continue;
    savedGradSlotOf.set(outSlot, upGradSlot);
    gradSlotOf.delete(outSlot);

    if (ctor === 'linear') {
      const [, inDim, outDim, bias] = decl.args;
      const w = s.weights?.[String(st.layer)] ?? new Array(inDim * outDim + (bias ? outDim : 0)).fill(0);
      const wT = transposeColumnMajorWeights(w, inDim, outDim);
      const linTIdx = layers.length;
      layers.push({ constructor: 'linear', args: [nextRegId++, outDim, inDim, false] });
      weights[String(linTIdx)] = wT;
      const dxSlot = nextSlot++;
      steps.push({ kind: 'unary', layer: linTIdx, slots: [upGradSlot, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else if (ctor === 'add') {
      accumulateGrad(st.slots[0], upGradSlot);
      accumulateGrad(st.slots[1], upGradSlot);
    } else if (ctor === 'sub') {
      accumulateGrad(st.slots[0], upGradSlot);
      const zeroSlot = nextSlot++;
      const negSlot = nextSlot++;
      const subIdx = layers.length;
      layers.push({ constructor: 'sub', args: [nextRegId++] });
      steps.push({ kind: 'binary', layer: subIdx, slots: [upGradSlot, upGradSlot, zeroSlot] });
      steps.push({ kind: 'binary', layer: subIdx, slots: [zeroSlot, upGradSlot, negSlot] });
      accumulateGrad(st.slots[1], negSlot);
    } else if (ctor === 'mul') {
      const [aSlot, bSlot] = st.slots;
      const mulIdx = layers.length;
      layers.push({ constructor: 'mul', args: [nextRegId++] });
      const daSlot = nextSlot++;
      const dbSlot = nextSlot++;
      steps.push({ kind: 'binary', layer: mulIdx, slots: [upGradSlot, bSlot, daSlot] });
      steps.push({ kind: 'binary', layer: mulIdx, slots: [upGradSlot, aSlot, dbSlot] });
      accumulateGrad(aSlot, daSlot);
      accumulateGrad(bSlot, dbSlot);
    } else if (ctor === 'div') {
      // y = a / b => dA = G / b, dB = -(G * y) / b
      const [aSlot, bSlot] = st.slots;
      const divIdx = layers.length;
      layers.push({ constructor: 'div', args: [nextRegId++] });
      const mulIdx = layers.length;
      layers.push({ constructor: 'mul', args: [nextRegId++] });
      const subIdx = layers.length;
      layers.push({ constructor: 'sub', args: [nextRegId++] });
      const daSlot = nextSlot++;
      const gySlot = nextSlot++;
      const posDbSlot = nextSlot++;
      const zeroSlot = nextSlot++;
      const dbSlot = nextSlot++;
      steps.push({ kind: 'binary', layer: divIdx, slots: [upGradSlot, bSlot, daSlot] });
      steps.push({ kind: 'binary', layer: mulIdx, slots: [upGradSlot, outSlot, gySlot] });
      steps.push({ kind: 'binary', layer: divIdx, slots: [gySlot, bSlot, posDbSlot] });
      steps.push({ kind: 'binary', layer: subIdx, slots: [posDbSlot, posDbSlot, zeroSlot] });
      steps.push({ kind: 'binary', layer: subIdx, slots: [zeroSlot, posDbSlot, dbSlot] });
      accumulateGrad(aSlot, daSlot);
      accumulateGrad(bSlot, dbSlot);
    } else if (ctor === 'sqrt') {
      // y = sqrt(x) => dX = G / (y + y)
      const addIdx = layers.length;
      layers.push({ constructor: 'add', args: [nextRegId++] });
      const divIdx = layers.length;
      layers.push({ constructor: 'div', args: [nextRegId++] });
      const twoYSlot = nextSlot++;
      const dxSlot = nextSlot++;
      steps.push({ kind: 'binary', layer: addIdx, slots: [outSlot, outSlot, twoYSlot] });
      steps.push({ kind: 'binary', layer: divIdx, slots: [upGradSlot, twoYSlot, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else if (ctor === 'sigmoid') {
      const sh = slotShapes.get(outSlot);
      const oneMinusY = emitOneMinus(outSlot, sh);
      const mulIdx = layers.length;
      layers.push({ constructor: 'mul', args: [nextRegId++] });
      const yOneMinusY = nextSlot++;
      const dxSlot = nextSlot++;
      steps.push({ kind: 'binary', layer: mulIdx, slots: [outSlot, oneMinusY, yOneMinusY] });
      steps.push({ kind: 'binary', layer: mulIdx, slots: [upGradSlot, yOneMinusY, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else if (ctor === 'tanh') {
      const sh = slotShapes.get(outSlot);
      const mulIdx = layers.length;
      layers.push({ constructor: 'mul', args: [nextRegId++] });
      const ySq = nextSlot++;
      steps.push({ kind: 'binary', layer: mulIdx, slots: [outSlot, outSlot, ySq] });
      const oneMinusYSq = emitOneMinus(ySq, sh);
      const dxSlot = nextSlot++;
      steps.push({ kind: 'binary', layer: mulIdx, slots: [upGradSlot, oneMinusYSq, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else if (ctor === 'transpose') {
      const [, d0, d1] = decl.args;
      const trIdx = layers.length;
      layers.push({ constructor: 'transpose', args: [nextRegId++, d0, d1] });
      const dxSlot = nextSlot++;
      steps.push({ kind: 'unary', layer: trIdx, slots: [upGradSlot, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else if (ctor === 'matmul') {
      const [aSlot, bSlot] = st.slots;
      const trIdx = layers.length;
      layers.push({ constructor: 'transpose', args: [nextRegId++, 2, 3] });
      const mmIdx = layers.length;
      layers.push({ constructor: 'matmul', args: [nextRegId++] });
      const bTSlot = nextSlot++;
      const aTSlot = nextSlot++;
      const daSlot = nextSlot++;
      const dbSlot = nextSlot++;
      steps.push({ kind: 'unary', layer: trIdx, slots: [bSlot, bTSlot] });
      steps.push({ kind: 'binary', layer: mmIdx, slots: [upGradSlot, bTSlot, daSlot] });
      steps.push({ kind: 'unary', layer: trIdx, slots: [aSlot, aTSlot] });
      steps.push({ kind: 'binary', layer: mmIdx, slots: [aTSlot, upGradSlot, dbSlot] });
      accumulateGrad(aSlot, daSlot);
      accumulateGrad(bSlot, dbSlot);
    } else if (SHIFT_INV[ctor]) {
      const [, shiftSize] = decl.args;
      const shIdx = layers.length;
      layers.push({ constructor: SHIFT_INV[ctor], args: [nextRegId++, shiftSize] });
      const dxSlot = nextSlot++;
      steps.push({ kind: 'unary', layer: shIdx, slots: [upGradSlot, dxSlot] });
      accumulateGrad(st.slots[0], dxSlot);
    } else {
      throw new CompositionError(
        'unsupported_symbolic_grad_op',
        `symbolic graph differentiation does not support constructor '${ctor}' at step ${si} (supported in-graph VJP: linear, add, sub, mul, div, sqrt, sigmoid, tanh, transpose, matmul, shift*)`,
      );
    }
  }

  const isPortSlot = (s.ports ?? []).some((p) => p.slot === resolvedWrt);
  const ssaWrtSlot = isPortSlot ? resolvedWrt : (finalSlotMap?.get(resolvedWrt) ?? resolvedWrt);
  let targetGradSlot = isPortSlot
    ? (gradSlotOf.get(resolvedWrt) ?? savedGradSlotOf.get(resolvedWrt) ?? gradSlotOf.get(ssaWrtSlot) ?? savedGradSlotOf.get(ssaWrtSlot))
    : (gradSlotOf.get(ssaWrtSlot) ?? savedGradSlotOf.get(ssaWrtSlot) ?? gradSlotOf.get(resolvedWrt) ?? savedGradSlotOf.get(resolvedWrt));

  if (targetGradSlot === undefined) {
    const zeroRefSlot = slotShapes.has(resolvedWrt) ? resolvedWrt : (slotShapes.has(ssaWrtSlot) ? ssaWrtSlot : 0);
    const subIdx = layers.length;
    layers.push({ constructor: 'sub', args: [nextRegId++] });
    targetGradSlot = nextSlot++;
    steps.push({ kind: 'binary', layer: subIdx, slots: [zeroRefSlot, zeroRefSlot, targetGradSlot] });
  }

  let compiled = {
    numSlots: nextSlot,
    layers,
    steps,
    outputSlot: targetGradSlot,
    ports,
    weights,
    inputs,
  };
  compiled = autoRepairSpec(compiled).spec;

  return {
    ...compiled,
    meta: {
      ...(forwardSpec.meta ?? {}),
      kind: 'symbolic_backward_spec',
      wrt_slot: resolvedWrt,
      dout_slot: doutSlot,
      forward_output_slot: s.outputSlot,
    },
  };
}
