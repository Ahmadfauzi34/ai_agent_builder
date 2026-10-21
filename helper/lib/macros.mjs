// burn-math #1: macro/sub-graph expansion.
//
// A macro is a named composite block built from primitive layer
// constructors. expandMacro() rewrites one macro invocation into a flat
// spec fragment (layers + steps + weights) with namespaced layer ids and
// freshly allocated intermediate slots. The expansion is PROVEN by running
// it against a hand-written flat reference on the engine and requiring
// bit-identical outputs (see bin/macro_expand.mjs prove).
//
// Slot model: the caller names the macro's input/output slots; all
// intermediate slots are allocated by the expander starting at slotBase
// (default: one past the highest slot mentioned in params/ports).

function need(p, name) {
  if (p[name] === undefined) throw new Error(`macro param '${name}' is required`);
  return p[name];
}

function makeAlloc(idBase, slotBase) {
  let lid = idBase, slot = slotBase;
  return {
    layerId: () => lid++,
    slot: () => slot++,
    layersUsed: () => lid - idBase,
    slotsUsed: () => slot - slotBase,
  };
}

const lin = (id, inDim, outDim, bias = true) => ({ constructor: 'linear', args: [id, inDim, outDim, bias] });
const unary = (layer, a, b) => ({ kind: 'unary', layer, slots: [a, b] });
const binary = (layer, a, b, c) => ({ kind: 'binary', layer, slots: [a, b, c] });

export const MACROS = {
  mlp: {
    description: 'linear -> relu -> linear feedforward block',
    params: ['in_dim', 'hidden_dim', 'out_dim', 'in_slot', 'out_slot'],
    expand(p, alloc) {
      const inDim = need(p, 'in_dim'), hid = need(p, 'hidden_dim'), outDim = need(p, 'out_dim');
      const sIn = need(p, 'in_slot'), sOut = need(p, 'out_slot');
      const sH1 = alloc.slot(), sH2 = alloc.slot();
      const l0 = alloc.layerId(), l1 = alloc.layerId(), l2 = alloc.layerId();
      return {
        layers: [lin(l0, inDim, hid), { constructor: 'relu', args: [l1] }, lin(l2, hid, outDim)],
        steps: [unary(0, sIn, sH1), unary(1, sH1, sH2), unary(2, sH2, sOut)],
        weighted: [0, 2], // macro-local layer indices expecting weights
      };
    },
  },
  gated_unit: {
    description: 'sigmoid(linear(x)) * linear(x) gating unit (mul is elementwise)',
    params: ['in_dim', 'dim', 'in_slot', 'out_slot'],
    expand(p, alloc) {
      const inDim = need(p, 'in_dim'), d = need(p, 'dim');
      const sIn = need(p, 'in_slot'), sOut = need(p, 'out_slot');
      const sA1 = alloc.slot(), sA2 = alloc.slot(), sB = alloc.slot();
      const l0 = alloc.layerId(), l1 = alloc.layerId(), l2 = alloc.layerId(), l3 = alloc.layerId();
      return {
        layers: [
          lin(l0, inDim, d),
          { constructor: 'sigmoid', args: [l1] },
          lin(l2, inDim, d),
          { constructor: 'mul', args: [l3] },
        ],
        steps: [unary(0, sIn, sA1), unary(1, sA1, sA2), unary(2, sIn, sB), binary(3, sA2, sB, sOut)],
        weighted: [0, 2],
      };
    },
  },
  residual_mlp: {
    description: 'mlp block with an add skip connection; requires in_dim == out_dim',
    params: ['in_dim', 'hidden_dim', 'in_slot', 'out_slot'],
    expand(p, alloc) {
      const inDim = need(p, 'in_dim'), hid = need(p, 'hidden_dim');
      const outDim = inDim;
      const sIn = need(p, 'in_slot'), sOut = need(p, 'out_slot');
      if (p.out_dim !== undefined && p.out_dim !== inDim)
        throw new Error('residual_mlp requires in_dim == out_dim');
      const sH1 = alloc.slot(), sH2 = alloc.slot(), sH3 = alloc.slot();
      const l0 = alloc.layerId(), l1 = alloc.layerId(), l2 = alloc.layerId(), l3 = alloc.layerId();
      return {
        layers: [
          lin(l0, inDim, hid),
          { constructor: 'relu', args: [l1] },
          lin(l2, hid, outDim),
          { constructor: 'add', args: [l3] },
        ],
        steps: [
          unary(0, sIn, sH1), unary(1, sH1, sH2), unary(2, sH2, sH3),
          binary(3, sH3, sIn, sOut),
        ],
        weighted: [0, 2],
      };
    },
  },
  transformer_ffn: {
    description: 'transformer feed-forward block: linear -> gelu -> linear + residual add; requires in_dim == out_dim (d_ff defaults to 4*d_model)',
    params: ['d_model', 'd_ff', 'in_slot', 'out_slot'],
    expand(p, alloc) {
      const d = need(p, 'd_model'), dff = p.d_ff ?? 4 * d;
      const sIn = need(p, 'in_slot'), sOut = need(p, 'out_slot');
      const sH1 = alloc.slot(), sH2 = alloc.slot(), sH3 = alloc.slot();
      const l0 = alloc.layerId(), l1 = alloc.layerId(), l2 = alloc.layerId(), l3 = alloc.layerId();
      return {
        layers: [
          lin(l0, d, dff),
          { constructor: 'gelu', args: [l1] },
          lin(l2, dff, d),
          { constructor: 'add', args: [l3] },
        ],
        steps: [
          unary(0, sIn, sH1), unary(1, sH1, sH2), unary(2, sH2, sH3),
          binary(3, sH3, sIn, sOut),
        ],
        weighted: [0, 2],
      };
    },
  },
};

// Expand one macro invocation into a complete runnable spec.
// invocation: {macro, params, weights? (macro-local layer idx -> flat array),
//   ports, inputs, idBase?, slotBase?, numSlots?}
export function expandMacro(inv) {
  const def = MACROS[inv.macro];
  if (!def) throw new Error(`unknown macro '${inv.macro}' (known: ${Object.keys(MACROS).join(', ')})`);
  const p = inv.params ?? {};
  const idBase = inv.idBase ?? 0;
  let slotBase = inv.slotBase;
  if (slotBase === undefined) {
    const mentioned = [p.in_slot, p.out_slot, ...(inv.ports ?? []).map((x) => x.slot), inv.outputSlot ?? p.out_slot];
    slotBase = Math.max(...mentioned.filter((x) => x !== undefined)) + 1;
  }
  const alloc = makeAlloc(idBase, slotBase);
  const frag = def.expand(p, alloc);
  // Remap macro-local layer indices to global (idBase-based) indices.
  const layers = frag.layers;
  const steps = frag.steps.map((s) => ({ ...s, layer: s.layer + idBase }));
  const weights = {};
  for (const li of frag.weighted ?? []) {
    const w = inv.weights?.[String(li)];
    if (!w) throw new Error(`macro '${inv.macro}' needs weights for macro-local layer ${li}`);
    weights[String(li + idBase)] = w;
  }
  // Drain otherwise-unused input ports through an identity linear so the
  // spec passes engine port validation (minimum 2 ports, all consumed).
  // The drain writes to slots nothing reads: semantically a no-op.
  const readSlots = new Set();
  for (const s of steps) {
    const nRead = s.kind === 'binary' ? 2 : 1;
    for (let i = 0; i < nRead; i++) readSlots.add(s.slots[i]);
  }
  for (const port of inv.ports ?? []) {
    if (readSlots.has(port.slot)) continue;
    const dim = port.shape[1];
    const lid = idBase + layers.length;
    const sOut = alloc.slot();
    layers.push({ constructor: 'linear', args: [lid, dim, dim, false] });
    steps.push({ kind: 'unary', layer: layers.length - 1, slots: [port.slot, sOut] });
    const eye = [];
    for (let k = 0; k < dim; k++) for (let i = 0; i < dim; i++) eye.push(i === k ? 1 : 0);
    weights[String(layers.length - 1)] = eye;
    readSlots.add(port.slot);
  }
  const numSlots = Math.max(inv.numSlots ?? 0, slotBase + alloc.slotsUsed(), (p.out_slot ?? 0) + 1);
  return {
    layers, steps, weights, numSlots,
    outputSlot: inv.outputSlot ?? p.out_slot,
    ports: inv.ports ?? [],
    inputs: inv.inputs ?? {},
    meta: { macro: inv.macro, params: p, idBase, slotBase },
  };
}

export function macroNames() {
  return Object.keys(MACROS);
}
