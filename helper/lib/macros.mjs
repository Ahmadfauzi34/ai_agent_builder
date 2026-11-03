// burn-math #1: macro/sub-graph expansion (v3.6: includes stateful recurrent_cell
// and recursive depth unrolling with optional weight-tying).
//
// A macro is a named composite block built from primitive layer
// constructors. expandMacro() rewrites one macro invocation (or a depth-D
// recursive unrolling of it) into a flat spec fragment (layers + steps + weights)
// with namespaced layer ids and freshly allocated intermediate slots.
// The expansion is PROVEN by running it against a hand-written flat reference
// on the engine and requiring bit-identical outputs (see bin/macro_expand.mjs prove).
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
  recurrent_cell: {
    description: 'stateful residual recurrent cell: s_next = tanh(W_s * s + W_x * x) + s',
    params: ['state_dim', 'in_dim', 'state_slot', 'in_slot', 'out_slot'],
    expand(p, alloc) {
      const sDim = p.state_dim ?? need(p, 'in_dim');
      const inDim = p.in_dim ?? sDim;
      const sState = p.state_slot ?? need(p, 'in_slot');
      const sIn = p.in_slot ?? sState;
      const sOut = need(p, 'out_slot');
      const bias = p.bias ?? true;
      const sWs = alloc.slot(), sWx = alloc.slot(), sSum = alloc.slot(), sAct = alloc.slot();
      const l0 = alloc.layerId(), l1 = alloc.layerId(), l2 = alloc.layerId(), l3 = alloc.layerId(), l4 = alloc.layerId();
      return {
        layers: [
          lin(l0, sDim, sDim, bias),
          lin(l1, inDim, sDim, false),
          { constructor: 'add', args: [l2] },
          { constructor: 'tanh', args: [l3] },
          { constructor: 'add', args: [l4] },
        ],
        steps: [
          unary(0, sState, sWs),
          unary(1, sIn, sWx),
          binary(2, sWs, sWx, sSum),
          unary(3, sSum, sAct),
          binary(4, sAct, sState, sOut),
        ],
        weighted: [0, 1],
      };
    },
  },
  attention_head: {
    description: 'single-head dot-product attention: softmax(Q @ K^T, dim=keys) @ V. Unscaled (fold 1/sqrt(d) into Q upstream if needed). Layouts (explicit — the chain is layout-specific, and the wrong layout silently computes something else): seq_axis1 (default): Q/K/V are [B,S,D,*] (sequence axis 1, dim axis 2); heads_axis1: [B,H,S,D] (heads axis 1, sequence axis 2, dim axis 3). Cross-attention (Sq != Skv) works in both layouts; output seq = Sq.',
    params: ['d_model', 'q_slot', 'k_slot', 'v_slot', 'out_slot', 'layout'],
    expand(p, alloc) {
      need(p, 'd_model');
      const sQ = need(p, 'q_slot'), sK = need(p, 'k_slot'), sV = need(p, 'v_slot');
      const sOut = need(p, 'out_slot');
      const layout = p.layout ?? 'seq_axis1';
      if (layout === 'heads_axis1') {
        // [B,H,S,D] layout: K^T = transpose(2,3)(K) -> [B,H,D,Skv];
        // scores = Q @ K^T -> [B,H,Sq,Skv]; W = softmax(scores, dim=3);
        // out = W @ V -> [B,H,Sq,D].
        const sTk = alloc.slot(), sScores = alloc.slot(), sW = alloc.slot();
        const L = [];
        for (let i = 0; i < 4; i++) L.push(alloc.layerId());
        return {
          layers: [
            { constructor: 'transpose', args: [L[0], 2, 3] },
            { constructor: 'matmul', args: [L[1]] },
            { constructor: 'softmax', args: [L[2], 3] },
            { constructor: 'matmul', args: [L[3]] },
          ],
          steps: [
            unary(0, sK, sTk),
            binary(1, sQ, sTk, sScores),
            unary(2, sScores, sW),
            binary(3, sW, sV, sOut),
          ],
          weighted: [],
        };
      }
      if (layout !== 'seq_axis1') throw new Error(`attention_head layout must be 'seq_axis1' or 'heads_axis1' (got ${JSON.stringify(p.layout)})`);
      // seq_axis1 (default, pinned by incident test 2026-10-22):
      // Q/K/V are [B,S,D,*] (sequence axis 1, dim axis 2). The chain
      // transposes into [B,*,S,D] form, scores [B,*,Sq,Skv], softmax
      // dim 3 over keys, and transposes back to [B,Sq,Dv,*].
      // The engine matmul contracts the last two axes, so QK^T is built via
      // a [B,1] batch trick: Q -> [B,1,Sq,F], K -> [B,1,F,Skv].
      const sTq1 = alloc.slot(), sTq2 = alloc.slot(), sTk = alloc.slot();
      const sScores = alloc.slot(), sW = alloc.slot();
      const sTv1 = alloc.slot(), sTv2 = alloc.slot(), sO = alloc.slot(), sTo = alloc.slot();
      const L = [];
      for (let i = 0; i < 10; i++) L.push(alloc.layerId());
      return {
        layers: [
          { constructor: 'transpose', args: [L[0], 2, 3] }, // Q: [B,Sq,F,1] -> [B,Sq,1,F]
          { constructor: 'transpose', args: [L[1], 1, 2] }, //   -> [B,1,Sq,F]
          { constructor: 'transpose', args: [L[2], 1, 3] }, // K: [B,Skv,F,1] -> [B,1,F,Skv]
          { constructor: 'matmul', args: [L[3]] },          // scores [B,1,Sq,Skv]
          { constructor: 'softmax', args: [L[4], 3] },      // weights over keys (axis 3)
          { constructor: 'transpose', args: [L[5], 2, 3] }, // V: [B,Skv,F,1] -> [B,Skv,1,F]
          { constructor: 'transpose', args: [L[6], 1, 2] }, //   -> [B,1,Skv,F]
          { constructor: 'matmul', args: [L[7]] },          // out [B,1,Sq,F]
          { constructor: 'transpose', args: [L[8], 1, 2] }, //   -> [B,Sq,1,F]
          { constructor: 'transpose', args: [L[9], 2, 3] }, //   -> [B,Sq,F,1]
        ],
        steps: [
          unary(0, sQ, sTq1), unary(1, sTq1, sTq2),
          unary(2, sK, sTk),
          binary(3, sTq2, sTk, sScores),
          unary(4, sScores, sW),
          unary(5, sV, sTv1), unary(6, sTv1, sTv2),
          binary(7, sW, sTv2, sO),
          unary(8, sO, sTo), unary(9, sTo, sOut),
        ],
        weighted: [],
      };
    },
  },
};

// Expand one macro invocation (or a depth-D recursive unrolling of it) into a complete runnable spec.
// invocation: {macro, params, depth?: N, tie_weights?: bool, weights?, weights_per_depth?, ports, inputs, idBase?, slotBase?, numSlots?}
export function expandMacro(inv) {
  const def = MACROS[inv.macro];
  if (!def) throw new Error(`unknown macro '${inv.macro}' (known: ${Object.keys(MACROS).join(', ')})`);
  const p = { ...(inv.params ?? {}) };
  const depth = Number(inv.depth ?? p.depth ?? 1);
  if (!Number.isInteger(depth) || depth < 1 || depth > 32) {
    throw new Error(`macro depth must be an integer in 1..32, got ${JSON.stringify(inv.depth ?? p.depth)}`);
  }
  const tieWeights = Boolean(inv.tie_weights ?? p.tie_weights ?? false);
  const idBase = inv.idBase ?? 0;
  let slotBase = inv.slotBase;
  const primaryInKey = inv.macro === 'recurrent_cell'
    ? (p.state_slot !== undefined ? 'state_slot' : 'in_slot')
    : (inv.macro === 'attention_head' ? 'q_slot' : 'in_slot');
  const initialInSlot = p[primaryInKey];
  const finalOutSlot = inv.outputSlot ?? p.out_slot;
  if (slotBase === undefined) {
    const mentioned = [
      p.in_slot, p.state_slot, p.q_slot, p.k_slot, p.v_slot, p.out_slot,
      ...(inv.ports ?? []).map((x) => x.slot), finalOutSlot,
    ];
    slotBase = Math.max(...mentioned.filter((x) => x !== undefined)) + 1;
  }
  const alloc = makeAlloc(idBase, slotBase);
  const layers = [];
  const steps = [];
  const weights = {};

  // Remap macro-local layer indices to global indices (per-depth offset;
  // tied depths reuse the depth-0 layer indices).
  let curInSlot = initialInSlot;
  for (let d = 0; d < depth; d++) {
    const iterOutSlot = d === depth - 1 ? finalOutSlot : alloc.slot();
    const iterParams = { ...p, [primaryInKey]: curInSlot, out_slot: iterOutSlot };
    const frag = def.expand(iterParams, alloc);

    if (d === 0 || !tieWeights) {
      const layerOffset = layers.length;
      for (const l of frag.layers) layers.push(l);
      for (const s of frag.steps) {
        steps.push({ ...s, layer: s.layer + layerOffset });
      }
      const depthWeights = (Array.isArray(inv.weights_per_depth) && !tieWeights)
        ? inv.weights_per_depth[d]
        : inv.weights;
      for (const li of frag.weighted ?? []) {
        const w = depthWeights?.[String(li)];
        if (!w) throw new Error(`macro '${inv.macro}' needs weights for macro-local layer ${li}`);
        weights[String(li + layerOffset)] = w;
      }
    } else {
      // Weight-tied recurrence (d >= 1): reuse the exact layer indices created at depth 0!
      for (const s of frag.steps) {
        steps.push({ ...s, layer: s.layer });
      }
    }
    curInSlot = iterOutSlot;
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
    const lid = idBase + alloc.layersUsed() + layers.length;
    const sOut = alloc.slot();
    layers.push({ constructor: 'linear', args: [lid, dim, dim, false] });
    steps.push({ kind: 'unary', layer: layers.length - 1, slots: [port.slot, sOut] });
    const eye = [];
    for (let k = 0; k < dim; k++) for (let i = 0; i < dim; i++) eye.push(i === k ? 1 : 0);
    weights[String(layers.length - 1)] = eye;
    readSlots.add(port.slot);
  }
  const numSlots = Math.max(inv.numSlots ?? 0, slotBase + alloc.slotsUsed(), (finalOutSlot ?? 0) + 1);
  return {
    layers, steps, weights, numSlots,
    outputSlot: finalOutSlot,
    ports: inv.ports ?? [],
    inputs: inv.inputs ?? {},
    meta: { macro: inv.macro, params: p, depth, tie_weights: tieWeights, idBase, slotBase },
  };
}

export function macroNames() {
  return Object.keys(MACROS);
}
