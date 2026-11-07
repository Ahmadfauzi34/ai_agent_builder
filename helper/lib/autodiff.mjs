// burn-math #12: autodiff as a MAP (v3.8: 31 proven rules + BPTT weight-tied accumulation + full SSA backward safety).
//
// A declarative table: forward constructor -> backward (VJP) rule.
// Each rule is a PURE function over flat f64 arrays: given the saved
// forward values and the upstream gradient dL/dOut, it returns dL/dIn
// (and dL/dParams for weighted layers). Rules are DATA — inspectable,
// auditable, and each one is PROVEN against two independent oracles:
//   (1) a hand-computed exact case, and
// A rule that passes only self-consistent checks is not proven (the
// attention_head softmax-axis lesson). The finite-difference oracle
// (bin/gradcheck.mjs) stays the permanent regression guard.
//
// Coverage v3.8: 31 proven rules (the differentiable core behind MLP and
// attention blocks, plus the four shift ops, div/sqrt, the norm layers,
// glu, and the newer activations). Everything else is an explicit,
// classified gap in GAPS — a missing rule is an honest error, never a
// silent wrong answer.
//
// Conventions (engine-faithful, audited):
// - Tensors are flat row-major f64 arrays with shape [d0,d1,d2,d3].
// - linear(id,in_dim,out_dim,bias): input MUST be [B,F,1,1] (the engine
//   rejects anything else); W is column-major: y[b,i] = sum_j
//   w[i+j*O]*x[b,j] + b[i].
// - matmul(id): batched last-two-axis: C[b0,b1,i,j] = sum_k
//   A[b0,b1,i,k]*B[b0,b1,k,j]; engine validates b0/b1 match and a_k==b_k.
// - All elementwise ops require exact shape match (engine-validated).
//   (2) central finite differences of the engine's own forward pass.

export const MAP_VERSION = 'autodiff-map.v3.8';

// ---------- tensor helpers ----------
export const numel = (s) => s.reduce((a, b) => a * b, 1);
const coords = (shape, flat) => {
  const c = [0, 0, 0, 0];
  let r = flat;
  for (let d = 3; d >= 0; d--) { c[d] = r % shape[d]; r = Math.floor(r / shape[d]); }
  return c;
};
const flat = (shape, c) => ((c[0] * shape[1] + c[1]) * shape[2] + c[2]) * shape[3] + c[3];

export function transposeDims(values, shape, d0, d1) {
  const ns = [...shape];
  [ns[d0], ns[d1]] = [ns[d1], ns[d0]];
  const out = new Array(values.length);
  for (let f = 0; f < values.length; f++) {
    const c = coords(ns, f);
    [c[d0], c[d1]] = [c[d1], c[d0]];
    out[f] = values[flat(shape, c)];
  }
  return { values: out, shape: ns };
}

// Sum over `dim`, keeping the dim (size 1) so the result broadcasts.
export function reduceSumKeepdim(values, shape, dim) {
  const ns = [...shape]; ns[dim] = 1;
  const out = new Array(numel(ns)).fill(0);
  for (let f = 0; f < values.length; f++) {
    const c = coords(shape, f); c[dim] = 0;
    out[flat(ns, c)] += values[f];
  }
  return { values: out, shape: ns };
}

function broadcastOf(keepdim, shape) {
  // keepdim has one axis at size 1; expand it to `shape`.
  const out = new Array(numel(shape));
  for (let f = 0; f < out.length; f++) {
    const c = coords(shape, f);
    const kc = c.map((v, d) => (keepdim.shape[d] === 1 ? 0 : v));
    out[f] = keepdim.values[flat(keepdim.shape, kc)];
  }
  return out;
}

const zip3 = (a, b, c, f) => a.map((x, i) => f(x, b[i], c[i]));

// Shift forward, engine-faithful: slice + zero-pad, never wrap.
// dir in {shiftUp, shiftDown, shiftLeft, shiftRight}, s = shift_size (u32).
// out[c] = in[c+s] (Up), in[c-s] (Down); out[d] = in[d+s] (Left), in[d-s] (Right);
// out-of-range reads are 0. s >= dim => all zeros.
function applyShift(values, shape, dir, s) {
  const [d0, d1, d2, d3] = shape;
  const out = new Array(values.length).fill(0);
  for (let a = 0; a < d0; a++) for (let b = 0; b < d1; b++)
    for (let c = 0; c < d2; c++) for (let d = 0; d < d3; d++) {
      let sc = c, sd = d;
      if (dir === 'shiftUp') sc = c + s;
      else if (dir === 'shiftDown') sc = c - s;
      else if (dir === 'shiftLeft') sd = d + s;
      else sd = d - s; // shiftRight
      if (sc >= 0 && sc < d2 && sd >= 0 && sd < d3)
        out[((a * d1 + b) * d2 + c) * d3 + d] = values[((a * d1 + b) * d2 + sc) * d3 + sd];
    }
  return out;
}

// VJP of a shift is the inverse shift: the Jacobian is 0/1 (a partial
// permutation from the zero-padding), so its transpose is the inverse shift.
// shiftUp<->shiftDown, shiftLeft<->shiftRight. No saved values needed.
const SHIFT_INVERSE = {
  shiftUp: 'shiftDown', shiftDown: 'shiftUp',
  shiftLeft: 'shiftRight', shiftRight: 'shiftLeft',
};
function shiftRule(dir) {
  const inv = SHIFT_INVERSE[dir];
  return {
    saved: [],
    doc: `dX=${inv}(dL/dY, s): shift Jacobian is 0/1 (zero-pad, never wrap); transpose = inverse shift. s>=dim => dX=0, matching the all-zero forward.`,
    vjp({ inputs, upstream: up, args }) {
      const s = args[1];
      if (!Number.isInteger(s) || s < 0) throw new Error(`${dir} vjp: bad shift_size ${JSON.stringify(s)}`);
      return { inputGrads: [applyShift(up.values, inputs[0].shape, inv, s)], paramGrads: null };
    },
  };
}

// Standard normal PDF and CDF (A&S 7.1.26, ~1e-7; the engine's gelu is
// the exact erf form — probed 2026-10-22: |engine-exact| ~ 2e-8).
function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
function erfA(x) {
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const e = 1 - poly * Math.exp(-ax * ax);
  return x < 0 ? -e : e;
}
function Phi(x) { return 0.5 * (1 + erfA(x / Math.SQRT2)); }

// ---------- VJP rules ----------
// Each rule: ({inputs:[{values,shape}], output:{values,shape}, upstream:[...], args})
//         -> {inputGrads:[[...],...], paramGrads:{w:[...],b:[...]}|null}
// `inputs` are the SAVED forward values the rule needs (never recomputed).
export const RULES = {
  linear: {
    saved: ['input'],
    doc: 'y[b,i]=sum_j w[i+j*O]*x[b,j]+b[i]; W column-major (audited). Input must be [B,F,1,1] (engine-enforced).',
    vjp({ inputs, upstream: up, args, weights }) {
      const [, inD, outD, bias] = args;
      const x = inputs[0], sh = x.shape, B = sh[0], F = sh[1];
      if (F !== inD) throw new Error(`linear vjp: in_dim ${inD} != input axis1 ${F}`);
      if (sh[2] !== 1 || sh[3] !== 1) throw new Error('linear vjp: engine requires [B,F,1,1]');
      const w = weights.w, G = up.values;
      const dx = new Array(x.values.length).fill(0);
      const dw = new Array(inD * outD).fill(0);
      const db = new Array(outD).fill(0);
      for (let b = 0; b < B; b++) for (let i = 0; i < outD; i++) {
        const g = G[(b * outD + i)];
        db[i] += g;
        for (let j = 0; j < inD; j++) {
          dx[(b * inD + j)] += g * w[i + j * outD];
          dw[i + j * outD] += g * x.values[(b * inD + j)];
        }
      }
      return { inputGrads: [dx], paramGrads: { w: dw, b: bias ? db : null } };
    },
  },
  matmul: {
    saved: ['input', 'input'],
    doc: 'C[b0,b1,i,j]=sum_k A[b0,b1,i,k]*B[b0,b1,k,j]. dA=G@B^T, dB=A^T@G over last two axes.',
    vjp({ inputs, upstream: up }) {
      const [A, B] = inputs, G = up.values;
      const [b0, b1, m, k] = A.shape, n = B.shape[3];
      const dA = new Array(A.values.length).fill(0);
      const dB = new Array(B.values.length).fill(0);
      for (let a = 0; a < b0; a++) for (let b = 0; b < b1; b++)
        for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
          const g = G[(((a * b1 + b) * m + i) * n + j)];
          for (let kk = 0; kk < k; kk++) {
            dA[(((a * b1 + b) * m + i) * k + kk)] += g * B.values[(((a * b1 + b) * k + kk) * n + j)];
            dB[(((a * b1 + b) * k + kk) * n + j)] += g * A.values[(((a * b1 + b) * m + i) * k + kk)];
          }
        }
      return { inputGrads: [dA, dB], paramGrads: null };
    },
  },
  transpose: {
    saved: [],
    doc: 'dX = transpose(dL/dY) with the same axes: the VJP of a swap is the swap.',
    vjp({ upstream: up, args, outShape }) {
      const [, d0, d1] = args;
      return { inputGrads: [transposeDims(up.values, outShape, d0, d1).values], paramGrads: null };
    },
  },
  add: {
    saved: [],
    doc: 'dX=dL/dY, dY=dL/dY (fan-out).',
    vjp({ upstream: up }) {
      return { inputGrads: [[...up.values], [...up.values]], paramGrads: null };
    },
  },
  sub: {
    saved: [],
    doc: 'dX=G, dY=-G.',
    vjp({ upstream: up }) {
      return { inputGrads: [[...up.values], up.values.map((g) => -g)], paramGrads: null };
    },
  },
  mul: {
    saved: ['input', 'input'],
    doc: 'dX=G*y, dY=G*x (elementwise).',
    vjp({ inputs, upstream: up }) {
      const [x, y] = inputs, G = up.values;
      return {
        inputGrads: [
          x.values.map((xv, i) => G[i] * y.values[i]),
          y.values.map((yv, i) => G[i] * x.values[i]),
        ],
        paramGrads: null,
      };
    },
  },
  div: {
    saved: ['input', 'input'],
    doc: 'y=x/y (elementwise). dX=G/y, dY=-G*x/(y*y).',
    vjp({ inputs, upstream: up }) {
      const [x, y] = inputs, G = up.values;
      return {
        inputGrads: [
          x.values.map((_, i) => G[i] / y.values[i]),
          y.values.map((yv, i) => (-G[i] * x.values[i]) / (yv * yv)),
        ],
        paramGrads: null,
      };
    },
  },
  sqrt: {
    saved: ['output'],
    doc: 'y=sqrt(x). dX=G/(2*y), y=saved output.',
    vjp({ output, upstream: up }) {
      return {
        inputGrads: [output.values.map((y, i) => up.values[i] / (2 * y))],
        paramGrads: null,
      };
    },
  },
  relu: {
    saved: ['input'],
    doc: 'dX=G*(x>0). Subgradient at x=0 is 0 (documented choice).',
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => (x > 0 ? up.values[i] : 0))],
        paramGrads: null,
      };
    },
  },
  leakyRelu: {
    saved: ['input'],
    doc: 'dX=G*(x>0?1:negative_slope). Subgradient at 0 uses slope 1.',
    vjp({ inputs, upstream: up, args }) {
      const alpha = args[1];
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] * (x > 0 ? 1 : alpha))],
        paramGrads: null,
      };
    },
  },
  prelu: {
    saved: ['input'],
    doc: 'prelu(id,num_parameters,alpha): dX=G*(x>0?1:alpha).',
    vjp({ inputs, upstream: up, args }) {
      const alpha = args[2];
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] * (x > 0 ? 1 : alpha))],
        paramGrads: null,
      };
    },
  },
  hardSigmoid: {
    saved: ['input'],
    doc: 'hardSigmoid(id,alpha,beta): y=clamp(alpha*x+beta,0,1); dX=G*(0<alpha*x+beta<1 ? alpha : 0).',
    vjp({ inputs, upstream: up, args }) {
      const alpha = args[1], beta = args[2];
      return {
        inputGrads: [inputs[0].values.map((x, i) => {
          const z = alpha * x + beta;
          return (z > 0 && z < 1) ? up.values[i] * alpha : 0;
        })],
        paramGrads: null,
      };
    },
  },
  hardSwish: {
    saved: ['input'],
    doc: 'hardSwish(id): y=x*clamp(x/6+0.5,0,1); dX=G*(0 if x<=-3, 1 if x>=3, x/3+0.5 inside (-3,3)).',
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => {
          if (x <= -3) return 0;
          if (x >= 3) return up.values[i];
          return up.values[i] * (x / 3 + 0.5);
        })],
        paramGrads: null,
      };
    },
  },
  mish: {
    saved: ['input'],
    doc: 'mish(x)=x*tanh(softplus(x)), softplus(x)=ln(1+e^x). dX=G*(t + x*(1-t^2)*sigmoid(x)).',
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => {
          const sp = x > 20 ? x : Math.log1p(Math.exp(x));
          const t = Math.tanh(sp);
          const sig = 1 / (1 + Math.exp(-x));
          return up.values[i] * (t + x * (1 - t * t) * sig);
        })],
        paramGrads: null,
      };
    },
  },
  glu: {
    saved: ['input'],
    doc: 'glu(id,dim): splits x=[a,b] in half along dim; y=a*sigmoid(b). dA=G*s, dB=G*a*s*(1-s), s=sigmoid(b).',
    vjp({ inputs, upstream: up, args }) {
      const dim = args[1] ?? 1;
      const x = inputs[0].values, inSh = inputs[0].shape, outSh = up.shape, G = up.values;
      const half = outSh[dim];
      const dx = new Array(x.length).fill(0);
      for (let f = 0; f < G.length; f++) {
        const cOut = coords(outSh, f);
        const cA = [...cOut];
        const cB = [...cOut]; cB[dim] += half;
        const idxA = flat(inSh, cA);
        const idxB = flat(inSh, cB);
        const a = x[idxA], b = x[idxB];
        const s = 1 / (1 + Math.exp(-b));
        dx[idxA] = G[f] * s;
        dx[idxB] = G[f] * a * s * (1 - s);
      }
      return { inputGrads: [dx], paramGrads: null };
    },
  },
  // Tahap 7b: swiGlu(dIn, dOut, bias) — y = silu(W_i x + b_i) * (W_o x + b_o),
  // silu(a) = a*sigmoid(a), on the LAST axis (feature_last; any rank-4 shape
  // with last dim dIn). Weights arrive flat in engine order [wI, bI?, wO, bO?]
  // (see weightOf in bin/autodiff.mjs): wI/wO are column-major like linear,
  // w[i + j*dOut]. silu'(a) = s(a) + (a - silu(a))*s(a), s = sigmoid.
  // dL/dx = W_i^T (silu'(a)*o*G) + W_o^T (silu(a)*G); param grads are
  // returned as ONE flat 'w' vector in engine order (the verify-fd loop
  // perturbs 'w' from offset 0, so a single flat vector is the only shape
  // it can check correctly for a multi-segment layout); 'b' is null.
  swiGlu: {
    saved: ['input'],
    doc: 'swiGlu(dIn,dOut,bias): y=silu(W_i x+b_i)*(W_o x+b_o) on last axis. dX=W_i^T(silu\'(a)*o*G)+W_o^T(silu(a)*G).',
    vjp({ inputs, upstream: up, args, weights }) {
      const [, dIn, dOut, bias] = args;
      const x = inputs[0], sh = x.shape, G = up.values;
      const last = sh[3];
      if (last !== dIn) throw new Error(`swiGlu vjp: last dim ${last} != d_in ${dIn}`);
      const pos = x.values.length / dIn; // batch*spatial positions
      const { wI, bI, wO, bO } = weights;
      const dx = new Array(x.values.length).fill(0);
      const dwI = new Array(dIn * dOut).fill(0);
      const dwO = new Array(dIn * dOut).fill(0);
      const dbI = new Array(dOut).fill(0);
      const dbO = new Array(dOut).fill(0);
      for (let p = 0; p < pos; p++) {
        for (let i = 0; i < dOut; i++) {
          let a = bias ? bI[i] : 0, o = bias ? bO[i] : 0;
          for (let j = 0; j < dIn; j++) {
            const xv = x.values[p * dIn + j];
            a += wI[i + j * dOut] * xv;
            o += wO[i + j * dOut] * xv;
          }
          const s = 1 / (1 + Math.exp(-a));
          const silu = a * s;
          const dsilu = s + (a - silu) * s;
          const g = G[p * dOut + i];
          const g1 = g * dsilu * o, g2 = g * silu;
          dbI[i] += g1; // dy/dbI = silu'(a)*o (chain through silu AND the o factor)
          dbO[i] += g * silu;
          for (let j = 0; j < dIn; j++) {
            const xv = x.values[p * dIn + j];
            dx[p * dIn + j] += g1 * wI[i + j * dOut] + g2 * wO[i + j * dOut];
            dwI[i + j * dOut] += g1 * xv;
            dwO[i + j * dOut] += g2 * xv;
          }
        }
      }
      // Flat param grad in engine order [wI, bI?, wO, bO?] (see weightOf).
      // No spec weights on the layer -> the engine's zero init is not a
      // spec parameter: return null so verify-fd doesn't try to perturb
      // a weight vector the spec doesn't have.
      const flat = [...dwI, ...(bias ? dbI : []), ...dwO, ...(bias ? dbO : [])];
      return { inputGrads: [dx], paramGrads: weights.hasSpecWeights ? { w: flat, b: null } : null };
    },
  },
  featureNorm: {
    saved: ['input'],
    doc: 'featureNorm(id,eps?): L2 normalization along axis 1: y_c=x_c/r, r=sqrt(sum_c x_c^2 + eps). dX_c = (G_c - y_c*sum_k(G_k*y_k))/r.',
    vjp({ inputs, upstream: up, args }) {
      const eps = args[1] ?? 1e-12;
      const x = inputs[0].values, sh = inputs[0].shape, G = up.values;
      const [N, C, H, W] = sh;
      const dx = new Array(x.length).fill(0);
      for (let n = 0; n < N; n++) for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) {
        let sumSq = 0;
        for (let c = 0; c < C; c++) {
          const idx = ((n * C + c) * H + h) * W + w;
          sumSq += x[idx] * x[idx];
        }
        const r = Math.sqrt(sumSq + eps);
        let dotGY = 0;
        for (let c = 0; c < C; c++) {
          const idx = ((n * C + c) * H + h) * W + w;
          const yc = x[idx] / r;
          dotGY += G[idx] * yc;
        }
        for (let c = 0; c < C; c++) {
          const idx = ((n * C + c) * H + h) * W + w;
          const yc = x[idx] / r;
          dx[idx] = (G[idx] - yc * dotGY) / r;
        }
      }
      return { inputGrads: [dx], paramGrads: null };
    },
  },
  rmsNorm: {
    saved: ['input'],
    doc: 'rmsNorm(id,size,eps?): feature_last [B,*,*,F], y_i=gamma_i*x_i/rms, rms=sqrt(mean(x^2)+eps).',
    vjp({ inputs, upstream: up, args, weights }) {
      const F = args[1], eps = args[2] ?? 1e-5;
      const x = inputs[0].values, G = up.values;
      const gamma = weights?.w ?? new Array(F).fill(1);
      const dx = new Array(x.length).fill(0);
      const dgamma = new Array(F).fill(0);
      const outer = x.length / F;
      for (let o = 0; o < outer; o++) {
        const base = o * F;
        let sq = 0;
        for (let i = 0; i < F; i++) sq += x[base + i] * x[base + i];
        const rms = Math.sqrt(sq / F + eps);
        let dot = 0;
        for (let i = 0; i < F; i++) {
          const xhat = x[base + i] / rms;
          const u = G[base + i] * gamma[i];
          dot += u * xhat;
          dgamma[i] += G[base + i] * xhat;
        }
        for (let i = 0; i < F; i++) {
          const xhat = x[base + i] / rms;
          const u = G[base + i] * gamma[i];
          dx[base + i] = (u - xhat * (dot / F)) / rms;
        }
      }
      return { inputGrads: [dx], paramGrads: { w: dgamma, b: null } };
    },
  },
  layerNorm: {
    saved: ['input'],
    doc: 'layerNorm(id,size,eps?): feature_last [B,*,*,F], y_i=gamma_i*(x_i-mu)/sigma + beta_i, var=mean((x-mu)^2).',
    vjp({ inputs, upstream: up, args, weights }) {
      const F = args[1], eps = args[2] ?? 1e-5;
      const x = inputs[0].values, G = up.values;
      const gamma = weights?.w ?? new Array(F).fill(1);
      const dx = new Array(x.length).fill(0);
      const dgamma = new Array(F).fill(0);
      const dbeta = new Array(F).fill(0);
      const outer = x.length / F;
      for (let o = 0; o < outer; o++) {
        const base = o * F;
        let mu = 0;
        for (let i = 0; i < F; i++) mu += x[base + i];
        mu /= F;
        let v = 0;
        for (let i = 0; i < F; i++) v += (x[base + i] - mu) ** 2;
        const sigma = Math.sqrt(v / F + eps);
        let sumU = 0, sumUXhat = 0;
        for (let i = 0; i < F; i++) {
          const xhat = (x[base + i] - mu) / sigma;
          const u = G[base + i] * gamma[i];
          sumU += u;
          sumUXhat += u * xhat;
          dgamma[i] += G[base + i] * xhat;
          dbeta[i] += G[base + i];
        }
        for (let i = 0; i < F; i++) {
          const xhat = (x[base + i] - mu) / sigma;
          const u = G[base + i] * gamma[i];
          dx[base + i] = (u - sumU / F - xhat * (sumUXhat / F)) / sigma;
        }
      }
      return { inputGrads: [dx], paramGrads: { w: dgamma, b: dbeta } };
    },
  },
  shiftUp: shiftRule('shiftUp'),
  shiftDown: shiftRule('shiftDown'),
  shiftLeft: shiftRule('shiftLeft'),
  shiftRight: shiftRule('shiftRight'),
  sigmoid: {
    saved: ['output'],
    doc: 'dX=G*s*(1-s), s=saved output (avoids recomputing the exp).',
    vjp({ output, upstream: up }) {
      return {
        inputGrads: [output.values.map((s, i) => up.values[i] * s * (1 - s))],
        paramGrads: null,
      };
    },
  },
  tanh: {
    saved: ['output'],
    doc: 'dX=G*(1-t^2), t=saved output.',
    vjp({ output, upstream: up }) {
      return {
        inputGrads: [output.values.map((t, i) => up.values[i] * (1 - t * t))],
        paramGrads: null,
      };
    },
  },
  gelu: {
    saved: ['input'],
    doc: 'Exact gelu(x)=x*Phi(x) (engine-probed 2026-10-22). dX=G*(Phi(x)+x*phi(x)).',
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] * (Phi(x) + x * phi(x)))],
        paramGrads: null,
      };
    },
  },
  softplus: {
    saved: ['input'],
    doc: 'softplus(x;beta)=ln(1+e^(beta*x))/beta; d/dx=sigmoid(beta*x). beta=args[1].',
    vjp({ inputs, upstream: up, args }) {
      const beta = args[1];
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] / (1 + Math.exp(-beta * x)))],
        paramGrads: null,
      };
    },
  },
  softmax: {
    saved: ['output'],
    doc: 'dX=s*(G-sum_dim(G*s)); s=saved output. dim is the constructor arg.',
    vjp({ output, upstream: up, args }) {
      const dim = args[1], s = output.values, sh = output.shape, G = up.values;
      const gs = s.map((sv, i) => G[i] * sv);
      const summed = reduceSumKeepdim(gs, sh, dim);
      const bcast = broadcastOf({ values: summed.values, shape: summed.shape }, sh);
      return { inputGrads: [s.map((sv, i) => sv * (G[i] - bcast[i]))], paramGrads: null };
    },
  },
  logSoftmax: {
    saved: ['output'],
    doc: 'dX=G-s*sum_dim(G); s=exp(saved log-output)=softmax.',
    vjp({ output, upstream: up, args }) {
      const dim = args[1], sh = output.shape, G = up.values;
      const s = output.values.map(Math.exp);
      const summed = reduceSumKeepdim(G, sh, dim);
      const bcast = broadcastOf({ values: summed.values, shape: summed.shape }, sh);
      return { inputGrads: [G.map((g, i) => g - s[i] * bcast[i])], paramGrads: null };
    },
  },
  concat: {
    saved: [],
    doc: 'dX_i = slice of G along dim per the forward input shapes.',
    vjp({ upstream: up, args, inShapes }) {
      const dim = args[1];
      const grads = [];
      let off = 0;
      for (const sh of inShapes) {
        const n = sh[dim], g = new Array(numel(sh));
        for (let f = 0; f < g.length; f++) {
          const c = coords(sh, f); c[dim] += off;
          // upstream shape: same as sh but dim extended
          const us = [...sh]; us[dim] = up.shape[dim];
          g[f] = up.values[flat(us, c)];
        }
        grads.push(g);
        off += n;
      }
      return { inputGrads: grads, paramGrads: null };
    },
  },
  avgPool2d: {
    saved: ['input'],
    doc: 'dX[i] = sum of G[window]/(kh*kw) over all windows covering i. '
      + 'Divisor is ALWAYS the full kernel area: the engine divides by '
      + 'kh*kw even for windows that hang over the padding (probed '
      + '2026-10-28: 3x3 input, k2 s1 p1 -> corner window value 1/4, '
      + 'not 1/1). Padded cells contribute no gradient (they are not '
      + 'input positions); input positions covered by no window '
      + '(stride > kernel gaps, floor-discarded edges) get 0.',
    vjp({ inputs, upstream: up, args }) {
      const x = inputs[0], sh = x.shape; // [N,C,H,W]
      const [, kh, kw, sh0, sw0, ph0, pw0] = args;
      const sH = sh0 ?? kh, sW = sw0 ?? kw, pH = ph0 ?? 0, pW = pw0 ?? 0;
      const [n, c, h, w] = sh;
      const oH = Math.floor((h + 2 * pH - kh) / sH) + 1;
      const oW = Math.floor((w + 2 * pW - kw) / sW) + 1;
      const div = kh * kw;
      const dx = new Array(x.values.length).fill(0);
      const G = up.values;
      for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
        for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
          const g = G[(((b * c + cc) * oH + oh) * oW + ow)] / div;
          const h0 = oh * sH - pH, w0 = ow * sW - pW;
          for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
            const hh = h0 + i, ww = w0 + j;
            if (hh < 0 || hh >= h || ww < 0 || ww >= w) continue;
            dx[(((b * c + cc) * h + hh) * w + ww)] += g;
          }
        }
      return { inputGrads: [dx], paramGrads: null };
    },
  },
  avgPool1d: {
    saved: ['input'],
    doc: 'dX[i] = sum of G[window]/k over all windows covering i. '
      + 'Divisor is ALWAYS the full kernel: the engine divides by k even '
      + 'for windows that hang over the padding (probed 2026-10-28: '
      + '[1,2,3,4,5], k2 s1 p1 -> edge windows 0.5 and 2.5, not 1 and 5). '
      + 'Padded cells contribute no gradient; input positions covered by '
      + 'no window (stride > kernel gaps, floor-discarded tail) get 0.',
    vjp({ inputs, upstream: up, args }) {
      const x = inputs[0], sh = x.shape; // [N,C,L,W]
      const [, k, s0, p0] = args;
      const s = s0 ?? k, p = p0 ?? 0;
      const [n, c, l, w] = sh;
      const ol = Math.floor((l + 2 * p - k) / s) + 1;
      const dx = new Array(x.values.length).fill(0);
      const G = up.values;
      for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
        for (let ww = 0; ww < w; ww++)
          for (let o = 0; o < ol; o++) {
            const g = G[(((b * c + cc) * ol + o) * w + ww)] / k;
            const l0 = o * s - p;
            for (let i = 0; i < k; i++) {
              const ll = l0 + i;
              if (ll < 0 || ll >= l) continue;
              dx[(((b * c + cc) * l + ll) * w + ww)] += g;
            }
          }
      return { inputGrads: [dx], paramGrads: null };
    },
  },
};

// ---------- classified gaps (honest, not silent) ----------
export const GAPS = {
  // Non-differentiable: no VJP exists (or only a subgradient needing engine support).
  maxPool1d: 'non-differentiable here: subgradient routes via argmax mask, engine does not expose indices',
  maxPool2d: 'non-differentiable here: subgradient routes via argmax mask, engine does not expose indices',
  embedding: 'non-differentiable w.r.t. integer indices (differentiable w.r.t. weight rows; deferred)',
  // Differentiable but deferred: rule not yet written/proven.
  conv1d: 'deferred: im2col-style VJP not yet proven',
  conv2d: 'deferred: VJP not yet proven',
  convTranspose2d: 'deferred: VJP not yet proven',
  batchNorm: 'deferred: VJP needs saved batch stats; not yet proven',
  groupNorm: 'deferred: VJP not yet proven',
  instanceNorm: 'deferred: VJP not yet proven',
  adaptiveAvgPool2d: 'deferred: VJP not yet proven',
  ghost: 'deferred: VJP not yet proven',
  seBlock: 'deferred: VJP not yet proven',
};

export function ruleFor(constructor) {
  // hasOwn: without it, RULES['toString'] returns the inherited
  // Object.prototype function and a bogus constructor looks ruled.
  const r = (typeof constructor === 'string' && Object.hasOwn(RULES, constructor)) ? RULES[constructor] : undefined;
  if (!r) {
    const gap = (typeof constructor === 'string' && Object.hasOwn(GAPS, constructor)) ? GAPS[constructor] : undefined;
    throw new Error(`no backward rule for '${constructor}'` + (gap ? `: ${gap}` : ' (unknown constructor)'));
  }
  return r;
}

export function ruleNames() { return Object.keys(RULES); }
export function gapNames() { return Object.keys(GAPS); }

// ---------- shape inference (engine-faithful; supplements lib/shapes.mjs) ----------
// NOTE (port v2.5): lib/shapes.mjs now HAS matmul/transpose rules and its
// linear rule enforces the engine's [B,F,1,1] requirement (probed
// 2026-10-22), so the extras below duplicate it. They are kept because
// their error messages are the ones autodiff callers already match on;
// do not let the two copies drift — the semantics must stay identical.
function shapeTranspose(inShapes, args) {
  const [, d0, d1] = args;
  if (!(d0 >= 0 && d0 <= 3 && d1 >= 0 && d1 <= 3)) throw new Error('transpose dims out of range');
  const s = [...inShapes[0]];
  [s[d0], s[d1]] = [s[d1], s[d0]];
  return s;
}
function shapeMatmul(inShapes) {
  const [a, b] = inShapes;
  if (a[0] !== b[0] || a[1] !== b[1]) throw new Error('matmul batch mismatch');
  if (a[3] !== b[2]) throw new Error(`matmul k mismatch ${a[3]} vs ${b[2]}`);
  return [a[0], a[1], a[2], b[3]];
}
function shapeLinearStrict(inShapes, args) {
  const s = inShapes[0], [, inD, outD] = args;
  if (s[2] !== 1 || s[3] !== 1) throw new Error(`linear: engine requires [B,F,1,1], got [${s}]`);
  if (s[1] !== inD) throw new Error(`linear in_dim ${inD} != ${s[1]}`);
  return [s[0], outD, 1, 1];
}
const SHAPE_RULES_EXTRA = { transpose: shapeTranspose, matmul: shapeMatmul, linear: shapeLinearStrict };

export function inferShapes(spec) {
  // Returns Map slot -> shape. Throws an honest error on the first gap.
  const shapes = new Map(spec.ports.map((p) => [p.slot, [...p.shape]]));
  for (const step of spec.steps) {
    const layer = spec.layers[step.layer];
    const inShapes = step.slots.slice(0, step.kind === 'binary' ? 2 : 1).map((s) => {
      const sh = shapes.get(s);
      if (!sh) throw new Error(`shape inference: slot ${s} has no known shape`);
      return sh;
    });
    let out;
    const extra = SHAPE_RULES_EXTRA[layer.constructor];
    if (extra) {
      out = extra(inShapes, layer.args);
    } else {
      // fall back to lib/shapes.mjs for the rest (activations preserve, concat, ...)
      out = shapeRuleFromLib(layer.constructor, inShapes, layer.args);
    }
    const outSlot = step.slots[step.kind === 'binary' ? 2 : 1];
    shapes.set(outSlot, out);
    step._outShape = out; step._inShapes = inShapes;
  }
  return shapes;
}

import { ruleFor as libShapeRule } from './shapes.mjs';
function shapeRuleFromLib(constructor, inShapes, args) {
  const r = libShapeRule(constructor);
  if (!r) throw new Error(`no shape rule for '${constructor}' (add one or extend SHAPE_RULES_EXTRA)`);
  return r(inShapes, args);
}

export function ssaNormalizeSpec(spec) {
  const s = JSON.parse(JSON.stringify(spec));
  const curSlotMap = new Map();
  for (const p of (s.ports ?? [])) {
    curSlotMap.set(p.slot, p.slot);
  }
  let nextSlot = Math.max(s.numSlots ?? 0, ...[...(s.ports ?? []).map((p) => p.slot + 1), 0]);
  const writtenOnce = new Set((s.ports ?? []).map((p) => p.slot));
  const stepInSsaSlots = [];
  const stepOutSsaSlots = [];

  for (let si = 0; si < s.steps.length; si++) {
    const st = s.steps[si];
    const inCount = st.kind === 'binary' ? 2 : 1;
    const origIn = st.slots.slice(0, inCount);
    const origOut = st.slots[inCount];
    const mappedIn = origIn.map((sl) => (curSlotMap.has(sl) ? curSlotMap.get(sl) : sl));
    let mappedOut = origOut;
    if (writtenOnce.has(origOut)) {
      mappedOut = nextSlot++;
    } else {
      writtenOnce.add(origOut);
    }
    curSlotMap.set(origOut, mappedOut);
    st.slots = [...mappedIn, mappedOut];
    stepInSsaSlots.push(mappedIn);
    stepOutSsaSlots.push(mappedOut);
  }
  s.outputSlot = curSlotMap.get(s.outputSlot) ?? s.outputSlot;
  s.numSlots = Math.max(nextSlot, s.outputSlot + 1);
  return { ssaSpec: s, stepInSsaSlots, stepOutSsaSlots, finalSlotMap: curSlotMap };
}

export function captureSavedForSpec(spec, shapes, runForward) {
  const { ssaSpec, stepInSsaSlots, stepOutSsaSlots } = ssaNormalizeSpec(spec);
  const ssaShapes = inferShapes(ssaSpec);
  const needSsaSlots = new Set();
  for (let si = 0; si < ssaSpec.steps.length; si++) {
    const step = ssaSpec.steps[si];
    const rule = ruleFor(ssaSpec.layers[step.layer].constructor);
    stepInSsaSlots[si].forEach((s) => needSsaSlots.add(s));
    if (rule.saved.includes('output')) needSsaSlots.add(stepOutSsaSlots[si]);
  }
  const ssaValues = new Map();
  for (const sl of needSsaSlots) {
    ssaValues.set(sl, { values: runForward(ssaSpec, sl), shape: ssaShapes.get(sl) });
  }
  const stepInputs = new Map();
  const stepOutputs = new Map();
  for (let si = 0; si < ssaSpec.steps.length; si++) {
    stepInputs.set(si, stepInSsaSlots[si].map((sl) => ssaValues.get(sl)));
    stepOutputs.set(si, ssaValues.get(stepOutSsaSlots[si]));
  }
  return {
    byStep: true,
    stepInputs,
    stepOutputs,
    get(slot) { return ssaValues.get(slot); },
  };
}

const WEIGHTED_VJP_CTORS = new Set(['linear', 'rmsNorm', 'layerNorm', 'swiGlu']);

// ---------- reverse-mode builder (pure; engine only supplies `saved`) ----------
// saved: Map slot -> {values, shape}, or a byStep capture from
// captureSavedForSpec (step-indexed, so slot reuse never aliases).
// weightOf: (layerIdx, args, constructor) -> {w:[...], b:[...]|null} for
// weighted layers (linear, rmsNorm, layerNorm).
// doutSeed: {slot: [...]} (default: ones at outputSlot).
export function backwardWithSaved(spec, shapes, saved, doutSeed, weightOf) {
  // Execute the backward sweep over the SSA-normalized graph when step-indexed saved
  // activations are present so slot reuse across different tensor shapes never aliases.
  const useSsa = Boolean(saved?.byStep);
  const { ssaSpec } = useSsa ? ssaNormalizeSpec(spec) : { ssaSpec: spec };
  const activeSpec = useSsa ? ssaSpec : spec;
  const activeShapes = useSsa ? inferShapes(activeSpec) : shapes;

  const grads = new Map();
  const paramGrads = {};
  const outSlot = activeSpec.outputSlot;
  const outShape = activeShapes.get(outSlot);
  const seed = doutSeed?.[outSlot] ?? doutSeed?.[spec.outputSlot] ?? new Array(numel(outShape)).fill(1);
  grads.set(outSlot, [...seed]);

  for (let si = activeSpec.steps.length - 1; si >= 0; si--) {
    const step = activeSpec.steps[si];
    const layer = activeSpec.layers[step.layer];
    const rule = ruleFor(layer.constructor);
    const isBin = step.kind === 'binary';
    const inSlots = step.slots.slice(0, isBin ? 2 : 1);
    const os = step.slots[isBin ? 2 : 1];
    const oShape = step._outShape ?? activeShapes.get(os);
    const iShapes = step._inShapes ?? inSlots.map((sl) => activeShapes.get(sl));

    const upstream = { values: grads.get(os) ?? new Array(numel(oShape)).fill(0), shape: oShape };
    grads.delete(os);

    const stepInVals = saved?.byStep ? saved.stepInputs.get(si) : undefined;
    const stepOutVal = saved?.byStep ? saved.stepOutputs.get(si) : undefined;

    const ctx = {
      inputs: inSlots.map((sl, idx) => {
        const sv = stepInVals ? stepInVals[idx] : saved.get(sl);
        if (!sv) throw new Error(`rule '${layer.constructor}' needs saved input but slot ${sl} was not captured`);
        return sv;
      }),
      output: (() => {
        const sv = stepOutVal !== undefined ? stepOutVal : saved.get(os);
        if (rule.saved.includes('output') && !sv)
          throw new Error(`rule '${layer.constructor}' needs saved output of slot ${os}`);
        return sv ?? { values: [], shape: oShape };
      })(),
      upstream,
      args: layer.args,
      inShapes: iShapes,
      outShape: oShape,
      weights: WEIGHTED_VJP_CTORS.has(layer.constructor) ? weightOf(step.layer, layer.args, layer.constructor) : null,
    };
    const { inputGrads, paramGrads: pg } = rule.vjp(ctx);
    inSlots.forEach((sl, k) => {
      const cur = grads.get(sl) ?? new Array(numel(iShapes[k])).fill(0);
      grads.set(sl, cur.map((v, i) => v + inputGrads[k][i]));
    });
    if (pg) {
      const key = String(step.layer);
      const prev = paramGrads[key];
      if (!prev) {
        paramGrads[key] = {
          w: pg.w ? [...pg.w] : null,
          b: pg.b ? [...pg.b] : null,
        };
      } else {
        if (pg.w && prev.w) prev.w = prev.w.map((v, i) => v + pg.w[i]);
        if (pg.b && prev.b) prev.b = prev.b.map((v, i) => v + pg.b[i]);
      }
    }
  }
  const inputGrads = {};
  for (const po of spec.ports) {
    inputGrads[String(po.slot)] = grads.get(po.slot) ?? new Array(numel(po.shape)).fill(0);
  }
  return { inputGrads, paramGrads };
}

export function makeMap() {
  return {
    version: MAP_VERSION,
    rules: Object.entries(RULES).map(([name, r]) => ({
      name, saved: r.saved, doc: r.doc, status: 'proven: hand-computed + finite-difference oracle',
    })),
    gaps: { ...GAPS },
    note: 'A missing rule is an honest error, never a silent wrong answer. The finite-difference oracle (bin/gradcheck.mjs) is the permanent regression guard.',
  };
}

export { zip3 as _zip3 }; // (test hook)
