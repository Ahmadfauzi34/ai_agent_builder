// Independent f64 reference interpreter for burn-math specs (used by diff_ref.mjs).
//
// This is a SEPARATE code path from the WASM engine: plain JS, float64,
// written directly from the documented op semantics (spec-schema.md documents
// linear as column-major: y_i = sum_j w[i + j*outDim] * x_j). Disagreement
// with the engine beyond f32 rounding is a signal, not a verdict —
// investigate which side is wrong (each op below was probed against the
// engine during development; see SUPPORTED).
//
// Extended in v3.7 to cover 34 constructors across linear, activations (incl. mish, glu),
// norms (featureNorm, rmsNorm, layerNorm), matmul, transpose, concat, pooling
// (avgPool1d, avgPool2d, maxPool1d, maxPool2d), embedding, and shifts.
// Extended in Tahap 6 (PR6) to all 44 f32 constructors: + conv1d/conv2d/
// convTranspose2d, batchNorm/groupNorm/instanceNorm, adaptiveAvgPool2d,
// ghost, seBlock, swiGlu. ghost/seBlock/swiGlu then implemented the
// engine's effective zero-weight behavior (their spec-path weights were
// unsettable; see design-6.md) — parity was the guard.
// Extended in Tahap 7b (PR7b) to WEIGHTED ghost/seBlock/swiGlu + prelu:
// the engine now exposes flat-weight bridges for all four, so the refs
// read the flat weight vector (falling back to the zero-weight behavior
// when the spec carries no weights — the Tahap 6 parity cases still
// exercise exactly that path). ghost output channels are 2*primary_ch
// (engine quirk: cat([primary, cheap]) — equals out_ch only at ratio=2).
//
// Every op receives one context object:
//   { xs: [values, ...], shapes: [[N,C,H,W], ...], args, weights }
// and returns { values, shape? } (shape defaults to the first input's shape).
import { transposeDims, numel } from './autodiff.mjs';

function linearF64(x, w, inDim, outDim, bias) {
  const batch = x.length / inDim;
  const y = new Array(batch * outDim).fill(0);
  const bVec = bias ? w.slice(inDim * outDim, inDim * outDim + outDim) : null;
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < outDim; i++) {
      let acc = bVec ? bVec[i] : 0;
      for (let j = 0; j < inDim; j++) acc += w[i + j * outDim] * x[b * inDim + j];
      y[b * outDim + i] = acc;
    }
  }
  return y;
}

// Softmax over one axis of a 4-D [N,C,H,W] tensor.
// With logMode=true it returns log-softmax instead.
function softmaxAxis(x, shape, dim, logMode = false) {
  const dims = [...shape];
  const strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
  const y = new Array(x.length);
  const off = (idx) => idx[0] * strides[0] + idx[1] * strides[1] + idx[2] * strides[2] + idx[3];
  const idx = [0, 0, 0, 0];
  const walk = (d, fn) => {
    if (d === 4) { fn(); return; }
    if (d === dim) { walk(d + 1, fn); return; }
    for (let k = 0; k < dims[d]; k++) { idx[d] = k; walk(d + 1, fn); }
  };
  walk(0, () => {
    let m = -Infinity;
    for (let k = 0; k < dims[dim]; k++) { idx[dim] = k; const v = x[off(idx)]; if (v > m) m = v; }
    let s = 0;
    for (let k = 0; k < dims[dim]; k++) { idx[dim] = k; s += Math.exp(x[off(idx)] - m); }
    const logS = Math.log(s);
    for (let k = 0; k < dims[dim]; k++) {
      idx[dim] = k;
      const o = off(idx);
      y[o] = logMode ? (x[o] - m - logS) : (Math.exp(x[o] - m) / s);
    }
  });
  return y;
}

function erfA(x) {
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const e = 1 - poly * Math.exp(-ax * ax);
  return x < 0 ? -e : e;
}

function shiftF64(values, shape, dir, s) {
  const [d0, d1, d2, d3] = shape;
  const out = new Array(values.length).fill(0);
  for (let a = 0; a < d0; a++) for (let b = 0; b < d1; b++)
    for (let c = 0; c < d2; c++) for (let d = 0; d < d3; d++) {
      let sc = c, sd = d;
      if (dir === 'shiftUp') sc = c + s;
      else if (dir === 'shiftDown') sc = c - s;
      else if (dir === 'shiftLeft') sd = d + s;
      else sd = d - s;
      if (sc >= 0 && sc < d2 && sd >= 0 && sd < d3)
        out[((a * d1 + b) * d2 + c) * d3 + d] = values[((a * d1 + b) * d2 + sc) * d3 + sd];
    }
  return out;
}

const coords4 = (sh, f) => {
  const c = [0, 0, 0, 0];
  let r = f;
  for (let d = 3; d >= 0; d--) { c[d] = r % sh[d]; r = Math.floor(r / sh[d]); }
  return c;
};
const flatIdx4 = (sh, c) => ((c[0] * sh[1] + c[1]) * sh[2] + c[2]) * sh[3] + c[3];

const OPS = {
  linear: ({ xs, shapes, args, weights }) => {
    const [x] = xs;
    const [, inDim, outDim, bias] = args;
    if (shapes[0][1] !== inDim) throw new Error(`ref linear: input dim ${shapes[0][1]} != in_dim ${inDim}`);
    if (!weights) throw new Error('ref linear: missing weights for layer');
    return { values: linearF64(x, weights, inDim, outDim, bias), shape: [shapes[0][0], outDim, shapes[0][2], shapes[0][3]] };
  },
  relu: ({ xs }) => ({ values: xs[0].map((v) => Math.max(0, v)) }),
  leakyRelu: ({ xs, args }) => {
    const alpha = args[1] ?? 0.01;
    return { values: xs[0].map((v) => (v > 0 ? v : alpha * v)) };
  },
  prelu: ({ xs, shapes, args, weights }) => {
    // Tahap 7b: alpha is the flat weight vector (length num_parameters:
    // 1 = shared, C = per-channel). No weights on the spec -> the engine
    // uses the init alpha from args (default 0.25), shared.
    const [, numParams, alphaArg] = args;
    const [, c, h, w] = shapes[0];
    const hw = h * w;
    if (weights) {
      if (weights.length !== numParams) throw new Error(`ref prelu: weights length ${weights.length} != num_parameters ${numParams}`);
      if (numParams !== 1 && numParams !== c) throw new Error(`ref prelu: num_parameters ${numParams} must be 1 or C (${c})`);
      return { values: xs[0].map((v, idx) => {
        const alpha = numParams === 1 ? weights[0] : weights[Math.floor(idx / hw) % c];
        return v > 0 ? v : alpha * v;
      }) };
    }
    const alpha = alphaArg ?? 0.25;
    return { values: xs[0].map((v) => (v > 0 ? v : alpha * v)) };
  },
  sigmoid: ({ xs }) => ({ values: xs[0].map((v) => 1 / (1 + Math.exp(-v))) }),
  tanh: ({ xs }) => ({ values: xs[0].map((v) => Math.tanh(v)) }),
  gelu: ({ xs }) => ({ values: xs[0].map((v) => 0.5 * v * (1 + erfA(v / Math.SQRT2))) }),
  softplus: ({ xs, args }) => {
    const beta = args[1] ?? 1.0;
    return { values: xs[0].map((v) => Math.log(1 + Math.exp(beta * v)) / beta) };
  },
  hardSigmoid: ({ xs, args }) => {
    const alpha = args[1], beta = args[2];
    return { values: xs[0].map((v) => Math.min(1, Math.max(0, alpha * v + beta))) };
  },
  hardSwish: ({ xs }) => ({
    values: xs[0].map((v) => v * Math.min(1, Math.max(0, v / 6 + 0.5))),
  }),
  mish: ({ xs }) => ({
    values: xs[0].map((v) => {
      const sp = v > 20 ? v : Math.log1p(Math.exp(v));
      return v * Math.tanh(sp);
    }),
  }),
  glu: ({ xs, shapes, args }) => {
    const dim = args[1] ?? 1;
    const inSh = shapes[0];
    const outSh = [...inSh];
    outSh[dim] = inSh[dim] / 2;
    const half = outSh[dim];
    const out = new Array(numel(outSh));
    for (let f = 0; f < out.length; f++) {
      const cA = coords4(outSh, f);
      const cB = [...cA]; cB[dim] += half;
      const a = xs[0][flatIdx4(inSh, cA)];
      const b = xs[0][flatIdx4(inSh, cB)];
      out[f] = a / (1 + Math.exp(-b));
    }
    return { values: out, shape: outSh };
  },
  add: ({ xs }) => ({ values: xs[0].map((v, i) => v + xs[1][i]) }),
  sub: ({ xs }) => ({ values: xs[0].map((v, i) => v - xs[1][i]) }),
  mul: ({ xs }) => ({ values: xs[0].map((v, i) => v * xs[1][i]) }),
  div: ({ xs }) => ({ values: xs[0].map((v, i) => v / xs[1][i]) }),
  sqrt: ({ xs }) => ({ values: xs[0].map((v) => Math.sqrt(v)) }),
  softmax: ({ xs, shapes, args }) => ({ values: softmaxAxis(xs[0], shapes[0], args[1] ?? 1, false) }),
  logSoftmax: ({ xs, shapes, args }) => ({ values: softmaxAxis(xs[0], shapes[0], args[1] ?? 1, true) }),
  transpose: ({ xs, shapes, args }) => transposeDims(xs[0], shapes[0], args[1], args[2]),
  matmul: ({ xs, shapes }) => {
    const [A, B] = xs;
    const [b0, b1, m, k] = shapes[0], n = shapes[1][3];
    const out = new Array(b0 * b1 * m * n).fill(0);
    for (let a = 0; a < b0; a++) for (let b = 0; b < b1; b++)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        let s = 0;
        for (let kk = 0; kk < k; kk++) {
          s += A[(((a * b1 + b) * m + i) * k + kk)] * B[(((a * b1 + b) * k + kk) * n + j)];
        }
        out[(((a * b1 + b) * m + i) * n + j)] = s;
      }
    return { values: out, shape: [b0, b1, m, n] };
  },
  concat: ({ xs, shapes, args }) => {
    const dim = args[1];
    const outShape = [...shapes[0]];
    outShape[dim] = shapes[0][dim] + shapes[1][dim];
    const out = new Array(numel(outShape));
    for (let f = 0; f < out.length; f++) {
      const c = coords4(outShape, f);
      if (c[dim] < shapes[0][dim]) {
        out[f] = xs[0][flatIdx4(shapes[0], c)];
      } else {
        const c2 = [...c]; c2[dim] -= shapes[0][dim];
        out[f] = xs[1][flatIdx4(shapes[1], c2)];
      }
    }
    return { values: out, shape: outShape };
  },
  shiftUp: ({ xs, shapes, args }) => ({ values: shiftF64(xs[0], shapes[0], 'shiftUp', args[1]) }),
  shiftDown: ({ xs, shapes, args }) => ({ values: shiftF64(xs[0], shapes[0], 'shiftDown', args[1]) }),
  shiftLeft: ({ xs, shapes, args }) => ({ values: shiftF64(xs[0], shapes[0], 'shiftLeft', args[1]) }),
  shiftRight: ({ xs, shapes, args }) => ({ values: shiftF64(xs[0], shapes[0], 'shiftRight', args[1]) }),
  featureNorm: ({ xs, shapes, args }) => {
    const eps = args[1] ?? 1e-12;
    const x = xs[0], [N, C, H, W] = shapes[0];
    const y = new Array(x.length);
    for (let n = 0; n < N; n++) for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) {
      let sq = 0;
      for (let c = 0; c < C; c++) sq += x[((n * C + c) * H + h) * W + w] ** 2;
      const r = Math.sqrt(sq + eps);
      for (let c = 0; c < C; c++) {
        const idx = ((n * C + c) * H + h) * W + w;
        y[idx] = x[idx] / r;
      }
    }
    return { values: y };
  },
  rmsNorm: ({ xs, shapes, args, weights }) => {
    const F = args[1], eps = args[2] ?? 1e-5;
    const x = xs[0];
    const gamma = weights ? weights.slice(0, F) : new Array(F).fill(1);
    const y = new Array(x.length);
    const outer = x.length / F;
    for (let o = 0; o < outer; o++) {
      const base = o * F;
      let sq = 0;
      for (let i = 0; i < F; i++) sq += x[base + i] * x[base + i];
      const rms = Math.sqrt(sq / F + eps);
      for (let i = 0; i < F; i++) y[base + i] = (x[base + i] / rms) * gamma[i];
    }
    return { values: y, shape: [...shapes[0]] };
  },
  layerNorm: ({ xs, shapes, args, weights }) => {
    const F = args[1], eps = args[2] ?? 1e-5;
    const x = xs[0];
    const gamma = weights ? weights.slice(0, F) : new Array(F).fill(1);
    const beta = weights && weights.length >= 2 * F ? weights.slice(F, 2 * F) : new Array(F).fill(0);
    const y = new Array(x.length);
    const outer = x.length / F;
    for (let o = 0; o < outer; o++) {
      const base = o * F;
      let mu = 0;
      for (let i = 0; i < F; i++) mu += x[base + i];
      mu /= F;
      let v = 0;
      for (let i = 0; i < F; i++) v += (x[base + i] - mu) ** 2;
      const sigma = Math.sqrt(v / F + eps);
      for (let i = 0; i < F; i++) y[base + i] = ((x[base + i] - mu) / sigma) * gamma[i] + beta[i];
    }
    return { values: y, shape: [...shapes[0]] };
  },
  avgPool1d: ({ xs, shapes, args }) => {
    const x = xs[0], [n, c, l, w] = shapes[0];
    const [, k, s0, p0] = args;
    const s = s0 ?? k, p = p0 ?? 0;
    const ol = Math.floor((l + 2 * p - k) / s) + 1;
    const out = new Array(n * c * ol * w).fill(0);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let ww = 0; ww < w; ww++)
        for (let o = 0; o < ol; o++) {
          let sum = 0;
          const l0 = o * s - p;
          for (let i = 0; i < k; i++) {
            const ll = l0 + i;
            if (ll >= 0 && ll < l) sum += x[(((b * c + cc) * l + ll) * w + ww)];
          }
          out[(((b * c + cc) * ol + o) * w + ww)] = sum / k;
        }
    return { values: out, shape: [n, c, ol, w] };
  },
  maxPool1d: ({ xs, shapes, args }) => {
    const x = xs[0], [n, c, l, w] = shapes[0];
    const [, k, s0, p0] = args;
    const s = s0 ?? k, p = p0 ?? 0;
    const ol = Math.floor((l + 2 * p - k) / s) + 1;
    const out = new Array(n * c * ol * w).fill(0);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let ww = 0; ww < w; ww++)
        for (let o = 0; o < ol; o++) {
          let mx = -Infinity;
          const l0 = o * s - p;
          for (let i = 0; i < k; i++) {
            const ll = l0 + i;
            if (ll >= 0 && ll < l) {
              const v = x[(((b * c + cc) * l + ll) * w + ww)];
              if (v > mx) mx = v;
            }
          }
          out[(((b * c + cc) * ol + o) * w + ww)] = mx;
        }
    return { values: out, shape: [n, c, ol, w] };
  },
  avgPool2d: ({ xs, shapes, args }) => {
    const x = xs[0], [n, c, h, w] = shapes[0];
    const [, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sH = sh0 ?? kh, sW = sw0 ?? kw, pH = ph0 ?? 0, pW = pw0 ?? 0;
    const oH = Math.floor((h + 2 * pH - kh) / sH) + 1;
    const oW = Math.floor((w + 2 * pW - kw) / sW) + 1;
    const div = kh * kw;
    const out = new Array(n * c * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
        let sum = 0;
        const h0 = oh * sH - pH, w0 = ow * sW - pW;
        for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
          const hh = h0 + i, ww = w0 + j;
          if (hh >= 0 && hh < h && ww >= 0 && ww < w) {
            sum += x[(((b * c + cc) * h + hh) * w + ww)];
          }
        }
        out[(((b * c + cc) * oH + oh) * oW + ow)] = sum / div;
      }
    return { values: out, shape: [n, c, oH, oW] };
  },
  maxPool2d: ({ xs, shapes, args }) => {
    const x = xs[0], [n, c, h, w] = shapes[0];
    const [, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sH = sh0 ?? kh, sW = sw0 ?? kw, pH = ph0 ?? 0, pW = pw0 ?? 0;
    const oH = Math.floor((h + 2 * pH - kh) / sH) + 1;
    const oW = Math.floor((w + 2 * pW - kw) / sW) + 1;
    const out = new Array(n * c * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
        let mx = -Infinity;
        const h0 = oh * sH - pH, w0 = ow * sW - pW;
        for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
          const hh = h0 + i, ww = w0 + j;
          if (hh >= 0 && hh < h && ww >= 0 && ww < w) {
            const v = x[(((b * c + cc) * h + hh) * w + ww)];
            if (v > mx) mx = v;
          }
        }
        out[(((b * c + cc) * oH + oh) * oW + ow)] = mx;
      }
    return { values: out, shape: [n, c, oH, oW] };
  },
  embedding: ({ xs, shapes, args, weights }) => {
    const [x] = xs;
    const [, vocabSize, dModel] = args;
    if (!weights) throw new Error('ref embedding: missing weights for layer');
    const [B, S] = shapes[0];
    const out = new Array(B * S * dModel);
    for (let i = 0; i < x.length; i++) {
      const tokenIdx = Math.trunc(x[i]);
      if (tokenIdx < 0 || tokenIdx >= vocabSize) {
        throw new Error(`ref embedding: token index ${tokenIdx} out of bounds [0, ${vocabSize})`);
      }
      for (let d = 0; d < dModel; d++) {
        out[i * dModel + d] = weights[tokenIdx * dModel + d];
      }
    }
    return { values: out, shape: [B, S, dModel, 1] };
  },
  // --- Tahap 6 (PR6): the remaining 10 f32 constructors. Weight layouts
  // and semantics were fitted against the engine (see pr6/semantics_probe.mjs).
  conv1d: ({ xs, shapes, args, weights }) => {
    // args: [id, in_ch, out_ch, kernel, stride?, padding?].
    // Weights row-major [out][in][k], bias [out] appended.
    const x = xs[0], [n, cIn, len] = shapes[0];
    const [, inCh, outCh, k, s0, p0] = args;
    const s = s0 ?? 1, p = p0 ?? 0;
    const oL = Math.floor((len + 2 * p - k) / s) + 1;
    if (!weights) throw new Error('ref conv1d: missing weights for layer');
    const out = new Array(n * outCh * oL).fill(0);
    for (let b = 0; b < n; b++) for (let o = 0; o < outCh; o++)
      for (let l = 0; l < oL; l++) {
        let acc = weights[inCh * outCh * k + o];
        for (let i = 0; i < inCh; i++) for (let kk = 0; kk < k; kk++) {
          const ll = l * s + kk - p;
          if (ll >= 0 && ll < len) acc += weights[(o * inCh + i) * k + kk] * x[((b * cIn + i) * len + ll)];
        }
        out[(b * outCh + o) * oL + l] = acc;
      }
    return { values: out, shape: [n, outCh, oL, 1] };
  },
  conv2d: ({ xs, shapes, args, weights }) => {
    // args: [id, in, out, kh, kw, sh?, sw?, ph?, pw?].
    // Weights row-major [out][in][kh][kw], bias [out] appended.
    const x = xs[0], [n, cIn, h, w] = shapes[0];
    const [, inCh, outCh, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sH = sh0 ?? 1, sW = sw0 ?? 1, pH = ph0 ?? 0, pW = pw0 ?? 0;
    const oH = Math.floor((h + 2 * pH - kh) / sH) + 1;
    const oW = Math.floor((w + 2 * pW - kw) / sW) + 1;
    if (!weights) throw new Error('ref conv2d: missing weights for layer');
    const out = new Array(n * outCh * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let o = 0; o < outCh; o++)
      for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
        let acc = weights[inCh * outCh * kh * kw + o];
        for (let i = 0; i < inCh; i++) for (let khh = 0; khh < kh; khh++) for (let kww = 0; kww < kw; kww++) {
          const hh = oh * sH + khh - pH, ww = ow * sW + kww - pW;
          if (hh >= 0 && hh < h && ww >= 0 && ww < w)
            acc += weights[((o * inCh + i) * kh + khh) * kw + kww] * x[(((b * cIn + i) * h + hh) * w + ww)];
        }
        out[(((b * outCh + o) * oH + oh) * oW + ow)] = acc;
      }
    return { values: out, shape: [n, outCh, oH, oW] };
  },
  convTranspose2d: ({ xs, shapes, args, weights }) => {
    // args: [id, in, out, kh, kw, sh?, sw?, ph?, pw?].
    // Weights row-major [in][out][kh][kw] (burn ConvTranspose2d shape),
    // bias [out] appended. Scatter (gradient-of-conv) form.
    const x = xs[0], [n, cIn, h, w] = shapes[0];
    const [, inCh, outCh, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sH = sh0 ?? 1, sW = sw0 ?? 1, pH = ph0 ?? 0, pW = pw0 ?? 0;
    const oH = (h - 1) * sH + kh - 2 * pH;
    const oW = (w - 1) * sW + kw - 2 * pW;
    if (!weights) throw new Error('ref convTranspose2d: missing weights for layer');
    const out = new Array(n * outCh * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let o = 0; o < outCh; o++)
      for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++)
        out[(((b * outCh + o) * oH + oh) * oW + ow)] = weights[inCh * outCh * kh * kw + o];
    for (let b = 0; b < n; b++) for (let i = 0; i < inCh; i++)
      for (let hh = 0; hh < h; hh++) for (let ww = 0; ww < w; ww++)
        for (let khh = 0; khh < kh; khh++) for (let kww = 0; kww < kw; kww++) {
          const oh = hh * sH + khh - pH, ow = ww * sW + kww - pW;
          if (oh < 0 || oh >= oH || ow < 0 || ow >= oW) continue;
          const xv = x[(((b * cIn + i) * h + hh) * w + ww)];
          for (let o = 0; o < outCh; o++)
            out[(((b * outCh + o) * oH + oh) * oW + ow)] += xv * weights[((i * outCh + o) * kh + khh) * kw + kww];
        }
    return { values: out, shape: [n, outCh, oH, oW] };
  },
  // Norms: weights are [gamma(C), beta(C)] (defaults gamma=1, beta=0).
  // batchNorm uses RUNNING stats (mean 0, var 1): the engine's BatchNorm
  // never leaves eval mode on the spec path (probed 2026-11-07).
  batchNorm: ({ xs, shapes, args, weights }) => {
    const x = xs[0], [n, c] = shapes[0];
    const [, numF, eps0] = args;
    const eps = eps0 ?? 1e-5;
    const gamma = weights ? weights.slice(0, numF) : new Array(numF).fill(1);
    const beta = weights ? weights.slice(numF, 2 * numF) : new Array(numF).fill(0);
    const inv = 1 / Math.sqrt(1 + eps);
    const out = new Array(x.length);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let f = 0; f < x.length / (n * c); f++)
        out[(b * c + cc) * (x.length / (n * c)) + f] = gamma[cc] * x[(b * c + cc) * (x.length / (n * c)) + f] * inv + beta[cc];
    return { values: out };
  },
  groupNorm: ({ xs, shapes, args, weights }) => {
    const x = xs[0], [n, c, h, w] = shapes[0];
    const [, numG, numC, eps0] = args;
    const eps = eps0 ?? 1e-5;
    const gamma = weights ? weights.slice(0, numC) : new Array(numC).fill(1);
    const beta = weights ? weights.slice(numC, 2 * numC) : new Array(numC).fill(0);
    const chPerGroup = Math.floor(c / numG);
    const out = new Array(x.length);
    for (let b = 0; b < n; b++) for (let g = 0; g < numG; g++) {
      const idxs = [];
      for (let cc = g * chPerGroup; cc < (g + 1) * chPerGroup; cc++)
        for (let f = 0; f < h * w; f++) idxs.push((b * c + cc) * h * w + f);
      let mean = 0;
      for (const i of idxs) mean += x[i];
      mean /= idxs.length;
      let va = 0;
      for (const i of idxs) va += (x[i] - mean) * (x[i] - mean);
      va /= idxs.length;
      const inv = 1 / Math.sqrt(va + eps);
      for (const i of idxs) {
        const cc = Math.floor((i / (h * w)) % c);
        out[i] = gamma[cc] * (x[i] - mean) * inv + beta[cc];
      }
    }
    return { values: out };
  },
  instanceNorm: ({ xs, shapes, args, weights }) => {
    const x = xs[0], [n, c, h, w] = shapes[0];
    const [, numC, eps0] = args;
    const eps = eps0 ?? 1e-5;
    const gamma = weights ? weights.slice(0, numC) : new Array(numC).fill(1);
    const beta = weights ? weights.slice(numC, 2 * numC) : new Array(numC).fill(0);
    const out = new Array(x.length);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++) {
      let mean = 0;
      for (let f = 0; f < h * w; f++) mean += x[(b * c + cc) * h * w + f];
      mean /= h * w;
      let va = 0;
      for (let f = 0; f < h * w; f++) va += (x[(b * c + cc) * h * w + f] - mean) ** 2;
      va /= h * w;
      const inv = 1 / Math.sqrt(va + eps);
      for (let f = 0; f < h * w; f++)
        out[(b * c + cc) * h * w + f] = gamma[cc] * (x[(b * c + cc) * h * w + f] - mean) * inv + beta[cc];
    }
    return { values: out };
  },
  adaptiveAvgPool2d: ({ xs, shapes, args }) => {
    const x = xs[0], [n, c, h, w] = shapes[0];
    const [, oH, oW] = args;
    const out = new Array(n * c * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let cc = 0; cc < c; cc++)
      for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
        const h0 = Math.floor((oh * h) / oH), h1 = Math.ceil(((oh + 1) * h) / oH);
        const w0 = Math.floor((ow * w) / oW), w1 = Math.ceil(((ow + 1) * w) / oW);
        let s = 0, cnt = 0;
        for (let hh = h0; hh < h1; hh++) for (let ww = w0; ww < w1; ww++) {
          s += x[(((b * c + cc) * h + hh) * w + ww)];
          cnt++;
        }
        out[(((b * c + cc) * oH + oh) * oW + ow)] = s / cnt;
      }
    return { values: out, shape: [n, c, oH, oW] };
  },
  // Tahap 7b: ghost/seBlock/swiGlu are WEIGHTED now (flat-weight bridge).
  // With no weights on the spec the engine still uses its zero init, so
  // the fallback below reproduces the Tahap 6 zero behavior exactly
  // (ghost: zeros; seBlock: x*0.5; swiGlu: zeros). With weights the refs
  // implement the full formulas; parity is the guard.
  ghost: ({ xs, shapes, args, weights }) => {
    const [, inCh, outCh, kh, kw, ratio, sh0, sw0, ph0, pw0] = args;
    const [n, , h, w] = shapes[0];
    const sH = sh0 ?? 1, sW = sw0 ?? 1, pH = ph0 ?? 0, pW = pw0 ?? 0;
    const oH = Math.floor((h + 2 * pH - kh) / sH) + 1;
    const oW = Math.floor((w + 2 * pW - kw) / sW) + 1;
    const primaryCh = outCh / ratio;
    // Engine quirk: forward cats [primary, cheap] -> 2*primary_ch channels
    // (equals outCh only at ratio=2). The ref mirrors the engine, not the
    // declared out_ch.
    const outC = 2 * primaryCh;
    if (!weights) return { values: new Array(n * outC * oH * oW).fill(0), shape: [n, outC, oH, oW] };
    const primaryLen = primaryCh * (inCh * kh * kw + 1);
    if (weights.length !== primaryLen + primaryCh) throw new Error(`ref ghost: weights length ${weights.length} != ${primaryLen + primaryCh}`);
    const primaryW = weights.slice(0, primaryLen);
    const cheapW = weights.slice(primaryLen);
    const primary = OPS.conv2d({ xs, shapes, args: [0, inCh, primaryCh, kh, kw, sH, sW, pH, pW], weights: primaryW });
    const pVals = primary.values;
    const out = new Array(n * outC * oH * oW).fill(0);
    for (let b = 0; b < n; b++) for (let c = 0; c < primaryCh; c++) for (let oh = 0; oh < oH; oh++) for (let ow = 0; ow < oW; ow++) {
      const pv = pVals[(((b * primaryCh + c) * oH + oh) * oW + ow)];
      out[(((b * outC + c) * oH + oh) * oW + ow)] = pv;
      // cheap: 1x1 depthwise (groups=primary_ch), no bias -> per-channel scale.
      out[(((b * outC + primaryCh + c) * oH + oh) * oW + ow)] = cheapW[c] * pv;
    }
    return { values: out, shape: [n, outC, oH, oW] };
  },
  seBlock: ({ xs, shapes, args, weights }) => {
    const [, channels, reduction] = args;
    const [n, c, h, w] = shapes[0];
    const x = xs[0];
    if (!weights) return { values: x.map((v) => v * 0.5) };
    const reduced = Math.floor(channels / reduction);
    const fc1Len = channels * reduced + reduced;
    if (weights.length !== fc1Len + reduced * channels + channels) throw new Error(`ref seBlock: weights length ${weights.length} != ${fc1Len + reduced * channels + channels}`);
    const fc1W = weights.slice(0, fc1Len);
    const fc2W = weights.slice(fc1Len);
    // squeeze: mean over H*W per (b, c) -> [n, channels]; fc1 -> relu -> fc2 -> sigmoid.
    const hw = h * w;
    const out = new Array(x.length);
    for (let b = 0; b < n; b++) {
      const squeezed = new Array(channels);
      for (let ci = 0; ci < channels; ci++) {
        let acc = 0;
        for (let k = 0; k < hw; k++) acc += x[((b * c + ci) * hw) + k];
        squeezed[ci] = acc / hw;
      }
      const hidden = linearF64(squeezed, fc1W, channels, reduced, true).map((v) => Math.max(0, v));
      const gate = linearF64(hidden, fc2W, reduced, channels, true).map((v) => 1 / (1 + Math.exp(-v)));
      for (let ci = 0; ci < channels; ci++) for (let k = 0; k < hw; k++) {
        out[((b * c + ci) * hw) + k] = x[((b * c + ci) * hw) + k] * gate[ci];
      }
    }
    return { values: out };
  },
  swiGlu: ({ xs, shapes, args, weights }) => {
    const [, dIn, dOut, bias] = args;
    const inSh = shapes[0];
    const outSh = [...inSh];
    outSh[3] = dOut;
    if (!weights) return { values: new Array(numel(outSh)).fill(0), shape: outSh };
    const innerLen = dIn * dOut + (bias ? dOut : 0);
    if (weights.length !== 2 * innerLen) throw new Error(`ref swiGlu: weights length ${weights.length} != ${2 * innerLen}`);
    const inner = linearF64(xs[0], weights.slice(0, innerLen), dIn, dOut, bias);
    const outer = linearF64(xs[0], weights.slice(innerLen), dIn, dOut, bias);
    const values = inner.map((a, i) => (a / (1 + Math.exp(-a))) * outer[i]);
    return { values, shape: outSh };
  },
};

export const SUPPORTED = Object.keys(OPS);

// Run a spec's dataflow in f64. slots: Map slot -> {values, shape}.
// weights: {layerIndex -> number[]} (column-major, as in the spec).
export function runReference(spec, inputs) {
  const slots = new Map();
  for (const p of spec.ports) {
    const v = inputs[String(p.slot)];
    if (v === undefined) throw new Error(`ref: no input for port slot ${p.slot}`);
    if (v.length !== p.shape.reduce((a, b) => a * b, 1))
      throw new Error(`ref: input len ${v.length} != port shape ${JSON.stringify(p.shape)}`);
    slots.set(p.slot, { values: v.map(Number), shape: [...p.shape] });
  }
  for (const st of spec.steps) {
    const layerDef = spec.layers[st.layer];
    if (!layerDef) throw new Error(`ref: step references missing layer ${st.layer}`);
    const op = OPS[layerDef.constructor];
    if (!op) throw new Error(`ref: unsupported constructor '${layerDef.constructor}' (supported: ${SUPPORTED.join(', ')})`);
    const inSlots = st.kind === 'unary' ? [st.slots[0]] : [st.slots[0], st.slots[1]];
    const outSlot = st.slots[st.slots.length - 1];
    const ins = inSlots.map((s) => {
      const r = slots.get(s);
      if (!r) throw new Error(`ref: slot ${s} read before written`);
      return r;
    });
    if (ins.length === 2 && layerDef.constructor !== 'concat' && layerDef.constructor !== 'matmul' && ins[0].values.length !== ins[1].values.length)
      throw new Error(`ref: binary length mismatch ${ins[0].values.length} vs ${ins[1].values.length}`);
    const r = op({
      xs: ins.map((x) => x.values),
      shapes: ins.map((x) => x.shape),
      args: layerDef.args,
      weights: spec.weights?.[String(st.layer)],
    });
    slots.set(outSlot, { values: r.values, shape: r.shape ?? [...ins[0].shape] });
  }
  const out = slots.get(spec.outputSlot);
  if (!out) throw new Error('ref: outputSlot never written');
  return out.values;
}
