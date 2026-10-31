#!/usr/bin/env node
// Shared CLI plumbing for the f32 vector tools (arith.mjs, kernel.mjs).
//
// Extracted verbatim from arith.mjs (2026-11-01, kernel-CLI foundation):
// the second f32 tool must NOT copy-paste these functions — a duplicated
// parser/warning/provenance drifts the next time one of them is fixed
// (the quantization warnings below already carry one such audit fix).
// Pure functions only: no engine objects, no tool-specific dispatch, so
// importing this module can never change a tool's computed values.
// Tools that need different semantics (int.mjs: BigInt strings) keep
// their own parsers — this module is f32 CLI parsing, nothing more.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isFiniteF32 } from './spec.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

export function engineProvenance() {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'MANIFEST.json'), 'utf8'));
    const full = m?.sha256?.['burn_research_bg.wasm'] ?? m?.engine?.wasm_sha256 ?? null;
    return { wasm_sha256_short: full ? full.slice(0, 16) : null, mode: 'direct' };
  } catch { return { wasm_sha256_short: null, mode: 'direct' }; }
}

// Quantization-loss warnings (audit finding #5, closed 2026-10-31):
// the a/b echo keeps the given values (what the user typed), while the
// engine computes in f32. When a given v differs from Math.fround(v),
// mantissa precision was lost on entry — name it instead of hiding it.
// Always present (possibly empty) so the output shape is stable.
export function quantizationWarnings(values, name) {
  const out = [];
  values.forEach((v, i) => {
    if (v !== Math.fround(v)) out.push({ arg: name, index: i, given: v, as_f32: Math.fround(v) });
  });
  return out;
}
export function matrixQuantizationWarnings(rows, name) {
  const out = [];
  rows.forEach((row, ri) => row.forEach((v, ci) => {
    if (v !== Math.fround(v)) out.push({ arg: name, row: ri, col: ci, given: v, as_f32: Math.fround(v) });
  }));
  return out;
}

export function parseVector(text, name, usage, allowSubnormalZeroCheck = false) {
  if (text === undefined || text === '') throw new Error(usage);
  const values = text.split(',').map((t, i) => {
    const trimmed = t.trim();
    const v = Number(trimmed);
    if (trimmed === '' || !Number.isFinite(v)) {
      throw new Error(`${name}: not a finite number: ${JSON.stringify(t)}`);
    }
    if (allowSubnormalZeroCheck && v !== 0 && Math.fround(v) === 0) {
      throw new Error(`divisor underflow at index ${i} (${name}[${i}] is ${v}: its reciprocal is not finite; in f32 it is zero)`);
    }
    if (!isFiniteF32(v)) {
      throw new Error(`${name}: not a finite number in f32 range (overflows f32): ${JSON.stringify(t)}`);
    }
    return v;
  });
  if (values.length === 0) throw new Error(`${name}: empty vector`);
  return values;
}

export function parseVectorValues(arr, name, allowSubnormalZeroCheck = false) {
  if (!Array.isArray(arr)) throw new Error(`${name}: expected an array of finite numbers`);
  if (arr.length === 0) throw new Error(`${name}: empty vector`);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new Error(`${name}: not a finite number: ${JSON.stringify(v)}`);
    }
    if (allowSubnormalZeroCheck && v !== 0 && Math.fround(v) === 0) {
      throw new Error(`divisor underflow at index ${i} (${name}[${i}] is ${v}: its reciprocal is not finite; in f32 it is zero)`);
    }
    if (!isFiniteF32(v)) {
      throw new Error(`${name}: not a finite number in f32 range (overflows f32): ${JSON.stringify(v)}`);
    }
  }
  return arr.slice();
}

export function parseMatrix(text, name, usage) {
  if (text === undefined || text === '') throw new Error(usage);
  const rows = text.split(';').map((row, ri) => {
    const vals = row.split(',').map((t) => {
      const trimmed = t.trim();
      const v = Number(trimmed);
      if (trimmed === '' || !Number.isFinite(v) || !isFiniteF32(v)) {
        throw new Error(`${name} row ${ri}: not a finite number: ${JSON.stringify(t)}`);
      }
      return v;
    });
    if (vals.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    return vals;
  });
  if (rows.length === 0) throw new Error(`${name}: empty matrix`);
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) {
    throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  }
  return rows;
}

export function parseMatrixValues(arr, name) {
  if (!Array.isArray(arr)) throw new Error(`${name}: expected an array of rows`);
  if (arr.length === 0) throw new Error(`${name}: empty matrix`);
  const rows = arr.map((row, ri) => {
    if (!Array.isArray(row)) throw new Error(`${name} row ${ri}: expected an array of finite numbers`);
    if (row.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    for (const v of row) {
      if (typeof v !== 'number' || !Number.isFinite(v) || !isFiniteF32(v)) {
        throw new Error(`${name} row ${ri}: not a finite number: ${JSON.stringify(v)}`);
      }
    }
    return row.slice();
  });
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) {
    throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  }
  return rows;
}

export function assertAllFiniteOutput(op, values) {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`${op} output overflow/NaN at index ${i}: result is not finite in f32`);
    }
  }
  return values;
}
