//! # Kontrak: `registry`
//!
//! ## Tanggung jawab
//! `LayerRegistry`: pemilik semua instance layer (`HashMap<LayerId, _>` per
//! tipe). Menyediakan siklus hidup init → forward → get_state/load_state →
//! destroy, plus proyeksi `runtime_contract` untuk inventarisasi runtime.
//!
//! ## Invariant
//! - Satu `(tipe_layer, layer_id)` = satu instance; init ulang pada kunci
//!   yang sama bersifat deterministik.
//! - Isolasi per-call: satu panggilan yang gagal TIDAK PERNAH me-wedge
//!   instance maupun registry — panggilan jujur berikutnya tetap berhasil
//!   (keluhan #14).
//! - Input `forwardLayer` serta output linear/matmul/embedding divalidasi
//!   ukurannya SEBELUM materialisasi (keluhan #17/#18).
//! - `runtime_contract` dideklarasikan di sini (bukan via `#[path]` dari
//!   modul lain) agar dapat membaca field privat registry: privasi Rust
//!   mengikuti pohon modul yang dideklarasikan, bukan path file.
//!
//! ## Error yang dijamin
//! - String terstruktur; `tensor_too_large:` untuk tensor melebihi budget,
//!   selalu per-call dan tidak me-wedge.
//!
//! ## Bukan tanggung jawab modul ini
//! - Membangun paket init dari parameter ramah-agen → `agent.rs`.
//! - Engine komputasi per layer → `layers/*`.

use crate::layers::activation::WasmActivation;
use crate::layers::binary::WasmBinary;
use crate::layers::conv::WasmConv;
use crate::layers::custom::feature_norm::WasmFeatureNorm;
use crate::layers::custom::ghost::WasmGhostModule;
use crate::layers::custom::seblock::WasmSeBlock;
use crate::layers::custom::shift::WasmShift;
use crate::layers::embedding::WasmEmbedding;
use crate::layers::linear::WasmLinear;
use crate::layers::norm::WasmNorm;
use crate::layers::pool::WasmPool;
use crate::protocol::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use wasm_bindgen::prelude::*;

// The runtime-contract projection over the live registry. It lives here (not
// under `graph`) so it can read `LayerRegistry`'s private fields: privacy in
// Rust is scoped to the defining module and its descendants.
pub(crate) mod runtime_contract;

pub(crate) type LayerId = u32;
type LayerKey = (u8, LayerId);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerInitIdentity {
    variant: u8,
    flags: u8,
    payload: Vec<u8>,
}

impl LayerInitIdentity {
    pub(crate) fn new(header: &PacketHeader, payload: &[u8]) -> Self {
        Self {
            variant: header.variant,
            flags: header.flags,
            payload: payload.to_vec(),
        }
    }

    pub(crate) fn fingerprint(&self, layer_type: u8, layer_id: LayerId) -> String {
        let mut out = format!(
            "type={layer_type:02x};id={layer_id};variant={:02x};flags={:02x};payload=",
            self.variant, self.flags
        );
        for byte in &self.payload {
            let _ = write!(&mut out, "{byte:02x}");
        }
        out
    }
}

#[wasm_bindgen]
pub struct LayerRegistry {
    pub(crate) linears: HashMap<LayerId, WasmLinear>,
    pub(crate) norms: HashMap<LayerId, WasmNorm>,
    pub(crate) convs: HashMap<LayerId, WasmConv>,
    pub(crate) activations: HashMap<LayerId, WasmActivation>,
    pub(crate) embeddings: HashMap<LayerId, WasmEmbedding>,
    pub(crate) pools: HashMap<LayerId, WasmPool>,
    pub(crate) shifts: HashMap<LayerId, WasmShift>,
    pub(crate) ghosts: HashMap<LayerId, WasmGhostModule>,
    pub(crate) seblocks: HashMap<LayerId, WasmSeBlock>,
    pub(crate) binaries: HashMap<LayerId, WasmBinary>,
    pub(crate) feature_norms: HashMap<LayerId, WasmFeatureNorm>,
    pub(crate) init_identities: HashMap<LayerKey, LayerInitIdentity>,
    pub(crate) cached_params: usize,
}

// #[wasm_bindgen] impl LayerRegistry — dipindah ke src/facade/registry.rs (Opsi C Fase 2).

// #[wasm_bindgen] impl LayerRegistry — dipindah ke src/facade/registry.rs (Opsi C Fase 2).

// #[wasm_bindgen] impl LayerRegistry — dipindah ke src/facade/registry.rs (Opsi C Fase 2).

// ============================================================
// GRAPH EXECUTOR — plan 9 byte/step (unary + binary)
// ============================================================
const MAX_SLOTS: u32 = 64;

#[derive(Clone, Copy)]
pub(crate) struct RunStep {
    pub(crate) arity: u8,
    pub(crate) layer_type: u8,
    pub(crate) layer_id: u32,
    pub(crate) in_slot: u8,
    pub(crate) in_slot2: u8,
    pub(crate) out_slot: u8,
}

pub(crate) fn read_run_step(c: &mut PayloadCursor) -> Result<RunStep, String> {
    Ok(RunStep {
        arity: c.read_u8()?,
        layer_type: c.read_u8()?,
        layer_id: c.read_u32()?,
        in_slot: c.read_u8()?,
        in_slot2: c.read_u8()?,
        out_slot: c.read_u8()?,
    })
}

fn contains_layer(reg: &LayerRegistry, layer_type: u8, layer_id: u32) -> bool {
    match layer_type {
        LAYER_LINEAR => reg.linears.contains_key(&layer_id),
        LAYER_NORM => reg.norms.contains_key(&layer_id),
        LAYER_CONV => reg.convs.contains_key(&layer_id),
        LAYER_ACTIVATION => reg.activations.contains_key(&layer_id),
        LAYER_EMBEDDING => reg.embeddings.contains_key(&layer_id),
        LAYER_POOL => reg.pools.contains_key(&layer_id),
        LAYER_SHIFT => reg.shifts.contains_key(&layer_id),
        LAYER_GHOST => reg.ghosts.contains_key(&layer_id),
        LAYER_SEBLOCK => reg.seblocks.contains_key(&layer_id),
        LAYER_BINARY => reg.binaries.contains_key(&layer_id),
        _ => false,
    }
}

pub(crate) fn validate_plan(reg: &LayerRegistry, plan: &[u8]) -> Result<(u32, u32, u8), String> {
    let mut c = PayloadCursor::new(plan);
    let num_steps = c.read_u32()?;
    let num_slots = c.read_u32()?;
    if num_steps == 0 {
        return Err("run_graph: plan has no steps".into());
    }
    if !(1..=MAX_SLOTS).contains(&num_slots) {
        return Err(format!(
            "run_graph: num_slots must be 1..={}, got {}",
            MAX_SLOTS, num_slots
        ));
    }
    let mut filled: u64 = 1;
    for _ in 0..num_steps {
        let s = read_run_step(&mut c)?;
        let in_slot = s.in_slot as u32;
        let in_slot2 = s.in_slot2 as u32;
        let out_slot = s.out_slot as u32;
        if in_slot >= num_slots || in_slot2 >= num_slots || out_slot >= num_slots {
            return Err(format!(
                "run_graph: slot index out of range (num_slots={})",
                num_slots
            ));
        }
        if s.arity == crate::graph::ARITY_BINARY {
            if s.layer_type != LAYER_BINARY {
                return Err(format!(
                    "run_graph: arity 2 requires LAYER_BINARY, got 0x{:02X}",
                    s.layer_type
                ));
            }
            if (filled >> in_slot) & 1 == 0 {
                return Err(format!("run_graph: input slot {} is empty", in_slot));
            }
            if (filled >> in_slot2) & 1 == 0 {
                return Err(format!("run_graph: input slot {} is empty", in_slot2));
            }
        } else if s.arity == crate::graph::ARITY_UNARY {
            if s.layer_type == LAYER_BINARY {
                return Err("run_graph: arity 1 cannot use LAYER_BINARY (needs 2 inputs)".into());
            }
            if (filled >> in_slot) & 1 == 0 {
                return Err(format!("run_graph: input slot {} is empty", in_slot));
            }
        } else {
            return Err(format!(
                "run_graph: invalid arity {} (expected 1 or 2)",
                s.arity
            ));
        }
        if !contains_layer(reg, s.layer_type, s.layer_id) {
            return Err(format!(
                "run_graph: layer type 0x{:02X} id {} not found",
                s.layer_type, s.layer_id
            ));
        }
        filled |= 1u64 << out_slot;
    }
    let out_slot = c.read_u8()? as u32;
    if out_slot >= num_slots {
        return Err(format!("run_graph: output slot {} out of range", out_slot));
    }
    if (filled >> out_slot) & 1 == 0 {
        return Err(format!(
            "run_graph: output slot {} is never written",
            out_slot
        ));
    }
    Ok((num_steps, num_slots, out_slot as u8))
}

// #[wasm_bindgen] impl LayerRegistry — dipindah ke src/facade/registry.rs (Opsi C Fase 2).

// #[wasm_bindgen] impl LayerRegistry — dipindah ke src/facade/registry.rs (Opsi C Fase 2).
