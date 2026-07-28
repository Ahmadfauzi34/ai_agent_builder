use wasm_bindgen::prelude::*;
use crate::protocol::{PayloadCursor, LAYER_BINARY};
use crate::registry::LayerRegistry;
use crate::WasmTensor;

// Arity = jumlah input sebuah langkah graph. Satu sumber untuk graph + registry.
pub(crate) const ARITY_UNARY: u8 = 1;
pub(crate) const ARITY_BINARY: u8 = 2;

// ============================================================
// COMPILE-ONCE GRAPH EXECUTOR  (format plan v2: mendukung langkah binary)
// ============================================================
// Wire format `plan` (little-endian), per langkah 9 byte (fixed stride):
//   num_steps : u32
//   num_slots : u32            // 1..=64 ; slot 0 = input eksternal
//   steps     : ulang num_steps kali ->
//               { arity:u8, layer_type:u8, layer_id:u32,
//                 in_slot:u8, in_slot2:u8, out_slot:u8 }
//   out_slot  : u8
//   arity 1 -> registry.forward_layer        (in_slot2 = don't-care, tulis 0)
//   arity 2 -> registry.forward_binary_layer (in_slot + in_slot2)
// Aturan: steps urut topologis; kipas-out boleh; arity<->type harus cocok.
// ============================================================

const CG_MAX_SLOTS: u32 = 64;

#[derive(Clone, Copy)]
struct CompiledStep {
    arity: u8,
    layer_type: u8,
    layer_id: u32,
    in_slot: u8,
    in_slot2: u8,
    out_slot: u8,
}

/// Plan yang sudah di-parse + divalidasi. Field private → tidak menyeberang boundary.
#[wasm_bindgen]
pub struct CompiledGraph {
    steps: Vec<CompiledStep>,
    num_slots: u32,
    out_slot: u8,
}

// --- logika internal (bukan API JS) ---
impl CompiledGraph {
    fn read_step(c: &mut PayloadCursor) -> Result<CompiledStep, String> {
        Ok(CompiledStep {
            arity: c.read_u8()?,
            layer_type: c.read_u8()?,
            layer_id: c.read_u32()?,
            in_slot: c.read_u8()?,
            in_slot2: c.read_u8()?,
            out_slot: c.read_u8()?,
        })
    }

    /// Parse + validasi SEMUA hal TANPA eksekusi. Gagal = Result::Err rapi.
    pub(crate) fn build(reg: &LayerRegistry, plan: &[u8]) -> Result<CompiledGraph, String> {
        let mut c = PayloadCursor::new(plan);
        let num_steps = c.read_u32()?;
        let num_slots = c.read_u32()?;

        if num_steps == 0 {
            return Err("compile_graph: plan has no steps".into());
        }
        if !(1..=CG_MAX_SLOTS).contains(&num_slots) {
            return Err(format!(
                "compile_graph: num_slots must be 1..={}, got {}",
                CG_MAX_SLOTS, num_slots
            ));
        }

        let mut steps: Vec<CompiledStep> = Vec::with_capacity(num_steps as usize);
        let mut filled: u64 = 1; // bit 0 = slot 0 (input) sudah terisi
        for _ in 0..num_steps {
            let s = Self::read_step(&mut c)?;
            let in_slot = s.in_slot as u32;
            let in_slot2 = s.in_slot2 as u32;
            let out_slot = s.out_slot as u32;

            if in_slot >= num_slots || in_slot2 >= num_slots || out_slot >= num_slots {
                return Err(format!(
                    "compile_graph: slot index out of range (num_slots={})",
                    num_slots
                ));
            }
            if s.arity == ARITY_BINARY {
                if s.layer_type != LAYER_BINARY {
                    return Err(format!(
                        "compile_graph: arity 2 requires LAYER_BINARY, got 0x{:02X}",
                        s.layer_type
                    ));
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
                if (filled >> in_slot2) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot2));
                }
            } else if s.arity == ARITY_UNARY {
                if s.layer_type == LAYER_BINARY {
                    return Err("compile_graph: arity 1 cannot use LAYER_BINARY (needs 2 inputs)".into());
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
            } else {
                return Err(format!("compile_graph: invalid arity {} (expected 1 or 2)", s.arity));
            }
            if !reg.layer_exists(s.layer_type, s.layer_id) {
                return Err(format!(
                    "compile_graph: layer type 0x{:02X} id {} not found",
                    s.layer_type, s.layer_id
                ));
            }
            filled |= 1u64 << out_slot; // out_slot < 64 dijamin → shift aman
            steps.push(s);
        }

        let out_slot = c.read_u8()? as u32;
        if out_slot >= num_slots {
            return Err(format!("compile_graph: output slot {} out of range", out_slot));
        }
        if (filled >> out_slot) & 1 == 0 {
            return Err(format!("compile_graph: output slot {} is never written", out_slot));
        }

        Ok(CompiledGraph { steps, num_slots, out_slot: out_slot as u8 })
    }
}

// --- API yang dilihat JS ---
#[wasm_bindgen]
impl CompiledGraph {
    /// Jalankan plan yang sudah di-compile. TIDAK ada parsing byte di sini.
    /// Borrow aman: ambil ref input, hasil owned, baru assign ke slot.
    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        registry: &LayerRegistry,
        input: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        slots[0] = Some(input.clone());

        for s in &self.steps {
            let out = if s.arity == ARITY_BINARY {
                let a = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot))?;
                let b = slots[s.in_slot2 as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot2))?;
                registry.forward_binary_layer(s.layer_id, a, b)?
            } else {
                let inp = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot))?;
                registry.forward_layer(s.layer_id, s.layer_type, inp)?
            };
            slots[s.out_slot as usize] = Some(out);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("run: empty output slot {}", self.out_slot))
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn step_count(&self) -> u32 { self.steps.len() as u32 }
    #[wasm_bindgen(js_name = numSlots)]
    pub fn slot_count(&self) -> u32 { self.num_slots }
    #[wasm_bindgen(js_name = outSlot)]
    pub fn output_slot(&self) -> u8 { self.out_slot }
}
