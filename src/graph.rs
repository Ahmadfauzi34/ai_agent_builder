use wasm_bindgen::prelude::*;
use crate::protocol::PayloadCursor;
use crate::registry::LayerRegistry;
use crate::WasmTensor;

// ============================================================
// COMPILE-ONCE GRAPH EXECUTOR
// ============================================================
// parse + validasi plan DILAKUKAN SEKALI di registry.compileGraph();
// hasilnya (CompiledGraph) dipegang JS. Loop panas memanggil
// compiled.run(registry, input) yang TIDAK mem-parse byte lagi, dan
// tensor intermediate TIDAK menyeberang boundary.
//
// run() komposisi penuh dari registry.forward_layer() → 9 cabang dispatch
// tidak diduplikasi; konsistensi shape/Result otomatis terwarisi.
//
// Wire format `plan` (little-endian):
//   num_steps : u32
//   num_slots : u32            // 1..=64 ; slot 0 = input eksternal
//   steps     : ulang num_steps kali ->
//               { layer_type: u8, layer_id: u32, in_slot: u8, out_slot: u8 }
//   out_slot  : u8             // slot mana yang dikembalikan
//
// ATURAN: steps HARUS urut topologis (sebuah langkah hanya boleh membaca
// slot yang sudah diisi input atau oleh langkah sebelumnya). Kipas-out
// (satu slot dibaca banyak langkah) boleh.
// ============================================================

const CG_MAX_SLOTS: u32 = 64;

#[derive(Clone, Copy)]
struct CompiledStep {
    layer_type: u8,
    layer_id: u32,
    in_slot: u8,
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
            layer_type: c.read_u8()?,
            layer_id: c.read_u32()?,
            in_slot: c.read_u8()?,
            out_slot: c.read_u8()?,
        })
    }

    /// Parse + validasi SEMUA hal (format, rentang slot, dataflow, keberadaan
    /// layer) TANPA eksekusi. Gagal = Result::Err rapi, bukan panic.
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
            let out_slot = s.out_slot as u32;

            if in_slot >= num_slots || out_slot >= num_slots {
                return Err(format!(
                    "compile_graph: slot index out of range (num_slots={})",
                    num_slots
                ));
            }
            if (filled >> in_slot) & 1 == 0 {
                return Err(format!("compile_graph: input slot {} is empty", in_slot));
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
    /// Jalankan plan yang sudah di-compile di atas `registry` dengan `input`
    /// di slot 0. TIDAK ada parsing byte di sini — ini jalur panas.
    /// Kalau sebuah layer di-destroy setelah compile, forward_layer() di bawah
    /// mengembalikan Err rapi (bukan trap).
    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        registry: &LayerRegistry,
        input: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        // invariant: build() menjamin num_slots >= 1 → slots[0] aman.
        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        slots[0] = Some(input.clone());

        for s in &self.steps {
            let inp = slots[s.in_slot as usize]
                .as_ref()
                .ok_or_else(|| format!("run: empty input slot {}", s.in_slot))?;
            // dispatch yang SUDAH ADA — tidak ada duplikasi 9 cabang.
            let out = registry.forward_layer(s.layer_id, s.layer_type, inp)?;
            slots[s.out_slot as usize] = Some(out);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("run: empty output slot {}", self.out_slot))
    }

    // Introspeksi murah — berguna untuk assertion di test JS-mu.
    #[wasm_bindgen(js_name = numSteps)]
    pub fn step_count(&self) -> u32 { self.steps.len() as u32 }
    #[wasm_bindgen(js_name = numSlots)]
    pub fn slot_count(&self) -> u32 { self.num_slots }
    #[wasm_bindgen(js_name = outSlot)]
    pub fn output_slot(&self) -> u8 { self.out_slot }
      }
