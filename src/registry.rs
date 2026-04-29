// src/registry.rs
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};
use crate::layers::linear::WasmLinear;
use crate::layers::norm::WasmNorm;
use crate::layers::conv::WasmConv; // nanti
use crate::layers::pool::WasmPool; // nanti

type LayerId = u32;

#[wasm_bindgen]
pub struct LayerRegistry {
    linears: HashMap<LayerId, WasmLinear>,
    norms: HashMap<LayerId, WasmNorm>,
    // convs, pools, dll.
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        LayerRegistry {
            linears: HashMap::new(),
            norms: HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = createLinear)]
    pub fn create_linear(&mut self, id: LayerId, in_dim: usize, out_dim: usize, bias: bool) {
        let layer = WasmLinear::new(in_dim, out_dim, bias);
        self.linears.insert(id, layer);
    }

    #[wasm_bindgen(js_name = createRmsNorm)]
    pub fn create_rms_norm(&mut self, id: LayerId, size: usize, epsilon: Option<f64>) {
        let layer = WasmNorm::new_rms_norm(size, epsilon);
        self.norms.insert(id, layer);
    }

    #[wasm_bindgen(js_name = forward)]
    pub fn forward(&self, layer_id: LayerId, layer_type: u8, input: &WasmTensor) -> Option<WasmTensor> {
        match layer_type {
            0x01 => self.linears.get(&layer_id).map(|l| l.forward(input)),
            0x02 => self.norms.get(&layer_id).map(|l| l.forward(input)),
            _ => None,
        }
    }
}

