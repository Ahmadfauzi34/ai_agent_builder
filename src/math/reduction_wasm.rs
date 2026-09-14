use wasm_bindgen::prelude::*;

use crate::math::{reduction_capabilities, TensorReduction};
use crate::WasmTensor;

#[wasm_bindgen(js_name = reductionCapabilities)]
pub fn wasm_reduction_capabilities() -> String {
    reduction_capabilities()
}

#[wasm_bindgen]
pub struct WasmReduction {
    inner: TensorReduction,
}

#[wasm_bindgen]
impl WasmReduction {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmReduction {
        WasmReduction {
            inner: TensorReduction::new(),
        }
    }

    #[wasm_bindgen(js_name = sumAxis)]
    pub fn sum_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.sum_axis(input, axis)
    }

    #[wasm_bindgen(js_name = meanAxis)]
    pub fn mean_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.mean_axis(input, axis)
    }

    #[wasm_bindgen(js_name = minAxis)]
    pub fn min_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.min_axis(input, axis)
    }

    #[wasm_bindgen(js_name = maxAxis)]
    pub fn max_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.max_axis(input, axis)
    }
}
