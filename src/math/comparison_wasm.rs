use wasm_bindgen::prelude::*;

use crate::math::{comparison_capabilities, TensorComparison};
use crate::WasmTensor;

#[wasm_bindgen(js_name = comparisonCapabilities)]
pub fn wasm_comparison_capabilities() -> String {
    comparison_capabilities()
}

#[wasm_bindgen]
pub struct WasmComparison {
    inner: TensorComparison,
}

#[wasm_bindgen]
impl WasmComparison {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmComparison {
        WasmComparison {
            inner: TensorComparison::new(),
        }
    }

    #[wasm_bindgen(js_name = lessEqual01)]
    pub fn less_equal_01(
        &self,
        lhs: &WasmTensor,
        rhs: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.inner.less_equal_01(lhs, rhs)
    }
}
