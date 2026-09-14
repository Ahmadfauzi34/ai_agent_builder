use wasm_bindgen::prelude::*;

use crate::math::{index_source_capabilities, TensorIndexSource};
use crate::WasmTensor;

#[wasm_bindgen(js_name = indexSourceCapabilities)]
pub fn wasm_index_source_capabilities() -> String {
    index_source_capabilities()
}

#[wasm_bindgen]
pub struct WasmIndexSource {
    inner: TensorIndexSource,
}

#[wasm_bindgen]
impl WasmIndexSource {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmIndexSource {
        WasmIndexSource {
            inner: TensorIndexSource::new(),
        }
    }

    #[wasm_bindgen(js_name = indicesLike)]
    pub fn indices_like(
        &self,
        reference: &WasmTensor,
        axis: u32,
    ) -> Result<WasmTensor, String> {
        self.inner.indices_like(reference, axis)
    }
}
