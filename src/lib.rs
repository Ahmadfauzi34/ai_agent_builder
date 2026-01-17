use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData; // PENTING: Pakai TensorData, bukan Data

// Import modul layer
pub mod layers;
use layers::linear::{StrictLinear, StrictLinearConfig};

// Kunci Backend ke NdArray (CPU) untuk WASM
type WasmBackend = burn_ndarray::NdArray<f32>;

#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<WasmBackend, 2>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], rows: usize, cols: usize) -> WasmTensor {
        let device = Default::default();
        
        // PERBAIKAN BURN 0.15:
        // Gunakan TensorData::new(data, shape)
        let tensor_data = TensorData::new(
            data.to_vec(), 
            [rows, cols] // Shape langsung array, bukan .into()
        );

        let tensor = Tensor::from_data(tensor_data, &device);
        
        WasmTensor { inner: tensor }
    }

    pub fn to_array(&self) -> Vec<f32> {
        // PERBAIKAN BURN 0.15:
        // Tidak ada field .value. Kita harus convert ke slice lalu to_vec.
        let data = self.inner.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
}

#[wasm_bindgen]
pub struct WasmStrictLinear {
    inner: StrictLinear<WasmBackend>,
}

#[wasm_bindgen]
impl WasmStrictLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize) -> WasmStrictLinear {
        let device = Default::default();
        let config = StrictLinearConfig::new(in_dim, out_dim);
        let layer = config.init(&device);
        
        WasmStrictLinear { inner: layer }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let output = self.inner.forward(x);
        WasmTensor { inner: output }
    }
}
