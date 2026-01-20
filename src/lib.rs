use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;

// 1. REGISTRASI MODUL
pub mod layers;

// 2. KUNCI BACKEND (CPU)
// Kita buat pub agar bisa dipakai di file lain
pub type WasmBackend = burn_ndarray::NdArray<f32>;

// 3. JEMBATAN DATA (WasmTensor)
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmTensor {
    pub(crate) inner: Tensor<WasmBackend, 4>, 
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let device = Default::default();
        
        let mut dims = [1, 1, 1, 1];
        for (i, &d) in shape.iter().enumerate().take(4) {
            dims[i] = d;
        }

        let tensor_data = TensorData::new(data.to_vec(), dims);
        let tensor = Tensor::from_data(tensor_data, &device);
        
        WasmTensor { inner: tensor }
    }

    pub fn to_array(&self) -> Vec<f32> {
        let data = self.inner.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
}
