use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;

// 1. REGISTRASI MODUL
// Pastikan folder src/layers/ ada dan berisi activation.rs & mod.rs
pub mod layers;

// Import Enum Aktivasi yang baru kita buat
use layers::activation::{Activation, ActivationConfig};

// Import Config bawaan Burn untuk parameter aktivasi
use burn::nn::{
    LeakyReluConfig, PreluConfig, HardSigmoidConfig, SoftplusConfig, SwiGluConfig
};

// 2. KUNCI BACKEND (CPU)
type WasmBackend = burn_ndarray::NdArray<f32>;

// 3. JEMBATAN DATA (WasmTensor)
#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<WasmBackend, 4>, 
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let device = Default::default();
        
        // Paksa shape menjadi 4 Dimensi [d1, d2, d3, d4]
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

// 4. WRAPPER AKTIVASI (Menggunakan Enum Activation)
#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    // --- Factory Methods (Pembuat Layer) ---

    #[wasm_bindgen]
    pub fn new_relu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Relu.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_gelu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Gelu.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_sigmoid() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Sigmoid.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_tanh() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Tanh.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_hard_swish() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::HardSwish.init(&device) }
    }

    // --- Factory Methods dengan Parameter ---

    #[wasm_bindgen]
    pub fn new_leaky_relu(slope: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::LeakyRelu(
            LeakyReluConfig::new().with_negative_slope(slope)
        );
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_prelu() -> WasmActivation {
        let device = Default::default();
        // Prelu punya weights yang bisa dilatih, diinit default dulu
        let config = ActivationConfig::Prelu(PreluConfig::new());
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_swiglu(d_model: usize) -> WasmActivation {
        let device = Default::default();
        // SwiGlu butuh dimensi input karena dia memecah tensor
        let config = ActivationConfig::SwiGlu(SwiGluConfig::new(d_model));
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_hard_sigmoid(alpha: f64, beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::HardSigmoid(
            HardSigmoidConfig::new().with_alpha(alpha).with_beta(beta)
        );
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_softplus(beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::Softplus(
            SoftplusConfig::new().with_beta(beta)
        );
        WasmActivation { inner: config.init(&device) }
    }

    // --- Forward Pass ---
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}
