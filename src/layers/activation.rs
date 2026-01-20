use burn::prelude::*;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu,
    PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
use burn::nn::activation::HardSwish; // Import spesifik untuk HardSwish

// --- TAMBAHAN UNTUK WASM ---
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// ============================================================
// 1. LOGIKA BURN (Internal Rust)
// ============================================================

#[derive(Config, Debug)]
pub enum ActivationConfig {
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    HardSwish,
    LeakyRelu(LeakyReluConfig),
    PRelu(PReluConfig),
    SwiGlu(SwiGluConfig),
    HardSigmoid(HardSigmoidConfig),
    Softplus(SoftplusConfig),
}

impl ActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Gelu => Activation::Gelu(Gelu::new()),
            ActivationConfig::Relu => Activation::Relu(Relu::new()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            ActivationConfig::Tanh => Activation::Tanh(Tanh::new()),
            ActivationConfig::HardSwish => Activation::HardSwish(HardSwish::new()),
            ActivationConfig::LeakyRelu(config) => Activation::LeakyRelu(config.init()),
            ActivationConfig::PRelu(config) => Activation::PRelu(config.init(device)),
            ActivationConfig::SwiGlu(config) => Activation::SwiGlu(config.init(device)),
            ActivationConfig::HardSigmoid(config) => Activation::HardSigmoid(config.init()),
            ActivationConfig::Softplus(config) => Activation::Softplus(config.init()),
        }
    }
}

#[derive(Module, Debug)]
pub enum Activation<B: Backend> {
    Gelu(Gelu),
    Relu(Relu),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    HardSwish(HardSwish),
    LeakyRelu(LeakyRelu),
    PRelu(PRelu<B>),
    SwiGlu(SwiGlu<B>),
    HardSigmoid(HardSigmoid),
    Softplus(Softplus),
}

impl<B: Backend> Activation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activation::Gelu(layer) => layer.forward(input),
            Activation::Relu(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::HardSwish(layer) => layer.forward(input),
            Activation::LeakyRelu(layer) => layer.forward(input),
            Activation::PRelu(layer) => layer.forward(input),
            Activation::SwiGlu(layer) => layer.forward(input),
            Activation::HardSigmoid(layer) => layer.forward(input),
            Activation::Softplus(layer) => layer.forward(input),
        }
    }
}

// ============================================================
// 2. WRAPPER WASM (Ekspor ke JavaScript)
// ============================================================

#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    // --- Factory Methods (Tanpa Parameter) ---

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

    // --- Factory Methods (Dengan Parameter) ---

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
        let config = ActivationConfig::PRelu(PReluConfig::new());
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_swiglu(d_model: usize) -> WasmActivation {
        let device = Default::default();
        // SwiGlu butuh input dan output dim. Kita samakan saja.
        let config = ActivationConfig::SwiGlu(SwiGluConfig::new(d_model, d_model));
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
