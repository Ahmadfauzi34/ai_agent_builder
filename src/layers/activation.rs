use burn::prelude::*;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig, PRelu,
    PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
use wasm_bindgen::prelude::*; // <-- Butuh ini buat Wrapper
use crate::{WasmBackend, WasmTensor}; // <-- Import tipe dari lib.rs

// --- 1. CONFIGURATION ENUM (Burn Logic) ---
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
    // Manual Ops
    Mish,
    Softmax { dim: usize },
    LogSoftmax { dim: usize },
    Glu { dim: usize },
}

impl ActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Gelu => Activation::Gelu(Gelu::new()),
            ActivationConfig::Relu => Activation::Relu(Relu::new()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            ActivationConfig::Tanh => Activation::Tanh(Tanh::new()),
            ActivationConfig::HardSwish => Activation::HardSwish(HardSwish::new()),
            ActivationConfig::LeakyRelu(c) => Activation::LeakyRelu(c.init()),
            ActivationConfig::PRelu(c) => Activation::PRelu(c.init(device)),
            ActivationConfig::SwiGlu(c) => Activation::SwiGlu(c.init(device)),
            ActivationConfig::HardSigmoid(c) => Activation::HardSigmoid(c.init()),
            ActivationConfig::Softplus(c) => Activation::Softplus(c.init()),
            ActivationConfig::Mish => Activation::Mish,
            ActivationConfig::Softmax { dim } => Activation::Softmax(*dim),
            ActivationConfig::LogSoftmax { dim } => Activation::LogSoftmax(*dim),
            ActivationConfig::Glu { dim } => Activation::Glu(*dim),
        }
    }
}

// --- 2. MODULE ENUM (Burn Logic) ---
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
    Mish,
    Softmax(usize),
    LogSoftmax(usize),
    Glu(usize),
}

impl<B: Backend> Activation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activation::Gelu(m) => m.forward(input),
            Activation::Relu(m) => m.forward(input),
            Activation::Sigmoid(m) => m.forward(input),
            Activation::Tanh(m) => m.forward(input),
            Activation::HardSwish(m) => m.forward(input),
            Activation::LeakyRelu(m) => m.forward(input),
            Activation::PRelu(m) => m.forward(input),
            Activation::SwiGlu(m) => m.forward(input),
            Activation::HardSigmoid(m) => m.forward(input),
            Activation::Softplus(m) => m.forward(input),
            Activation::Mish => burn::tensor::activation::mish(input),
            Activation::Softmax(dim) => burn::tensor::activation::softmax(input, *dim),
            Activation::LogSoftmax(dim) => burn::tensor::activation::log_softmax(input, *dim),
            Activation::Glu(dim) => burn::tensor::activation::glu(input, *dim),
        }
    }
}

// --- 3. WASM WRAPPER (Pindahan dari lib.rs) ---
#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    // Basic
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
    #[wasm_bindgen]
    pub fn new_mish() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Mish.init(&device) }
    }

    // Configurable
    #[wasm_bindgen]
    pub fn new_leaky_relu(slope: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::LeakyRelu(LeakyReluConfig::new().with_negative_slope(slope));
        WasmActivation { inner: config.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_prelu() -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::PRelu(PReluConfig::new());
        WasmActivation { inner: config.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_swiglu(d_input: usize, d_output: usize) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::SwiGlu(SwiGluConfig::new(d_input, d_output));
        WasmActivation { inner: config.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_hard_sigmoid(alpha: f64, beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::HardSigmoid(HardSigmoidConfig::new().with_alpha(alpha).with_beta(beta));
        WasmActivation { inner: config.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_softplus(beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::Softplus(SoftplusConfig::new().with_beta(beta));
        WasmActivation { inner: config.init(&device) }
    }

    // Dimension Aware
    #[wasm_bindgen]
    pub fn new_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Softmax { dim }.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_log_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::LogSoftmax { dim }.init(&device) }
    }
    #[wasm_bindgen]
    pub fn new_glu(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Glu { dim }.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}
