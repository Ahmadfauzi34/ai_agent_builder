use burn::prelude::*;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig, Prelu,
    PreluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};

// --- 1. CONFIGURATION ENUM ---
// Ini adalah "Menu" untuk memilih jenis aktivasi dan parameternya.
#[derive(Config, Debug)]
pub enum ActivationConfig {
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    HardSwish,
    
    // Aktivasi yang punya parameter konfigurasi:
    LeakyRelu(LeakyReluConfig),
    Prelu(PreluConfig),
    SwiGlu(SwiGluConfig),
    HardSigmoid(HardSigmoidConfig),
    Softplus(SoftplusConfig),
}

impl ActivationConfig {
    /// Mengubah Config menjadi Layer Hidup (Module)
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            // Stateless (Tidak butuh config khusus)
            ActivationConfig::Gelu => Activation::Gelu(Gelu::new()),
            ActivationConfig::Relu => Activation::Relu(Relu::new()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            ActivationConfig::Tanh => Activation::Tanh(Tanh::new()),
            ActivationConfig::HardSwish => Activation::HardSwish(HardSwish::new()),

            // Stateful / Configurable
            ActivationConfig::LeakyRelu(config) => Activation::LeakyRelu(config.init()),
            ActivationConfig::Prelu(config) => Activation::Prelu(config.init(device)),
            ActivationConfig::SwiGlu(config) => Activation::SwiGlu(config.init(device)),
            ActivationConfig::HardSigmoid(config) => Activation::HardSigmoid(config.init()),
            ActivationConfig::Softplus(config) => Activation::Softplus(config.init()),
        }
    }
}

// --- 2. MODULE ENUM ---
// Ini adalah wadah "Layer Hidup" yang menyimpan state (jika ada).
#[derive(Module, Debug)]
pub enum Activation<B: Backend> {
    Gelu(Gelu),
    Relu(Relu),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    HardSwish(HardSwish),
    
    LeakyRelu(LeakyRelu),
    Prelu(Prelu<B>),
    SwiGlu(SwiGlu<B>),
    HardSigmoid(HardSigmoid),
    Softplus(Softplus),
}

impl<B: Backend> Activation<B> {
    /// Forward pass universal.
    /// Kita menggunakan Tensor 4D agar konsisten dengan WasmTensor.
    /// (Aktivasi bersifat element-wise, jadi dimensi berapapun tidak masalah sebenarnya).
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activation::Gelu(layer) => layer.forward(input),
            Activation::Relu(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::HardSwish(layer) => layer.forward(input),
            
            Activation::LeakyRelu(layer) => layer.forward(input),
            Activation::Prelu(layer) => layer.forward(input),
            Activation::SwiGlu(layer) => layer.forward(input),
            Activation::HardSigmoid(layer) => layer.forward(input),
            Activation::Softplus(layer) => layer.forward(input),
        }
    }
}
