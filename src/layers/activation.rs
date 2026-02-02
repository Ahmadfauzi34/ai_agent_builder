use burn::prelude::*;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu,
    PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
// PERBAIKAN 1: Import HardSwish dari path yang benar
use burn::nn::activation::HardSwish; 

// --- HELPER STRUCTS (SOLUSI ERROR DERIVE) ---
// Kita bungkus usize ke dalam struct Module agar Enum Activation tidak panic.

#[derive(Module, Debug, Clone)]
pub struct StrictSoftmax {
    pub dim: usize,
}
impl StrictSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictLogSoftmax {
    pub dim: usize,
}
impl StrictLogSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::log_softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictGlu {
    pub dim: usize,
}
impl StrictGlu {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::glu(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictMish;
impl StrictMish {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::mish(input)
    }
}

// --- CONFIGURATION ENUM ---
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
    
    // Manual Ops Config
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

            // PERBAIKAN 2: Init ke Struct Wrapper, bukan langsung enum variant
            ActivationConfig::Mish => Activation::Mish(StrictMish),
            ActivationConfig::Softmax { dim } => Activation::Softmax(StrictSoftmax { dim: *dim }),
            ActivationConfig::LogSoftmax { dim } => Activation::LogSoftmax(StrictLogSoftmax { dim: *dim }),
            ActivationConfig::Glu { dim } => Activation::Glu(StrictGlu { dim: *dim }),
        }
    }
}

// --- MODULE ENUM ---
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
    
    // PERBAIKAN 3: Varian sekarang berisi Struct Module, bukan usize
    Mish(StrictMish),
    Softmax(StrictSoftmax),
    LogSoftmax(StrictLogSoftmax),
    Glu(StrictGlu),
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
            
            // Panggil forward dari struct wrapper
            Activation::Mish(m) => m.forward(input),
            Activation::Softmax(m) => m.forward(input),
            Activation::LogSoftmax(m) => m.forward(input),
            Activation::Glu(m) => m.forward(input),
        }
    }
}
