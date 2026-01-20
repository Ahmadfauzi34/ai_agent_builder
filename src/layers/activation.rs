use burn::prelude::*;
// PERBAIKAN: Gunakan PRelu (R besar) dan pastikan path import benar
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu,
    PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
// HardSwish kadang ada di submodul activation, kita coba akses langsung atau via activation
use burn::nn::activation::HardSwish; 

#[derive(Config, Debug)]
pub enum ActivationConfig {
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    HardSwish,
    LeakyRelu(LeakyReluConfig),
    PRelu(PReluConfig), // Ganti Prelu -> PRelu
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
    PRelu(PRelu<B>), // Ganti Prelu -> PRelu
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
