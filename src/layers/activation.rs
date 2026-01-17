use burn::prelude::*;

// 1. ReLU (Rectified Linear Unit)
#[derive(Module, Debug, Clone)]
pub struct StrictRelu {
    inner: nn::Relu,
}

impl StrictRelu {
    pub fn new() -> Self {
        Self { inner: nn::Relu::new() }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }
}

// 2. Gelu (Gaussian Error Linear Unit - Dipakai di LLM)
#[derive(Module, Debug, Clone)]
pub struct StrictGelu {
    inner: nn::Gelu,
}

impl StrictGelu {
    pub fn new() -> Self {
        Self { inner: nn::Gelu::new() }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }
}

// 3. Sigmoid (Untuk probabilitas 0-1)
// Burn belum punya struct khusus Sigmoid di nn, jadi kita panggil tensor ops langsung
#[derive(Module, Debug, Clone)]
pub struct StrictSigmoid;

impl StrictSigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::sigmoid(input)
    }
}
