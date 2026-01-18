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

    // PERBAIKAN: Ubah Tensor<B, 2> menjadi Tensor<B, 4>
    // Agar kompatibel dengan WasmTensor yang sekarang universal 4D
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inner.forward(input)
    }
}

// 2. Gelu (Gaussian Error Linear Unit)
#[derive(Module, Debug, Clone)]
pub struct StrictGelu {
    inner: nn::Gelu,
}

impl StrictGelu {
    pub fn new() -> Self {
        Self { inner: nn::Gelu::new() }
    }

    // PERBAIKAN: Ubah ke 4D
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inner.forward(input)
    }
}

// 3. Sigmoid
#[derive(Module, Debug, Clone)]
pub struct StrictSigmoid;

impl StrictSigmoid {
    pub fn new() -> Self {
        Self
    }

    // PERBAIKAN: Ubah ke 4D
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::sigmoid(input)
    }
}
