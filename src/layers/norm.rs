use burn::prelude::*;

// 1. BatchNorm
#[derive(Config, Debug)]
pub struct StrictBatchNormConfig {
    pub num_features: usize,
}

impl StrictBatchNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictBatchNorm<B> {
        let norm = nn::BatchNormConfig::new(self.num_features).init(device);
        StrictBatchNorm { inner: norm }
    }
}

#[derive(Module, Debug)]
pub struct StrictBatchNorm<B: Backend> {
    // PERBAIKAN BURN 0.20: Hapus angka 2, cukup <B>
    inner: nn::BatchNorm<B>, 
}

impl<B: Backend> StrictBatchNorm<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inner.forward(input)
    }
}

// 2. LayerNorm
#[derive(Config, Debug)]
pub struct StrictLayerNormConfig {
    pub d_model: usize,
}

impl StrictLayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictLayerNorm<B> {
        let norm = nn::LayerNormConfig::new(self.d_model).init(device);
        StrictLayerNorm { inner: norm }
    }
}

#[derive(Module, Debug)]
pub struct StrictLayerNorm<B: Backend> {
    inner: nn::LayerNorm<B>,
}

impl<B: Backend> StrictLayerNorm<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.inner.forward(input)
    }
}