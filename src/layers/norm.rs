use burn::prelude::*;
use burn::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIG ENUM ---
#[derive(Config, Debug)]
pub enum NormalizationConfig {
    Batch(BatchNormConfig),
    Group(GroupNormConfig),
    Instance(InstanceNormConfig),
    Layer(LayerNormConfig),
    Rms(RmsNormConfig),
}

impl NormalizationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Normalization<B> {
        match self {
            NormalizationConfig::Batch(config) => Normalization::Batch(config.init(device)),
            NormalizationConfig::Group(config) => Normalization::Group(config.init(device)),
            NormalizationConfig::Instance(config) => Normalization::Instance(config.init(device)),
            NormalizationConfig::Layer(config) => Normalization::Layer(config.init(device)),
            NormalizationConfig::Rms(config) => Normalization::Rms(config.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)] 
pub enum Normalization<B: Backend> {
    Batch(BatchNorm<B>),
    Group(GroupNorm<B>),
    Instance(InstanceNorm<B>),
    Layer(LayerNorm<B>),
    Rms(RmsNorm<B>),
}

impl<B: Backend> Normalization<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Normalization::Batch(norm) => norm.forward(input),
            Normalization::Group(norm) => norm.forward(input),
            Normalization::Instance(norm) => norm.forward(input),
            Normalization::Layer(norm) => norm.forward(input),
            Normalization::Rms(norm) => norm.forward(input),
        }
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmNorm {
    inner: Normalization<WasmBackend>,
}

#[wasm_bindgen]
impl WasmNorm {
    #[wasm_bindgen]
    pub fn new_rms_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Rms(RmsNormConfig::new(size).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_layer_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Layer(LayerNormConfig::new(size).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_batch_norm(num_features: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Batch(BatchNormConfig::new(num_features).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_group_norm(num_groups: usize, num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Group(GroupNormConfig::new(num_groups, num_channels).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_instance_norm(num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Instance(InstanceNormConfig::new(num_channels).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &device)
            .map_err(|e| e.to_string())?;
            
        // PERBAIKAN: Clone dulu
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        let record = self.inner.clone().into_record();
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(record, ())
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layer_norm() {
        // Test with default epsilon
        let norm_default = WasmNorm::new_layer_norm(128, None);
        assert_eq!(norm_default.num_params(), 256); // gamma and beta for size 128 (128 * 2)

        // Test with custom epsilon
        let norm_custom = WasmNorm::new_layer_norm(64, Some(1e-6));
        assert_eq!(norm_custom.num_params(), 128);
    }
}
