use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, Relu, Sigmoid};
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;

use crate::{WasmBackend, WasmTensor};

#[inline]
fn reject_invalid_config<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

// --- CONFIGURATION ---
#[derive(Config, Debug)]
pub struct SeBlockConfig {
    pub channels: usize,
    #[config(default = 16)]
    pub reduction: usize,
}

impl SeBlockConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.channels == 0 {
            return Err("seblock: channels must be > 0".into());
        }
        if self.reduction == 0 {
            return Err("seblock: reduction must be > 0".into());
        }
        if self.channels < self.reduction {
            return Err(format!(
                "seblock: channels ({}) must be >= reduction ({})",
                self.channels, self.reduction
            ));
        }
        Ok(())
    }

    pub fn try_init<B: Backend>(&self, device: &B::Device) -> Result<SeBlock<B>, String> {
        self.validate()?;
        let reduced = self.channels / self.reduction;

        let squeeze = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let fc1 = LinearConfig::new(self.channels, reduced).init(device);
        let relu = Relu::new();
        let fc2 = LinearConfig::new(reduced, self.channels).init(device);
        let sigmoid = Sigmoid::new();

        Ok(SeBlock {
            squeeze,
            fc1,
            relu,
            fc2,
            sigmoid,
            channels: self.channels,
        })
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SeBlock<B> {
        self.try_init(device).unwrap_or_else(|e| panic!("{e}"))
    }
}

// --- MODULE ---
#[derive(Module, Debug)]
pub struct SeBlock<B: Backend> {
    squeeze: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    relu: Relu,
    fc2: Linear<B>,
    sigmoid: Sigmoid,
    channels: usize,
}

impl<B: Backend> SeBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, _h, _w] = input.dims();
        let pooled = self.squeeze.forward(input.clone());
        let flat = pooled.reshape([b, c]);
        let x = self.fc1.forward(flat);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x);
        let weights = self.sigmoid.forward(x);
        let weights_4d = weights.reshape([b, c, 1, 1]);
        input * weights_4d
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmSeBlock {
    inner: SeBlock<WasmBackend>,
}

#[wasm_bindgen]
impl WasmSeBlock {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: usize, reduction: Option<usize>) -> WasmSeBlock {
        let device = Default::default();
        let mut config = SeBlockConfig::new(channels);
        if let Some(r) = reduction {
            config.reduction = r;
        }
        let inner = config.try_init(&device).unwrap_or_else(reject_invalid_config);
        WasmSeBlock { inner }
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
    use super::SeBlockConfig;

    #[test]
    fn validation_rejects_zero_reduction_before_division() {
        let mut cfg = SeBlockConfig::new(16);
        cfg.reduction = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .contains("reduction must be > 0"));
    }

    #[test]
    fn validation_rejects_reduction_larger_than_channels() {
        let mut cfg = SeBlockConfig::new(8);
        cfg.reduction = 16;
        assert!(cfg.validate().unwrap_err().contains("must be >= reduction"));
    }

    #[test]
    fn validation_rejects_zero_channels() {
        let cfg = SeBlockConfig::new(0);
        assert!(cfg.validate().unwrap_err().contains("channels must be > 0"));
    }

    #[test]
    fn validation_accepts_existing_valid_contract() {
        let cfg = SeBlockConfig::new(16);
        assert!(cfg.validate().is_ok());
    }
}
