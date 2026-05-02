use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Relu, Sigmoid};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIGURATION ---
#[derive(Config, Debug)]
pub struct SeBlockConfig {
    pub channels: usize,
    #[config(default = 16)]
    pub reduction: usize,
}

impl SeBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SeBlock<B> {
        assert!(self.channels >= self.reduction, "channels must be >= reduction");
        let reduced = self.channels / self.reduction;

        // Squeeze: Global Average Pooling (AdaptiveAvgPool2d to [1,1])
        let squeeze = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        // Excitation: FC1 (channels → channels/reduction) + ReLU
        let fc1 = LinearConfig::new(self.channels, reduced).init(device);
        let relu = Relu::new();

        // Excitation: FC2 (channels/reduction → channels) + Sigmoid
        let fc2 = LinearConfig::new(reduced, self.channels).init(device);
        let sigmoid = Sigmoid::new();

        SeBlock {
            squeeze,
            fc1,
            relu,
            fc2,
            sigmoid,
            channels: self.channels,
        }
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

        // Squeeze: Global Average Pooling → [B, C, 1, 1]
        let pooled = self.squeeze.forward(input.clone());

        // Flatten → [B, C]
        let flat = pooled.reshape([b, c]);

        // Excitation: FC1 + ReLU → [B, C/r]
        let x = self.fc1.forward(flat);
        let x = self.relu.forward(x);

        // Excitation: FC2 + Sigmoid → [B, C]
        let x = self.fc2.forward(x);
        let weights = self.sigmoid.forward(x);

        // Reshape weights → [B, C, 1, 1] untuk broadcast multiply
        let weights_4d = weights.reshape([b, c, 1, 1]);

        // Scale: input * weights (broadcast)
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
        WasmSeBlock {
            inner: config.init(&device),
        }
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
