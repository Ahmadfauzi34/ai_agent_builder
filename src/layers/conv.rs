use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIG ---
#[derive(Config, Debug)]
pub struct Conv2dLayerConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    // PERBAIKAN: Pastikan tipe datanya Array [usize; 2]
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl Conv2dLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2dLayer<B> {
        let conv = Conv2dConfig::new(
            [self.channels_in, self.channels_out],
            self.kernel_size
        )
        .with_stride(self.stride)
        .with_padding(PaddingConfig2d::Explicit(self.padding[0], self.padding[1]))
        .init(device);

        Conv2dLayer { inner: conv }
    }
}

// --- MODULE ---
#[derive(Module, Debug)]
pub struct Conv2dLayer<B: Backend> {
    inner: Conv2d<B>,
}

impl<B: Backend> Conv2dLayer<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inner.forward(input)
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmConv2d {
    inner: Conv2dLayer<WasmBackend>,
}

#[wasm_bindgen]
impl WasmConv2d {
    #[wasm_bindgen(constructor)]
    pub fn new(c_in: usize, c_out: usize, k_h: usize, k_w: usize, pad_h: usize, pad_w: usize, stride: usize) -> WasmConv2d {
        let device = Default::default();
        // PERBAIKAN: Inisialisasi struct sesuai tipe data array
        let config = Conv2dLayerConfig {
            channels_in: c_in,
            channels_out: c_out,
            kernel_size: [k_h, k_w],
            stride: [stride, stride],
            padding: [pad_h, pad_w],
        };
        WasmConv2d { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}
