use burn::prelude::*;
use burn::nn::conv::{
    Conv1d, Conv1dConfig,
    Conv2d, Conv2dConfig,
    ConvTranspose2d, ConvTranspose2dConfig
};
use burn::nn::PaddingConfig2d;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum ConvolutionConfig {
    Conv1d(Conv1dConfig),
    Conv2d(Conv2dConfig),
    ConvTranspose2d(ConvTranspose2dConfig),
}

impl ConvolutionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Convolution<B> {
        match self {
            ConvolutionConfig::Conv1d(c) => Convolution::Conv1d(c.init(device)),
            ConvolutionConfig::Conv2d(c) => Convolution::Conv2d(c.init(device)),
            ConvolutionConfig::ConvTranspose2d(c) => Convolution::ConvTranspose2d(c.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Convolution<B: Backend> {
    Conv1d(Conv1d<B>),
    Conv2d(Conv2d<B>),
    ConvTranspose2d(ConvTranspose2d<B>),
}

impl<B: Backend> Convolution<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Convolution::Conv2d(layer) => layer.forward(input),
            Convolution::ConvTranspose2d(layer) => layer.forward(input),
            Convolution::Conv1d(layer) => {
                let [b, c, h, _w] = input.dims();
                let x_3d = input.reshape([b, c, h]);
                let out = layer.forward(x_3d);
                let [b_out, c_out, l_out] = out.dims();
                out.reshape([b_out, c_out, l_out, 1])
            }
        }
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmConv {
    inner: Convolution<WasmBackend>,
}

#[wasm_bindgen]
impl WasmConv {
    #[wasm_bindgen(js_name = newConv1d)]
    pub fn new_conv1d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = Conv1dConfig::new(in_channels, out_channels, kernel_size);
        if let Some(s) = stride {
            config.stride = s;  // usize, bukan [usize; 1]
        }
        if let Some(p) = padding {
            config.padding = burn::nn::PaddingConfig1d::Explicit(p);
        }
        WasmConv {
            inner: ConvolutionConfig::Conv1d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newConv2d)]
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = Conv2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];  // [usize; 2], bukan PaddingConfig2d
        }
        WasmConv {
            inner: ConvolutionConfig::Conv2d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newConvTranspose2d)]
    pub fn new_conv_transpose2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = ConvTranspose2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];  // [usize; 2]
        }
        WasmConv {
            inner: ConvolutionConfig::ConvTranspose2d(config).init(&device),
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
