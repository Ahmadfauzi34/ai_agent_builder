use burn::prelude::*;
use burn::nn::pool::{
    MaxPool1d, MaxPool1dConfig,
    MaxPool2d, MaxPool2dConfig,
    AvgPool1d, AvgPool1dConfig,
    AvgPool2d, AvgPool2dConfig,
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig,
};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum PoolingConfig {
    MaxPool1d(MaxPool1dConfig),
    MaxPool2d(MaxPool2dConfig),
    AvgPool1d(AvgPool1dConfig),
    AvgPool2d(AvgPool2dConfig),
    AdaptiveAvgPool2d(AdaptiveAvgPool2dConfig),
}

impl PoolingConfig {
    // Pooling layers tidak punya parameter, jadi init() tanpa device
    pub fn init(&self) -> Pooling {
        match self {
            PoolingConfig::MaxPool1d(c) => Pooling::MaxPool1d(c.init()),
            PoolingConfig::MaxPool2d(c) => Pooling::MaxPool2d(c.init()),
            PoolingConfig::AvgPool1d(c) => Pooling::AvgPool1d(c.init()),
            PoolingConfig::AvgPool2d(c) => Pooling::AvgPool2d(c.init()),
            PoolingConfig::AdaptiveAvgPool2d(c) => Pooling::AdaptiveAvgPool2d(c.init()),
        }
    }
}

// --- MODULE ENUM ---
// Pooling tidak generic (tidak ada trainable params)
#[derive(Module, Debug)]
pub enum Pooling {
    MaxPool1d(MaxPool1d),
    MaxPool2d(MaxPool2d),
    AvgPool1d(AvgPool1d),
    AvgPool2d(AvgPool2d),
    AdaptiveAvgPool2d(AdaptiveAvgPool2d),
}

impl Pooling {
    // Input selalu 4D [Batch, Channel, H, W] dari WasmTensor
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            // 2D pooling native support 4D
            Pooling::MaxPool2d(layer) => layer.forward(input),
            Pooling::AvgPool2d(layer) => layer.forward(input),
            Pooling::AdaptiveAvgPool2d(layer) => layer.forward(input),

            // 1D pooling butuh 3D [Batch, Channel, Length]
            // Squeeze dimensi terakhir (width), pool, lalu unsqueeze balik
            Pooling::MaxPool1d(layer) => {
                let [b, c, h, _w] = input.dims();
                let x_3d = input.reshape([b, c, h]);
                let out = layer.forward(x_3d);
                let [b_out, c_out, l_out] = out.dims();
                out.reshape([b_out, c_out, l_out, 1])
            }
            Pooling::AvgPool1d(layer) => {
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
pub struct WasmPool {
    inner: Pooling,
}

#[wasm_bindgen]
impl WasmPool {
    #[wasm_bindgen(js_name = newMaxPool1d)]
    pub fn new_max_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        let mut config = MaxPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::MaxPool1d(config).init(),
        }
    }

    // [usize; 2] dipecah jadi 2 parameter karena wasm_bindgen tidak support array
    #[wasm_bindgen(js_name = newMaxPool2d)]
    pub fn new_max_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmPool {
        let mut config = MaxPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_stride([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding([ph, pw]);
        }
        WasmPool {
            inner: PoolingConfig::MaxPool2d(config).init(),
        }
    }

    #[wasm_bindgen(js_name = newAvgPool1d)]
    pub fn new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        let mut config = AvgPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::AvgPool1d(config).init(),
        }
    }

    #[wasm_bindgen(js_name = newAvgPool2d)]
    pub fn new_avg_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmPool {
        let mut config = AvgPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_stride([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding([ph, pw]);
        }
        WasmPool {
            inner: PoolingConfig::AvgPool2d(config).init(),
        }
    }

    #[wasm_bindgen(js_name = newAdaptiveAvgPool2d)]
    pub fn new_adaptive_avg_pool2d(
        output_size_h: usize,
        output_size_w: usize,
    ) -> WasmPool {
        let config = AdaptiveAvgPool2dConfig::new([output_size_h, output_size_w]);
        WasmPool {
            inner: PoolingConfig::AdaptiveAvgPool2d(config).init(),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    // Pooling tidak punya trainable params
    pub fn num_params(&self) -> usize {
        0
    }
}
