use burn::prelude::*;
use burn::nn::pool::{
    MaxPool1d, MaxPool1dConfig,
    MaxPool2d, MaxPool2dConfig,
    AvgPool1d, AvgPool1dConfig,
    AvgPool2d, AvgPool2dConfig,
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig,
};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> Pooling<B> {
        match self {
            PoolingConfig::MaxPool1d(c) => Pooling::MaxPool1d(c.init(device)),
            PoolingConfig::MaxPool2d(c) => Pooling::MaxPool2d(c.init(device)),
            PoolingConfig::AvgPool1d(c) => Pooling::AvgPool1d(c.init(device)),
            PoolingConfig::AvgPool2d(c) => Pooling::AvgPool2d(c.init(device)),
            PoolingConfig::AdaptiveAvgPool2d(c) => Pooling::AdaptiveAvgPool2d(c.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Pooling<B: Backend> {
    MaxPool1d(MaxPool1d),
    MaxPool2d(MaxPool2d),
    AvgPool1d(AvgPool1d),
    AvgPool2d(AvgPool2d),
    AdaptiveAvgPool2d(AdaptiveAvgPool2d),
}

impl<B: Backend> Pooling<B> {
    // Input selalu 4D [Batch, Channel, H, W] dari WasmTensor
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
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
    inner: Pooling<WasmBackend>,
}

#[wasm_bindgen]
impl WasmPool {
    #[wasm_bindgen(js_name = newMaxPool1d)]
    pub fn new_max_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        let device = Default::default();
        let mut config = MaxPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::MaxPool1d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newMaxPool2d)]
    pub fn new_max_pool2d(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: Option<[usize; 2]>,
    ) -> WasmPool {
        let device = Default::default();
        let mut config = MaxPool2dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::MaxPool2d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newAvgPool1d)]
    pub fn new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        let device = Default::default();
        let mut config = AvgPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::AvgPool1d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newAvgPool2d)]
    pub fn new_avg_pool2d(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: Option<[usize; 2]>,
    ) -> WasmPool {
        let device = Default::default();
        let mut config = AvgPool2dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(p);
        }
        WasmPool {
            inner: PoolingConfig::AvgPool2d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newAdaptiveAvgPool2d)]
    pub fn new_adaptive_avg_pool2d(output_size: [usize; 2]) -> WasmPool {
        let device = Default::default();
        let config = AdaptiveAvgPool2dConfig::new(output_size);
        WasmPool {
            inner: PoolingConfig::AdaptiveAvgPool2d(config).init(&device),
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

