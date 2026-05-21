use burn::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIGURATION ---
#[derive(Config, Debug)]
pub struct GhostModuleConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    #[config(default = 2)]
    pub ratio: usize,
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
}

impl GhostModuleConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GhostModule<B> {
        assert!(
            self.out_channels % self.ratio == 0,
            "out_channels must be divisible by ratio"
        );

        let primary_ch = self.out_channels / self.ratio;

        // Primary conv: full Conv2d biasa
        // Conv2dConfig::new(channels, kernel_size) — channels = [in, out]
        let mut primary_cfg = Conv2dConfig::new(
            [self.in_channels, primary_ch],
            self.kernel_size,
        );
        primary_cfg.stride = self.stride;
        primary_cfg.padding = PaddingConfig2d::Explicit(self.padding[0], self.padding[1]);
        let primary = primary_cfg.init(device);

        // Cheap conv: depthwise (groups = primary_ch) pada output primary
        // Kernel 1x1 untuk efisiensi maksimal, no bias
        let mut cheap_cfg = Conv2dConfig::new(
            [primary_ch, primary_ch],
            [1, 1],
        );
        cheap_cfg.groups = primary_ch;
        cheap_cfg.bias = false;
        let cheap = cheap_cfg.init(device);

        GhostModule {
            primary,
            cheap,
            ratio: self.ratio,
            primary_ch,
        }
    }
}

// --- MODULE ---
#[derive(Module, Debug)]
pub struct GhostModule<B: Backend> {
    primary: Conv2d<B>,
    cheap: Conv2d<B>,
    ratio: usize,
    primary_ch: usize,
}

impl<B: Backend> GhostModule<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // 1. Primary conv → intrinsic feature maps
        let intrinsic = self.primary.forward(input);

        // 2. Cheap depthwise conv pada intrinsic → ghost feature maps
        let ghost = self.cheap.forward(intrinsic.clone());

        // 3. Concat intrinsic + ghost
        Tensor::cat(vec![intrinsic, ghost], 1)
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmGhostModule {
    inner: GhostModule<WasmBackend>,
}

#[wasm_bindgen]
impl WasmGhostModule {
    #[wasm_bindgen(constructor)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        ratio: Option<usize>,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmGhostModule {
        let device = Default::default();
        let mut config = GhostModuleConfig::new(in_channels, out_channels, [kernel_size_h, kernel_size_w]);
        if let Some(r) = ratio {
            config.ratio = r;
        }
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];
        }
        WasmGhostModule {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_ghost_module_config_init() {
        let device = Default::default();
        let config = GhostModuleConfig::new(16, 32, [3, 3])
            .with_ratio(2)
            .with_stride([1, 1])
            .with_padding([1, 1]);

        let module = config.init::<TestBackend>(&device);

        assert_eq!(module.ratio, 2);
        assert_eq!(module.primary_ch, 16);
    }

    #[test]
    #[should_panic(expected = "out_channels must be divisible by ratio")]
    fn test_ghost_module_config_init_panic() {
        let device = Default::default();
        let config = GhostModuleConfig::new(16, 33, [3, 3]).with_ratio(2);
        config.init::<TestBackend>(&device);
    }
}
