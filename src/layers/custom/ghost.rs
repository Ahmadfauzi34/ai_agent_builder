use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
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
    pub fn validate(&self) -> Result<(), String> {
        if self.in_channels == 0 {
            return Err("ghost: in_channels must be > 0".into());
        }
        if self.out_channels == 0 {
            return Err("ghost: out_channels must be > 0".into());
        }
        if self.kernel_size.iter().any(|&v| v == 0) {
            return Err("ghost: kernel dimensions must be > 0".into());
        }
        if self.stride.iter().any(|&v| v == 0) {
            return Err("ghost: stride dimensions must be > 0".into());
        }
        if self.ratio == 0 {
            return Err("ghost: ratio must be > 0".into());
        }
        if self.out_channels % self.ratio != 0 {
            return Err(format!(
                "ghost: out_channels ({}) must be divisible by ratio ({})",
                self.out_channels, self.ratio
            ));
        }
        Ok(())
    }

    pub fn try_init<B: Backend>(&self, device: &B::Device) -> Result<GhostModule<B>, String> {
        self.validate()?;

        let primary_ch = self.out_channels / self.ratio;

        let mut primary_cfg =
            Conv2dConfig::new([self.in_channels, primary_ch], self.kernel_size);
        primary_cfg.stride = self.stride;
        primary_cfg.padding = PaddingConfig2d::Explicit(self.padding[0], self.padding[1]);
        let primary = primary_cfg.init(device);

        let mut cheap_cfg = Conv2dConfig::new([primary_ch, primary_ch], [1, 1]);
        cheap_cfg.groups = primary_ch;
        cheap_cfg.bias = false;
        let cheap = cheap_cfg.init(device);

        Ok(GhostModule {
            primary,
            cheap,
            ratio: self.ratio,
            primary_ch,
        })
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> GhostModule<B> {
        self.try_init(device).unwrap_or_else(|e| panic!("{e}"))
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
        let intrinsic = self.primary.forward(input);
        let ghost = self.cheap.forward(intrinsic.clone());
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
        let mut config =
            GhostModuleConfig::new(in_channels, out_channels, [kernel_size_h, kernel_size_w]);
        if let Some(r) = ratio {
            config.ratio = r;
        }
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];
        }
        let inner = config.try_init(&device).unwrap_or_else(reject_invalid_config);
        WasmGhostModule { inner }
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
    use super::GhostModuleConfig;

    #[test]
    fn validation_rejects_zero_ratio_before_modulo() {
        let mut cfg = GhostModuleConfig::new(4, 8, [3, 3]);
        cfg.ratio = 0;
        assert!(cfg.validate().unwrap_err().contains("ratio must be > 0"));
    }

    #[test]
    fn validation_rejects_non_divisible_channels() {
        let mut cfg = GhostModuleConfig::new(4, 7, [3, 3]);
        cfg.ratio = 2;
        assert!(cfg.validate().unwrap_err().contains("must be divisible"));
    }

    #[test]
    fn validation_rejects_zero_kernel_or_stride() {
        let mut cfg = GhostModuleConfig::new(4, 8, [0, 3]);
        assert!(cfg.validate().is_err());
        cfg.kernel_size = [3, 3];
        cfg.stride = [1, 0];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_accepts_existing_valid_contract() {
        let cfg = GhostModuleConfig::new(4, 8, [3, 3]);
        assert!(cfg.validate().is_ok());
    }
}
