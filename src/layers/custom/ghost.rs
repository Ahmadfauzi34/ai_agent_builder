use super::super::state_record::deterministic_record_bytes;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub use crate::facade::wasm_types::WasmGhostModule;
use crate::{WasmBackend, WasmTensor};

#[inline]
pub(crate) fn reject_invalid_config<T>(message: String) -> T {
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
        if self.kernel_size.contains(&0) {
            return Err("ghost: kernel dimensions must be > 0".into());
        }
        if self.stride.contains(&0) {
            return Err("ghost: stride dimensions must be > 0".into());
        }
        if self.ratio == 0 {
            return Err("ghost: ratio must be > 0".into());
        }
        if !self.out_channels.is_multiple_of(self.ratio) {
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

        let mut primary_cfg = Conv2dConfig::new([self.in_channels, primary_ch], self.kernel_size)
            .with_initializer(burn::module::Initializer::Zeros);
        primary_cfg.stride = self.stride;
        primary_cfg.padding = PaddingConfig2d::Explicit(self.padding[0], self.padding[1]);
        let primary = primary_cfg.init(device);

        let mut cheap_cfg = Conv2dConfig::new([primary_ch, primary_ch], [1, 1])
            .with_initializer(burn::module::Initializer::Zeros);
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

fn validate_conv2d_params(
    context: &str,
    expected_weight: &burn::module::Param<Tensor<WasmBackend, 4>>,
    expected_bias: &Option<burn::module::Param<Tensor<WasmBackend, 1>>>,
    actual_weight: &burn::module::Param<Tensor<WasmBackend, 4>>,
    actual_bias: &Option<burn::module::Param<Tensor<WasmBackend, 1>>>,
) -> Result<(), String> {
    if expected_weight.dims() != actual_weight.dims() {
        return Err(format!(
            "GhostModule loadState: {context} weight shape mismatch: expected {:?}, got {:?}",
            expected_weight.dims(),
            actual_weight.dims()
        ));
    }
    match (expected_bias, actual_bias) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual)) if expected.dims() == actual.dims() => Ok(()),
        (Some(expected), Some(actual)) => Err(format!(
            "GhostModule loadState: {context} bias shape mismatch: expected {:?}, got {:?}",
            expected.dims(),
            actual.dims()
        )),
        _ => Err(format!(
            "GhostModule loadState: {context} bias presence mismatch"
        )),
    }
}

pub(crate) fn validate_ghost_state_structure(
    current: &GhostModuleRecord<WasmBackend>,
    incoming: &GhostModuleRecord<WasmBackend>,
) -> Result<(), String> {
    validate_conv2d_params(
        "primary",
        &current.primary.weight,
        &current.primary.bias,
        &incoming.primary.weight,
        &incoming.primary.bias,
    )?;
    validate_conv2d_params(
        "cheap",
        &current.cheap.weight,
        &current.cheap.bias,
        &incoming.cheap.weight,
        &incoming.cheap.bias,
    )
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
