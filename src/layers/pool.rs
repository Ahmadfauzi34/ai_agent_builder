pub use crate::facade::wasm_types::WasmPool;
use crate::layers::shape_contract::require_singleton_axis;
use crate::WasmTensor;
use burn::nn::pool::{
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool1d, AvgPool1dConfig, AvgPool2d,
    AvgPool2dConfig, MaxPool1d, MaxPool1dConfig, MaxPool2d, MaxPool2dConfig,
};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub(crate) fn validate_pool1d_params(
    kernel_size: usize,
    stride: Option<usize>,
    context: &str,
) -> Result<(), String> {
    if kernel_size == 0 {
        return Err(format!("{context}: kernel_size must be greater than 0"));
    }
    if stride == Some(0) {
        return Err(format!("{context}: stride must be greater than 0"));
    }
    Ok(())
}

pub(crate) fn validate_pool2d_params(
    kernel_size_h: usize,
    kernel_size_w: usize,
    stride_h: Option<usize>,
    stride_w: Option<usize>,
    context: &str,
) -> Result<(), String> {
    if kernel_size_h == 0 || kernel_size_w == 0 {
        return Err(format!(
            "{context}: kernel sizes must be greater than 0, got [{kernel_size_h}, {kernel_size_w}]"
        ));
    }

    // Preserve the existing wrapper contract: custom strides are applied only when both are Some.
    if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
        if sh == 0 || sw == 0 {
            return Err(format!(
                "{context}: strides must be greater than 0, got [{sh}, {sw}]"
            ));
        }
    }
    Ok(())
}

pub(crate) fn pool_fail<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

// --- CONFIGURATION ENUM ---
#[derive(Debug)]
pub enum PoolingConfig {
    MaxPool1d(MaxPool1dConfig),
    MaxPool2d(MaxPool2dConfig),
    AvgPool1d(AvgPool1dConfig),
    AvgPool2d(AvgPool2dConfig),
    AdaptiveAvgPool2d(AdaptiveAvgPool2dConfig),
}

impl PoolingConfig {
    // Pooling layers tidak punka parameter, init() tanpa device
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
// Tidak pakai #[derive(Module)] karena pooling layers tidak Clone dan tidak punya trainable params
#[derive(Debug)]
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
            Pooling::MaxPool2d(layer) => layer.forward(input),
            Pooling::AvgPool2d(layer) => layer.forward(input),
            Pooling::AdaptiveAvgPool2d(layer) => layer.forward(input),
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

#[cfg(test)]
mod tests {
    use super::{validate_pool1d_params, validate_pool2d_params};

    #[test]
    fn pool1d_rejects_zero_kernel() {
        assert!(validate_pool1d_params(0, None, "pool").is_err());
    }

    #[test]
    fn pool1d_rejects_zero_explicit_stride() {
        assert!(validate_pool1d_params(3, Some(0), "pool").is_err());
    }

    #[test]
    fn pool1d_accepts_positive_kernel_and_default_stride() {
        assert!(validate_pool1d_params(3, None, "pool").is_ok());
    }

    #[test]
    fn pool2d_rejects_zero_kernel_axis() {
        assert!(validate_pool2d_params(3, 0, None, None, "pool").is_err());
    }

    #[test]
    fn pool2d_rejects_zero_paired_stride_axis() {
        assert!(validate_pool2d_params(3, 3, Some(1), Some(0), "pool").is_err());
    }

    #[test]
    fn pool2d_preserves_partial_stride_behavior() {
        assert!(validate_pool2d_params(3, 3, Some(0), None, "pool").is_ok());
    }

    #[test]
    fn pool2d_accepts_positive_parameters() {
        assert!(validate_pool2d_params(3, 5, Some(2), Some(1), "pool").is_ok());
    }
}
