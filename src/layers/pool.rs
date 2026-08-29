use burn::prelude::*;
use burn::nn::pool::{
    MaxPool1d, MaxPool1dConfig,
    MaxPool2d, MaxPool2dConfig,
    AvgPool1d, AvgPool1dConfig,
    AvgPool2d, AvgPool2dConfig,
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig,
};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use wasm_bindgen::prelude::*;
use crate::WasmTensor;

fn validate_pool1d_params(kernel_size: usize, stride: Option<usize>, context: &str) -> Result<(), String> {
    if kernel_size == 0 {
        return Err(format!("{context}: kernel_size must be greater than 0"));
    }
    if stride == Some(0) {
        return Err(format!("{context}: stride must be greater than 0"));
    }
    Ok(())
}

fn validate_pool2d_params(
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

fn pool_fail<T>(message: String) -> T {
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
        Self::try_new_max_pool1d(kernel_size, stride, padding).unwrap_or_else(pool_fail)
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
        Self::try_new_max_pool2d(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(pool_fail)
    }

    #[wasm_bindgen(js_name = newAvgPool1d)]
    pub fn new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        Self::try_new_avg_pool1d(kernel_size, stride, padding).unwrap_or_else(pool_fail)
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
        Self::try_new_avg_pool2d(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(pool_fail)
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

impl WasmPool {
    pub(crate) fn try_new_max_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool1d_params(kernel_size, stride, "MaxPool1d")?;
        let mut config = MaxPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(PaddingConfig1d::Explicit(p));
        }
        Ok(WasmPool {
            inner: PoolingConfig::MaxPool1d(config).init(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_max_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool2d_params(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            "MaxPool2d",
        )?;
        let mut config = MaxPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_strides([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding(PaddingConfig2d::Explicit(ph, pw));
        }
        Ok(WasmPool {
            inner: PoolingConfig::MaxPool2d(config).init(),
        })
    }

    pub(crate) fn try_new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool1d_params(kernel_size, stride, "AvgPool1d")?;
        let mut config = AvgPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(PaddingConfig1d::Explicit(p));
        }
        Ok(WasmPool {
            inner: PoolingConfig::AvgPool1d(config).init(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_avg_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool2d_params(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            "AvgPool2d",
        )?;
        let mut config = AvgPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_strides([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding(PaddingConfig2d::Explicit(ph, pw));
        }
        Ok(WasmPool {
            inner: PoolingConfig::AvgPool2d(config).init(),
        })
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
        // Existing wrapper ignores custom stride unless both components are present.
        assert!(validate_pool2d_params(3, 3, Some(0), None, "pool").is_ok());
    }

    #[test]
    fn pool2d_accepts_positive_parameters() {
        assert!(validate_pool2d_params(3, 5, Some(2), Some(1), "pool").is_ok());
    }
}
