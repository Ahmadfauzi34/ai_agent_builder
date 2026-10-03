use super::state_record::deterministic_record_bytes;
pub use crate::facade::wasm_types::WasmConv;
use crate::layers::shape_contract::require_singleton_axis;
use crate::{WasmBackend, WasmTensor};
use burn::nn::conv::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub(crate) fn validate_conv1d_stride(stride: Option<usize>, context: &str) -> Result<(), String> {
    if stride == Some(0) {
        return Err(format!("{context}: stride must be greater than 0"));
    }
    Ok(())
}

pub(crate) fn validate_conv2d_stride(
    stride_h: Option<usize>,
    stride_w: Option<usize>,
    context: &str,
) -> Result<(), String> {
    // Preserve the existing wrapper contract: a custom 2D stride is applied only when both
    // components are present. A partial pair is ignored and leaves Burn's [1, 1] default intact.
    if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
        if sh == 0 || sw == 0 {
            return Err(format!(
                "{context}: strides must be greater than 0, got [{sh}, {sw}]"
            ));
        }
    }
    Ok(())
}

pub(crate) fn conv_fail<T>(message: String) -> T {
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

fn validate_conv_params<const D: usize>(
    expected_weight: &burn::module::Param<Tensor<WasmBackend, D>>,
    expected_bias: &Option<burn::module::Param<Tensor<WasmBackend, 1>>>,
    actual_weight: &burn::module::Param<Tensor<WasmBackend, D>>,
    actual_bias: &Option<burn::module::Param<Tensor<WasmBackend, 1>>>,
) -> Result<(), String> {
    if expected_weight.dims() != actual_weight.dims() {
        return Err(format!(
            "Conv loadState: weight shape mismatch: expected {:?}, got {:?}",
            expected_weight.dims(),
            actual_weight.dims()
        ));
    }
    match (expected_bias, actual_bias) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual)) if expected.dims() == actual.dims() => Ok(()),
        (Some(expected), Some(actual)) => Err(format!(
            "Conv loadState: bias shape mismatch: expected {:?}, got {:?}",
            expected.dims(),
            actual.dims()
        )),
        _ => Err("Conv loadState: bias presence mismatch".to_string()),
    }
}

pub(crate) fn validate_conv_state_structure(
    current: &ConvolutionRecord<WasmBackend>,
    incoming: &ConvolutionRecord<WasmBackend>,
) -> Result<(), String> {
    match (current, incoming) {
        (ConvolutionRecord::Conv1d(expected), ConvolutionRecord::Conv1d(actual)) => {
            validate_conv_params::<3>(
                &expected.weight,
                &expected.bias,
                &actual.weight,
                &actual.bias,
            )
        }
        (ConvolutionRecord::Conv2d(expected), ConvolutionRecord::Conv2d(actual)) => {
            validate_conv_params::<4>(
                &expected.weight,
                &expected.bias,
                &actual.weight,
                &actual.bias,
            )
        }
        (
            ConvolutionRecord::ConvTranspose2d(expected),
            ConvolutionRecord::ConvTranspose2d(actual),
        ) => validate_conv_params::<4>(
            &expected.weight,
            &expected.bias,
            &actual.weight,
            &actual.bias,
        ),
        _ => Err("Conv loadState: convolution variant mismatch".to_string()),
    }
}

// ============================================================
// FLOAT-BRIDGE (M1) — conv. Record per variant = { weight: Param<TD>, bias: Option<Param<T1>> }.
// D = 3 (conv1d) atau 4 (conv2d / transpose2d). Helper generic supaya rank statis & aman.
// ============================================================
pub(crate) fn push_param<B: Backend, const D: usize>(
    p: &burn::module::Param<Tensor<B, D>>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let t = <Tensor<B, D> as Clone>::clone(p).into_data();
    out.extend(
        t.as_slice::<f32>()
            .map_err(|_| "getWeightsFlat: conv param not f32".to_string())?,
    );
    Ok(())
}

pub(crate) fn set_conv_param<B: Backend, const D: usize>(
    weight: &mut burn::module::Param<Tensor<B, D>>,
    bias: &mut Option<burn::module::Param<Tensor<B, 1>>>,
    data: &[f32],
) -> Result<(), String> {
    let wd = weight.dims(); // [usize; D]
    let wlen = wd.iter().product::<usize>();
    let has_bias = bias.is_some();
    let blen = if has_bias {
        bias.as_ref().unwrap().dims().iter().product::<usize>()
    } else {
        0
    };
    if data.len() != wlen + blen {
        return Err(format!(
            "setWeightsFlat: conv expected {} floats (weight{}), got {}",
            wlen + blen,
            if has_bias { "+bias" } else { "" },
            data.len()
        ));
    }
    let device: <B as Backend>::Device = Default::default();
    *weight = burn::module::Param::from_data(
        burn::tensor::TensorData::new(data[..wlen].to_vec(), wd),
        &device,
    );
    if has_bias {
        *bias = Some(burn::module::Param::from_data(
            burn::tensor::TensorData::new(data[wlen..].to_vec(), [blen]),
            &device,
        ));
    }
    Ok(())
}

// ============================================================
// WEIGHT LAYOUT (M2) — conv. Mirror urutan getWeightsFlat per variant.
// ============================================================
pub(crate) fn push_conv_segs<B: Backend, const D: usize>(
    weight: &burn::module::Param<Tensor<B, D>>,
    bias: &Option<burn::module::Param<Tensor<B, 1>>>,
    segs: &mut Vec<(&'static str, usize)>,
) {
    segs.push(("weight", weight.dims().iter().product::<usize>()));
    if let Some(b) = bias {
        segs.push(("bias", b.dims().iter().product::<usize>()));
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_conv1d_stride, validate_conv2d_stride};

    #[test]
    fn conv1d_rejects_zero_explicit_stride() {
        assert!(validate_conv1d_stride(Some(0), "Conv1d").is_err());
    }

    #[test]
    fn conv1d_accepts_default_and_positive_stride() {
        assert!(validate_conv1d_stride(None, "Conv1d").is_ok());
        assert!(validate_conv1d_stride(Some(2), "Conv1d").is_ok());
    }

    #[test]
    fn conv2d_rejects_zero_paired_stride_axis() {
        assert!(validate_conv2d_stride(Some(0), Some(1), "Conv2d").is_err());
        assert!(validate_conv2d_stride(Some(1), Some(0), "Conv2d").is_err());
    }

    #[test]
    fn conv2d_preserves_partial_stride_behavior() {
        assert!(validate_conv2d_stride(Some(0), None, "Conv2d").is_ok());
        assert!(validate_conv2d_stride(None, Some(0), "Conv2d").is_ok());
    }

    #[test]
    fn conv2d_accepts_default_and_positive_stride() {
        assert!(validate_conv2d_stride(None, None, "Conv2d").is_ok());
        assert!(validate_conv2d_stride(Some(2), Some(3), "Conv2d").is_ok());
    }
}
