use super::state_record::deterministic_record_bytes;
pub use crate::facade::wasm_types::WasmActivation;
use crate::{WasmBackend, WasmTensor};
use burn::nn::activation::HardSwish;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu,
    Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub(crate) fn validate_rank4_axis(dim: usize, context: &str) -> Result<(), String> {
    if dim >= 4 {
        return Err(format!("{context}: dim must be in 0..4, got {dim}"));
    }
    Ok(())
}

pub(crate) fn validate_glu_shape(shape: [usize; 4], dim: usize) -> Result<(), String> {
    validate_rank4_axis(dim, "GLU forward")?;
    let n = shape[dim];
    if !n.is_multiple_of(2) {
        return Err(format!(
            "GLU forward: axis {dim} size must be divisible by 2, got {n} for shape {:?}",
            shape
        ));
    }
    Ok(())
}

pub(crate) fn validate_prelu_shape(shape: [usize; 4], num_parameters: usize) -> Result<(), String> {
    // Rank-4 tensors: channels live on axis 1. Burn's prelu accepts
    // alpha of length 1 (shared) or C (per-channel); anything else
    // traps inside Burn (unreachable), so reject it here with a
    // clean Err — same pattern as validate_glu_shape.
    let c = shape[1];
    if num_parameters != 1 && num_parameters != c {
        return Err(format!(
            "PRelu forward: num_parameters must be 1 or C (channels), got {num_parameters} for input shape {shape:?} (C={c})"
        ));
    }
    Ok(())
}

pub(crate) fn activation_forward_fail<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

// --- HELPER STRUCTS (SOLUSI ERROR DERIVE) ---
#[derive(Module, Debug, Clone)]
pub struct StrictSoftmax {
    pub dim: usize,
}
impl StrictSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictLogSoftmax {
    pub dim: usize,
}
impl StrictLogSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::log_softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictGlu {
    pub dim: usize,
}
impl StrictGlu {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::glu(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictMish;
impl StrictMish {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::mish(input)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictSqrt;
impl StrictSqrt {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.sqrt()
    }
}

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum ActivationConfig {
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    HardSwish,
    LeakyRelu(LeakyReluConfig),
    PRelu(PReluConfig),
    SwiGlu(SwiGluConfig),
    HardSigmoid(HardSigmoidConfig),
    Softplus(SoftplusConfig),
    Mish,
    Softmax { dim: usize },
    LogSoftmax { dim: usize },
    Glu { dim: usize },
    Sqrt,
}

impl ActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Gelu => Activation::Gelu(Gelu::new()),
            ActivationConfig::Relu => Activation::Relu(Relu::new()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            ActivationConfig::Tanh => Activation::Tanh(Tanh::new()),
            ActivationConfig::HardSwish => Activation::HardSwish(HardSwish::new()),
            ActivationConfig::LeakyRelu(c) => Activation::LeakyRelu(c.init()),
            ActivationConfig::PRelu(c) => Activation::PRelu(c.init(device)),
            ActivationConfig::SwiGlu(c) => Activation::SwiGlu(c.init(device)),
            ActivationConfig::HardSigmoid(c) => Activation::HardSigmoid(c.init()),
            ActivationConfig::Softplus(c) => Activation::Softplus(c.init()),
            ActivationConfig::Mish => Activation::Mish(StrictMish),
            ActivationConfig::Softmax { dim } => Activation::Softmax(StrictSoftmax { dim: *dim }),
            ActivationConfig::LogSoftmax { dim } => {
                Activation::LogSoftmax(StrictLogSoftmax { dim: *dim })
            }
            ActivationConfig::Glu { dim } => Activation::Glu(StrictGlu { dim: *dim }),
            ActivationConfig::Sqrt => Activation::Sqrt(StrictSqrt),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Activation<B: Backend> {
    Gelu(Gelu),
    Relu(Relu),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    HardSwish(HardSwish),
    LeakyRelu(LeakyRelu),
    PRelu(PRelu<B>),
    SwiGlu(SwiGlu<B>),
    HardSigmoid(HardSigmoid),
    Softplus(Softplus),
    Mish(StrictMish),
    Softmax(StrictSoftmax),
    LogSoftmax(StrictLogSoftmax),
    Glu(StrictGlu),
    Sqrt(StrictSqrt),
}

impl<B: Backend> Activation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activation::Gelu(m) => m.forward(input),
            Activation::Relu(m) => m.forward(input),
            Activation::Sigmoid(m) => m.forward(input),
            Activation::Tanh(m) => m.forward(input),
            Activation::HardSwish(m) => m.forward(input),
            Activation::LeakyRelu(m) => m.forward(input),
            Activation::PRelu(m) => m.forward(input),
            Activation::SwiGlu(m) => m.forward(input),
            Activation::HardSigmoid(m) => m.forward(input),
            Activation::Softplus(m) => m.forward(input),
            Activation::Mish(m) => m.forward(input),
            Activation::Softmax(m) => m.forward(input),
            Activation::LogSoftmax(m) => m.forward(input),
            Activation::Glu(m) => m.forward(input),
            Activation::Sqrt(m) => m.forward(input),
        }
    }
}

fn validate_activation_linear_state(
    context: &str,
    expected: &burn::nn::LinearRecord<WasmBackend>,
    actual: &burn::nn::LinearRecord<WasmBackend>,
) -> Result<(), String> {
    if expected.weight.dims() != actual.weight.dims() {
        return Err(format!(
            "Activation loadState: {context} weight shape mismatch: expected {:?}, got {:?}",
            expected.weight.dims(),
            actual.weight.dims()
        ));
    }
    match (&expected.bias, &actual.bias) {
        (None, None) => Ok(()),
        (Some(expected), Some(actual)) if expected.dims() == actual.dims() => Ok(()),
        (Some(expected), Some(actual)) => Err(format!(
            "Activation loadState: {context} bias shape mismatch: expected {:?}, got {:?}",
            expected.dims(),
            actual.dims()
        )),
        _ => Err(format!(
            "Activation loadState: {context} bias presence mismatch"
        )),
    }
}

pub(crate) fn validate_activation_state_structure(
    current: &ActivationRecord<WasmBackend>,
    incoming: &ActivationRecord<WasmBackend>,
) -> Result<(), String> {
    if std::mem::discriminant(current) != std::mem::discriminant(incoming) {
        return Err("Activation loadState: activation variant mismatch".to_string());
    }

    match (current, incoming) {
        (ActivationRecord::PRelu(expected), ActivationRecord::PRelu(actual)) => {
            if expected.alpha.dims() != actual.alpha.dims() {
                return Err(format!(
                    "Activation loadState: PRelu alpha shape mismatch: expected {:?}, got {:?}",
                    expected.alpha.dims(),
                    actual.alpha.dims()
                ));
            }
            Ok(())
        }
        (ActivationRecord::SwiGlu(expected), ActivationRecord::SwiGlu(actual)) => {
            validate_activation_linear_state(
                "SwiGlu.linear_inner",
                &expected.linear_inner,
                &actual.linear_inner,
            )?;
            validate_activation_linear_state(
                "SwiGlu.linear_outer",
                &expected.linear_outer,
                &actual.linear_outer,
            )
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::WasmActivation;
    use super::{validate_glu_shape, validate_prelu_shape, validate_rank4_axis};
    use crate::WasmTensor;

    #[test]
    fn rank4_axis_accepts_valid_dimensions() {
        for dim in 0..4 {
            assert!(validate_rank4_axis(dim, "test").is_ok());
        }
    }

    #[test]
    fn rank4_axis_rejects_out_of_range_dimension() {
        let err = validate_rank4_axis(4, "Softmax forward").unwrap_err();
        assert!(err.contains("0..4"));
    }

    #[test]
    fn glu_accepts_even_axis_size() {
        assert!(validate_glu_shape([2, 8, 3, 1], 1).is_ok());
    }

    #[test]
    fn glu_rejects_odd_axis_size() {
        let err = validate_glu_shape([2, 7, 3, 1], 1).unwrap_err();
        assert!(err.contains("divisible by 2"));
    }

    #[test]
    fn glu_rejects_out_of_range_axis() {
        assert!(validate_glu_shape([2, 8, 3, 1], 4).is_err());
    }

    #[test]
    fn prelu_accepts_one_or_channel_count() {
        assert!(validate_prelu_shape([2, 4, 3, 1], 1).is_ok());
        assert!(validate_prelu_shape([2, 4, 3, 1], 4).is_ok());
    }

    #[test]
    fn prelu_rejects_other_parameter_counts() {
        let err = validate_prelu_shape([2, 4, 3, 1], 2).unwrap_err();
        assert!(err.contains("num_parameters"), "prelu err: {err}");
    }

    #[test]
    fn prelu_try_forward_rejects_mismatch_without_trap() {
        // Previously trapped (unreachable) inside Burn; the
        // try_forward arm must turn it into a clean Err.
        let act = WasmActivation::new_prelu(Some(2), Some(0.25));
        let input = WasmTensor::new(&[1.0, -0.5, 3.0, -1.0], &[1, 4, 1, 1]);
        let err = act
            .try_forward(&input)
            .err()
            .expect("prelu(2)/C=4 must err, not trap");
        assert!(err.contains("num_parameters"), "prelu err: {err}");
    }

    #[test]
    fn prelu_try_forward_accepts_one_and_channel_count() {
        for n in [1, 4] {
            let act = WasmActivation::new_prelu(Some(n), Some(0.25));
            let input = WasmTensor::new(&[1.0, -0.5, 3.0, -1.0], &[1, 4, 1, 1]);
            let out = act.try_forward(&input).unwrap();
            // Positive passthrough, negative * alpha (0.25).
            assert_eq!(out.to_array(), vec![1.0, -0.125, 3.0, -0.25]);
        }
    }
}
