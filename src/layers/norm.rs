use super::state_record::deterministic_record_bytes;
pub use crate::facade::wasm_types::WasmNorm;
use crate::{WasmBackend, WasmTensor};
use burn::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub(crate) fn validate_group_config(num_groups: usize, num_channels: usize) -> Result<(), String> {
    if num_groups == 0 {
        return Err("GroupNorm: num_groups must be greater than 0".to_string());
    }
    if num_channels % num_groups != 0 {
        return Err(format!(
            "GroupNorm: num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        ));
    }
    Ok(())
}

pub(crate) fn validate_rms_epsilon(epsilon: f64) -> Result<(), String> {
    if !(epsilon > 0.0) {
        return Err(format!("RMSNorm: epsilon must be positive, got {epsilon}"));
    }
    Ok(())
}

pub(crate) fn validate_norm_axis(
    shape: [usize; 4],
    axis: usize,
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if shape[axis] != expected {
        return Err(format!(
            "{context}: expected axis {axis} size {expected}, got {} for shape {:?}",
            shape[axis], shape
        ));
    }
    Ok(())
}

pub(crate) fn norm_fail<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

// --- CONFIG ENUM ---
#[derive(Config, Debug)]
pub enum NormalizationConfig {
    Batch(BatchNormConfig),
    Group(GroupNormConfig),
    Instance(InstanceNormConfig),
    Layer(LayerNormConfig),
    Rms(RmsNormConfig),
}

impl NormalizationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Normalization<B> {
        match self {
            NormalizationConfig::Batch(config) => Normalization::Batch(config.init(device)),
            NormalizationConfig::Group(config) => Normalization::Group(config.init(device)),
            NormalizationConfig::Instance(config) => Normalization::Instance(config.init(device)),
            NormalizationConfig::Layer(config) => Normalization::Layer(config.init(device)),
            NormalizationConfig::Rms(config) => Normalization::Rms(config.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Normalization<B: Backend> {
    Batch(BatchNorm<B>),
    Group(GroupNorm<B>),
    Instance(InstanceNorm<B>),
    Layer(LayerNorm<B>),
    Rms(RmsNorm<B>),
}

impl<B: Backend> Normalization<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Normalization::Batch(norm) => norm.forward(input),
            Normalization::Group(norm) => norm.forward(input),
            Normalization::Instance(norm) => norm.forward(input),
            Normalization::Layer(norm) => norm.forward(input),
            Normalization::Rms(norm) => norm.forward(input),
        }
    }
}

pub(crate) fn validate_norm_state_structure(
    current: &NormalizationRecord<WasmBackend>,
    incoming: &NormalizationRecord<WasmBackend>,
) -> Result<(), String> {
    if std::mem::discriminant(current) != std::mem::discriminant(incoming) {
        return Err("Norm loadState: normalization variant mismatch".to_string());
    }

    let param_shapes = |record: &NormalizationRecord<WasmBackend>| {
        let (gamma, beta) = norm_trainable_refs(record);
        (
            gamma.map(|param| param.dims().to_vec()),
            beta.map(|param| param.dims().to_vec()),
        )
    };
    let expected = param_shapes(current);
    let actual = param_shapes(incoming);
    if expected != actual {
        return Err(format!(
            "Norm loadState: parameter structure mismatch: expected {:?}, got {:?}",
            expected, actual
        ));
    }
    Ok(())
}

// ============================================================
// FLOAT-BRIDGE + WEIGHT LAYOUT (M1b) — norm.
// KONTRAK TRAINABLE-ONLY: hanya gamma (+ beta kalau ada) yang diekspos.
// running_mean / running_var (BatchNorm) TIDAK disentuh -> mustahil di-perturb
// ES (aman-by-construction: kita hanya menyebut field trainable secara eksplisit).
// RmsNorm = gamma saja (tanpa beta).
// ============================================================

pub(crate) fn norm_param_len(p: &burn::module::Param<Tensor<WasmBackend, 1>>) -> usize {
    p.dims().iter().product::<usize>()
}

pub(crate) fn push_norm_param(
    p: &burn::module::Param<Tensor<WasmBackend, 1>>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let t = <Tensor<WasmBackend, 1> as Clone>::clone(p).into_data();
    out.extend(
        t.as_slice::<f32>()
            .map_err(|_| "getWeightsFlat: norm param not f32".to_string())?,
    );
    Ok(())
}

pub(crate) fn set_norm_param(
    p: &mut burn::module::Param<Tensor<WasmBackend, 1>>,
    data: &[f32],
) -> Result<(), String> {
    let n = p.dims().iter().product::<usize>();
    if data.len() != n {
        return Err(format!(
            "setWeightsFlat: norm segment expected {} floats, got {}",
            n,
            data.len()
        ));
    }
    let device: <WasmBackend as Backend>::Device = Default::default();
    *p = burn::module::Param::from_data(burn::tensor::TensorData::new(data.to_vec(), [n]), &device);
    Ok(())
}

// Seragamkan ekstraksi trainable: gamma selalu ada, beta kecuali Rms.
// TITIK API: nama variant record + field (gamma/beta) mengikuti Burn 0.20.
pub(crate) fn norm_trainable_refs(
    rec: &NormalizationRecord<WasmBackend>,
) -> (
    Option<&burn::module::Param<Tensor<WasmBackend, 1>>>,
    Option<&burn::module::Param<Tensor<WasmBackend, 1>>>,
) {
    match rec {
        NormalizationRecord::Batch(r) => (Some(&r.gamma), Some(&r.beta)),
        NormalizationRecord::Group(r) => (r.gamma.as_ref(), r.beta.as_ref()),
        NormalizationRecord::Instance(r) => (r.gamma.as_ref(), r.beta.as_ref()),
        NormalizationRecord::Layer(r) => (Some(&r.gamma), r.beta.as_ref()),
        NormalizationRecord::Rms(r) => (Some(&r.gamma), None),
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_group_config, validate_norm_axis, validate_rms_epsilon};

    #[test]
    fn group_config_rejects_zero_groups_before_modulo() {
        assert!(validate_group_config(0, 8).is_err());
    }

    #[test]
    fn group_config_rejects_non_divisible_channels() {
        assert!(validate_group_config(3, 8).is_err());
    }

    #[test]
    fn group_config_accepts_divisible_channels() {
        assert!(validate_group_config(4, 8).is_ok());
    }

    #[test]
    fn rms_epsilon_rejects_non_positive_and_nan() {
        assert!(validate_rms_epsilon(0.0).is_err());
        assert!(validate_rms_epsilon(-1e-5).is_err());
        assert!(validate_rms_epsilon(f64::NAN).is_err());
    }

    #[test]
    fn rms_epsilon_accepts_positive_value() {
        assert!(validate_rms_epsilon(1e-5).is_ok());
    }

    #[test]
    fn norm_axis_rejects_feature_mismatch() {
        let err = validate_norm_axis([2, 7, 4, 4], 1, 8, "GroupNorm forward").unwrap_err();
        assert!(err.contains("size 8"));
    }

    #[test]
    fn norm_axis_accepts_matching_feature_count() {
        assert!(validate_norm_axis([2, 8, 4, 4], 1, 8, "GroupNorm forward").is_ok());
        assert!(validate_norm_axis([2, 3, 4, 8], 3, 8, "RMSNorm forward").is_ok());
    }
}
