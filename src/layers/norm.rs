use burn::prelude::*;
use burn::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

fn validate_group_config(num_groups: usize, num_channels: usize) -> Result<(), String> {
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

fn validate_rms_epsilon(epsilon: f64) -> Result<(), String> {
    if !(epsilon > 0.0) {
        return Err(format!("RMSNorm: epsilon must be positive, got {epsilon}"));
    }
    Ok(())
}

fn validate_norm_axis(
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

fn norm_fail<T>(message: String) -> T {
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

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmNorm {
    inner: Normalization<WasmBackend>,
}

#[wasm_bindgen]
impl WasmNorm {
    #[wasm_bindgen]
    pub fn new_rms_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        Self::try_new_rms_norm(size, epsilon).unwrap_or_else(norm_fail)
    }

    #[wasm_bindgen]
    pub fn new_layer_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Layer(LayerNormConfig::new(size).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_batch_norm(num_features: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Batch(BatchNormConfig::new(num_features).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_group_norm(num_groups: usize, num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        Self::try_new_group_norm(num_groups, num_channels, epsilon).unwrap_or_else(norm_fail)
    }

    #[wasm_bindgen]
    pub fn new_instance_norm(num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Instance(InstanceNormConfig::new(num_channels).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        self.try_forward(input).unwrap_or_else(norm_fail)
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &device)
            .map_err(|e| e.to_string())?;
            
        // PERBAIKAN: Clone dulu
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

impl WasmNorm {
    pub(crate) fn try_new_rms_norm(size: usize, epsilon: Option<f64>) -> Result<Self, String> {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        validate_rms_epsilon(eps)?;
        let config = NormalizationConfig::Rms(RmsNormConfig::new(size).with_epsilon(eps));
        Ok(WasmNorm { inner: config.init(&device) })
    }

    pub(crate) fn try_new_group_norm(
        num_groups: usize,
        num_channels: usize,
        epsilon: Option<f64>,
    ) -> Result<Self, String> {
        validate_group_config(num_groups, num_channels)?;
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Group(
            GroupNormConfig::new(num_groups, num_channels).with_epsilon(eps),
        );
        Ok(WasmNorm { inner: config.init(&device) })
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let shape = input.inner.dims();
        match &self.inner {
            Normalization::Batch(norm) => {
                validate_norm_axis(shape, 1, norm.gamma.dims()[0], "BatchNorm forward")?;
            }
            Normalization::Group(norm) => {
                validate_norm_axis(shape, 1, norm.num_channels, "GroupNorm forward")?;
            }
            Normalization::Instance(norm) => {
                validate_norm_axis(shape, 1, norm.num_channels, "InstanceNorm forward")?;
            }
            Normalization::Layer(norm) => {
                validate_norm_axis(shape, 3, norm.gamma.dims()[0], "LayerNorm forward")?;
            }
            Normalization::Rms(norm) => {
                validate_norm_axis(shape, 3, norm.gamma.dims()[0], "RMSNorm forward")?;
            }
        }

        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

// ============================================================
// FLOAT-BRIDGE + WEIGHT LAYOUT (M1b) — norm.
// KONTRAK TRAINABLE-ONLY: hanya gamma (+ beta kalau ada) yang diekspos.
// running_mean / running_var (BatchNorm) TIDAK disentuh -> mustahil di-perturb
// ES (aman-by-construction: kita hanya menyebut field trainable secara eksplisit).
// RmsNorm = gamma saja (tanpa beta).
// ============================================================

fn norm_param_len(p: &burn::module::Param<Tensor<WasmBackend, 1>>) -> usize {
    p.dims().iter().product::<usize>()
}

fn push_norm_param(
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

fn set_norm_param(
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
    *p = burn::module::Param::from_data(
        burn::tensor::TensorData::new(data.to_vec(), [n]),
        &device,
    );
    Ok(())
}

// Seragamkan ekstraksi trainable: gamma selalu ada, beta kecuali Rms.
// TITIK API: nama variant record + field (gamma/beta) mengikuti Burn 0.20.
fn norm_trainable_refs(
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

#[wasm_bindgen]
impl WasmNorm {
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        let (gamma, beta) = norm_trainable_refs(&rec);
        let mut out = Vec::new();
        if let Some(g) = gamma {
            push_norm_param(g, &mut out)?;
        }
        if let Some(b) = beta {
            push_norm_param(b, &mut out)?;
        }
        Ok(out)
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.clone().into_record();
        // panjang trainable: pinjam immut, lalu lepas (blok tersendiri)
        let (gl, bl) = {
            let (g, b) = norm_trainable_refs(&rec);
            (
                g.map(norm_param_len).unwrap_or(0),
                b.map(norm_param_len).unwrap_or(0)
            )
        };
        let total = gl + bl;
        if data.len() != total {
            return Err(format!(
                "setWeightsFlat: norm expected {} floats ({}{}), got {}",
                total,
                if gl > 0 { "gamma" } else { "" },
                if bl > 0 { "+beta" } else { "" },
                data.len()
            ));
        }
        // tulis per-field sekuensial (tanpa pinjam-mut bersamaan)
        match &mut rec {
            NormalizationRecord::Batch(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
                set_norm_param(&mut r.beta, &data[gl..])?;
            }
            NormalizationRecord::Group(r) => {
                if let Some(ref mut g) = r.gamma {
                    set_norm_param(g, &data[..gl])?;
                }
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Instance(r) => {
                if let Some(ref mut g) = r.gamma {
                    set_norm_param(g, &data[..gl])?;
                }
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Layer(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Rms(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
            }
        }
        self.inner = self.inner.clone().load_record(rec);
        Ok(())
    }
}

impl WasmNorm {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        let (gamma, beta) = norm_trainable_refs(&rec);
        let mut segs = Vec::new();
        if let Some(g) = gamma {
            segs.push(("gamma", norm_param_len(g)));
        }
        if let Some(b) = beta {
            segs.push(("beta", norm_param_len(b)));
        }
        segs
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
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
