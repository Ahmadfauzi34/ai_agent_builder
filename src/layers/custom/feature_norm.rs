use burn::prelude::*;
use burn::tensor::linalg::{vector_normalize, Norm};
use wasm_bindgen::prelude::*;

use crate::WasmTensor;

pub(crate) const DEFAULT_EPSILON: f64 = 1e-12;

fn validate_epsilon(epsilon: f64) -> Result<(), String> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(format!(
            "FeatureNorm: epsilon must be finite and > 0, got {epsilon}"
        ));
    }
    Ok(())
}

fn validate_feature_shape(shape: [usize; 4]) -> Result<(), String> {
    let [_, features, h, w] = shape;
    if features == 0 {
        return Err("FeatureNorm: feature axis 1 must be non-empty".into());
    }
    if h != 1 || w != 1 {
        return Err(format!(
            "FeatureNorm: expected [B,F,1,1], got {shape:?}"
        ));
    }
    Ok(())
}

/// Parameter-free L2 normalization for feature vectors carried through the
/// canonical rank-4 bridge as `[B, F, 1, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct FeatureNorm {
    epsilon: f64,
}

impl FeatureNorm {
    pub fn try_new(epsilon: Option<f64>) -> Result<Self, String> {
        let epsilon = epsilon.unwrap_or(DEFAULT_EPSILON);
        validate_epsilon(epsilon)?;
        Ok(Self { epsilon })
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Result<Tensor<B, 4>, String> {
        validate_feature_shape(input.dims())?;
        Ok(vector_normalize(input, Norm::L2, 1, self.epsilon))
    }
}

#[wasm_bindgen]
pub struct WasmFeatureNorm {
    inner: FeatureNorm,
}

#[wasm_bindgen]
impl WasmFeatureNorm {
    #[wasm_bindgen(js_name = newFeatureNorm)]
    pub fn new_feature_norm(epsilon: Option<f64>) -> Result<WasmFeatureNorm, String> {
        Ok(Self {
            inner: FeatureNorm::try_new(epsilon)?,
        })
    }

    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let out = self.inner.forward(input.inner.clone())?;
        Ok(WasmTensor { inner: out })
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::WasmFeatureNorm;
    use crate::WasmTensor;

    #[test]
    fn normalizes_feature_vector_and_preserves_zero_vector() {
        let norm = WasmFeatureNorm::new_feature_norm(None).expect("FeatureNorm init");

        let input = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        let output = norm.forward(&input).expect("FeatureNorm forward");
        let values = output.to_array();
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
        assert_eq!(norm.num_params(), 0);

        let zero = WasmTensor::new(&[0.0, 0.0], &[1, 2, 1, 1]);
        let zero_out = norm.forward(&zero).expect("FeatureNorm zero forward");
        assert_eq!(zero_out.to_array(), vec![0.0, 0.0]);
    }

    #[test]
    fn rejects_non_feature_vector_layout_and_invalid_epsilon() {
        assert!(WasmFeatureNorm::new_feature_norm(Some(0.0)).is_err());
        assert!(WasmFeatureNorm::new_feature_norm(Some(f64::NAN)).is_err());

        let norm = WasmFeatureNorm::new_feature_norm(None).expect("FeatureNorm init");
        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        assert!(norm.forward(&input).is_err());
    }
}
