use burn::prelude::*;
use wasm_bindgen::prelude::*;

use crate::WasmTensor;

const NORMALIZATION_TOLERANCE: f32 = 1e-5;

fn validate_layout(input: &WasmTensor, context: &str) -> Result<[usize; 4], String> {
    let shape = input.inner.dims();
    let [batch, features, h, w] = shape;
    if batch == 0 {
        return Err(format!("{context}: batch axis must be non-empty"));
    }
    if features == 0 {
        return Err(format!("{context}: feature axis 1 must be non-empty"));
    }
    if h != 1 || w != 1 {
        return Err(format!(
            "{context}: expected probability layout [B,F,1,1], got {shape:?}"
        ));
    }
    Ok(shape)
}

fn validate_nonnegative_finite(
    input: &WasmTensor,
    context: &str,
) -> Result<([usize; 4], Vec<f32>), String> {
    let shape = validate_layout(input, context)?;
    let values = input.to_array();
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite value at index {index}: {value}"
            ));
        }
        if value < 0.0 {
            return Err(format!(
                "{context}: probabilities/weights must be >= 0; index {index} has {value}"
            ));
        }
    }
    Ok((shape, values))
}

fn batch_masses(values: &[f32], batch: usize, features: usize) -> Vec<f32> {
    (0..batch)
        .map(|b| values[b * features..(b + 1) * features].iter().sum())
        .collect()
}

fn validate_normalized(
    input: &WasmTensor,
    context: &str,
) -> Result<([usize; 4], Vec<f32>), String> {
    let (shape, values) = validate_nonnegative_finite(input, context)?;
    let masses = batch_masses(&values, shape[0], shape[1]);
    for (batch_index, mass) in masses.into_iter().enumerate() {
        if !mass.is_finite() || (mass - 1.0).abs() > NORMALIZATION_TOLERANCE {
            return Err(format!(
                "{context}: batch {batch_index} must sum to 1 within tolerance {NORMALIZATION_TOLERANCE}; got {mass}"
            ));
        }
    }
    Ok((shape, values))
}

fn validate_pair(
    p: &WasmTensor,
    q: &WasmTensor,
    context: &str,
) -> Result<([usize; 4], Vec<f32>, Vec<f32>), String> {
    let (p_shape, p_values) = validate_normalized(p, &format!("{context} P"))?;
    let (q_shape, q_values) = validate_normalized(q, &format!("{context} Q"))?;
    if p_shape != q_shape {
        return Err(format!(
            "{context}: distribution shape mismatch {p_shape:?} vs {q_shape:?}"
        ));
    }
    for (index, (&pv, &qv)) in p_values.iter().zip(q_values.iter()).enumerate() {
        if pv > 0.0 && qv == 0.0 {
            return Err(format!(
                "{context}: Q has zero support at index {index} where P is positive ({pv})"
            ));
        }
    }
    Ok((p_shape, p_values, q_values))
}

fn checked_scalar_output(
    inner: Tensor<crate::WasmBackend, 4>,
    batch: usize,
    context: &str,
) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    let shape = output.inner.dims();
    if shape != [batch, 1, 1, 1] {
        return Err(format!(
            "{context}: internal scalar-output shape invariant failed, got {shape:?}"
        ));
    }
    for (index, value) in output.to_array().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite output at index {index}: {value}"
            ));
        }
    }
    Ok(output)
}

fn safe_log_input(input: Tensor<crate::WasmBackend, 4>) -> Tensor<crate::WasmBackend, 4> {
    let zero_mask = input.clone().lower_equal_elem(0.0);
    input.mask_fill(zero_mask, 1.0)
}

#[wasm_bindgen(js_name = probabilityCapabilities)]
pub fn probability_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.probability.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"input_layout\":\"[B,F,1,1]\",",
        "\"scalar_output_layout\":\"[B,1,1,1]\",",
        "\"log_base\":\"e\",",
        "\"ops\":[\"normalize\",\"entropy\",\"crossEntropy\",\"klDivergence\"],",
        "\"contracts\":{",
        "\"non_negative\":true,",
        "\"normalization_tolerance\":1e-5,",
        "\"zero_p_contribution\":\"exact_zero\",",
        "\"q_zero_where_p_positive\":\"controlled_error\",",
        "\"rng_sampling\":\"deferred\"",
        "}",
        "}"
    )
    .to_string()
}

/// Deterministic probability-distribution math over `[B,F,1,1]` tensors.
///
/// Entropy, cross-entropy and KL require normalized distributions. Zero-probability
/// P entries contribute exactly zero. Q support must cover every positive P entry.
#[wasm_bindgen]
pub struct WasmProbability;

#[wasm_bindgen]
impl WasmProbability {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmProbability {
        WasmProbability
    }

    pub fn normalize(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, values) = validate_nonnegative_finite(input, "Probability.normalize")?;
        let masses = batch_masses(&values, shape[0], shape[1]);
        for (batch_index, mass) in masses.into_iter().enumerate() {
            if !mass.is_finite() || mass <= 0.0 {
                return Err(format!(
                    "Probability.normalize: batch {batch_index} must have strictly positive finite mass; got {mass}"
                ));
            }
        }

        let mass = input.inner.clone().sum_dim(1);
        let denominator = mass.repeat_dim(1, shape[1]);
        let output = WasmTensor {
            inner: input.inner.clone().div(denominator),
        };
        validate_normalized(&output, "Probability.normalize output")?;
        Ok(output)
    }

    pub fn entropy(&self, p: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, _) = validate_normalized(p, "Probability.entropy")?;
        let safe_p = safe_log_input(p.inner.clone());
        let terms = p.inner.clone().mul(safe_p.log());
        checked_scalar_output(
            terms.sum_dim(1).neg(),
            shape[0],
            "Probability.entropy output",
        )
    }

    #[wasm_bindgen(js_name = crossEntropy)]
    pub fn cross_entropy(
        &self,
        p: &WasmTensor,
        q: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        let (shape, _, _) = validate_pair(p, q, "Probability.crossEntropy")?;
        let safe_q = safe_log_input(q.inner.clone());
        let terms = p.inner.clone().mul(safe_q.log());
        checked_scalar_output(
            terms.sum_dim(1).neg(),
            shape[0],
            "Probability.crossEntropy output",
        )
    }

    #[wasm_bindgen(js_name = klDivergence)]
    pub fn kl_divergence(
        &self,
        p: &WasmTensor,
        q: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        let (shape, _, _) = validate_pair(p, q, "Probability.klDivergence")?;
        let safe_p = safe_log_input(p.inner.clone());
        let safe_q = safe_log_input(q.inner.clone());
        let log_ratio = safe_p.log().sub(safe_q.log());
        let terms = p.inner.clone().mul(log_ratio);
        checked_scalar_output(
            terms.sum_dim(1),
            shape[0],
            "Probability.klDivergence output",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{probability_capabilities, WasmProbability};
    use crate::WasmTensor;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected} within {tolerance}"
            );
        }
    }

    #[test]
    fn normalize_handles_batched_nonnegative_weights() {
        let probability = WasmProbability::new();
        let weights = WasmTensor::new(&[1.0, 1.0, 2.0, 0.0, 3.0, 1.0], &[2, 3, 1, 1]);
        let normalized = probability.normalize(&weights).unwrap();
        assert_eq!(normalized.shape(), vec![2, 3, 1, 1]);
        assert_close(
            &normalized.to_array(),
            &[0.25, 0.25, 0.5, 0.0, 0.75, 0.25],
            1e-6,
        );
    }

    #[test]
    fn entropy_cross_entropy_and_kl_match_references() {
        let probability = WasmProbability::new();
        let p = WasmTensor::new(&[0.5, 0.5], &[1, 2, 1, 1]);
        let q = WasmTensor::new(&[0.75, 0.25], &[1, 2, 1, 1]);

        let entropy = probability.entropy(&p).unwrap();
        assert_close(&entropy.to_array(), &[std::f32::consts::LN_2], 1e-6);

        let expected_ce = -0.5 * 0.75_f32.ln() - 0.5 * 0.25_f32.ln();
        let cross_entropy = probability.cross_entropy(&p, &q).unwrap();
        assert_close(&cross_entropy.to_array(), &[expected_ce], 1e-6);

        let expected_kl = expected_ce - std::f32::consts::LN_2;
        let kl = probability.kl_divergence(&p, &q).unwrap();
        assert_close(&kl.to_array(), &[expected_kl], 1e-6);
    }

    #[test]
    fn zero_probability_convention_is_finite_and_exact() {
        let probability = WasmProbability::new();
        let p = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        let q = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        assert_eq!(probability.entropy(&p).unwrap().to_array(), vec![0.0]);
        assert_eq!(probability.cross_entropy(&p, &q).unwrap().to_array(), vec![0.0]);
        assert_eq!(probability.kl_divergence(&p, &q).unwrap().to_array(), vec![0.0]);
    }

    #[test]
    fn invalid_probability_domains_are_controlled_errors() {
        let probability = WasmProbability::new();
        let negative = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
        assert!(probability.normalize(&negative).is_err());

        let zero_mass = WasmTensor::new(&[0.0, 0.0], &[1, 2, 1, 1]);
        assert!(probability.normalize(&zero_mass).is_err());

        let unnormalized = WasmTensor::new(&[0.2, 0.2], &[1, 2, 1, 1]);
        assert!(probability.entropy(&unnormalized).is_err());

        let p = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);
        let q_missing_support = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        assert!(probability.cross_entropy(&p, &q_missing_support).is_err());
        assert!(probability.kl_divergence(&p, &q_missing_support).is_err());

        let nonfinite = WasmTensor::new(&[f32::NAN, 1.0], &[1, 2, 1, 1]);
        assert!(probability.normalize(&nonfinite).is_err());
    }

    #[test]
    fn capabilities_are_machine_discoverable() {
        let caps = probability_capabilities();
        assert!(caps.contains("burn-research.probability.v1"));
        assert!(caps.contains("\"log_base\":\"e\""));
        assert!(caps.contains("\"rng_sampling\":\"deferred\""));
    }
}
