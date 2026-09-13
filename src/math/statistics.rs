use burn::prelude::*;
use wasm_bindgen::prelude::*;

use crate::WasmTensor;

fn validate_feature_tensor(input: &WasmTensor, context: &str) -> Result<(), String> {
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
            "{context}: expected feature-vector layout [B,F,1,1], got {shape:?}"
        ));
    }
    for (index, value) in input.to_array().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite value at index {index}: {value}"
            ));
        }
    }
    Ok(())
}

fn checked_output(
    inner: Tensor<crate::WasmBackend, 4>,
    context: &str,
) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    let shape = output.inner.dims();
    if shape[1] != 1 || shape[2] != 1 || shape[3] != 1 {
        return Err(format!(
            "{context}: internal reduction shape invariant failed, got {shape:?}"
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

#[wasm_bindgen(js_name = statisticsCapabilities)]
pub fn statistics_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.statistics.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"input_layout\":\"[B,F,1,1]\",",
        "\"output_layout\":\"[B,1,1,1]\",",
        "\"reduction_axis\":1,",
        "\"reducers\":[\"sum\",\"mean\",\"variancePopulation\",\"stdPopulation\",\"min\",\"max\"],",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"non_empty_batch\":true,",
        "\"non_empty_features\":true,",
        "\"variance_semantics\":\"population_no_bessel_correction\",",
        "\"probability_ops\":\"deferred\"",
        "}",
        "}"
    )
    .to_string()
}

/// Stateless descriptive-statistics surface for canonical feature tensors `[B,F,1,1]`.
///
/// All reducers operate across feature axis 1 and retain the canonical rank-4 bridge,
/// producing `[B,1,1,1]`. Variance and standard deviation use population semantics.
#[wasm_bindgen]
pub struct WasmStatistics;

#[wasm_bindgen]
impl WasmStatistics {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStatistics {
        WasmStatistics
    }

    pub fn sum(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.sum")?;
        checked_output(input.inner.clone().sum_dim(1), "Statistics.sum output")
    }

    pub fn mean(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.mean")?;
        checked_output(input.inner.clone().mean_dim(1), "Statistics.mean output")
    }

    #[wasm_bindgen(js_name = variancePopulation)]
    pub fn variance_population(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.variancePopulation")?;
        checked_output(
            input.inner.clone().var_bias(1),
            "Statistics.variancePopulation output",
        )
    }

    #[wasm_bindgen(js_name = stdPopulation)]
    pub fn std_population(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.stdPopulation")?;
        checked_output(
            input.inner.clone().var_bias(1).sqrt(),
            "Statistics.stdPopulation output",
        )
    }

    pub fn min(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.min")?;
        checked_output(input.inner.clone().min_dim(1), "Statistics.min output")
    }

    pub fn max(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.max")?;
        checked_output(input.inner.clone().max_dim(1), "Statistics.max output")
    }
}

#[cfg(test)]
mod tests {
    use super::{statistics_capabilities, WasmStatistics};
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
    fn reducers_compute_expected_batched_values() {
        let stats = WasmStatistics::new();
        let input = WasmTensor::new(
            &[2.0, 4.0, 6.0, 8.0, 1.0, 1.0, 3.0, 3.0],
            &[2, 4, 1, 1],
        );

        let sum = stats.sum(&input).unwrap();
        assert_eq!(sum.shape(), vec![2, 1, 1, 1]);
        assert_close(&sum.to_array(), &[20.0, 8.0], 1e-6);

        let mean = stats.mean(&input).unwrap();
        assert_close(&mean.to_array(), &[5.0, 2.0], 1e-6);

        let variance = stats.variance_population(&input).unwrap();
        assert_close(&variance.to_array(), &[5.0, 1.0], 1e-6);

        let std = stats.std_population(&input).unwrap();
        assert_close(&std.to_array(), &[5.0_f32.sqrt(), 1.0], 1e-6);

        let min = stats.min(&input).unwrap();
        assert_close(&min.to_array(), &[2.0, 1.0], 1e-6);

        let max = stats.max(&input).unwrap();
        assert_close(&max.to_array(), &[8.0, 3.0], 1e-6);
    }

    #[test]
    fn singleton_feature_has_zero_population_variance() {
        let stats = WasmStatistics::new();
        let input = WasmTensor::new(&[3.0, -2.0], &[2, 1, 1, 1]);
        assert_eq!(stats.variance_population(&input).unwrap().to_array(), vec![0.0, 0.0]);
        assert_eq!(stats.std_population(&input).unwrap().to_array(), vec![0.0, 0.0]);
    }

    #[test]
    fn invalid_layout_and_nonfinite_input_are_controlled_errors() {
        let stats = WasmStatistics::new();
        let bad_layout = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        assert!(stats.mean(&bad_layout).is_err());

        let nonfinite = WasmTensor::new(&[1.0, f32::INFINITY], &[1, 2, 1, 1]);
        assert!(stats.sum(&nonfinite).is_err());
        assert!(stats.variance_population(&nonfinite).is_err());
        assert!(stats.max(&nonfinite).is_err());
    }

    #[test]
    fn capabilities_are_machine_discoverable() {
        let caps = statistics_capabilities();
        assert!(caps.contains("burn-research.statistics.v1"));
        assert!(caps.contains("\"reduction_axis\":1"));
        assert!(caps.contains("population_no_bessel_correction"));
        assert!(caps.contains("\"probability_ops\":\"deferred\""));
    }
}
