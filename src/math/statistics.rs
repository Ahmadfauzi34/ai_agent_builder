pub use crate::facade::math::statistics_capabilities;
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub use crate::facade::wasm_types::WasmStatistics;
use crate::WasmTensor;

/// Fase 2: layout-only check assuming finiteness (see numeric.rs for the
/// safety invariant). Used by `_unchecked` kernel twins.
pub(crate) fn validate_feature_layout(input: &WasmTensor, context: &str) -> Result<(), String> {
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
    Ok(())
}

pub(crate) fn validate_feature_tensor(input: &WasmTensor, context: &str) -> Result<(), String> {
    validate_feature_layout(input, context)?;
    // One materialization, no intermediate Vec: iterate the slice directly.
    let data = input.inner.to_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{context}: expected f32 tensor"))?;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite value at index {index}: {value}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn checked_output(
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
    // One materialization, no intermediate Vec: iterate the slice directly.
    let data = output.inner.to_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{context}: expected f32 tensor"))?;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite output at index {index}: {value}"
            ));
        }
    }
    Ok(output)
}

/// f64 per-batch variances for one feature tensor [B,F,1,1].
/// Two-pass algorithm, exactly as specified: f64 mean, then f64 squared
/// deviations, then divide. `ddof` 0 = population, 1 = sample (Bessel
/// correction; requires at least 2 features per batch, else a controlled
/// error). Assumes layout [B,F,1,1] and finite inputs (checked by callers);
/// returns (batch, variances in f64) — the caller casts to f32 and runs
/// output validation, so a true variance above f32::MAX still fails
/// honestly instead of silently becoming inf.
pub(crate) fn variance_rows_f64(
    input: &WasmTensor,
    ddof: usize,
    context: &str,
) -> Result<(usize, Vec<f64>), String> {
    let [batch, features, _, _] = input.inner.dims();
    let data = input.inner.to_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{context}: expected f32 tensor"))?;
    if ddof > 0 && features < 2 {
        return Err(format!(
            "{context}: sample variance requires at least 2 features, got {features}"
        ));
    }
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &values[b * features..(b + 1) * features];
        let n = features as f64;
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / n;
        let sq: f64 = row
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum();
        out.push(sq / (n - ddof as f64));
    }
    Ok((batch, out))
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
        let input = WasmTensor::new(&[2.0, 4.0, 6.0, 8.0, 1.0, 1.0, 3.0, 3.0], &[2, 4, 1, 1]);

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
        assert_eq!(
            stats.variance_population(&input).unwrap().to_array(),
            vec![0.0, 0.0]
        );
        assert_eq!(
            stats.std_population(&input).unwrap().to_array(),
            vec![0.0, 0.0]
        );
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
    fn lane_e_batch4_sample_variance_and_wide_spread_population() {
        let stats = WasmStatistics::new();
        // The T5 case: f32 accumulation overflows, f64 does not.
        let wide = WasmTensor::new(&[1e19, -1e19, 1e19, -1e19], &[1, 4, 1, 1]);
        assert_close(
            &stats.variance_population(&wide).unwrap().to_array(),
            &[1e38],
            1e31,
        );
        assert_close(
            &stats.std_population(&wide).unwrap().to_array(),
            &[1e19],
            1e12,
        );

        // Sample variants (ddof=1) on [1,2,3,4]: var = 5/3, std = sqrt(5/3).
        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
        assert_close(
            &stats.variance_sample(&input).unwrap().to_array(),
            &[5.0 / 3.0],
            1e-6,
        );
        assert_close(
            &stats.std_sample(&input).unwrap().to_array(),
            &[(5.0f32 / 3.0).sqrt()],
            1e-6,
        );
        // Population values on the same input are unchanged in kind.
        assert_close(
            &stats.variance_population(&input).unwrap().to_array(),
            &[1.25],
            1e-6,
        );

        // A single feature has no defined sample variance.
        let singleton = WasmTensor::new(&[3.0], &[1, 1, 1, 1]);
        assert!(stats.variance_sample(&singleton).is_err());
        assert!(stats.std_sample(&singleton).is_err());
        assert_eq!(
            stats.variance_population(&singleton).unwrap().to_array(),
            vec![0.0]
        );
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
