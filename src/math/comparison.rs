//! Exact-shape numeric comparison primitives for rank-4 tensors.
//!
//! Comparison v1 deliberately returns canonical numeric predicates (`0.0` / `1.0`) instead of
//! exposing a new boolean tensor family. Inputs must be finite and exactly shape-compatible; no
//! implicit broadcasting is permitted.

use burn::prelude::*;

use crate::WasmTensor;

fn validate_same_shape(lhs: &WasmTensor, rhs: &WasmTensor, context: &str) -> Result<[usize; 4], String> {
    let lhs_shape = lhs.inner.dims();
    let rhs_shape = rhs.inner.dims();
    if lhs_shape != rhs_shape {
        return Err(format!(
            "{context}: shape mismatch {lhs_shape:?} vs {rhs_shape:?}; implicit broadcasting is not allowed in v1"
        ));
    }
    Ok(lhs_shape)
}

fn validate_finite(input: &WasmTensor, context: &str) -> Result<(), String> {
    for (index, value) in input.to_array().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite value at index {index}: {value}"
            ));
        }
    }
    Ok(())
}

fn validate_numeric_predicate_output(
    output: &WasmTensor,
    expected_shape: [usize; 4],
    context: &str,
) -> Result<(), String> {
    let actual_shape = output.inner.dims();
    if actual_shape != expected_shape {
        return Err(format!(
            "{context}: shape invariant failed; expected {expected_shape:?}, got {actual_shape:?}"
        ));
    }

    for (index, value) in output.to_array().into_iter().enumerate() {
        let bits = value.to_bits();
        if bits != 0.0f32.to_bits() && bits != 1.0f32.to_bits() {
            return Err(format!(
                "{context}: non-canonical predicate value at index {index}: value={value}, bits=0x{bits:08x}"
            ));
        }
    }
    Ok(())
}

pub fn comparison_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.comparison.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"rank\":4,",
        "\"comparison_ops\":[\"lessEqual01\"],",
        "\"predicate_representation\":\"canonical_f32_0_or_1\",",
        "\"broadcasting\":\"forbidden_v1\",",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"shape_preserved\":true,",
        "\"equality_is_true\":true,",
        "\"boolean_tensor_family\":false,",
        "\"stateless\":true,",
        "\"registry_independent\":true,",
        "\"grants_authority\":false",
        "}",
        "}"
    )
    .to_string()
}

/// Stateless Burn-backed exact-shape numeric comparison surface.
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorComparison;

impl TensorComparison {
    pub fn new() -> Self {
        Self
    }

    /// Element-wise `lhs <= rhs`, materialized as canonical f32 `0.0` / `1.0` values.
    pub fn less_equal_01(
        &self,
        lhs: &WasmTensor,
        rhs: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        let shape = validate_same_shape(lhs, rhs, "Comparison.lessEqual01")?;
        validate_finite(lhs, "Comparison.lessEqual01 lhs")?;
        validate_finite(rhs, "Comparison.lessEqual01 rhs")?;

        let output = WasmTensor {
            inner: lhs
                .inner
                .clone()
                .lower_equal(rhs.inner.clone())
                .float(),
        };
        validate_numeric_predicate_output(&output, shape, "Comparison.lessEqual01 output")?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: Vec<f32>) -> Vec<u32> {
        values.into_iter().map(f32::to_bits).collect()
    }

    #[test]
    fn less_equal_01_covers_less_equal_and_greater_with_exact_numeric_predicates() {
        let comparison = TensorComparison::new();
        let lhs = WasmTensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[1, 6, 1, 1]);
        let rhs = WasmTensor::new(&[-1.0, -1.0, 0.0, 0.0, 3.0, 2.0], &[1, 6, 1, 1]);

        let output = comparison.less_equal_01(&lhs, &rhs).unwrap();
        assert_eq!(output.shape(), lhs.shape());
        assert_eq!(output.to_array(), vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(
            bits(output.to_array()),
            vec![
                1.0f32.to_bits(),
                1.0f32.to_bits(),
                1.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                0.0f32.to_bits(),
            ]
        );
    }

    #[test]
    fn equality_including_signed_zero_returns_one() {
        let comparison = TensorComparison::new();
        let lhs = WasmTensor::new(&[-3.5, 0.0, -0.0, 7.0], &[1, 4, 1, 1]);
        let rhs = WasmTensor::new(&[-3.5, -0.0, 0.0, 7.0], &[1, 4, 1, 1]);
        let output = comparison.less_equal_01(&lhs, &rhs).unwrap();
        assert_eq!(output.to_array(), vec![1.0; 4]);
    }

    #[test]
    fn shape_mismatch_fails_closed_without_broadcasting() {
        let comparison = TensorComparison::new();
        let lhs = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let rhs = WasmTensor::new(&[1.0, 2.0], &[1, 1, 2, 1]);
        let err = comparison.less_equal_01(&lhs, &rhs).err().unwrap();
        assert!(err.contains("shape mismatch"));
        assert!(err.contains("implicit broadcasting is not allowed"));
    }

    #[test]
    fn nonfinite_values_on_either_side_fail_closed() {
        let comparison = TensorComparison::new();
        let finite = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        let nan = WasmTensor::new(&[0.0, f32::NAN], &[1, 2, 1, 1]);
        let pos_inf = WasmTensor::new(&[0.0, f32::INFINITY], &[1, 2, 1, 1]);
        let neg_inf = WasmTensor::new(&[0.0, f32::NEG_INFINITY], &[1, 2, 1, 1]);

        assert!(comparison.less_equal_01(&nan, &finite).is_err());
        assert!(comparison.less_equal_01(&pos_inf, &finite).is_err());
        assert!(comparison.less_equal_01(&finite, &nan).is_err());
        assert!(comparison.less_equal_01(&finite, &neg_inf).is_err());
    }

    #[test]
    fn repeated_execution_is_bit_deterministic_and_output_is_only_positive_zero_or_one() {
        let comparison = TensorComparison::new();
        let lhs = WasmTensor::new(&[-4.0, 2.0, 8.0, 8.0], &[1, 2, 2, 1]);
        let rhs = WasmTensor::new(&[-5.0, 2.0, 9.0, 7.0], &[1, 2, 2, 1]);

        let first = comparison.less_equal_01(&lhs, &rhs).unwrap();
        let second = comparison.less_equal_01(&lhs, &rhs).unwrap();
        let first_bits = bits(first.to_array());
        let second_bits = bits(second.to_array());
        assert_eq!(first_bits, second_bits);
        assert!(first_bits
            .iter()
            .all(|bits| *bits == 0.0f32.to_bits() || *bits == 1.0f32.to_bits()));
    }

    #[test]
    fn capabilities_publish_numeric_predicate_and_authority_boundaries() {
        let caps = comparison_capabilities();
        assert!(caps.contains("burn-research.comparison.v1"));
        assert!(caps.contains("\"predicate_representation\":\"canonical_f32_0_or_1\""));
        assert!(caps.contains("\"broadcasting\":\"forbidden_v1\""));
        assert!(caps.contains("\"finite_inputs\":true"));
        assert!(caps.contains("\"boolean_tensor_family\":false"));
        assert!(caps.contains("\"grants_authority\":false"));
    }
}
