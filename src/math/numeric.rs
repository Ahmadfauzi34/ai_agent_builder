pub use crate::facade::math::numeric_kernel_capabilities;
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub use crate::facade::wasm_types::WasmNumericKernel;
use crate::WasmTensor;

pub(crate) fn validate_same_shape(a: &WasmTensor, b: &WasmTensor, op: &str) -> Result<(), String> {
    let a_shape = a.inner.dims();
    let b_shape = b.inner.dims();
    if a_shape != b_shape {
        return Err(format!(
            "NumericKernel.{op}: shape mismatch {a_shape:?} vs {b_shape:?}; implicit broadcasting is not allowed in v1"
        ));
    }
    Ok(())
}

pub(crate) fn validate_finite(input: &WasmTensor, context: &str) -> Result<(), String> {
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

pub(crate) fn validate_nonnegative(input: &WasmTensor, context: &str) -> Result<(), String> {
    // Single materialization for both checks (was: validate_finite + to_array).
    // Two passes preserve the original error precedence (finiteness first).
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
    for (index, &value) in values.iter().enumerate() {
        if value < 0.0 {
            return Err(format!(
                "{context}: negative value at index {index}: {value}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_positive(input: &WasmTensor, context: &str) -> Result<(), String> {
    // Single materialization for both checks (was: validate_finite + to_array).
    // Two passes preserve the original error precedence (finiteness first).
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
    for (index, &value) in values.iter().enumerate() {
        if value <= 0.0 {
            return Err(format!(
                "{context}: value must be > 0 at index {index}, got {value}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_nonzero(input: &WasmTensor, context: &str) -> Result<(), String> {
    // Single materialization for both checks (was: validate_finite + to_array).
    // Two passes preserve the original error precedence (finiteness first).
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
    for (index, &value) in values.iter().enumerate() {
        if value == 0.0 {
            return Err(format!("{context}: zero denominator at index {index}"));
        }
    }
    Ok(())
}

pub(crate) fn checked_output(
    inner: Tensor<crate::WasmBackend, 4>,
    context: &str,
) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    validate_finite(&output, context)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{numeric_kernel_capabilities, WasmNumericKernel};
    use crate::WasmTensor;

    #[test]
    fn fused_validators_preserve_error_precedence() {
        // The fused validators (nonnegative/positive/nonzero) must report
        // non-finite before the specific violation, matching the old
        // validate_finite-then-check call sequence.
        let both = WasmTensor::new(&[-1.0, f32::NAN], &[1, 2, 1, 1]);
        let err = super::validate_nonnegative(&both, "test").unwrap_err();
        assert!(err.contains("non-finite"), "unexpected error: {err}");

        let negative = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
        let err = super::validate_nonnegative(&negative, "test").unwrap_err();
        assert!(err.contains("negative"), "unexpected error: {err}");

        let valid = WasmTensor::new(&[0.0, 2.0], &[1, 2, 1, 1]);
        assert!(super::validate_nonnegative(&valid, "test").is_ok());

        let both_positive = WasmTensor::new(&[f32::INFINITY, -1.0], &[1, 2, 1, 1]);
        let err = super::validate_positive(&both_positive, "test").unwrap_err();
        assert!(err.contains("non-finite"), "unexpected error: {err}");

        let zero = WasmTensor::new(&[f32::NAN, 0.0], &[1, 2, 1, 1]);
        let err = super::validate_nonzero(&zero, "test").unwrap_err();
        assert!(err.contains("non-finite"), "unexpected error: {err}");
    }

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
    fn binary_ops_preserve_shape_and_compute_expected_values() {
        let kernel = WasmNumericKernel::new();
        let a = WasmTensor::new(&[1.0, -2.0, 6.0], &[1, 3, 1, 1]);
        let b = WasmTensor::new(&[2.0, 4.0, 3.0], &[1, 3, 1, 1]);

        let add = kernel.add(&a, &b).unwrap();
        let sub = kernel.sub(&a, &b).unwrap();
        let mul = kernel.mul(&a, &b).unwrap();
        let div = kernel.div(&a, &b).unwrap();

        assert_eq!(add.shape(), a.shape());
        assert_eq!(add.to_array(), vec![3.0, 2.0, 9.0]);
        assert_eq!(sub.to_array(), vec![-1.0, -6.0, 3.0]);
        assert_eq!(mul.to_array(), vec![2.0, -8.0, 18.0]);
        assert_close(&div.to_array(), &[0.5, -0.5, 2.0], 1e-6);
    }

    #[test]
    fn binary_contract_rejects_shape_mismatch_zero_divisor_and_nonfinite_values() {
        let kernel = WasmNumericKernel::new();
        let a = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let mismatch = WasmTensor::new(&[1.0, 2.0], &[1, 1, 2, 1]);
        assert!(kernel.add(&a, &mismatch).is_err());

        let zero = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);
        assert!(kernel.div(&a, &zero).is_err());

        let nan = WasmTensor::new(&[1.0, f32::NAN], &[1, 2, 1, 1]);
        assert!(kernel.mul(&a, &nan).is_err());
    }

    #[test]
    fn unary_ops_enforce_domains_and_finite_outputs() {
        let kernel = WasmNumericKernel::new();

        let abs_input = WasmTensor::new(&[-2.0, 0.0, 3.0], &[1, 3, 1, 1]);
        assert_eq!(
            kernel.abs(&abs_input).unwrap().to_array(),
            vec![2.0, 0.0, 3.0]
        );

        let sqrt_input = WasmTensor::new(&[0.0, 4.0, 9.0], &[1, 3, 1, 1]);
        assert_eq!(
            kernel.sqrt(&sqrt_input).unwrap().to_array(),
            vec![0.0, 2.0, 3.0]
        );
        let negative = WasmTensor::new(&[-1.0], &[1, 1, 1, 1]);
        assert!(kernel.sqrt(&negative).is_err());

        let exp_input = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        assert_close(
            &kernel.exp(&exp_input).unwrap().to_array(),
            &[1.0, std::f32::consts::E],
            1e-6,
        );
        let overflow = WasmTensor::new(&[100.0], &[1, 1, 1, 1]);
        assert!(kernel.exp(&overflow).is_err());

        let log_input = WasmTensor::new(&[1.0, std::f32::consts::E], &[1, 2, 1, 1]);
        assert_close(
            &kernel.log(&log_input).unwrap().to_array(),
            &[0.0, 1.0],
            1e-6,
        );
        let zero = WasmTensor::new(&[0.0], &[1, 1, 1, 1]);
        assert!(kernel.log(&zero).is_err());
    }

    #[test]
    fn clamp_and_finite_predicate_have_controlled_contracts() {
        let kernel = WasmNumericKernel::new();
        let input = WasmTensor::new(&[-2.0, 0.5, 3.0], &[1, 3, 1, 1]);
        assert_eq!(
            kernel.clamp(&input, 0.0, 1.0).unwrap().to_array(),
            vec![0.0, 0.5, 1.0]
        );
        assert!(kernel.clamp(&input, 2.0, 1.0).is_err());
        assert!(kernel.clamp(&input, f32::NAN, 1.0).is_err());
        assert!(kernel.all_finite(&input));

        let nonfinite = WasmTensor::new(&[1.0, f32::INFINITY], &[1, 2, 1, 1]);
        assert!(!kernel.all_finite(&nonfinite));
        assert!(kernel.abs(&nonfinite).is_err());
    }

    #[test]
    fn capabilities_are_machine_discoverable() {
        let caps = numeric_kernel_capabilities();
        assert!(caps.contains("burn-research.numeric-kernel.v1"));
        assert!(caps.contains("\"broadcasting\":\"forbidden_v1\""));
        assert!(caps.contains("\"finite_outputs\":true"));
    }
}
