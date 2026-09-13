use burn::prelude::*;
use wasm_bindgen::prelude::*;

use crate::WasmTensor;

fn validate_same_shape(a: &WasmTensor, b: &WasmTensor, op: &str) -> Result<(), String> {
    let a_shape = a.inner.dims();
    let b_shape = b.inner.dims();
    if a_shape != b_shape {
        return Err(format!(
            "NumericKernel.{op}: shape mismatch {a_shape:?} vs {b_shape:?}; implicit broadcasting is not allowed in v1"
        ));
    }
    Ok(())
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

fn validate_nonnegative(input: &WasmTensor, context: &str) -> Result<(), String> {
    validate_finite(input, context)?;
    for (index, value) in input.to_array().into_iter().enumerate() {
        if value < 0.0 {
            return Err(format!(
                "{context}: negative value at index {index}: {value}"
            ));
        }
    }
    Ok(())
}

fn validate_positive(input: &WasmTensor, context: &str) -> Result<(), String> {
    validate_finite(input, context)?;
    for (index, value) in input.to_array().into_iter().enumerate() {
        if value <= 0.0 {
            return Err(format!(
                "{context}: value must be > 0 at index {index}, got {value}"
            ));
        }
    }
    Ok(())
}

fn validate_nonzero(input: &WasmTensor, context: &str) -> Result<(), String> {
    validate_finite(input, context)?;
    for (index, value) in input.to_array().into_iter().enumerate() {
        if value == 0.0 {
            return Err(format!(
                "{context}: zero denominator at index {index}"
            ));
        }
    }
    Ok(())
}

fn checked_output(inner: Tensor<crate::WasmBackend, 4>, context: &str) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    validate_finite(&output, context)?;
    Ok(output)
}

#[wasm_bindgen(js_name = numericKernelCapabilities)]
pub fn numeric_kernel_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.numeric-kernel.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"broadcasting\":\"forbidden_v1\",",
        "\"binary_ops\":[\"add\",\"sub\",\"mul\",\"div\"],",
        "\"unary_ops\":[\"abs\",\"sqrt\",\"exp\",\"log\"],",
        "\"bounded_ops\":[\"clamp\"],",
        "\"predicates\":[\"allFinite\"],",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"divisor\":\"finite_nonzero\",",
        "\"sqrt_domain\":\"x>=0\",",
        "\"log_domain\":\"x>0\",",
        "\"clamp_bounds\":\"finite_min_lte_max\"",
        "}",
        "}"
    )
    .to_string()
}

/// Stateless Burn-backed primitive math surface.
///
/// Numeric Kernel v1 intentionally forbids implicit broadcasting and rejects invalid numerical
/// domains as controlled errors instead of silently producing NaN/Inf. It is a lower-level math
/// primitive, not a neural layer or graph policy.
#[wasm_bindgen]
pub struct WasmNumericKernel;

#[wasm_bindgen]
impl WasmNumericKernel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmNumericKernel {
        WasmNumericKernel
    }

    pub fn add(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "add")?;
        validate_finite(a, "NumericKernel.add lhs")?;
        validate_finite(b, "NumericKernel.add rhs")?;
        checked_output(
            a.inner.clone().add(b.inner.clone()),
            "NumericKernel.add output",
        )
    }

    pub fn sub(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "sub")?;
        validate_finite(a, "NumericKernel.sub lhs")?;
        validate_finite(b, "NumericKernel.sub rhs")?;
        checked_output(
            a.inner.clone().sub(b.inner.clone()),
            "NumericKernel.sub output",
        )
    }

    pub fn mul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "mul")?;
        validate_finite(a, "NumericKernel.mul lhs")?;
        validate_finite(b, "NumericKernel.mul rhs")?;
        checked_output(
            a.inner.clone().mul(b.inner.clone()),
            "NumericKernel.mul output",
        )
    }

    pub fn div(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "div")?;
        validate_finite(a, "NumericKernel.div lhs")?;
        validate_nonzero(b, "NumericKernel.div rhs")?;
        checked_output(
            a.inner.clone().div(b.inner.clone()),
            "NumericKernel.div output",
        )
    }

    pub fn abs(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_finite(input, "NumericKernel.abs input")?;
        checked_output(input.inner.clone().abs(), "NumericKernel.abs output")
    }

    pub fn sqrt(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_nonnegative(input, "NumericKernel.sqrt input")?;
        checked_output(input.inner.clone().sqrt(), "NumericKernel.sqrt output")
    }

    pub fn exp(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_finite(input, "NumericKernel.exp input")?;
        checked_output(input.inner.clone().exp(), "NumericKernel.exp output")
    }

    pub fn log(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_positive(input, "NumericKernel.log input")?;
        checked_output(input.inner.clone().log(), "NumericKernel.log output")
    }

    pub fn clamp(
        &self,
        input: &WasmTensor,
        min: f32,
        max: f32,
    ) -> Result<WasmTensor, String> {
        if !min.is_finite() || !max.is_finite() {
            return Err(format!(
                "NumericKernel.clamp: bounds must be finite, got min={min}, max={max}"
            ));
        }
        if min > max {
            return Err(format!(
                "NumericKernel.clamp: min must be <= max, got min={min}, max={max}"
            ));
        }
        validate_finite(input, "NumericKernel.clamp input")?;
        checked_output(
            input.inner.clone().clamp(min, max),
            "NumericKernel.clamp output",
        )
    }

    #[wasm_bindgen(js_name = allFinite)]
    pub fn all_finite(&self, input: &WasmTensor) -> bool {
        input.to_array().into_iter().all(f32::is_finite)
    }
}

#[cfg(test)]
mod tests {
    use super::{numeric_kernel_capabilities, WasmNumericKernel};
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
        assert_eq!(kernel.abs(&abs_input).unwrap().to_array(), vec![2.0, 0.0, 3.0]);

        let sqrt_input = WasmTensor::new(&[0.0, 4.0, 9.0], &[1, 3, 1, 1]);
        assert_eq!(kernel.sqrt(&sqrt_input).unwrap().to_array(), vec![0.0, 2.0, 3.0]);
        let negative = WasmTensor::new(&[-1.0], &[1, 1, 1, 1]);
        assert!(kernel.sqrt(&negative).is_err());

        let exp_input = WasmTensor::new(&[0.0, 1.0], &[1, 2, 1, 1]);
        assert_close(&kernel.exp(&exp_input).unwrap().to_array(), &[1.0, std::f32::consts::E], 1e-6);
        let overflow = WasmTensor::new(&[100.0], &[1, 1, 1, 1]);
        assert!(kernel.exp(&overflow).is_err());

        let log_input = WasmTensor::new(&[1.0, std::f32::consts::E], &[1, 2, 1, 1]);
        assert_close(&kernel.log(&log_input).unwrap().to_array(), &[0.0, 1.0], 1e-6);
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
