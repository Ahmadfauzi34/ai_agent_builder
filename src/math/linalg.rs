pub use crate::facade::math::linear_algebra_capabilities;
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub use crate::facade::wasm_types::WasmLinearAlgebra;
use crate::WasmTensor;

pub(crate) const DEFAULT_EPSILON: f64 = 1e-12;

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

pub(crate) fn validate_feature_shape(shape: [usize; 4], context: &str) -> Result<(), String> {
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

pub(crate) fn validate_feature_pair(
    a: &WasmTensor,
    b: &WasmTensor,
    context: &str,
) -> Result<(), String> {
    let a_shape = a.inner.dims();
    let b_shape = b.inner.dims();
    validate_feature_shape(a_shape, context)?;
    validate_feature_shape(b_shape, context)?;
    if a_shape != b_shape {
        return Err(format!(
            "{context}: feature shape mismatch {a_shape:?} vs {b_shape:?}"
        ));
    }
    validate_finite(a, &format!("{context} lhs"))?;
    validate_finite(b, &format!("{context} rhs"))?;
    Ok(())
}

pub(crate) fn validate_epsilon(epsilon: f64, context: &str) -> Result<f32, String> {
    if !epsilon.is_finite() || epsilon <= 0.0 || epsilon > f32::MAX as f64 {
        return Err(format!(
            "{context}: epsilon must be finite, > 0, and representable as f32; got {epsilon}"
        ));
    }
    Ok(epsilon as f32)
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
    use super::{linear_algebra_capabilities, WasmLinearAlgebra};
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
    fn vector_ops_compute_expected_batchwise_values() {
        let linalg = WasmLinearAlgebra::new();
        let a = WasmTensor::new(&[1.0, 2.0, 3.0, 3.0, 4.0, 0.0], &[2, 3, 1, 1]);
        let b = WasmTensor::new(&[3.0, 1.0, 2.0, 0.0, 4.0, 3.0], &[2, 3, 1, 1]);

        let dot = linalg.dot(&a, &b).unwrap();
        assert_eq!(dot.shape(), vec![2, 1, 1, 1]);
        assert_close(&dot.to_array(), &[11.0, 16.0], 1e-6);

        let norm = linalg.l2_norm(&a).unwrap();
        assert_close(&norm.to_array(), &[14.0_f32.sqrt(), 5.0], 1e-6);

        let cosine = linalg.cosine_similarity(&a, &b, None).unwrap();
        assert_close(&cosine.to_array(), &[11.0 / 14.0, 16.0 / 25.0], 1e-6);

        let distance = linalg.l2_distance(&a, &b).unwrap();
        assert_close(
            &distance.to_array(),
            &[6.0_f32.sqrt(), 18.0_f32.sqrt()],
            1e-6,
        );
    }

    #[test]
    fn cosine_zero_vector_is_stabilized_to_zero() {
        let linalg = WasmLinearAlgebra::new();
        let zero = WasmTensor::new(&[0.0, 0.0], &[1, 2, 1, 1]);
        let other = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        assert_eq!(
            linalg
                .cosine_similarity(&zero, &other, None)
                .unwrap()
                .to_array(),
            vec![0.0]
        );
    }

    #[test]
    fn matmul_matches_reference_result() {
        let linalg = WasmLinearAlgebra::new();
        let a = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]);
        let b = WasmTensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[1, 1, 3, 2]);
        let out = linalg.matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        assert_eq!(out.to_array(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn invalid_shapes_epsilon_and_nonfinite_inputs_are_controlled_errors() {
        let linalg = WasmLinearAlgebra::new();
        let a = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let mismatch = WasmTensor::new(&[1.0, 2.0], &[1, 1, 2, 1]);
        assert!(linalg.dot(&a, &mismatch).is_err());
        assert!(linalg.cosine_similarity(&a, &a, Some(0.0)).is_err());
        assert!(linalg.cosine_similarity(&a, &a, Some(f64::NAN)).is_err());

        let bad_mat = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let bad_rhs = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        assert!(linalg.matmul(&bad_mat, &bad_rhs).is_err());

        let nonfinite = WasmTensor::new(&[1.0, f32::INFINITY], &[1, 2, 1, 1]);
        assert!(linalg.l2_norm(&nonfinite).is_err());
        assert!(linalg.dot(&a, &nonfinite).is_err());
    }

    #[test]
    fn capabilities_are_machine_discoverable() {
        let caps = linear_algebra_capabilities();
        assert!(caps.contains("burn-research.linear-algebra.v1"));
        assert!(caps.contains("\"vector_layout\":\"[B,F,1,1]\""));
        assert!(caps.contains("\"solve\":\"deferred_v1\""));
    }
}
