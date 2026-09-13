use burn::prelude::*;
use wasm_bindgen::prelude::*;

use crate::WasmTensor;

const DEFAULT_EPSILON: f64 = 1e-12;

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

fn validate_feature_shape(shape: [usize; 4], context: &str) -> Result<(), String> {
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

fn validate_feature_pair(a: &WasmTensor, b: &WasmTensor, context: &str) -> Result<(), String> {
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

fn validate_epsilon(epsilon: f64, context: &str) -> Result<f32, String> {
    if !epsilon.is_finite() || epsilon <= 0.0 || epsilon > f32::MAX as f64 {
        return Err(format!(
            "{context}: epsilon must be finite, > 0, and representable as f32; got {epsilon}"
        ));
    }
    Ok(epsilon as f32)
}

fn checked_output(
    inner: Tensor<crate::WasmBackend, 4>,
    context: &str,
) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    validate_finite(&output, context)?;
    Ok(output)
}

#[wasm_bindgen(js_name = linearAlgebraCapabilities)]
pub fn linear_algebra_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.linear-algebra.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"matrix_ops\":[\"matmul\"],",
        "\"vector_ops\":[\"dot\",\"l2Norm\",\"cosineSimilarity\",\"l2Distance\"],",
        "\"vector_layout\":\"[B,F,1,1]\",",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"paired_vector_shape\":\"exact_match\",",
        "\"matmul_layout\":\"[B,G,M,K]@[B,G,K,N]\",",
        "\"cosine_zero_vector\":\"stabilized_to_zero\",",
        "\"solve\":\"deferred_v1\"",
        "}",
        "}"
    )
    .to_string()
}

/// Stateless Burn-backed linear algebra surface for the canonical rank-4 bridge.
///
/// Vector operations use `[B,F,1,1]`. Matrix multiplication follows Burn's rank-4
/// batched semantics `[B,G,M,K] @ [B,G,K,N] -> [B,G,M,N]`.
#[wasm_bindgen]
pub struct WasmLinearAlgebra;

#[wasm_bindgen]
impl WasmLinearAlgebra {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLinearAlgebra {
        WasmLinearAlgebra
    }

    pub fn matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        let da = a.inner.dims();
        let db = b.inner.dims();
        if da.iter().any(|&d| d == 0) || db.iter().any(|&d| d == 0) {
            return Err(format!(
                "LinearAlgebra.matmul: zero-sized dimensions are not supported in v1: {da:?} @ {db:?}"
            ));
        }
        if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
            return Err(format!(
                "LinearAlgebra.matmul: incompatible shapes {da:?} @ {db:?}; expected [B,G,M,K] @ [B,G,K,N]"
            ));
        }
        validate_finite(a, "LinearAlgebra.matmul lhs")?;
        validate_finite(b, "LinearAlgebra.matmul rhs")?;
        checked_output(
            a.inner.clone().matmul(b.inner.clone()),
            "LinearAlgebra.matmul output",
        )
    }

    pub fn dot(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.dot")?;
        checked_output(
            a.inner.clone().mul(b.inner.clone()).sum_dim(1),
            "LinearAlgebra.dot output",
        )
    }

    #[wasm_bindgen(js_name = l2Norm)]
    pub fn l2_norm(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_shape(input.inner.dims(), "LinearAlgebra.l2Norm")?;
        validate_finite(input, "LinearAlgebra.l2Norm input")?;
        let squared = input.inner.clone().mul(input.inner.clone());
        checked_output(
            squared.sum_dim(1).sqrt(),
            "LinearAlgebra.l2Norm output",
        )
    }

    #[wasm_bindgen(js_name = cosineSimilarity)]
    pub fn cosine_similarity(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        epsilon: Option<f64>,
    ) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.cosineSimilarity")?;
        let epsilon = validate_epsilon(
            epsilon.unwrap_or(DEFAULT_EPSILON),
            "LinearAlgebra.cosineSimilarity",
        )?;

        let dot = a.inner.clone().mul(b.inner.clone()).sum_dim(1);
        let norm_a = a
            .inner
            .clone()
            .mul(a.inner.clone())
            .sum_dim(1)
            .sqrt();
        let norm_b = b
            .inner
            .clone()
            .mul(b.inner.clone())
            .sum_dim(1)
            .sqrt();
        let denominator = norm_a.mul(norm_b).clamp_min(epsilon);
        checked_output(
            dot.div(denominator),
            "LinearAlgebra.cosineSimilarity output",
        )
    }

    #[wasm_bindgen(js_name = l2Distance)]
    pub fn l2_distance(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.l2Distance")?;
        let delta = a.inner.clone().sub(b.inner.clone());
        let squared = delta.clone().mul(delta);
        checked_output(
            squared.sum_dim(1).sqrt(),
            "LinearAlgebra.l2Distance output",
        )
    }
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
        let a = WasmTensor::new(
            &[1.0, 2.0, 3.0, 3.0, 4.0, 0.0],
            &[2, 3, 1, 1],
        );
        let b = WasmTensor::new(
            &[3.0, 1.0, 2.0, 0.0, 4.0, 3.0],
            &[2, 3, 1, 1],
        );

        let dot = linalg.dot(&a, &b).unwrap();
        assert_eq!(dot.shape(), vec![2, 1, 1, 1]);
        assert_close(&dot.to_array(), &[11.0, 16.0], 1e-6);

        let norm = linalg.l2_norm(&a).unwrap();
        assert_close(&norm.to_array(), &[14.0_f32.sqrt(), 5.0], 1e-6);

        let cosine = linalg.cosine_similarity(&a, &b, None).unwrap();
        assert_close(
            &cosine.to_array(),
            &[11.0 / 14.0, 16.0 / 20.0],
            1e-6,
        );

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
        let a = WasmTensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1, 1, 2, 3],
        );
        let b = WasmTensor::new(
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[1, 1, 3, 2],
        );
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
