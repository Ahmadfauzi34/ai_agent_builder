use wasm_bindgen::prelude::*;

fn validate_tolerance(abs_tol: f64, rel_tol: f64) -> Result<(), String> {
    if !abs_tol.is_finite() || abs_tol < 0.0 {
        return Err(format!(
            "mathVerifyVectors: abs_tol must be finite and >= 0, got {abs_tol}"
        ));
    }
    if !rel_tol.is_finite() || rel_tol < 0.0 {
        return Err(format!(
            "mathVerifyVectors: rel_tol must be finite and >= 0, got {rel_tol}"
        ));
    }
    Ok(())
}

pub(crate) fn verify_vectors_report(
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> Result<String, String> {
    validate_tolerance(abs_tol, rel_tol)?;

    if reference.len() != candidate.len() {
        return Err(format!(
            "mathVerifyVectors: length mismatch: expected {}, got {}",
            reference.len(),
            candidate.len()
        ));
    }
    if reference.is_empty() {
        return Err("mathVerifyVectors: empty reference proof is not allowed".into());
    }

    let mut max_abs_error = 0.0f64;
    let mut max_rel_error = 0.0f64;
    let mut sum_sq_error = 0.0f64;
    let mut first_failure: Option<usize> = None;

    for (index, (&reference_value, &candidate_value)) in
        reference.iter().zip(candidate.iter()).enumerate()
    {
        if !reference_value.is_finite() {
            return Err(format!(
                "mathVerifyVectors: non-finite reference value at index {index}: {reference_value}"
            ));
        }
        if !candidate_value.is_finite() {
            return Err(format!(
                "mathVerifyVectors: non-finite candidate value at index {index}: {candidate_value}"
            ));
        }

        let reference_value = reference_value as f64;
        let candidate_value = candidate_value as f64;
        let abs_error = (reference_value - candidate_value).abs();
        let scale = reference_value
            .abs()
            .max(candidate_value.abs())
            .max(f64::EPSILON);
        let rel_error = abs_error / scale;
        let allowed_error = abs_tol + rel_tol * scale;

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        sum_sq_error += abs_error * abs_error;

        if first_failure.is_none() && abs_error > allowed_error {
            first_failure = Some(index);
        }
    }

    let rmse = (sum_sq_error / reference.len() as f64).sqrt();
    let passed = first_failure.is_none();
    let first_failure_json = first_failure
        .map(|index| index.to_string())
        .unwrap_or_else(|| "null".to_string());

    Ok(format!(
        "{{\"passed\":{passed},\"len\":{},\"max_abs_error\":{max_abs_error},\"max_rel_error\":{max_rel_error},\"rmse\":{rmse},\"first_failure\":{first_failure_json}}}",
        reference.len()
    ))
}

/// Compare an external implementation result with a trusted numerical reference.
///
/// This is intentionally dependency-free and returns compact JSON so an agent can
/// consume the proof result without coupling the produced artifact to this WASM runtime.
#[wasm_bindgen(js_name = mathVerifyVectors)]
pub fn math_verify_vectors(
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> Result<String, String> {
    verify_vectors_report(reference, candidate, abs_tol, rel_tol)
}

#[cfg(test)]
mod tests {
    use super::verify_vectors_report;
    use crate::graph::CompiledGraph;
    use crate::protocol::{ACT_RELU, LAYER_ACTIVATION, OP_INIT, PacketHeader};
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    #[test]
    fn exact_match_passes() {
        let report = verify_vectors_report(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 0.0, 0.0)
            .unwrap();
        assert!(report.contains("\"passed\":true"));
        assert!(report.contains("\"first_failure\":null"));
    }

    #[test]
    fn tolerance_match_passes() {
        let report = verify_vectors_report(&[1.0, 2.0], &[1.001, 1.999], 0.002, 0.0).unwrap();
        assert!(report.contains("\"passed\":true"));
    }

    #[test]
    fn numerical_mismatch_is_a_failed_proof_not_a_transport_error() {
        let report = verify_vectors_report(&[1.0, 2.0], &[1.0, 2.1], 0.01, 0.0).unwrap();
        assert!(report.contains("\"passed\":false"));
        assert!(report.contains("\"first_failure\":1"));
    }

    #[test]
    fn malformed_inputs_fail_before_comparison() {
        assert!(verify_vectors_report(&[1.0], &[], 0.0, 0.0).is_err());
        assert!(verify_vectors_report(&[], &[], 0.0, 0.0).is_err());
        assert!(verify_vectors_report(&[1.0], &[f32::NAN], 0.0, 0.0).is_err());
        assert!(verify_vectors_report(&[1.0], &[1.0], -1.0, 0.0).is_err());
    }

    #[test]
    fn compiled_graph_is_a_burn_backed_reference_oracle() {
        let mut registry = LayerRegistry::new();
        let layer_id = 7u32;
        let payload = layer_id.to_le_bytes();
        let header = PacketHeader {
            opcode: OP_INIT,
            layer_type: LAYER_ACTIVATION,
            variant: ACT_RELU,
            flags: 0,
            payload_len: payload.len() as u32,
        };
        registry.init_layer(&header, &payload).unwrap();

        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&2u32.to_le_bytes());
        plan.push(1);
        plan.push(LAYER_ACTIVATION);
        plan.extend_from_slice(&layer_id.to_le_bytes());
        plan.push(0);
        plan.push(0);
        plan.push(1);
        plan.push(1);

        let graph = CompiledGraph::build(&registry, &plan).unwrap();
        let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);

        let pass = graph
            .verify_flat(&registry, &input, &[0.0, 2.0], 0.0, 0.0)
            .unwrap();
        assert!(pass.contains("\"passed\":true"));

        let fail = graph
            .verify_flat(&registry, &input, &[0.0, 2.25], 0.01, 0.0)
            .unwrap();
        assert!(fail.contains("\"passed\":false"));
        assert!(fail.contains("\"first_failure\":1"));
    }
}
