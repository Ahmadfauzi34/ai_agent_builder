//! Burn-backed baseline/candidate comparison over the same bounded input cases.
//! The receipt is a reproducible observation, not a promotion or host authorization.

use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

use super::{bytes_hex, CompiledMultiInputGraph};
use crate::coprocessor::{validate_tolerance, verify_vectors_metrics};
use crate::multi_input_graph::{MultiInputGraphPlan, MultiInputInputBundle};
use crate::program_bundle::export_multi_input_program_bundle;
use crate::registry::LayerRegistry;
use crate::WasmTensor;

const MAX_CASES: usize = 128;
const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STATE_BYTES: usize = 16 * 1024 * 1024;

struct VerificationCase {
    baseline: MultiInputInputBundle,
    candidate: MultiInputInputBundle,
    input_digest: String,
    input_bytes: u64,
}

#[wasm_bindgen]
pub struct MultiInputVerificationCases {
    baseline_plan: MultiInputGraphPlan,
    candidate_plan: MultiInputGraphPlan,
    baseline_identity: String,
    candidate_identity: String,
    cases: Vec<VerificationCase>,
    input_bytes: u64,
}

fn digest(bytes: &[u8]) -> String {
    format!("sha256:{}", bytes_hex(&Sha256::digest(bytes)))
}

fn tensor_bytes(tensor: &WasmTensor) -> Result<u64, String> {
    tensor.inner.dims().into_iter().try_fold(4u64, |total, dimension| {
        u64::try_from(dimension).ok().and_then(|value| total.checked_mul(value))
            .ok_or_else(|| "verification: tensor byte length overflow".to_string())
    })
}

fn case_digest(baseline: &MultiInputInputBundle, candidate: &MultiInputInputBundle) -> Result<(String, u64), String> {
    let left = baseline.bound_inputs();
    let right = candidate.bound_inputs();
    if left.len() != right.len() {
        return Err("verification.addCase: input port count differs".into());
    }
    let mut hash = Sha256::new();
    let mut total = 0u64;
    for ((slot_left, tensor_left), (slot_right, tensor_right)) in left.iter().zip(right.iter()) {
        if slot_left != slot_right || tensor_left.inner.dims() != tensor_right.inner.dims() {
            return Err("verification.addCase: input slot or shape differs".into());
        }
        let bytes = tensor_bytes(tensor_left)?;
        total = total.checked_add(bytes).ok_or("verification.addCase: input budget overflow")?;
        if total > MAX_INPUT_BYTES {
            return Err("verification.addCase: input case exceeds 64 MiB".into());
        }
        hash.update([*slot_left]);
        for dimension in tensor_left.inner.dims() {
            hash.update(u64::try_from(dimension).map_err(|_| "verification: shape overflow")?.to_le_bytes());
        }
        let left_values = tensor_left.to_array();
        let right_values = tensor_right.to_array();
        for (a, b) in left_values.iter().zip(right_values.iter()) {
            if a.to_bits() != b.to_bits() {
                return Err(format!("verification.addCase: input value differs at slot {slot_left}"));
            }
            hash.update(a.to_bits().to_le_bytes());
        }
    }
    Ok((format!("sha256:{}", bytes_hex(&hash.finalize())), total))
}

fn state_digest(graph: &CompiledMultiInputGraph, registry: &LayerRegistry) -> Result<String, String> {
    let bytes = export_multi_input_program_bundle(graph, registry, true)?;
    if bytes.len() > MAX_STATE_BYTES {
        return Err("verification: stateful ProgramBundle exceeds 16 MiB".into());
    }
    Ok(digest(&bytes))
}

fn check_state(graph: &CompiledMultiInputGraph, registry: &LayerRegistry, expected: &str) -> Result<(), String> {
    if state_digest(graph, registry)? != expected {
        return Err("verification: mutable program state changed during comparison; no receipt issued".into());
    }
    Ok(())
}

fn output_observation(tensor: &WasmTensor) -> Result<([usize; 4], Vec<f32>, String), String> {
    if tensor_bytes(tensor)? > MAX_OUTPUT_BYTES {
        return Err("verification: output exceeds 16 MiB".into());
    }
    let shape = tensor.inner.dims();
    let values = tensor.to_array();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("verification: non-finite Burn output; no receipt issued".into());
    }
    let mut hash = Sha256::new();
    for value in &values {
        hash.update(value.to_bits().to_le_bytes());
    }
    Ok((shape, values, format!("sha256:{}", bytes_hex(&hash.finalize()))))
}

fn shape_json(shape: [usize; 4]) -> String {
    format!("[{},{},{},{}]", shape[0], shape[1], shape[2], shape[3])
}

#[wasm_bindgen]
impl MultiInputVerificationCases {
    #[wasm_bindgen(constructor)]
    pub fn new(baseline: &CompiledMultiInputGraph, candidate: &CompiledMultiInputGraph) -> Result<Self, String> {
        if baseline.plan.ports() != candidate.plan.ports() {
            return Err("verification: baseline and candidate input port contracts differ".into());
        }
        Ok(Self {
            baseline_plan: baseline.plan.clone(),
            candidate_plan: candidate.plan.clone(),
            baseline_identity: baseline.program_identity(),
            candidate_identity: candidate.program_identity(),
            cases: Vec::new(),
            input_bytes: 0,
        })
    }

    #[wasm_bindgen(js_name = addCase)]
    pub fn add_case(&mut self, baseline: &MultiInputInputBundle, candidate: &MultiInputInputBundle) -> Result<u32, String> {
        if self.cases.len() == MAX_CASES {
            return Err("verification.addCase: at most 128 test vectors".into());
        }
        if !baseline.input_preflight(&self.baseline_plan).ready
            || !candidate.input_preflight(&self.candidate_plan).ready {
            return Err("verification.addCase: input preflight failed".into());
        }
        let (input_digest, input_bytes) = case_digest(baseline, candidate)?;
        let total = self.input_bytes.checked_add(input_bytes).ok_or("verification.addCase: input budget overflow")?;
        if total > MAX_INPUT_BYTES {
            return Err("verification.addCase: total test vector bytes exceed 64 MiB".into());
        }
        self.cases.push(VerificationCase {
            baseline: baseline.clone(), candidate: candidate.clone(), input_digest, input_bytes,
        });
        self.input_bytes = total;
        Ok(self.cases.len() as u32)
    }

    #[wasm_bindgen(js_name = caseCount)]
    pub fn case_count(&self) -> u32 {
        self.cases.len() as u32
    }

    pub fn verify(&self,
        baseline_registry: &LayerRegistry, baseline: &CompiledMultiInputGraph,
        candidate_registry: &LayerRegistry, candidate: &CompiledMultiInputGraph,
        abs_tol: f64, rel_tol: f64,
    ) -> Result<String, String> {
        validate_tolerance(abs_tol, rel_tol)?;
        if self.cases.is_empty() {
            return Err("verification: at least one test vector is required".into());
        }
        if baseline.program_identity() != self.baseline_identity || candidate.program_identity() != self.candidate_identity
            || baseline.plan.ports() != self.baseline_plan.ports()
            || candidate.plan.ports() != self.candidate_plan.ports() {
            return Err("verification: baseline or candidate program identity changed".into());
        }
        // Validate the entire case set before the first numerical execution.
        for (index, case) in self.cases.iter().enumerate() {
            if !baseline.preflight_state(baseline_registry, &case.baseline).0
                || !candidate.preflight_state(candidate_registry, &case.candidate).0 {
                return Err(format!("verification: test vector {index} preflight failed; no execution started"));
            }
            let (current, _) = case_digest(&case.baseline, &case.candidate)?;
            if current != case.input_digest {
                return Err(format!("verification: test vector {index} changed after addCase"));
            }
        }
        let baseline_state = state_digest(baseline, baseline_registry)?;
        let candidate_state = state_digest(candidate, candidate_registry)?;
        let mut rows = Vec::with_capacity(self.cases.len());
        let mut equivalent = true;
        let mut compared_f32_count = 0usize;
        let mut max_abs_error = 0.0f64;
        let mut max_rel_error = 0.0f64;
        let mut observed_output_bytes = 0u64;
        let mut first_failure_case = None;
        for (index, case) in self.cases.iter().enumerate() {
            check_state(baseline, baseline_registry, &baseline_state)?;
            check_state(candidate, candidate_registry, &candidate_state)?;
            let baseline_output = baseline.run(baseline_registry, &case.baseline)
                .map_err(|error| format!("verification: baseline test vector {index}: {error}"))?;
            check_state(baseline, baseline_registry, &baseline_state)?;
            check_state(candidate, candidate_registry, &candidate_state)?;
            let candidate_output = candidate.run(candidate_registry, &case.candidate)
                .map_err(|error| format!("verification: candidate test vector {index}: {error}"))?;
            check_state(baseline, baseline_registry, &baseline_state)?;
            check_state(candidate, candidate_registry, &candidate_state)?;

            let pair_bytes = tensor_bytes(&baseline_output)?
                .checked_add(tensor_bytes(&candidate_output)?)
                .ok_or("verification: total output byte length overflow")?;
            observed_output_bytes = observed_output_bytes.checked_add(pair_bytes)
                .ok_or("verification: total output byte length overflow")?;
            if observed_output_bytes > MAX_TOTAL_OUTPUT_BYTES {
                return Err("verification: total observed output exceeds 64 MiB".into());
            }
            let (baseline_shape, baseline_values, baseline_digest) = output_observation(&baseline_output)?;
            let (candidate_shape, candidate_values, candidate_digest) = output_observation(&candidate_output)?;
            let shape_matches = baseline_shape == candidate_shape;
            let (passed, abs_error, rel_error, first_failure) = if shape_matches {
                let report = verify_vectors_metrics(&baseline_values, &candidate_values, abs_tol, rel_tol)?;
                compared_f32_count = compared_f32_count.checked_add(report.len).ok_or("verification: element count overflow")?;
                max_abs_error = max_abs_error.max(report.max_abs_error);
                max_rel_error = max_rel_error.max(report.max_rel_error);
                (report.passed, report.max_abs_error.to_string(), report.max_rel_error.to_string(),
                    report.first_failure.map_or_else(|| "null".to_string(), |value| value.to_string()))
            } else {
                (false, "null".to_string(), "null".to_string(), "null".to_string())
            };
            equivalent &= passed;
            if !passed && first_failure_case.is_none() {
                first_failure_case = Some(index);
            }
            rows.push(format!(
                "{{\"index\":{index},\"input_sha256\":\"{}\",\"input_bytes\":{},\"baseline_shape\":{},\"candidate_shape\":{},\"baseline_value_sha256\":\"{baseline_digest}\",\"candidate_value_sha256\":\"{candidate_digest}\",\"shape_matches\":{shape_matches},\"passed\":{passed},\"max_abs_error\":{abs_error},\"max_rel_error\":{rel_error},\"first_failure\":{first_failure}}}",
                case.input_digest, case.input_bytes, shape_json(baseline_shape), shape_json(candidate_shape),
            ));
        }
        let proof = format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.baseline-candidate-verification.v1\",\"authority\":\"wasm_burn_reference_observation\",\"baseline_program_identity\":{},\"candidate_program_identity\":{},\"baseline_state_checkpoint_bytes_sha256\":\"{baseline_state}\",\"candidate_state_checkpoint_bytes_sha256\":\"{candidate_state}\",\"abs_tol\":{abs_tol},\"rel_tol\":{rel_tol},\"tested_vector_count\":{},\"compared_f32_count\":{compared_f32_count},\"max_abs_error\":{},\"max_rel_error\":{},\"first_failure_case_index\":{},\"observed_output_bytes\":{observed_output_bytes},\"equivalent\":{equivalent},\"promotion_authorized\":false,\"cases\":[{}]}}",
            self.baseline_identity, self.candidate_identity, self.cases.len(),
            if compared_f32_count == 0 { "null".to_string() } else { max_abs_error.to_string() },
            if compared_f32_count == 0 { "null".to_string() } else { max_rel_error.to_string() },
            first_failure_case.map_or_else(|| "null".to_string(), |value| value.to_string()), rows.join(","),
        );
        Ok(format!("{},\"receipt_digest\":\"{}\"}}", &proof[..proof.len() - 1], digest(proof.as_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::protocol::LAYER_BINARY;

    fn build(spec: AgentLayerSpec) -> (LayerRegistry, CompiledMultiInputGraph, MultiInputGraphPlan) {
        let mut registry = LayerRegistry::new();
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_binary(&spec, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
        for (slot, role) in [(0, "observation"), (1, "state")] {
            plan.add_input_port(slot, role.into(), 1, 2, 1, 1,
                "feature_axis1_singleton".into(), false, 0).unwrap();
        }
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        (registry, graph, plan)
    }

    fn bundle(plan: &MultiInputGraphPlan, left: [f32; 2], right: [f32; 2]) -> MultiInputInputBundle {
        let mut bundle = MultiInputInputBundle::new(plan).unwrap();
        for (slot, role, values) in [(0, "observation", left), (1, "state", right)] {
            let tensor = WasmTensor::new(&values, &[1, 2, 1, 1]);
            bundle.bind_input(slot, &tensor, role.into(), "feature_axis1_singleton".into(),
                "test".into(), 0, String::new()).unwrap();
        }
        bundle
    }

    #[test]
    fn receipt_binds_two_executed_programs_cases_and_exact_state() {
        let (baseline_registry, baseline, baseline_plan) = build(AgentLayerSpec::add(21));
        let (candidate_registry, candidate, candidate_plan) = build(AgentLayerSpec::add(22));
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        let mut first = bundle(&baseline_plan, [1.0, 2.0], [3.0, 4.0]);
        let other = bundle(&candidate_plan, [1.0, 2.0], [3.0, 4.0]);
        assert_eq!(cases.add_case(&first, &other).unwrap(), 1);
        cases.add_case(&bundle(&baseline_plan, [-2.0, 8.0], [5.0, -3.0]),
            &bundle(&candidate_plan, [-2.0, 8.0], [5.0, -3.0])).unwrap();
        first.clear_input(0); // The accepted test vector was snapshotted.
        let receipt: serde_json::Value = serde_json::from_str(&cases.verify(
            &baseline_registry, &baseline, &candidate_registry, &candidate, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], true);
        assert_eq!(receipt["tested_vector_count"], 2);
        assert_eq!(receipt["compared_f32_count"], 4);
        assert_eq!(receipt["promotion_authorized"], false);
        assert_ne!(receipt["baseline_program_identity"], receipt["candidate_program_identity"]);
        assert_eq!(receipt["cases"][0]["baseline_value_sha256"], receipt["cases"][0]["candidate_value_sha256"]);
        let expected = digest(&export_multi_input_program_bundle(&baseline, &baseline_registry, true).unwrap());
        assert_eq!(receipt["baseline_state_checkpoint_bytes_sha256"], expected);
        assert!(receipt["receipt_digest"].as_str().unwrap().starts_with("sha256:"));
    }

    #[test]
    fn mismatch_is_a_failed_receipt_and_invalid_cases_do_not_run() {
        let (baseline_registry, baseline, baseline_plan) = build(AgentLayerSpec::add(21));
        let (mut candidate_registry, candidate, candidate_plan) = build(AgentLayerSpec::sub(21));
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        assert!(cases.verify(&baseline_registry, &baseline, &candidate_registry, &candidate, 0.0, 0.0).is_err());
        let input = bundle(&baseline_plan, [1.0, 2.0], [3.0, 4.0]);
        let different = bundle(&candidate_plan, [1.0, 3.0], [3.0, 4.0]);
        assert!(cases.add_case(&input, &different).is_err());
        assert_eq!(cases.case_count(), 0);
        cases.add_case(&input, &bundle(&candidate_plan, [1.0, 2.0], [3.0, 4.0])).unwrap();
        assert!(cases.verify(&baseline_registry, &baseline, &candidate_registry, &candidate, f64::NAN, 0.0).is_err());
        let receipt: serde_json::Value = serde_json::from_str(&cases.verify(
            &baseline_registry, &baseline, &candidate_registry, &candidate, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], false);
        assert_eq!(receipt["cases"][0]["first_failure"], 0);
        assert_eq!(receipt["cases"][0]["max_abs_error"], 8.0);
        assert_ne!(receipt["cases"][0]["baseline_value_sha256"], receipt["cases"][0]["candidate_value_sha256"]);
        assert!(candidate_registry.destroy_layer(21, LAYER_BINARY));
        assert!(cases.verify(&baseline_registry, &baseline, &candidate_registry, &candidate, 0.0, 0.0).is_err());
    }

    #[test]
    fn output_shape_difference_produces_negative_receipt() {
        let (baseline_registry, baseline, baseline_plan) = build(AgentLayerSpec::add(21));
        let (candidate_registry, candidate, candidate_plan) = build(AgentLayerSpec::concat(21, 1));
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases.add_case(&bundle(&baseline_plan, [1.0, 2.0], [3.0, 4.0]),
            &bundle(&candidate_plan, [1.0, 2.0], [3.0, 4.0])).unwrap();
        let receipt: serde_json::Value = serde_json::from_str(&cases.verify(
            &baseline_registry, &baseline, &candidate_registry, &candidate, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], false);
        assert_eq!(receipt["cases"][0]["shape_matches"], false);
        assert_eq!(receipt["compared_f32_count"], 0);
        assert!(receipt["cases"][0]["max_abs_error"].is_null());
    }
}
