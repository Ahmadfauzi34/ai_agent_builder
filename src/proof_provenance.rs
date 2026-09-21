use wasm_bindgen::prelude::*;

use crate::coprocessor::verify_vectors_metrics;
use crate::graph::CompiledGraph;
use crate::math::{
    MathProgram, MathProgramV4, MathProgramV5, MathProgramV6, MathProgramV7, MathProgramV8,
    MathProgramV9,
};
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

const PROOF_PROVENANCE_V1: &str = include_str!("../docs/proof-provenance.v1.json");
const MATH_PROOF_V1: &str = include_str!("../docs/math-proof.v1.json");
const MAX_PROOF_LABEL_BYTES: usize = 256;
const MAX_ATTESTATION_DETAIL_BYTES: usize = 1024;

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn validate_label(label: &str, context: &str) -> Result<(), String> {
    if label.is_empty() {
        return Err(format!("{context}: label must be non-empty"));
    }
    if label.len() > MAX_PROOF_LABEL_BYTES {
        return Err(format!(
            "{context}: label {} bytes exceeds limit {MAX_PROOF_LABEL_BYTES}",
            label.len()
        ));
    }
    Ok(())
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn bytes_fingerprint(bytes: &[u8]) -> String {
    format!("fnv1a64:{:016x}", fnv1a64(bytes.iter().copied()))
}

fn f32_fingerprint(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    bytes_fingerprint(&bytes)
}

fn tensor_fingerprint(tensor: &WasmTensor) -> String {
    let mut bytes = Vec::new();
    for dim in tensor.shape() {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    for value in tensor.to_array() {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes_fingerprint(&bytes)
}


fn tensors_fingerprint(inputs: &[WasmTensor]) -> String {
    let mut bytes = Vec::new();
    for (index, tensor) in inputs.iter().enumerate() {
        bytes.extend_from_slice(&(index as u64).to_le_bytes());
        for dim in tensor.shape() {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for value in tensor.to_array() {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes_fingerprint(&bytes)
}

fn math_program_version(plan: &[u8]) -> Result<u8, String> {
    if plan.len() < 5 {
        return Err("MathProgram proof: plan is truncated before version byte".to_string());
    }
    let version = plan[4];
    if !(1..=9).contains(&version) {
        return Err(format!(
            "MathProgram proof: unsupported canonical plan version {version}; expected 1..=9"
        ));
    }
    Ok(version)
}

fn math_program_identity_from_plan(plan: &[u8]) -> Result<(u8, String), String> {
    let version = math_program_version(plan)?;
    let identity = match version {
        1..=3 => MathProgram::from_plan(plan)?.program_identity(),
        4 => MathProgramV4::from_plan(plan)?.program_identity(),
        5 => MathProgramV5::from_plan(plan)?.program_identity(),
        6 => MathProgramV6::from_plan(plan)?.program_identity(),
        7 => MathProgramV7::from_plan(plan)?.program_identity(),
        8 => MathProgramV8::from_plan(plan)?.program_identity(),
        9 => MathProgramV9::from_plan(plan)?.program_identity(),
        _ => unreachable!("version validated above"),
    };
    Ok((version, identity))
}

fn run_math_program_plan(
    plan: &[u8],
    inputs: &[WasmTensor],
) -> Result<(u8, String, Vec<f32>), String> {
    let version = math_program_version(plan)?;
    let (identity, output) = match version {
        1..=3 => {
            let program = MathProgram::from_plan(plan)?;
            let output = match inputs {
                [input] => program.run1(input)?,
                [a, b] => program.run2(a, b)?,
                _ => {
                    return Err(format!(
                        "MathProgram proof: v1-v3 verifier supports exactly 1 or 2 inputs, got {}",
                        inputs.len()
                    ))
                }
            };
            (program.program_identity(), output)
        }
        4 => {
            let program = MathProgramV4::from_plan(plan)?;
            let output = match inputs {
                [input] => program.run1(input)?,
                [a, b] => program.run2(a, b)?,
                _ => {
                    return Err(format!(
                        "MathProgram proof: v4 verifier supports exactly 1 or 2 inputs, got {}",
                        inputs.len()
                    ))
                }
            };
            (program.program_identity(), output)
        }
        5 => {
            let program = MathProgramV5::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        6 => {
            let program = MathProgramV6::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        7 => {
            let program = MathProgramV7::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        8 => {
            let program = MathProgramV8::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        9 => {
            let program = MathProgramV9::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        _ => unreachable!("version validated above"),
    };
    Ok((version, identity, output.to_array()))
}

fn tolerances_json(abs_tol: f64, rel_tol: f64) -> String {
    format!("{{\"abs\":{abs_tol},\"rel\":{rel_tol}}}")
}

fn ledger_receipt_json(
    receipt_id: u32,
    authority: &str,
    verifier: &str,
    reference_authority: &str,
    label: &str,
    primary_fingerprint_name: &str,
    primary_fingerprint: &str,
    candidate_fingerprint: &str,
    program_identity_fingerprint: Option<&str>,
    runtime_subject_json: Option<&str>,
    abs_tol: f64,
    rel_tol: f64,
    result_json: &str,
) -> String {
    let program = program_identity_fingerprint
        .map(|value| format!(",\"program_identity_fingerprint\":\"{}\"", json_escape(value)))
        .unwrap_or_default();
    let runtime_subject = runtime_subject_json
        .map(|value| format!(",\"runtime_subject\":{value}"))
        .unwrap_or_default();
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"{}\",",
            "\"verifier\":\"{}\",",
            "\"reference_authority\":\"{}\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"{}\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\"",
            "{}",
            "{}",
            ",\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(authority),
        json_escape(verifier),
        json_escape(reference_authority),
        json_escape(label),
        json_escape(primary_fingerprint_name),
        json_escape(primary_fingerprint),
        json_escape(candidate_fingerprint),
        program,
        runtime_subject,
        tolerances_json(abs_tol, rel_tol),
        result_json,
    )
}

/// Return the embedded proof-provenance contract.
#[wasm_bindgen(js_name = proofProvenanceCapabilities)]
pub fn proof_provenance_capabilities() -> String {
    PROOF_PROVENANCE_V1.to_string()
}


/// Return the MathProgram proof/correlation authority contract.
#[wasm_bindgen(js_name = mathProofCapabilities)]
pub fn math_proof_capabilities() -> String {
    MATH_PROOF_V1.to_string()
}

/// Record an explicit caller claim. This never upgrades into verifier authority.
#[wasm_bindgen(js_name = workspaceRecordAttestation)]
pub fn workspace_record_attestation(
    workspace: &mut AgentWorkspace,
    label: String,
    claimed_passed: bool,
    detail: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceRecordAttestation")?;
    if detail.len() > MAX_ATTESTATION_DETAIL_BYTES {
        return Err(format!(
            "workspaceRecordAttestation: detail {} bytes exceeds limit {MAX_ATTESTATION_DETAIL_BYTES}",
            detail.len()
        ));
    }

    let attestation_id =
        workspace.record_attestation_internal(label.clone(), claimed_passed, detail.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.caller-attestation.v1\",",
            "\"attestation_id\":{},",
            "\"authority\":\"caller_attestation\",",
            "\"label\":\"{}\",",
            "\"claimed_passed\":{},",
            "\"detail\":\"{}\"",
            "}}"
        ),
        attestation_id,
        json_escape(&label),
        claimed_passed,
        json_escape(&detail),
    ))
}

/// Verify against a caller-supplied reference and record a lower-authority comparator receipt.
#[wasm_bindgen(js_name = workspaceVerifyVectorReceipt)]
pub fn workspace_verify_vector_receipt(
    workspace: &mut AgentWorkspace,
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceVerifyVectorReceipt")?;

    let report = verify_vectors_metrics(reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();
    let reference_fingerprint = f32_fingerprint(reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_comparator",
        "mathVerifyVectors",
        "caller_supplied",
        &label,
        "reference_fingerprint",
        &reference_fingerprint,
        &candidate_fingerprint,
        None,
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "mathVerifyVectors".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    let runtime_subject = runtime_subject_binding_json(workspace);
    Ok(ledger_receipt_json(
        receipt_id,
        "wasm_comparator",
        "mathVerifyVectors",
        "caller_supplied",
        &label,
        "reference_fingerprint",
        &reference_fingerprint,
        &candidate_fingerprint,
        None,
        Some(&runtime_subject),
        abs_tol,
        rel_tol,
        &result_json,
    ))
}

/// Execute the compiled Burn graph as reference, compare the candidate, and record the receipt.
///
/// The returned receipt includes the exact programIdentity object. The workspace ledger stores a
/// compact receipt keyed by the deterministic program-identity fingerprint.
#[wasm_bindgen(js_name = workspaceVerifyGraphReceipt)]
pub fn workspace_verify_graph_receipt(
    workspace: &mut AgentWorkspace,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceVerifyGraphReceipt")?;

    let program_identity = graph.program_identity();
    workspace.require_runtime_program_identity_if_bound(
        &program_identity,
        "workspaceVerifyGraphReceipt",
    )?;

    let reference = graph.run(registry, input)?.to_array();
    let report = verify_vectors_metrics(&reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();

    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensor_fingerprint(input);
    let reference_fingerprint = f32_fingerprint(&reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_verifier",
        "CompiledGraph.verifyFlat",
        "burn_compiled_graph",
        &label,
        "input_fingerprint",
        &input_fingerprint,
        &candidate_fingerprint,
        Some(&program_identity_fingerprint),
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "CompiledGraph.verifyFlat".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"wasm_verifier\",",
            "\"verifier\":\"CompiledGraph.verifyFlat\",",
            "\"reference_authority\":\"burn_compiled_graph\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"program_identity\":{},",
            "\"program_identity_fingerprint\":\"{}\",",
            "\"mutable_state_in_program_identity\":false,",
            "\"runtime_subject\":{},",
            "\"input_fingerprint\":\"{}\",",
            "\"reference_fingerprint\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\",",
            "\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(&label),
        program_identity,
        json_escape(&program_identity_fingerprint),
        runtime_subject_binding_json(workspace),
        json_escape(&input_fingerprint),
        json_escape(&reference_fingerprint),
        json_escape(&candidate_fingerprint),
        tolerances_json(abs_tol, rel_tol),
        result_json,
    ))
}



/// Bind a replay-derived MathProgram programIdentity to an already-bound runtime subject.
///
/// Callers supply canonical plan bytes, never a free-form identity string. Replay validation
/// derives the authoritative exact programIdentity before any binding mutation occurs.
#[wasm_bindgen(js_name = workspaceBindRuntimeMathProgramPlan)]
pub fn workspace_bind_runtime_math_program_plan(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
) -> Result<String, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceBindRuntimeMathProgramPlan: runtime subject must be bound first".to_string(),
        );
    }

    let (version, program_identity) = math_program_identity_from_plan(plan)?;
    let newly_bound = workspace.bind_runtime_program_identity(program_identity.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.runtime-math-program-binding.v1\",",
            "\"program_plan_version\":{},",
            "\"program_identity\":{},",
            "\"identity_policy\":\"exact_program_identity\",",
            "\"identity_source\":\"canonical_plan_replay\",",
            "\"newly_bound\":{},",
            "\"binding_count\":{},",
            "\"mutation\":\"runtime_program_binding_metadata_only\"",
            "}}"
        ),
        version,
        program_identity,
        newly_bound,
        workspace.runtime_program_binding_count(),
    ))
}

fn workspace_verify_math_program_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    inputs: &[WasmTensor],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
    context: &str,
) -> Result<String, String> {
    validate_label(&label, context)?;
    if !(1..=2).contains(&inputs.len()) {
        return Err(format!(
            "{context}: verifier supports exactly 1 or 2 external inputs, got {}",
            inputs.len()
        ));
    }

    let (identity_version, program_identity) = math_program_identity_from_plan(plan)?;

    if workspace.runtime_subject_binding().is_some()
        && !workspace.runtime_program_identity_bound(&program_identity)
    {
        return Err(format!(
            "{context}: MathProgram programIdentity is not bound to this runtime subject; bind the canonical plan with workspaceBindRuntimeMathProgramPlan first"
        ));
    }

    let (run_version, run_identity, reference) = run_math_program_plan(plan, inputs)?;
    debug_assert_eq!(identity_version, run_version);
    debug_assert_eq!(program_identity, run_identity);

    let report = verify_vectors_metrics(&reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();

    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensors_fingerprint(inputs);
    let reference_fingerprint = f32_fingerprint(&reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_verifier",
        "MathProgram.verifyFlat",
        "burn_math_program",
        &label,
        "input_fingerprint",
        &input_fingerprint,
        &candidate_fingerprint,
        Some(&program_identity_fingerprint),
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "MathProgram.verifyFlat".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"wasm_verifier\",",
            "\"verifier\":\"MathProgram.verifyFlat\",",
            "\"reference_authority\":\"burn_math_program\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"program_plan_version\":{},",
            "\"program_identity\":{},",
            "\"program_identity_fingerprint\":\"{}\",",
            "\"mutable_state_in_program_identity\":false,",
            "\"runtime_subject\":{},",
            "\"input_count\":{},",
            "\"input_fingerprint\":\"{}\",",
            "\"reference_fingerprint\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\",",
            "\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(&label),
        run_version,
        program_identity,
        json_escape(&program_identity_fingerprint),
        runtime_subject_binding_json(workspace),
        inputs.len(),
        json_escape(&input_fingerprint),
        json_escape(&reference_fingerprint),
        json_escape(&candidate_fingerprint),
        tolerances_json(abs_tol, rel_tol),
        result_json,
    ))
}

/// Replay a canonical 1-input MathProgram as the independent numerical reference.
#[wasm_bindgen(js_name = workspaceVerifyMathProgram1Receipt)]
pub fn workspace_verify_math_program_1_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_math_program_receipt(
        workspace,
        plan,
        &[input.clone()],
        candidate,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyMathProgram1Receipt",
    )
}

/// Replay a canonical 2-input MathProgram as the independent numerical reference.
#[wasm_bindgen(js_name = workspaceVerifyMathProgram2Receipt)]
pub fn workspace_verify_math_program_2_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    lhs: &WasmTensor,
    rhs: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_math_program_receipt(
        workspace,
        plan,
        &[lhs.clone(), rhs.clone()],
        candidate,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyMathProgram2Receipt",
    )
}


/// Return proof-related workspace collections without merging their authority classes.
#[wasm_bindgen(js_name = workspaceProofLedger)]
pub fn workspace_proof_ledger(workspace: &AgentWorkspace) -> String {
    let legacy = workspace.query("_proofs".into(), None, None);
    let attestations = workspace.query("_attestations".into(), None, None);
    let receipts = workspace.query("_verifier_receipts".into(), None, None);
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.proof-ledger.v1\",",
            "\"legacy_recordProof_authority\":\"caller_controlled_legacy\",",
            "\"legacy_proofs\":{},",
            "\"attestations\":{},",
            "\"verifier_receipts\":{}",
            "}}"
        ),
        legacy,
        attestations,
        receipts,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        workspace_proof_ledger, workspace_record_attestation, workspace_verify_graph_receipt,
        workspace_verify_vector_receipt,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::{workspace_compile, workspace_init_unary};
    use crate::WasmTensor;

    #[test]
    fn caller_attestation_is_not_verifier_authority() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let value = workspace_record_attestation(
            &mut workspace,
            "self-claim".into(),
            true,
            "not independently checked".into(),
        )
        .unwrap();
        assert!(value.contains("\"authority\":\"caller_attestation\""));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("caller_attestation"));
        assert!(!ledger.contains("\"authority\":\"wasm_verifier\""));
    }

    #[test]
    fn vector_receipt_preserves_caller_reference_authority() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let receipt = workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0, 2.0],
            &[1.0, 2.0],
            0.0,
            0.0,
            "vector-check".into(),
        )
        .unwrap();
        assert!(receipt.contains("\"authority\":\"wasm_comparator\""));
        assert!(receipt.contains("\"reference_authority\":\"caller_supplied\""));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn graph_receipt_is_burn_backed_and_failed_comparison_is_still_recorded() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        let mut registry = LayerRegistry::new();

        let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let spec = AgentLayerSpec::relu(id);
        let out = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu".into(),
        )
        .unwrap();
        let graph = workspace_compile(&builder, &registry, out).unwrap();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);

        let receipt = workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &[0.0, 4.0],
            0.0,
            0.0,
            "graph-check".into(),
        )
        .unwrap();

        assert!(receipt.contains("\"authority\":\"wasm_verifier\""));
        assert!(receipt.contains("\"reference_authority\":\"burn_compiled_graph\""));
        assert!(receipt.contains("\"passed\":false"));
        assert!(receipt.contains("\"program_identity\":{"));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("CompiledGraph.verifyFlat"));
        assert!(ledger.contains("\"state\":\"failed\""));
    }

    #[test]
    fn malformed_verification_does_not_consume_receipt_id() {
        let mut workspace = AgentWorkspace::new(2).unwrap();

        assert!(workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0],
            &[f32::NAN],
            0.0,
            0.0,
            "bad".into(),
        )
        .is_err());

        let good = workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0],
            &[1.0],
            0.0,
            0.0,
            "good".into(),
        )
        .unwrap();
        assert!(good.contains("\"receipt_id\":1"));
    }
}
