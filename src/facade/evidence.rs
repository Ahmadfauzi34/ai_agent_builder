//! Fasad WASM tunggal — domain `evidence` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::evidence::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::coprocessor::verify_vectors_metrics;
use crate::evidence::program_bundle::checked_u32;
use crate::evidence::program_bundle::decode_multi_input_program_bundle;
use crate::evidence::program_bundle::decode_program_bundle;
use crate::evidence::program_bundle::multi_input_referenced_layer_keys;
use crate::evidence::program_bundle::parse_init_fingerprint;
use crate::evidence::program_bundle::push_u32;
use crate::evidence::program_bundle::referenced_layer_keys;
use crate::evidence::program_bundle::BUNDLE_FLAG_STATE_INCLUDED;
use crate::evidence::program_bundle::BUNDLE_MAGIC;
use crate::evidence::program_bundle::BUNDLE_SCHEMA_VERSION;
use crate::evidence::program_bundle::MULTI_INPUT_BUNDLE_MAGIC;
use crate::evidence::proof_provenance::bytes_fingerprint;
use crate::evidence::proof_provenance::direct_math_reference_identity;
use crate::evidence::proof_provenance::f32_fingerprint;
use crate::evidence::proof_provenance::json_escape;
use crate::evidence::proof_provenance::ledger_receipt_json;
use crate::evidence::proof_provenance::math_program_identity_from_plan;
use crate::evidence::proof_provenance::semantic_graph_ledger_receipt_json;
use crate::evidence::proof_provenance::tensor_fingerprint;
use crate::evidence::proof_provenance::tolerances_json;
use crate::evidence::proof_provenance::validate_label;
use crate::evidence::proof_provenance::workspace_verify_direct_math_receipt;
use crate::evidence::proof_provenance::workspace_verify_math_program_receipt;
use crate::evidence::proof_provenance::MATH_PROOF_V1;
use crate::evidence::proof_provenance::MAX_ATTESTATION_DETAIL_BYTES;
use crate::evidence::proof_provenance::PROOF_PROVENANCE_V1;
use crate::evidence::runtime_resolution_evidence::RUNTIME_RESOLUTION_EVIDENCE_V1;
use crate::graph::CompiledGraph;
use crate::graph::CompiledMultiInputGraph;
use crate::math::math_check_operation;
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::protocol::PacketHeader;
use crate::protocol::OP_INIT;
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::semantic_execution_context::semantic_execution_context_for;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

#[wasm_bindgen(js_name = programBundleCapabilities)]
pub fn program_bundle_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.program-bundle.v1\",",
        "\"schema_version\":1,",
        "\"export\":\"exportProgramBundle\",",
        "\"import\":\"importProgramBundle\",",
        "\"structural_identity\":\"burn-research.program-identity.v1\",",
        "\"structural_source\":\"program-identity.v1_layer_init_fingerprint\",",
        "\"target_registry\":\"atomic_replace_on_success\",",
        "\"import_commit\":\"atomic_after_identity_validation\",",
        "\"mutable_state\":\"optional_separate_section\"",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = exportProgramBundle)]
pub fn export_program_bundle(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    include_state: bool,
) -> Result<Vec<u8>, String> {
    graph
        .validate_registry_binding(registry)
        .map_err(|error| format!("exportProgramBundle: {error}"))?;

    let plan = graph.program_plan();
    let identity = graph.program_identity();
    let keys = referenced_layer_keys(&plan)?;
    let mut records = Vec::with_capacity(keys.len());
    for (layer_type, layer_id) in keys {
        let fingerprint = registry
            .layer_init_fingerprint(layer_type, layer_id)
            .map_err(|error| format!("exportProgramBundle: {error}"))?;
        let (variant, flags, init_payload) =
            parse_init_fingerprint(&fingerprint, layer_type, layer_id)?;
        let state = if include_state {
            registry
                .get_layer_state(layer_id, layer_type)
                .map_err(|error| format!("exportProgramBundle: state for type 0x{layer_type:02X} id {layer_id}: {error}"))?
        } else {
            Vec::new()
        };
        records.push((layer_type, variant, flags, layer_id, init_payload, state));
    }

    let mut out = Vec::new();
    out.extend_from_slice(BUNDLE_MAGIC);
    push_u32(&mut out, BUNDLE_SCHEMA_VERSION);
    push_u32(
        &mut out,
        if include_state {
            BUNDLE_FLAG_STATE_INCLUDED
        } else {
            0
        },
    );
    push_u32(
        &mut out,
        checked_u32(plan.len(), "exportProgramBundle plan")?,
    );
    push_u32(
        &mut out,
        checked_u32(identity.len(), "exportProgramBundle identity")?,
    );
    push_u32(
        &mut out,
        checked_u32(records.len(), "exportProgramBundle layer count")?,
    );
    out.extend_from_slice(&plan);
    out.extend_from_slice(identity.as_bytes());

    for (layer_type, variant, flags, layer_id, init_payload, state) in records {
        out.push(layer_type);
        out.push(variant);
        out.push(flags);
        out.push(0);
        push_u32(&mut out, layer_id);
        push_u32(
            &mut out,
            checked_u32(init_payload.len(), "exportProgramBundle init payload")?,
        );
        push_u32(
            &mut out,
            checked_u32(state.len(), "exportProgramBundle layer state")?,
        );
        out.extend_from_slice(&init_payload);
        out.extend_from_slice(&state);
    }
    Ok(out)
}

#[wasm_bindgen(js_name = importProgramBundle)]
pub fn import_program_bundle(
    registry: &mut LayerRegistry,
    bundle: &[u8],
) -> Result<CompiledGraph, String> {
    let decoded =
        decode_program_bundle(bundle).map_err(|error| format!("importProgramBundle: {error}"))?;
    let mut staged = LayerRegistry::new();
    for (index, layer) in decoded.layers.iter().enumerate() {
        let payload_len =
            checked_u32(layer.init_payload.len(), "importProgramBundle init payload")?;
        let header = PacketHeader {
            opcode: OP_INIT,
            layer_type: layer.layer_type,
            variant: layer.variant,
            flags: layer.flags,
            payload_len,
        };
        staged
            .init_layer(&header, &layer.init_payload)
            .map_err(|error| format!("importProgramBundle: layer {index} init failed: {error}"))?;
        if decoded.state_included {
            staged
                .load_layer_state(layer.layer_id, layer.layer_type, &layer.state)
                .map_err(|error| {
                    format!("importProgramBundle: layer {index} state failed: {error}")
                })?;
        }
    }

    let graph = staged
        .compile_graph(&decoded.plan)
        .map_err(|error| format!("importProgramBundle: compile failed: {error}"))?;
    let actual_identity = graph.program_identity();
    if actual_identity != decoded.expected_identity {
        return Err(format!(
            "importProgramBundle: structural identity mismatch: expected {}, got {}",
            decoded.expected_identity, actual_identity
        ));
    }
    graph
        .validate_registry_binding(&staged)
        .map_err(|error| format!("importProgramBundle: staged binding invalid: {error}"))?;

    *registry = staged;
    Ok(graph)
}

#[wasm_bindgen(js_name = multiInputProgramBundleCapabilities)]
pub fn multi_input_program_bundle_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.multi-input-program-bundle.v1\",",
        "\"schema_version\":1,",
        "\"export\":\"exportMultiInputProgramBundle\",",
        "\"import\":\"importMultiInputProgramBundle\",",
        "\"structural_identity\":\"burn-research.multi-input-program-identity.v1\",",
        "\"structural_source\":\"exact_multi_input_plan_and_layer_init_fingerprints\",",
        "\"mutable_state\":\"optional_separate_layer_state_section\",",
        "\"state_integrity\":\"no_signature_or_authentication\",",
        "\"target_registry\":\"atomic_replace_on_success\",",
        "\"import_commit\":\"atomic_after_identity_validation\",",
        "\"authorization\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = exportMultiInputProgramBundle)]
pub fn export_multi_input_program_bundle(
    graph: &CompiledMultiInputGraph,
    registry: &LayerRegistry,
    include_state: bool,
) -> Result<Vec<u8>, String> {
    graph
        .validate_registry_binding(registry)
        .map_err(|error| format!("exportMultiInputProgramBundle: {error}"))?;

    let plan = graph.input_plan_v1();
    let identity = graph.program_identity();
    let keys = multi_input_referenced_layer_keys(&plan)?;
    let mut records = Vec::with_capacity(keys.len());
    for (layer_type, layer_id) in keys {
        let fingerprint = registry
            .layer_init_fingerprint(layer_type, layer_id)
            .map_err(|error| format!("exportMultiInputProgramBundle: {error}"))?;
        let (variant, flags, init_payload) =
            parse_init_fingerprint(&fingerprint, layer_type, layer_id)?;
        let state = if include_state {
            registry
                .get_layer_state(layer_id, layer_type)
                .map_err(|error| format!("exportMultiInputProgramBundle: state for type 0x{layer_type:02X} id {layer_id}: {error}"))?
        } else {
            Vec::new()
        };
        records.push((layer_type, variant, flags, layer_id, init_payload, state));
    }

    let mut out = Vec::new();
    out.extend_from_slice(MULTI_INPUT_BUNDLE_MAGIC);
    push_u32(&mut out, BUNDLE_SCHEMA_VERSION);
    push_u32(
        &mut out,
        if include_state {
            BUNDLE_FLAG_STATE_INCLUDED
        } else {
            0
        },
    );
    push_u32(
        &mut out,
        checked_u32(plan.len(), "exportMultiInputProgramBundle plan")?,
    );
    push_u32(
        &mut out,
        checked_u32(identity.len(), "exportMultiInputProgramBundle identity")?,
    );
    push_u32(
        &mut out,
        checked_u32(records.len(), "exportMultiInputProgramBundle layer count")?,
    );
    out.extend_from_slice(&plan);
    out.extend_from_slice(identity.as_bytes());
    for (layer_type, variant, flags, layer_id, init_payload, state) in records {
        out.push(layer_type);
        out.push(variant);
        out.push(flags);
        out.push(0);
        push_u32(&mut out, layer_id);
        push_u32(
            &mut out,
            checked_u32(
                init_payload.len(),
                "exportMultiInputProgramBundle init payload",
            )?,
        );
        push_u32(
            &mut out,
            checked_u32(state.len(), "exportMultiInputProgramBundle layer state")?,
        );
        out.extend_from_slice(&init_payload);
        out.extend_from_slice(&state);
    }
    Ok(out)
}

#[wasm_bindgen(js_name = importMultiInputProgramBundle)]
pub fn import_multi_input_program_bundle(
    registry: &mut LayerRegistry,
    bundle: &[u8],
) -> Result<crate::graph::CompiledMultiInputGraph, String> {
    let decoded = decode_multi_input_program_bundle(bundle)
        .map_err(|error| format!("importMultiInputProgramBundle: {error}"))?;
    let input_plan = MultiInputGraphPlan::from_bytes(&decoded.plan)
        .map_err(|error| format!("importMultiInputProgramBundle: {error}"))?;

    let mut staged = LayerRegistry::new();
    for (index, layer) in decoded.layers.iter().enumerate() {
        let payload_len = checked_u32(
            layer.init_payload.len(),
            "importMultiInputProgramBundle init payload",
        )?;
        let header = PacketHeader {
            opcode: OP_INIT,
            layer_type: layer.layer_type,
            variant: layer.variant,
            flags: layer.flags,
            payload_len,
        };
        staged
            .init_layer(&header, &layer.init_payload)
            .map_err(|error| {
                format!("importMultiInputProgramBundle: layer {index} init failed: {error}")
            })?;
        if decoded.state_included {
            staged
                .load_layer_state(layer.layer_id, layer.layer_type, &layer.state)
                .map_err(|error| {
                    format!("importMultiInputProgramBundle: layer {index} state failed: {error}")
                })?;
        }
    }

    let graph = staged
        .compile_multi_input_graph(&input_plan)
        .map_err(|error| format!("importMultiInputProgramBundle: compile failed: {error}"))?;
    let actual_identity = graph.program_identity();
    if actual_identity != decoded.expected_identity {
        return Err(format!(
            "importMultiInputProgramBundle: structural identity mismatch: expected {}, got {}",
            decoded.expected_identity, actual_identity
        ));
    }
    graph.validate_registry_binding(&staged).map_err(|error| {
        format!("importMultiInputProgramBundle: staged binding invalid: {error}")
    })?;

    *registry = staged;
    Ok(graph)
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

/// Verify a Burn graph while binding the current semantic graph/lifecycle identities
/// into the returned proof context. Numerical authority remains CompiledGraph.verifyFlat.
#[wasm_bindgen(js_name = workspaceVerifySemanticGraphReceipt)]
pub fn workspace_verify_semantic_graph_receipt(
    workspace: &mut AgentWorkspace,
    builder: &AgentGraphBuilder,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceVerifySemanticGraphReceipt")?;

    let semantic_context = semantic_execution_context_for(builder, graph, registry)?;
    let program_identity = graph.program_identity();
    workspace.require_runtime_program_identity_if_bound(
        &program_identity,
        "workspaceVerifySemanticGraphReceipt",
    )?;

    let reference = graph.run(registry, input)?.to_array();
    let report = verify_vectors_metrics(&reference, candidate, abs_tol, rel_tol)?;

    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensor_fingerprint(input);
    let reference_fingerprint = f32_fingerprint(&reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();
    let runtime_subject = runtime_subject_binding_json(workspace);
    let receipt_id = workspace.next_verifier_receipt_id();

    let compact = semantic_graph_ledger_receipt_json(
        receipt_id,
        &label,
        &input_fingerprint,
        &candidate_fingerprint,
        &program_identity_fingerprint,
        &semantic_context.context_fingerprint,
        &runtime_subject,
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
            "\"semantic_execution_context\":{},",
            "\"semantic_context_fingerprint\":\"{}\",",
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
        runtime_subject,
        semantic_context.json(),
        json_escape(&semantic_context.context_fingerprint),
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

/// Bind the canonical one-step MathProgramV9 reference identity for a direct math operation.
///
/// This does not execute the direct operation. It validates metadata through mathCheckOperation,
/// builds the verifier-owned V9 reference program, and binds only that exact replayable identity
/// to the already-bound runtime subject.
#[wasm_bindgen(js_name = workspaceBindRuntimeDirectMathOperation)]
pub fn workspace_bind_runtime_direct_math_operation(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<String, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceBindRuntimeDirectMathOperation: runtime subject must be bound first"
                .to_string(),
        );
    }

    if operation_id == "linalg.cosine_similarity" && f32_params.len() != 1 {
        return Err(
            "workspaceBindRuntimeDirectMathOperation: cosine verification requires explicit f32_params=[epsilon]"
                .to_string(),
        );
    }

    let preflight = math_check_operation(
        operation_id.clone(),
        lhs_shape,
        rhs_shape,
        u32_params,
        f32_params,
    )?;
    if !preflight.contains("\"status\":\"admissible\"") {
        return Err(format!(
            "workspaceBindRuntimeDirectMathOperation: operation metadata rejected: {preflight}"
        ));
    }

    let (_, program_identity) =
        direct_math_reference_identity(&operation_id, u32_params, f32_params)?;
    let newly_bound = workspace.bind_runtime_program_identity(program_identity.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.runtime-direct-math-binding.v1\",",
            "\"operation_id\":\"{}\",",
            "\"candidate_authority\":\"burn_direct_math\",",
            "\"reference_authority\":\"burn_math_program\",",
            "\"reference_program_generation\":\"v9\",",
            "\"reference_program_identity\":{},",
            "\"identity_source\":\"canonical_operation_to_math_program_v9\",",
            "\"newly_bound\":{},",
            "\"binding_count\":{},",
            "\"mutation\":\"runtime_program_binding_metadata_only\"",
            "}}"
        ),
        json_escape(&operation_id),
        program_identity,
        newly_bound,
        workspace.runtime_program_binding_count(),
    ))
}

/// Execute a canonical unary direct math operation and compare it to a verifier-owned V9 reference.
#[wasm_bindgen(js_name = workspaceVerifyDirectMath1Receipt)]
pub fn workspace_verify_direct_math_1_receipt(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    input: &WasmTensor,
    u32_params: &[u32],
    f32_params: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_direct_math_receipt(
        workspace,
        &operation_id,
        &[input.clone()],
        u32_params,
        f32_params,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyDirectMath1Receipt",
    )
}

/// Execute a canonical binary direct math operation and compare it to a verifier-owned V9 reference.
#[wasm_bindgen(js_name = workspaceVerifyDirectMath2Receipt)]
pub fn workspace_verify_direct_math_2_receipt(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    lhs: &WasmTensor,
    rhs: &WasmTensor,
    u32_params: &[u32],
    f32_params: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_direct_math_receipt(
        workspace,
        &operation_id,
        &[lhs.clone(), rhs.clone()],
        u32_params,
        f32_params,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyDirectMath2Receipt",
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
        legacy, attestations, receipts,
    )
}

#[wasm_bindgen(js_name = runtimeResolutionEvidenceCapabilities)]
pub fn runtime_resolution_evidence_capabilities() -> String {
    RUNTIME_RESOLUTION_EVIDENCE_V1.to_string()
}
