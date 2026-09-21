use burn_research::authorization::AuthorizationPolicy;
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration,
};
use burn_research::interaction_fault::interaction_check_compile;
use burn_research::registry::LayerRegistry;
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
use burn_research::resolution_subject::SubjectBoundReviewSession;
use burn_research::runtime_resolution_evidence::{
    runtime_resolution_evidence_capabilities, RejoinStatus, ResolutionEvidenceInbox,
    RuntimeEvidence,
};
use burn_research::agent::AgentGraphBuilder;

fn forward_bridge_fixture() -> (
    burn_research::resolution::ResolutionSnapshot,
    RuntimeSubjectProjection,
) {
    let spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "feature_transform").unwrap(),
        SpecDeclaration::new("planner.note", "agent_selects_graph").unwrap(),
    ])
    .unwrap();

    let subject = spec.approval_subject().unwrap();
    let mut review =
        SubjectBoundReviewSession::new("intent-evidence-rejoin", subject).unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve("owner").unwrap();
    let resolution_snapshot = review.snapshot().review.workflow;

    let approved = ApprovedEffectiveSpec::bind_root(spec, approval.clone()).unwrap();
    let policy = AuthorizationPolicy::new("runtime-policy", 11, "owner", vec![]).unwrap();
    let authorization = policy.authorize(&approval).unwrap();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

    assert_eq!(resolution_snapshot.intent_id, projection.intent_id);
    assert_eq!(
        resolution_snapshot.revision,
        projection.workflow_revision
    );
    assert!(resolution_snapshot.compile_eligible());

    (resolution_snapshot, projection)
}

#[test]
fn capability_contract_advertises_direct_math_reverse_adapter() {
    let caps: serde_json::Value =
        serde_json::from_str(&runtime_resolution_evidence_capabilities()).unwrap();

    assert_eq!(
        caps["structured_adapters"]["direct_math_verifier_receipt"]["constructor"],
        "RuntimeEvidence::from_direct_math_verifier_receipt_json"
    );
    assert_eq!(
        caps["structured_adapters"]["direct_math_verifier_receipt"]["authority"],
        "wasm_verifier source label / burn_direct_math candidate / burn_math_program reference / observation_only transport"
    );
}

#[test]
fn actual_forward_projection_rejoins_to_exact_resolution_snapshot() {
    let (snapshot, projection) = forward_bridge_fixture();
    let before = snapshot.clone();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        1,
        "graph-proof",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"relu\"}",
        true,
        "reference matched",
    )
    .unwrap();

    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
    assert!(inbox.record(evidence).unwrap());
    assert_eq!(inbox.len(), 1);

    // The reverse bridge owns only its inbox. The already-resolved workflow
    // snapshot is observationally unchanged.
    assert_eq!(snapshot, before);

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["target"]["intent_id"], "intent-evidence-rejoin");
    assert_eq!(json["summary"]["graph_verifier_passed"], 1);
    assert_eq!(json["resolution_effect"]["diagnostic_created"], false);
    assert_eq!(json["resolution_effect"]["state_transition"], "none");
    assert_eq!(json["resolution_effect"]["revision_created"], false);
}

#[test]
fn structured_agent_fault_fields_rejoin_without_legacy_error_parsing() {
    let (snapshot, projection) = forward_bridge_fixture();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    // No graph step writes slot 1, so the runtime preflight emits a structured
    // AgentFault envelope. The host transports its fields directly.
    let raw = interaction_check_compile(&builder, &registry, 1);
    let envelope: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(envelope["status"], "fault");

    let evidence =
        RuntimeEvidence::from_bound_agent_fault_envelope(&projection, &raw).unwrap();

    assert_eq!(evidence.source_authority(), "agent_fault_preflight");
    assert_eq!(evidence.evidence_authority(), "observation_only");
    assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
    assert!(inbox.record(evidence).unwrap());

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["summary"]["agent_faults"], 1);
    assert_eq!(
        json["entries"][0]["payload"]["code"],
        "E_OUTPUT_NOT_WRITTEN"
    );
    assert_eq!(json["resolution_effect"]["interpretation_required"], true);
}

#[test]
fn failed_graph_receipt_does_not_demote_resolved_state() {
    let (snapshot, projection) = forward_bridge_fixture();
    let before = snapshot.clone();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let failed = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        7,
        "candidate-mismatch",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"linear\"}",
        false,
        "max_abs_error exceeded tolerance",
    )
    .unwrap();

    assert!(inbox.record(failed).unwrap());
    assert_eq!(snapshot, before);
    assert!(snapshot.compile_eligible());

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["summary"]["graph_verifier_failed"], 1);
    assert_eq!(json["resolution_effect"]["diagnostic_created"], false);
    assert_eq!(json["resolution_effect"]["action_selected"], false);
}

#[test]
fn stale_projection_cannot_receive_current_resolution_evidence() {
    let (snapshot, projection) = forward_bridge_fixture();
    let mut stale = projection.clone();
    stale.workflow_revision = stale.workflow_revision.saturating_sub(1);

    let error = ResolutionEvidenceInbox::new(&snapshot, &stale).unwrap_err();
    assert!(error.contains("revision"));
}

#[test]
fn canonical_inbox_revalidates_current_authorization_policy() {
    let spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "feature_transform").unwrap(),
    ])
    .unwrap();
    let subject = spec.approval_subject().unwrap();

    let mut review =
        SubjectBoundReviewSession::new("intent-canonical-evidence", subject).unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve("owner").unwrap();
    let resolution = review.snapshot().review.workflow;

    let approved = ApprovedEffectiveSpec::bind_root(spec, approval.clone()).unwrap();
    let policy = AuthorizationPolicy::new("runtime-policy", 21, "owner", vec![]).unwrap();
    let authorization = policy.authorize(&approval).unwrap();

    let inbox = ResolutionEvidenceInbox::from_authorized(
        &resolution,
        &approved,
        &policy,
        &authorization,
    )
    .unwrap();
    assert!(inbox.is_empty());

    let newer_policy =
        AuthorizationPolicy::new("runtime-policy", 22, "owner", vec![]).unwrap();
    let error = ResolutionEvidenceInbox::from_authorized(
        &resolution,
        &approved,
        &newer_policy,
        &authorization,
    )
    .unwrap_err();
    assert!(
        error.contains("stale")
            || error.contains("policy")
            || error.contains("authorization")
    );
}


#[test]
fn graph_receipt_adapter_accepts_exact_wasm_receipt_shape_without_host_field_mapping() {
    let (snapshot, projection) = forward_bridge_fixture();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 41,
        "authority": "wasm_verifier",
        "verifier": "CompiledGraph.verifyFlat",
        "reference_authority": "burn_compiled_graph",
        "label": "graph-check",
        "fingerprint_algorithm": "fnv1a64_noncryptographic",
        "program_identity": {
            "schema": "burn-research.program-identity.v1",
            "plan_hex": "010000000300000001040100000000000101",
            "layer_init_fingerprints": [
                "type=04;id=1;variant=01;flags=00;payload=01000000"
            ]
        },
        "program_identity_fingerprint": "fnv1a64:9ad0e8000d20be73",
        "mutable_state_in_program_identity": false,
        "runtime_subject": {
            "status": "bound",
            "intent_id": projection.intent_id.clone(),
            "workflow_revision": projection.workflow_revision,
            "approval_id": projection.approval_id.clone(),
            "subject_kind": projection.subject_kind.clone(),
            "subject_identity": projection.subject_identity.clone(),
            "authorization_policy_id": projection.authorization_policy_id.clone(),
            "authorization_policy_revision": projection.authorization_policy_revision,
            "authorization_is_revision": projection.authorization_is_revision
        },
        "input_fingerprint": "fnv1a64:ce407a98af1ab857",
        "reference_fingerprint": "fnv1a64:a8c83832281aa685",
        "candidate_fingerprint": "fnv1a64:a8c83832281aa685",
        "tolerances": {"abs": 0.0, "rel": 0.0},
        "result": {
            "passed": true,
            "len": 2,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "rmse": 0.0,
            "first_failure": null
        }
    })
    .to_string();

    let evidence = RuntimeEvidence::from_graph_verifier_receipt_json(&receipt).unwrap();
    assert_eq!(evidence.kind(), "graph_verifier_receipt");
    assert_eq!(evidence.outcome(), "passed");
    assert_eq!(evidence.source_authority(), "wasm_verifier");
    assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
    assert!(inbox.record(evidence).unwrap());

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["entries"][0]["payload"]["receipt_id"], 41);
    assert_eq!(json["entries"][0]["payload"]["passed"], true);
    assert!(json["entries"][0]["payload"]["detail"]
        .as_str()
        .unwrap()
        .contains("\"max_abs_error\":0.0"));
}

#[test]
fn direct_math_receipt_adapter_preserves_dual_runtime_authority_and_exact_subject() {
    let (snapshot, projection) = forward_bridge_fixture();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();
    let before = snapshot.clone();

    let receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 43,
        "authority": "wasm_verifier",
        "verifier": "DirectMath.verifyAgainstMathProgramV9",
        "candidate_authority": "burn_direct_math",
        "reference_authority": "burn_math_program",
        "reference_program_generation": "v9",
        "operation_id": "numeric.add",
        "label": "direct-add",
        "program_identity": {
            "schema": "burn-research.math-program-identity.v1",
            "plan_hex": "42524d5009"
        },
        "runtime_subject": {
            "status": "bound",
            "intent_id": projection.intent_id.clone(),
            "workflow_revision": projection.workflow_revision,
            "approval_id": projection.approval_id.clone(),
            "subject_kind": projection.subject_kind.clone(),
            "subject_identity": projection.subject_identity.clone(),
            "authorization_policy_id": projection.authorization_policy_id.clone(),
            "authorization_policy_revision": projection.authorization_policy_revision,
            "authorization_is_revision": projection.authorization_is_revision
        },
        "result": {
            "passed": true,
            "len": 2,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "rmse": 0.0,
            "first_failure": null
        }
    })
    .to_string();

    let evidence =
        RuntimeEvidence::from_direct_math_verifier_receipt_json(&receipt).unwrap();
    assert_eq!(evidence.kind(), "direct_math_verifier_receipt");
    assert_eq!(evidence.source_authority(), "wasm_verifier");
    assert_eq!(evidence.evidence_authority(), "observation_only");
    assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
    assert!(inbox.record(evidence).unwrap());
    assert_eq!(snapshot, before);

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["summary"]["direct_math_verifier_passed"], 1);
    assert_eq!(
        json["entries"][0]["payload"]["candidate_authority"],
        "burn_direct_math"
    );
    assert_eq!(
        json["entries"][0]["payload"]["reference_authority"],
        "burn_math_program"
    );
    assert_eq!(
        json["entries"][0]["payload"]["reference_program_generation"],
        "v9"
    );
    assert!(json["entries"][0]["payload"]["reference_program_identity"]
        .as_str()
        .unwrap()
        .contains("burn-research.math-program-identity.v1"));
    assert!(json["entries"][0]["payload"].get("program_identity").is_none());
    assert_eq!(json["resolution_effect"]["state_transition"], "none");
}

#[test]
fn failed_direct_math_receipt_remains_observation_only() {
    let (snapshot, projection) = forward_bridge_fixture();
    let before = snapshot.clone();
    let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 44,
        "authority": "wasm_verifier",
        "verifier": "DirectMath.verifyAgainstMathProgramV9",
        "candidate_authority": "burn_direct_math",
        "reference_authority": "burn_math_program",
        "reference_program_generation": "v9",
        "operation_id": "numeric.add",
        "label": "direct-add-mismatch",
        "program_identity": {
            "schema": "burn-research.math-program-identity.v1",
            "plan_hex": "42524d5009"
        },
        "runtime_subject": {
            "status": "bound",
            "intent_id": projection.intent_id.clone(),
            "workflow_revision": projection.workflow_revision,
            "approval_id": projection.approval_id.clone(),
            "subject_kind": projection.subject_kind.clone(),
            "subject_identity": projection.subject_identity.clone(),
            "authorization_policy_id": projection.authorization_policy_id.clone(),
            "authorization_policy_revision": projection.authorization_policy_revision,
            "authorization_is_revision": projection.authorization_is_revision
        },
        "result": {
            "passed": false,
            "len": 2,
            "max_abs_error": 1.0,
            "max_rel_error": 1.0,
            "rmse": 0.7071067811865476,
            "first_failure": 0
        }
    })
    .to_string();

    let evidence =
        RuntimeEvidence::from_direct_math_verifier_receipt_json(&receipt).unwrap();
    assert_eq!(evidence.outcome(), "failed");
    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
    assert!(inbox.record(evidence).unwrap());

    let json: serde_json::Value = serde_json::from_str(&inbox.to_json()).unwrap();
    assert_eq!(json["summary"]["direct_math_verifier_failed"], 1);
    assert_eq!(json["resolution_effect"]["diagnostic_created"], false);
    assert_eq!(json["resolution_effect"]["state_transition"], "none");
    assert_eq!(json["resolution_effect"]["revision_created"], false);
    assert_eq!(json["resolution_effect"]["action_selected"], false);
    assert_eq!(json["resolution_effect"]["interpretation_required"], true);
    assert_eq!(snapshot, before);
    assert!(snapshot.compile_eligible());
}

#[test]
fn vector_receipt_adapter_preserves_caller_reference_authority_and_bound_subject() {
    let (snapshot, projection) = forward_bridge_fixture();
    let inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

    let receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 42,
        "authority": "wasm_comparator",
        "verifier": "mathVerifyVectors",
        "reference_authority": "caller_supplied",
        "label": "vector-check",
        "runtime_subject": {
            "status": "bound",
            "intent_id": projection.intent_id.clone(),
            "workflow_revision": projection.workflow_revision,
            "approval_id": projection.approval_id.clone(),
            "subject_kind": projection.subject_kind.clone(),
            "subject_identity": projection.subject_identity.clone(),
            "authorization_policy_id": projection.authorization_policy_id.clone(),
            "authorization_policy_revision": projection.authorization_policy_revision,
            "authorization_is_revision": projection.authorization_is_revision
        },
        "reference_fingerprint": "fnv1a64:097a69ee2da301d8",
        "candidate_fingerprint": "fnv1a64:097a69ee2da301d8",
        "tolerances": {"abs": 0.0, "rel": 0.0},
        "result": {
            "passed": true,
            "len": 2,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "rmse": 0.0,
            "first_failure": null
        }
    })
    .to_string();

    let evidence = RuntimeEvidence::from_vector_verifier_receipt_json(&receipt).unwrap();
    assert_eq!(evidence.kind(), "vector_verifier_receipt");
    assert_eq!(evidence.source_authority(), "wasm_comparator");
    assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
}

#[test]
fn receipt_adapter_rejects_authority_class_escalation() {
    let (_, projection) = forward_bridge_fixture();

    let fake_graph = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 7,
        "authority": "wasm_comparator",
        "verifier": "CompiledGraph.verifyFlat",
        "reference_authority": "burn_compiled_graph",
        "label": "forged-class",
        "program_identity": {
            "schema": "burn-research.program-identity.v1",
            "plan_hex": "00",
            "layer_init_fingerprints": []
        },
        "runtime_subject": {
            "status": "bound",
            "intent_id": projection.intent_id.clone(),
            "workflow_revision": projection.workflow_revision,
            "approval_id": projection.approval_id.clone(),
            "subject_kind": projection.subject_kind.clone(),
            "subject_identity": projection.subject_identity.clone(),
            "authorization_policy_id": projection.authorization_policy_id.clone(),
            "authorization_policy_revision": projection.authorization_policy_revision,
            "authorization_is_revision": projection.authorization_is_revision
        },
        "result": {"passed": true}
    })
    .to_string();

    let error = RuntimeEvidence::from_graph_verifier_receipt_json(&fake_graph).unwrap_err();
    assert!(error.contains("authority"));
    assert!(error.contains("wasm_verifier"));
}

#[test]
fn unbound_receipt_stays_unbound_after_structured_adaptation() {
    let receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 3,
        "authority": "wasm_comparator",
        "verifier": "mathVerifyVectors",
        "reference_authority": "caller_supplied",
        "label": "unbound-vector",
        "runtime_subject": {"status": "unbound"},
        "result": {"passed": false}
    })
    .to_string();

    let evidence = RuntimeEvidence::from_vector_verifier_receipt_json(&receipt).unwrap();
    assert!(evidence.subject().is_none());
}
