use burn_research::authorization::AuthorizationPolicy;
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration,
};
use burn_research::interaction_fault::interaction_check_compile;
use burn_research::registry::LayerRegistry;
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
use burn_research::resolution_subject::SubjectBoundReviewSession;
use burn_research::runtime_resolution_evidence::{
    RejoinStatus, ResolutionEvidenceInbox, RuntimeEvidence,
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

    let fault = &envelope["fault"];
    let evidence = RuntimeEvidence::bound_agent_fault(
        &projection,
        fault["code"].as_str().unwrap(),
        fault["class"].as_str().unwrap(),
        fault["operation"].as_str().unwrap(),
        fault["predicate"].as_str().unwrap(),
        fault["recoverable"].as_bool().unwrap(),
        fault["message"].as_str().unwrap(),
    )
    .unwrap();

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
