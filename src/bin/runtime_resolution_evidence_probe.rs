use burn_research::authorization::AuthorizationPolicy;
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration,
};
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
use burn_research::resolution_subject::SubjectBoundReviewSession;
use burn_research::runtime_resolution_evidence::{
    RejoinStatus, ResolutionEvidenceInbox, RuntimeEvidence, RuntimeEvidenceSubject,
    MAX_RUNTIME_EVIDENCE_ENTRIES,
};

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{{\"schema\":\"burn-research.runtime-resolution-evidence-native-probe.v1\",\"status\":\"failed\",\"message\":\"{}\"}}",
        message.as_ref().replace('\\', "\\\\").replace('"', "\\\"")
    );
    std::process::exit(1);
}

fn ensure(condition: bool, message: &str) {
    if !condition {
        fail(message);
    }
}

fn main() {
    if let Err(error) = run() {
        fail(error);
    }
}

fn run() -> Result<(), String> {
    let spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "feature_transform")?,
        SpecDeclaration::new("planner.note", "agent_selects_graph")?,
    ])?;
    let subject = spec.approval_subject()?;

    let mut review =
        SubjectBoundReviewSession::new("intent-runtime-evidence-probe", subject)?;
    review.submit("agent")?;
    let approval = review.approve("owner")?;
    let resolution = review.snapshot().review.workflow;
    ensure(resolution.compile_eligible(), "resolution must be compile eligible");

    let approved = ApprovedEffectiveSpec::bind_root(spec, approval.clone())?;
    let policy = AuthorizationPolicy::new("runtime-policy", 13, "owner", vec![])?;
    let authorization = policy.authorize(&approval)?;
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization)?;

    ensure(
        resolution.intent_id == projection.intent_id
            && resolution.revision == projection.workflow_revision,
        "forward projection drifted from resolution snapshot",
    );

    let before_resolution = resolution.clone();
    let mut inbox = ResolutionEvidenceInbox::from_authorized(
        &resolution,
        &approved,
        &policy,
        &authorization,
    )?;

    let fault_envelope = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.agent-fault.v1",
        "status": "fault",
        "fault": {
            "code": "E_LAYOUT_PREFLIGHT",
            "class": "semantic_precondition",
            "operation": "workspaceInitUnary",
            "predicate": "layout.edge_not_known_incompatible",
            "expected": "compatible|unknown",
            "actual": "known_incompatible",
            "mutation": "none",
            "recoverable": true,
            "suggested_actions": ["choose_compatible_layer"],
            "message": "known incompatible layout"
        }
    })
    .to_string();
    let fault =
        RuntimeEvidence::from_bound_agent_fault_envelope(&projection, &fault_envelope)?;
    ensure(
        inbox.classify(&fault) == RejoinStatus::Exact,
        "exact AgentFault failed rejoin classification",
    );
    ensure(inbox.record(fault.clone())?, "first fault insert was not recorded");
    ensure(!inbox.record(fault)?, "duplicate fault was not idempotent");

    let verifier_receipt = serde_json::json!({
        "schema_version": 1,
        "schema_id": "burn-research.verifier-receipt.v1",
        "receipt_id": 1,
        "authority": "wasm_verifier",
        "verifier": "CompiledGraph.verifyFlat",
        "reference_authority": "burn_compiled_graph",
        "label": "graph-mismatch",
        "fingerprint_algorithm": "fnv1a64_noncryptographic",
        "program_identity": {
            "schema": "burn-research.program-identity.v1",
            "plan_hex": "probe",
            "layer_init_fingerprints": []
        },
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
        "input_fingerprint": "fnv1a64:probe-input",
        "reference_fingerprint": "fnv1a64:probe-reference",
        "candidate_fingerprint": "fnv1a64:probe-candidate",
        "tolerances": {"abs": 0.0, "rel": 0.0},
        "result": {
            "passed": false,
            "len": 2,
            "max_abs_error": 1.0,
            "max_rel_error": 0.25,
            "rmse": 0.7071067811865476,
            "first_failure": 1
        }
    })
    .to_string();
    let verifier = RuntimeEvidence::from_graph_verifier_receipt_json(&verifier_receipt)?;
    ensure(
        verifier.source_authority() == "wasm_verifier"
            && verifier.evidence_authority() == "observation_only"
            && verifier.transport_integrity() == "host_structured_unverified"
            && verifier.outcome() == "failed"
            && verifier.kind() == "graph_verifier_receipt",
        "graph verifier source-authority/transport semantics drift",
    );
    ensure(inbox.record(verifier)?, "graph verifier evidence was not recorded");
    ensure(
        resolution == before_resolution && resolution.compile_eligible(),
        "recording runtime evidence mutated Resolution state",
    );

    let mut stale_subject = RuntimeEvidenceSubject::from_projection(&projection);
    stale_subject.workflow_revision = stale_subject.workflow_revision.saturating_sub(1);
    let stale = RuntimeEvidence::agent_fault(
        Some(stale_subject),
        "E_STALE",
        "control_precondition",
        "runtimeEvidence",
        "revision.match",
        true,
        "stale runtime revision",
    )?;
    ensure(
        inbox.classify(&stale) == RejoinStatus::StaleRuntimeRevision,
        "stale revision was not classified",
    );
    ensure(inbox.record(stale).is_err(), "stale evidence entered inbox");

    let mut foreign_subject = RuntimeEvidenceSubject::from_projection(&projection);
    foreign_subject.subject_identity = "foreign-spec".to_string();
    let foreign = RuntimeEvidence::agent_fault(
        Some(foreign_subject),
        "E_FOREIGN",
        "control_precondition",
        "runtimeEvidence",
        "subject.match",
        true,
        "foreign subject",
    )?;
    ensure(
        inbox.classify(&foreign) == RejoinStatus::SubjectMismatch,
        "foreign subject was not classified",
    );
    ensure(inbox.record(foreign).is_err(), "foreign subject evidence entered inbox");

    let unbound = RuntimeEvidence::agent_fault(
        None,
        "E_UNBOUND",
        "control_precondition",
        "runtimeEvidence",
        "subject.bound",
        true,
        "unbound runtime",
    )?;
    ensure(
        inbox.classify(&unbound) == RejoinStatus::Unbound,
        "unbound evidence was not classified",
    );
    ensure(inbox.record(unbound).is_err(), "unbound evidence entered inbox");

    for index in inbox.len()..MAX_RUNTIME_EVIDENCE_ENTRIES {
        let evidence = RuntimeEvidence::bound_agent_fault(
            &projection,
            format!("E_FILL_{index}"),
            "control_precondition",
            "runtimeEvidence",
            "bounded",
            true,
            format!("fill-{index}"),
        )?;
        ensure(inbox.record(evidence)?, "bounded evidence fill failed early");
    }
    ensure(
        inbox.len() == MAX_RUNTIME_EVIDENCE_ENTRIES,
        "inbox did not reach declared evidence bound",
    );

    let overflow = RuntimeEvidence::bound_agent_fault(
        &projection,
        "E_OVERFLOW",
        "resource_precondition",
        "runtimeEvidence",
        "inbox.capacity",
        true,
        "overflow",
    )?;
    ensure(inbox.record(overflow).is_err(), "evidence inbox accepted overflow");

    let summary = inbox.to_json();
    ensure(
        summary.contains("\"diagnostic_created\":false")
            && summary.contains("\"state_transition\":\"none\"")
            && summary.contains("\"revision_created\":false")
            && summary.contains("\"interpretation_required\":true"),
        "inbox overclaimed a Resolution effect",
    );

    println!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.runtime-resolution-evidence-native-probe.v1\",",
            "\"status\":\"passed\",",
            "\"forward_projection_match\":true,",
            "\"agent_fault_rejoin\":true,",
            "\"structured_payload_adaptation\":true,",
            "\"verifier_source_authority_preserved\":true,",
            "\"transport_observation_only\":true,",
            "\"resolution_state_unchanged\":true,",
            "\"stale_revision_rejected\":true,",
            "\"foreign_subject_rejected\":true,",
            "\"unbound_rejected\":true,",
            "\"duplicate_idempotent\":true,",
            "\"bounded_inbox\":true,",
            "\"entry_count\":{}",
            "}}"
        ),
        inbox.len(),
    );

    Ok(())
}
