#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution::ResolutionWorkflow;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_dispatch_requirements::response_dispatch_requirements;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::EvidenceResponseAction;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{
    ResolutionEvidenceInbox, RuntimeEvidence,
};

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-dispatch-requirements-probe.v1",
            "status": "failed",
            "message": message.as_ref(),
        })
    );
    std::process::exit(1);
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure(condition: bool, message: &str) {
    if !condition {
        fail(message);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn projection(
    intent_id: &str,
    workflow_revision: u64,
    subject_identity: &str,
) -> RuntimeSubjectProjection {
    RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: intent_id.to_string(),
        workflow_revision,
        approval_id: "approval-dispatch-requirements-probe".to_string(),
        subject_kind: "effective-spec".to_string(),
        subject_identity: subject_identity.to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: subject_identity.to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-dispatch-requirements-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "probe".to_string(),
        fields: Vec::new(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = run() {
        fail(error);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<(), String> {
    let mut workflow = ResolutionWorkflow::new("intent-dispatch-requirements-probe")?;
    workflow.submit()?;
    workflow.finalize_resolution()?;
    let resolution = workflow.snapshot();
    let before_resolution = resolution.clone();

    let subject = projection(
        &resolution.intent_id,
        resolution.revision,
        "spec-dispatch-requirements-probe",
    );

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &subject)?;
    let failed = RuntimeEvidence::bound_graph_verifier_receipt(
        &subject,
        1,
        "failed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
        false,
        "candidate mismatch",
    )?;
    ensure(inbox.record(failed)?, "failed graph evidence was not recorded");
    let before_len = inbox.len();

    let reverify_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent-requirements-probe",
    )?;
    let reverify = response_dispatch_requirements(&inbox, &reverify_intent);
    ensure(reverify.ready, "reverify requirements were not ready");
    ensure(
        reverify.executor_contract.as_deref() == Some("CompiledGraph.verifyFlat"),
        "reverify executor contract drift",
    );
    ensure(
        reverify
            .requirements
            .iter()
            .any(|item| item.name == "compiled_graph" && item.required),
        "reverify compiled_graph requirement missing",
    );
    ensure(
        reverify
            .requirements
            .iter()
            .any(|item| item.name == "candidate" && item.required),
        "reverify candidate requirement missing",
    );
    ensure(
        reverify.requirements_fingerprint.is_some(),
        "reverify requirements fingerprint missing",
    );
    ensure(
        !reverify.execution_authorized,
        "requirements projection must not authorize reverify",
    );

    let revision_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent-requirements-probe",
    )?;
    let revision = response_dispatch_requirements(&inbox, &revision_intent);
    ensure(revision.ready, "revision requirements were not ready");
    ensure(
        revision.executor_contract.as_deref() == Some("ResolutionRevisionChain"),
        "revision executor contract drift",
    );
    ensure(
        revision.operation.as_deref() == Some("open_revision"),
        "revision operation drift",
    );
    ensure(
        revision
            .requirements
            .iter()
            .any(|item| item.name == "revision_key" && item.required),
        "required revision_key missing",
    );
    ensure(
        revision
            .requirements
            .iter()
            .any(|item| item.name == "parent_revision_id" && !item.required),
        "optional parent_revision_id missing",
    );
    ensure(
        !revision.execution_authorized,
        "requirements projection must not authorize revision",
    );

    let ignore_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Ignore,
        "agent-requirements-probe",
    )?;
    let ignore = response_dispatch_requirements(&inbox, &ignore_intent);
    ensure(ignore.ready, "ignore requirements were not ready");
    ensure(
        ignore.payload_mode.as_deref() == Some("none") && ignore.requirements.is_empty(),
        "ignore must have explicit empty payload",
    );

    let mut changed_inbox = ResolutionEvidenceInbox::new(&resolution, &subject)?;
    let passed = RuntimeEvidence::bound_graph_verifier_receipt(
        &subject,
        2,
        "passed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"passed\"}",
        true,
        "match",
    )?;
    ensure(changed_inbox.record(passed)?, "changed evidence was not recorded");
    let stale = response_dispatch_requirements(&changed_inbox, &reverify_intent);
    ensure(!stale.ready, "stale response intent reopened requirements");
    ensure(
        stale.requirements.is_empty() && stale.requirements_fingerprint.is_none(),
        "closed requirements must expose no executable input profile",
    );
    ensure(
        !stale.execution_authorized,
        "closed requirements unexpectedly authorize execution",
    );

    ensure(inbox.len() == before_len, "requirements projection mutated inbox");
    ensure(
        resolution == before_resolution,
        "requirements projection mutated Resolution snapshot",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-dispatch-requirements-probe.v1",
            "status": "passed",
            "reverify": {
                "executor_contract": reverify.executor_contract,
                "operation": reverify.operation,
                "requirements_count": reverify.requirements.len(),
                "requirements_fingerprint": reverify.requirements_fingerprint,
                "execution_authorized": reverify.execution_authorized,
            },
            "propose_revision": {
                "executor_contract": revision.executor_contract,
                "operation": revision.operation,
                "requirements_count": revision.requirements.len(),
                "execution_authorized": revision.execution_authorized,
            },
            "ignore": {
                "payload_mode": ignore.payload_mode,
                "requirements_count": ignore.requirements.len(),
            },
            "stale_intent_closed": !stale.ready,
            "inbox_unchanged": inbox.len() == before_len,
            "resolution_unchanged": resolution == before_resolution,
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
