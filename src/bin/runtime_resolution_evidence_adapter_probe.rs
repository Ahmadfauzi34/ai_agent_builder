#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Read};

#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution::ResolutionWorkflow;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{
    RejoinStatus, ResolutionEvidenceInbox, RuntimeEvidence,
};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = run() {
        eprintln!(
            "{}",
            serde_json::json!({
                "schema": "burn-research.runtime-resolution-evidence-adapter-probe.v1",
                "status": "failed",
                "error": error,
            })
        );
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<(), String> {
    let mut raw = String::new();
    io::stdin()
        .read_to_string(&mut raw)
        .map_err(|error| format!("stdin read failed: {error}"))?;
    if raw.trim().is_empty() {
        return Err("stdin must contain one MathProgram verifier receipt JSON object".to_string());
    }

    let evidence = RuntimeEvidence::from_math_program_verifier_receipt_json(raw.trim())?;
    let subject = evidence
        .subject()
        .ok_or_else(|| "adapter probe requires a bound runtime_subject".to_string())?
        .clone();

    let mut workflow = ResolutionWorkflow::new(subject.intent_id.clone())?;
    workflow.submit()?;
    workflow.finalize_resolution()?;
    let resolution = workflow.snapshot();
    if resolution.revision != subject.workflow_revision {
        return Err(format!(
            "adapter probe resolved workflow revision {} does not match receipt workflow_revision {}; generate the audit receipt for revision {}",
            resolution.revision,
            subject.workflow_revision,
            resolution.revision
        ));
    }
    let before_resolution = resolution.clone();

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: subject.intent_id.clone(),
        workflow_revision: subject.workflow_revision,
        approval_id: subject.approval_id.clone(),
        subject_kind: subject.subject_kind.clone(),
        subject_identity: subject.subject_identity.clone(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: subject.subject_identity.clone(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: subject.authorization_policy_id.clone(),
        authorization_policy_revision: subject.authorization_policy_revision,
        authorization_is_revision: subject.authorization_is_revision,
        approver: "adapter-probe".to_string(),
        fields: Vec::new(),
    };

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let rejoin_status = inbox.classify(&evidence);
    if rejoin_status != RejoinStatus::Exact {
        return Err(format!(
            "adapter probe expected exact rejoin, got {}",
            rejoin_status.as_str()
        ));
    }
    let recorded = inbox.record(evidence.clone())?;
    if !recorded || inbox.len() != 1 {
        return Err("adapter probe failed to record exact evidence".to_string());
    }
    if resolution != before_resolution {
        return Err("adapter probe evidence recording mutated Resolution state".to_string());
    }

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.runtime-resolution-evidence-adapter-probe.v1",
            "status": "passed",
            "kind": evidence.kind(),
            "source_authority": evidence.source_authority(),
            "evidence_authority": evidence.evidence_authority(),
            "transport_integrity": evidence.transport_integrity(),
            "outcome": evidence.outcome(),
            "subject_bound": true,
            "subject": {
                "intent_id": subject.intent_id,
                "workflow_revision": subject.workflow_revision,
                "approval_id": subject.approval_id,
                "subject_kind": subject.subject_kind,
                "subject_identity": subject.subject_identity,
                "authorization_policy_id": subject.authorization_policy_id,
                "authorization_policy_revision": subject.authorization_policy_revision,
                "authorization_is_revision": subject.authorization_is_revision,
            },
            "rejoin_status": rejoin_status.as_str(),
            "inbox_recorded": recorded,
            "inbox_entry_count": inbox.len(),
            "resolution_unchanged": true,
            "resolution_effect": {
                "diagnostic_created": false,
                "state_transition": "none",
                "revision_created": false,
                "action_selected": false,
                "interpretation_required": true
            }
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
