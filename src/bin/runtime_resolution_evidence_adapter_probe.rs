#[cfg(not(target_arch = "wasm32"))]
use std::io::{self, Read};

#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::RuntimeEvidence;

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
    let subject = evidence.subject();

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
            "subject_bound": subject.is_some(),
            "subject": subject.map(|subject| serde_json::json!({
                "intent_id": subject.intent_id,
                "workflow_revision": subject.workflow_revision,
                "approval_id": subject.approval_id,
                "subject_kind": subject.subject_kind,
                "subject_identity": subject.subject_identity,
                "authorization_policy_id": subject.authorization_policy_id,
                "authorization_policy_revision": subject.authorization_policy_revision,
                "authorization_is_revision": subject.authorization_is_revision,
            })),
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
