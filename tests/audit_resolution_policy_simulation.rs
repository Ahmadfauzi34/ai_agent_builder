mod resolution_subject {
    pub use burn_research::resolution_subject::*;
}

#[path = "../src/effective_spec.rs"]
mod effective_spec;

use std::collections::BTreeSet;

use burn_research::resolution::{
    DiagnosticKind, ResolutionDiagnostic, ResolutionResponse,
};
use burn_research::resolution_review::{ApprovalSnapshot, ResolutionReviewSession};
use burn_research::resolution_subject::SubjectBoundReviewSession;
use effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration, SpecDirective,
};
use serde_json::json;

#[derive(Debug)]
struct Finding {
    id: &'static str,
    score: u8,
    status: &'static str,
    evidence: String,
    recommendation: &'static str,
}

#[derive(Debug)]
struct CustomerAuthorityPolicy {
    owner: String,
    delegates: BTreeSet<String>,
}

impl CustomerAuthorityPolicy {
    fn new(owner: impl Into<String>, delegates: impl IntoIterator<Item = String>) -> Self {
        Self {
            owner: owner.into(),
            delegates: delegates.into_iter().collect(),
        }
    }

    fn authorizes(&self, approval: &ApprovalSnapshot) -> bool {
        approval.approver == self.owner || self.delegates.contains(&approval.approver)
    }
}

#[derive(Clone, Debug)]
struct InheritRemainderCommand {
    parent_spec_identity: String,
    parent_approval_id: String,
}

fn ambiguity_diagnostic() -> ResolutionDiagnostic {
    ResolutionDiagnostic::blocking(
        "E-MATH-001",
        DiagnosticKind::MultipleInterpretations,
        "normalization",
        "multiple valid normalization semantics exist",
        "choose one normalization semantic",
        vec!["l2".to_string(), "probability".to_string()],
    )
    .unwrap()
}

fn build_review(final_approver: &str) -> ApprovalSnapshot {
    let mut review = ResolutionReviewSession::new(format!("intent-policy-{final_approver}")).unwrap();
    review.submit("agent").unwrap();
    review
        .request_diagnostic("validator", ambiguity_diagnostic())
        .unwrap();
    review
        .respond(
            "agent",
            ResolutionResponse::new("E-MATH-001", "l2").unwrap(),
        )
        .unwrap();
    review.approve(final_approver).unwrap()
}

fn directive_key(directive: &SpecDirective) -> &str {
    match directive {
        SpecDirective::Inherit { key }
        | SpecDirective::Override { key, .. }
        | SpecDirective::Remove { key }
        | SpecDirective::Declare { key, .. } => key,
    }
}

fn expand_explicit_inherit_remainder(
    parent: &ApprovedEffectiveSpec,
    command: &InheritRemainderCommand,
    mut explicit: Vec<SpecDirective>,
) -> Result<Vec<SpecDirective>, String> {
    if command.parent_spec_identity != parent.spec.identity {
        return Err("bulk policy: parent spec identity mismatch".to_string());
    }
    if command.parent_approval_id != parent.approval_id() {
        return Err("bulk policy: parent approval id mismatch".to_string());
    }

    let mut decided = BTreeSet::new();
    for directive in &explicit {
        let key = directive_key(directive).to_string();
        if !decided.insert(key.clone()) {
            return Err(format!("bulk policy: duplicate explicit decision: {key}"));
        }
    }

    for field in &parent.spec.fields {
        if !decided.contains(&field.key) {
            explicit.push(SpecDirective::inherit(field.key.clone())?);
        }
    }
    Ok(explicit)
}

fn approved_large_parent() -> ApprovedEffectiveSpec {
    let declarations = (0..64)
        .map(|index| {
            SpecDeclaration::new(format!("k{index:02}"), format!("v{index:02}")).unwrap()
        })
        .collect::<Vec<_>>();
    let spec = EffectiveSpec::root(declarations).unwrap();
    let mut review =
        SubjectBoundReviewSession::new("intent-bulk-parent", spec.approval_subject().unwrap())
            .unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve("customer").unwrap();
    ApprovedEffectiveSpec::bind_root(spec, approval).unwrap()
}

#[test]
fn audit_candidate_customer_authority_and_bulk_policy() {
    let mut findings = Vec::<Finding>::new();

    // 1) Existing core still accepts an agent self-approval; candidate policy rejects it externally.
    {
        let policy = CustomerAuthorityPolicy::new("customer", Vec::<String>::new());
        let agent_approval = build_review("agent");
        assert_eq!(agent_approval.approver, "agent");
        assert!(!policy.authorizes(&agent_approval));

        let customer_approval = build_review("customer");
        assert!(policy.authorizes(&customer_approval));

        findings.push(Finding {
            id: "customer_final_authority_can_be_layered_above_core",
            score: 0,
            status: "feasible_without_core_weakening",
            evidence: "the current compiler-like review core can stay unchanged while a separate policy rejects agent self-approval and accepts customer final approval".to_string(),
            recommendation: "PROTOTYPE_SEPARATE_AUTHORIZATION_LAYER",
        });
    }

    // 2) Explicit delegation can be represented without giving every agent blanket approval authority.
    {
        let policy = CustomerAuthorityPolicy::new(
            "customer",
            vec!["customer-delegate".to_string()],
        );
        let delegated = build_review("customer-delegate");
        let unrelated = build_review("agent");
        assert!(policy.authorizes(&delegated));
        assert!(!policy.authorizes(&unrelated));

        findings.push(Finding {
            id: "authority_can_be_delegated_explicitly",
            score: 1,
            status: "natural_policy_extension",
            evidence: "an explicit customer delegate can authorize while an unrelated agent remains non-authoritative".to_string(),
            recommendation: "BIND_DELEGATION_TO_INTENT_OR_SUBJECT_PROVENANCE",
        });
    }

    // 3) Simulate an explicit identity-bound `inherit remainder` command as a desugaring layer.
    let parent = approved_large_parent();
    {
        let command = InheritRemainderCommand {
            parent_spec_identity: parent.spec.identity.clone(),
            parent_approval_id: parent.approval_id().to_string(),
        };
        let expanded = expand_explicit_inherit_remainder(
            &parent,
            &command,
            vec![SpecDirective::override_value("k00", "changed").unwrap()],
        )
        .unwrap();
        assert_eq!(expanded.len(), 64);

        let bulk_spec = parent
            .materialize_child(expanded)
            .unwrap()
            .resolved()
            .unwrap();

        let mut manual = vec![SpecDirective::override_value("k00", "changed").unwrap()];
        for index in 1..64 {
            manual.push(SpecDirective::inherit(format!("k{index:02}")).unwrap());
        }
        let manual_spec = parent
            .materialize_child(manual)
            .unwrap()
            .resolved()
            .unwrap();

        assert_eq!(bulk_spec.identity, manual_spec.identity);
        assert_eq!(bulk_spec.fields, manual_spec.fields);
        assert_eq!(bulk_spec.changes, manual_spec.changes);

        findings.push(Finding {
            id: "explicit_bulk_inherit_can_desugar_to_existing_core",
            score: 0,
            status: "feasible_without_implicit_inheritance",
            evidence: "one explicit override plus one identity-bound inherit-remainder command can expand to the same 64 explicit directives and produce the exact same canonical EffectiveSpec identity as manual decisions".to_string(),
            recommendation: "PROTOTYPE_AS_EXPLICIT_COMMUNICATION_LAYER",
        });
    }

    // 4) The bulk command must fail closed if replayed against a different parent identity or approval.
    {
        let wrong_identity = InheritRemainderCommand {
            parent_spec_identity: "wrong-spec".to_string(),
            parent_approval_id: parent.approval_id().to_string(),
        };
        assert!(expand_explicit_inherit_remainder(
            &parent,
            &wrong_identity,
            vec![SpecDirective::override_value("k00", "changed").unwrap()],
        )
        .is_err());

        let wrong_approval = InheritRemainderCommand {
            parent_spec_identity: parent.spec.identity.clone(),
            parent_approval_id: "wrong-approval".to_string(),
        };
        assert!(expand_explicit_inherit_remainder(
            &parent,
            &wrong_approval,
            vec![SpecDirective::override_value("k00", "changed").unwrap()],
        )
        .is_err());

        findings.push(Finding {
            id: "bulk_policy_is_exact_parent_bound",
            score: 0,
            status: "natural",
            evidence: "bulk inheritance simulation rejects both parent-spec identity mismatch and parent-approval mismatch before directive expansion".to_string(),
            recommendation: "KEEP_IF_PRODUCTIZED",
        });
    }

    // 5) The communication layer should reduce customer choices, not proof detail.
    {
        let manual_customer_decisions = 64usize;
        let bulk_customer_decisions = 2usize; // override k00 + explicit inherit-remainder command
        assert!(bulk_customer_decisions < manual_customer_decisions);
        assert_eq!(manual_customer_decisions - bulk_customer_decisions, 62);

        findings.push(Finding {
            id: "bulk_policy_reduces_interaction_not_proof",
            score: 0,
            status: "ergonomic_gain",
            evidence: "the 64-field one-override case drops from 64 customer-level decisions to 2 explicit communication acts while the core still receives all 64 directives".to_string(),
            recommendation: "PREFER_DESUGARING_OVER_IMPLICIT_DEFAULTS",
        });
    }

    let report = json!({
        "verdict": "PASS_WITH_FINDINGS",
        "task": "Candidate authorization and explicit bulk-policy simulation",
        "summary": {
            "findingCount": findings.len(),
            "productionSemanticsChanged": false,
            "coreProofWeakened": false,
        },
        "findings": findings.iter().map(|finding| json!({
            "id": finding.id,
            "score": finding.score,
            "status": finding.status,
            "evidence": finding.evidence,
            "recommendation": finding.recommendation,
        })).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}