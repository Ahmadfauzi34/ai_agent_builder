mod resolution_subject {
    pub use burn_research::resolution_subject::*;
}

#[path = "../src/effective_spec.rs"]
mod effective_spec;

use burn_research::resolution::{
    DiagnosticKind, ResolutionDiagnostic, ResolutionResponse, ResolutionState, ResolutionWorkflow,
};
use burn_research::resolution_subject::{
    SubjectBoundReviewSession, SubjectBoundReviewSnapshot, SubjectBoundRevisionChain,
};
use effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, EffectiveSpecMaterialization, SpecDeclaration,
    SpecDirective,
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

fn approve_root(
    intent_id: &str,
    spec: EffectiveSpec,
    approver: &str,
) -> (ApprovedEffectiveSpec, SubjectBoundReviewSnapshot) {
    let mut review =
        SubjectBoundReviewSession::new(intent_id, spec.approval_subject().unwrap()).unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve(approver).unwrap();
    let snapshot = review.snapshot();
    let approved = ApprovedEffectiveSpec::bind_root(spec, approval).unwrap();
    (approved, snapshot)
}

#[test]
fn audit_resolution_and_effective_spec_communication_boundaries() {
    let mut findings = Vec::<Finding>::new();

    // 1) Ambiguity behaves like a blocking compiler diagnostic until explicitly answered.
    {
        let mut workflow = ResolutionWorkflow::new("intent-audit-ambiguity").unwrap();
        workflow.submit().unwrap();
        workflow.add_diagnostic(ambiguity_diagnostic()).unwrap();
        assert_eq!(workflow.state(), ResolutionState::NeedsResolution);
        assert!(!workflow.compile_eligible());
        assert!(workflow.finalize_resolution().is_err());

        workflow
            .apply_response(ResolutionResponse::new("E-MATH-001", "l2").unwrap())
            .unwrap();
        assert!(!workflow.compile_eligible());
        workflow.finalize_resolution().unwrap();
        assert!(workflow.compile_eligible());

        findings.push(Finding {
            id: "ambiguity_requires_explicit_resolution",
            score: 0,
            status: "natural",
            evidence: "blocking diagnostic prevents compile eligibility until an explicit allowed response is applied and resolution is finalized".to_string(),
            recommendation: "KEEP",
        });
    }

    // 2) Current actor values are provenance labels, not authorization. Demonstrate the gap explicitly.
    {
        let spec = EffectiveSpec::root(vec![SpecDeclaration::new("mode", "stable").unwrap()]).unwrap();
        let mut review = SubjectBoundReviewSession::new(
            "intent-self-approved",
            spec.approval_subject().unwrap(),
        )
        .unwrap();
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
        let approval = review.approve("agent").unwrap();
        assert!(review.compile_eligible());
        assert!(ApprovedEffectiveSpec::bind_root(spec, approval).is_ok());

        findings.push(Finding {
            id: "actor_authorization_policy_absent",
            score: 4,
            status: "hard_gap_for_customer_authority",
            evidence: "the same actor label 'agent' can submit, answer the ambiguity, approve the exact subject, and reach compile eligibility; actor names are provenance only".to_string(),
            recommendation: "EXTEND_ABOVE_CORE_WITH_AUTHORIZATION_POLICY",
        });
    }

    // 3) A large parent spec makes omission visible as communication, never implicit inheritance.
    let (approved_large, large_root_snapshot) = {
        let declarations = (0..64)
            .map(|index| {
                SpecDeclaration::new(format!("k{index:02}"), format!("v{index:02}"))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let root = EffectiveSpec::root(declarations).unwrap();
        approve_root("intent-large-root", root, "customer")
    };

    {
        let sparse = approved_large
            .materialize_child(vec![
                SpecDirective::override_value("k00", "changed").unwrap(),
            ])
            .unwrap();
        let diagnostics = match sparse {
            EffectiveSpecMaterialization::NeedsResolution(state) => state.diagnostics,
            EffectiveSpecMaterialization::Resolved(_) => {
                panic!("sparse child unexpectedly inherited omitted parent fields")
            }
        };
        assert_eq!(diagnostics.len(), 63);
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic.code == "E-SPEC-001"
                && diagnostic.candidates
                    == vec![
                        "inherit".to_string(),
                        "override".to_string(),
                        "remove".to_string(),
                    ]
        }));

        findings.push(Finding {
            id: "omission_becomes_resolution_request",
            score: 0,
            status: "natural",
            evidence: "overriding 1 of 64 parent fields produces 63 E-SPEC-001 diagnostics with explicit inherit/override/remove choices; nothing is inherited silently".to_string(),
            recommendation: "KEEP",
        });

        findings.push(Finding {
            id: "large_spec_explicitness_burden",
            score: 3,
            status: "interaction_burden",
            evidence: "a one-field change to a 64-field parent requires 64 explicit child decisions before materialization can resolve".to_string(),
            recommendation: "MEASURE_EXPLICIT_BULK_POLICY_WITHOUT_IMPLICIT_INHERITANCE",
        });
    }

    // 4) Explicit decisions remain deterministic regardless of directive order.
    {
        let mut directives = Vec::new();
        directives.push(SpecDirective::override_value("k00", "changed").unwrap());
        for index in 1..64 {
            directives.push(SpecDirective::inherit(format!("k{index:02}")).unwrap());
        }

        let forward = approved_large
            .materialize_child(directives.clone())
            .unwrap()
            .resolved()
            .unwrap();
        directives.reverse();
        let reverse = approved_large
            .materialize_child(directives)
            .unwrap()
            .resolved()
            .unwrap();
        assert_eq!(forward.identity, reverse.identity);
        assert_eq!(forward.fields, reverse.fields);
        assert_eq!(forward.changes, reverse.changes);

        findings.push(Finding {
            id: "directive_order_canonicalization",
            score: 0,
            status: "natural",
            evidence: "the same 64 explicit decisions in reverse order produce the same fields, changes, and canonical effective-spec identity".to_string(),
            recommendation: "KEEP",
        });
    }

    // 5) Exact approval subject prevents swapping a sibling spec under another approval.
    {
        let root = EffectiveSpec::root(vec![SpecDeclaration::new("mode", "stable").unwrap()]).unwrap();
        let (approved_root, root_snapshot) = approve_root("intent-binding-root", root, "customer");

        let child_a = approved_root
            .materialize_child(vec![SpecDirective::override_value("mode", "a").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        let child_b = approved_root
            .materialize_child(vec![SpecDirective::override_value("mode", "b").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();

        let mut chain = SubjectBoundRevisionChain::from_approved_root(root_snapshot).unwrap();
        let revision_a = chain
            .open_revision(None, "child-a", child_a.approval_subject().unwrap())
            .unwrap();
        chain.submit_revision(&revision_a, "agent").unwrap();
        let approval_a = chain.approve_revision(&revision_a, "customer").unwrap();

        assert!(ApprovedEffectiveSpec::bind_revision(child_b, approval_a.clone()).is_err());
        assert!(ApprovedEffectiveSpec::bind_revision(child_a, approval_a).is_ok());

        findings.push(Finding {
            id: "exact_subject_binding",
            score: 0,
            status: "natural",
            evidence: "an approval for child A cannot bind child B even when both share the same approved parent and review workflow".to_string(),
            recommendation: "KEEP",
        });
    }

    // 6) Parent-approval lineage is independently enforced even when the subject identity matches.
    {
        let root = EffectiveSpec::root(vec![SpecDeclaration::new("mode", "stable").unwrap()]).unwrap();
        let (approved_root, root_snapshot) = approve_root("intent-lineage-root", root, "customer");

        let child = approved_root
            .materialize_child(vec![SpecDirective::override_value("mode", "child").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();

        let mut chain = SubjectBoundRevisionChain::from_approved_root(root_snapshot).unwrap();
        let child_revision = chain
            .open_revision(None, "child", child.approval_subject().unwrap())
            .unwrap();
        chain.submit_revision(&child_revision, "agent").unwrap();
        let child_approval = chain
            .approve_revision(&child_revision, "customer")
            .unwrap();
        let approved_child = ApprovedEffectiveSpec::bind_revision(child, child_approval).unwrap();

        let grandchild = approved_child
            .materialize_child(vec![SpecDirective::inherit("mode").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();

        // Deliberately approve the grandchild subject as a sibling of child (wrong parent lineage).
        let wrong_lineage_revision = chain
            .open_revision(
                None,
                "grandchild-wrong-lineage",
                grandchild.approval_subject().unwrap(),
            )
            .unwrap();
        chain
            .submit_revision(&wrong_lineage_revision, "agent")
            .unwrap();
        let wrong_lineage_approval = chain
            .approve_revision(&wrong_lineage_revision, "customer")
            .unwrap();

        let err = ApprovedEffectiveSpec::bind_revision(grandchild, wrong_lineage_approval)
            .unwrap_err();
        assert!(err.contains("parent approval mismatch"));

        findings.push(Finding {
            id: "parent_approval_lineage_binding",
            score: 0,
            status: "natural",
            evidence: "matching subject identity is insufficient when revision approval points at the wrong parent approval lineage".to_string(),
            recommendation: "KEEP",
        });
    }

    // 7) Terminal resolution states stay closed and non-compile-eligible.
    {
        for terminal in ["contradiction", "unsupported", "invalid"] {
            let mut workflow = ResolutionWorkflow::new(format!("intent-{terminal}")).unwrap();
            workflow.submit().unwrap();
            match terminal {
                "contradiction" => workflow.mark_contradiction("conflicting constraints").unwrap(),
                "unsupported" => workflow.mark_unsupported("capability absent").unwrap(),
                "invalid" => workflow.mark_invalid("malformed request").unwrap(),
                _ => unreachable!(),
            }
            assert!(!workflow.compile_eligible());
            assert!(workflow.finalize_resolution().is_err());
        }

        findings.push(Finding {
            id: "terminal_states_are_closed",
            score: 0,
            status: "natural",
            evidence: "Contradiction, Unsupported, and Invalid remain distinct terminal states and never become compile eligible".to_string(),
            recommendation: "KEEP",
        });
    }

    // 8) Resolution/specification layers must not import or invoke Math Program execution.
    {
        let sources = [
            include_str!("../src/resolution.rs"),
            include_str!("../src/resolution_review.rs"),
            include_str!("../src/resolution_revision.rs"),
            include_str!("../src/resolution_subject.rs"),
            include_str!("../src/effective_spec.rs"),
        ];
        let forbidden = [
            "use crate::math",
            "crate::math::",
            "MathProgramBuilder",
            "MathProgramV4Builder",
            ".run1(",
            ".run2(",
        ];
        for source in sources {
            for needle in forbidden {
                assert!(
                    !source.contains(needle),
                    "resolution/effective-spec layer contains forbidden execution coupling: {needle}"
                );
            }
        }

        findings.push(Finding {
            id: "no_execution_authority_coupling",
            score: 0,
            status: "natural",
            evidence: "Resolution, review, revision, subject-binding, and EffectiveSpec sources contain no Math Program builder import or run1/run2 execution call".to_string(),
            recommendation: "KEEP",
        });
    }

    // 9) Capacity is explicit: 64 fields accepted, 65 rejected. Do not relax without evidence.
    {
        let too_many = (0..65)
            .map(|index| {
                SpecDeclaration::new(format!("z{index:02}"), format!("v{index:02}"))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(EffectiveSpec::root(too_many).is_err());

        findings.push(Finding {
            id: "effective_spec_field_limit_64",
            score: 2,
            status: "explicit_capacity_limit",
            evidence: "64 fields are accepted and 65 are rejected; this is a bounded capacity rule rather than an ambiguity rule".to_string(),
            recommendation: "MEASURE_BEFORE_RELAX",
        });
    }

    // Keep the prior large-root snapshot alive as an integration sanity check.
    assert!(large_root_snapshot.compile_eligible());

    let hard = findings
        .iter()
        .filter(|finding| finding.score >= 4)
        .map(|finding| finding.id)
        .collect::<Vec<_>>();
    let awkward = findings
        .iter()
        .filter(|finding| (2..4).contains(&finding.score))
        .map(|finding| finding.id)
        .collect::<Vec<_>>();
    let natural = findings
        .iter()
        .filter(|finding| finding.score <= 1)
        .map(|finding| finding.id)
        .collect::<Vec<_>>();

    let report = json!({
        "verdict": "PASS_WITH_FINDINGS",
        "task": "Resolution Workflow and Effective Specification communication audit",
        "summary": {
            "naturalCount": natural.len(),
            "awkwardCount": awkward.len(),
            "hardGapCount": hard.len(),
            "natural": natural,
            "awkward": awkward,
            "hardGaps": hard,
        },
        "findings": findings
            .iter()
            .map(|finding| json!({
                "id": finding.id,
                "score": finding.score,
                "status": finding.status,
                "evidence": finding.evidence,
                "recommendation": finding.recommendation,
            }))
            .collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}