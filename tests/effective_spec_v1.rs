mod resolution_subject {
    pub use ai_agent_builder::resolution_subject::*;
}

#[path = "../src/effective_spec.rs"]
mod effective_spec;

use effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, EffectiveSpecApprovalEvidence, SpecDeclaration,
    SpecDirective,
};
use resolution_subject::{SubjectBoundReviewSession, SubjectBoundRevisionChain};

#[test]
fn eight_explicit_inherit_revisions_remain_materializable() {
    let root_spec = EffectiveSpec::root(vec![SpecDeclaration::new("mode", "stable").unwrap()]).unwrap();
    let mut root_review = SubjectBoundReviewSession::new(
        "intent-depth",
        root_spec.approval_subject().unwrap(),
    )
    .unwrap();
    root_review.submit("agent").unwrap();
    let root_approval = root_review.approve("customer").unwrap();
    let root_snapshot = root_review.snapshot();
    let mut approved = ApprovedEffectiveSpec::bind_root(root_spec, root_approval).unwrap();
    assert!(matches!(approved.evidence, EffectiveSpecApprovalEvidence::Root(_)));

    let mut chain = SubjectBoundRevisionChain::from_approved_root(root_snapshot).unwrap();
    let mut parent_revision_id: Option<String> = None;

    for depth in 1..=8 {
        let child = approved
            .materialize_child(vec![SpecDirective::inherit("mode").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        let revision_id = chain
            .open_revision(
                parent_revision_id.as_deref(),
                format!("depth-{depth}"),
                child.approval_subject().unwrap(),
            )
            .unwrap();
        chain.submit_revision(&revision_id, "agent").unwrap();
        let approval = chain.approve_revision(&revision_id, "customer").unwrap();
        approved = ApprovedEffectiveSpec::bind_revision(child, approval).unwrap();
        parent_revision_id = Some(revision_id);
    }

    assert_eq!(approved.spec.field("mode").unwrap().value, "stable");
}
