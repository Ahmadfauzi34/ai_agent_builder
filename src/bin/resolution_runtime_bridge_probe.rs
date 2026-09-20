use burn_research::authorization::AuthorizationPolicy;
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, EffectiveSpecApprovalEvidence, SpecDeclaration,
    SpecDirective,
};
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
use burn_research::resolution_subject::{
    SubjectBoundReviewSession, SubjectBoundRevisionChain,
};

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{{\"schema\":\"burn-research.resolution-runtime-bridge-native-probe.v1\",\"status\":\"failed\",\"message\":\"{}\"}}",
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
    let root_spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "feature_transform")?,
        SpecDeclaration::new("input.semantic", "dense_features")?,
        SpecDeclaration::new("planner.note", "agent_selects_runtime_graph")?,
    ])?;

    let root_subject = root_spec.approval_subject()?;
    let mut root_review = SubjectBoundReviewSession::new(
        "intent-native-bridge-probe",
        root_subject.clone(),
    )?;
    root_review.submit("owner")?;
    let root_approval = root_review.approve("owner")?;
    let approved_root = ApprovedEffectiveSpec::bind_root(
        root_spec,
        root_approval.clone(),
    )?;

    let policy = AuthorizationPolicy::new("runtime-policy", 7, "owner", vec![])?;
    let root_authorization = policy.authorize(&root_approval)?;
    let root_projection = RuntimeSubjectProjection::from_authorized(
        &approved_root,
        &policy,
        &root_authorization,
    )?;

    ensure(
        root_projection.subject_identity == approved_root.spec.identity,
        "root subject identity drift",
    );
    ensure(
        root_projection.approval_id == approved_root.approval_id(),
        "root approval identity drift",
    );
    ensure(
        !root_projection.authorization_is_revision,
        "root projection incorrectly marked as revision",
    );
    let root_json = root_projection.to_json();
    ensure(
        root_json.contains("\"planner_policy\":\"opaque_fields_agent_planner_required\""),
        "root projection lost opaque planner policy",
    );
    ensure(
        !root_json.contains("AgentLayerSpec") && !root_json.contains("runtime_subject_id"),
        "bridge invented runtime planning or replacement identity",
    );

    let stale_policy = AuthorizationPolicy::new("runtime-policy", 8, "owner", vec![])?;
    ensure(
        RuntimeSubjectProjection::from_authorized(
            &approved_root,
            &stale_policy,
            &root_authorization,
        )
        .is_err(),
        "stale authorization policy was accepted",
    );

    let alternate_spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "different_transform")?,
    ])?;
    let alternate_subject = alternate_spec.approval_subject()?;
    let mut alternate_review = SubjectBoundReviewSession::new(
        "intent-native-bridge-probe-alt",
        alternate_subject,
    )?;
    alternate_review.submit("owner")?;
    let alternate_approval = alternate_review.approve("owner")?;
    let alternate_approved =
        ApprovedEffectiveSpec::bind_root(alternate_spec, alternate_approval)?;

    ensure(
        RuntimeSubjectProjection::from_authorized(
            &alternate_approved,
            &policy,
            &root_authorization,
        )
        .is_err(),
        "authorization for a different immutable subject was accepted",
    );

    let child_spec = approved_root
        .materialize_child(vec![
            SpecDirective::inherit("objective")?,
            SpecDirective::inherit("input.semantic")?,
            SpecDirective::override_value(
                "planner.note",
                "agent_selects_revised_runtime_graph",
            )?,
        ])?
        .resolved()?;

    let root_snapshot = match &approved_root.evidence {
        EffectiveSpecApprovalEvidence::Root(bound) => {
            let mut replay = SubjectBoundReviewSession::new(
                bound.approval.intent_id.clone(),
                bound.subject.clone(),
            )?;
            replay.submit("owner")?;
            replay.approve("owner")?;
            replay.snapshot()
        }
        EffectiveSpecApprovalEvidence::Revision(_) => {
            return Err("native probe expected root approval evidence".to_string())
        }
    };

    let mut chain = SubjectBoundRevisionChain::from_approved_root(root_snapshot)?;
    let revision_id = chain.open_revision(
        None,
        "runtime-plan-revision",
        child_spec.approval_subject()?,
    )?;
    chain.submit_revision(&revision_id, "owner")?;
    let revision_approval = chain.approve_revision(&revision_id, "owner")?;
    let approved_child =
        ApprovedEffectiveSpec::bind_revision(child_spec, revision_approval.clone())?;
    let revision_authorization = policy.authorize_revision(&revision_approval)?;
    let revision_projection = RuntimeSubjectProjection::from_authorized(
        &approved_child,
        &policy,
        &revision_authorization,
    )?;

    ensure(
        revision_projection.authorization_is_revision,
        "revision projection lost revision class",
    );
    ensure(
        revision_projection.subject_identity == approved_child.spec.identity,
        "revision subject identity drift",
    );
    ensure(
        revision_projection.approval_id == approved_child.approval_id(),
        "revision approval identity drift",
    );
    ensure(
        revision_projection
            .fields
            .iter()
            .any(|field| field.origin == "overridden"),
        "revision field provenance was not preserved",
    );

    println!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.resolution-runtime-bridge-native-probe.v1\",",
            "\"status\":\"passed\",",
            "\"root_projection\":true,",
            "\"stale_policy_rejected\":true,",
            "\"foreign_subject_rejected\":true,",
            "\"revision_projection\":true,",
            "\"opaque_planner_boundary\":true,",
            "\"root_field_count\":{},",
            "\"revision_field_count\":{}",
            "}}"
        ),
        root_projection.fields.len(),
        revision_projection.fields.len(),
    );

    Ok(())
}
