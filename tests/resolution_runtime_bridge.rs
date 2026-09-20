use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::authorization::{AuthorizationPolicy, AuthorizationSnapshot};
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration,
};
use burn_research::introspection::{describe_graph, describe_workspace};
use burn_research::proof_provenance::workspace_verify_graph_receipt;
use burn_research::registry::LayerRegistry;
use burn_research::resolution_runtime_bridge::{
    bind_runtime_subject_projection, resolution_runtime_bridge_capabilities,
    workspace_bind_runtime_subject, workspace_clear_runtime_subject, workspace_runtime_subject,
    RuntimeSubjectProjection,
};
use burn_research::resolution_subject::SubjectBoundReviewSession;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{workspace_compile, workspace_init_unary};
use burn_research::WasmTensor;

fn fixture() -> (
    ApprovedEffectiveSpec,
    AuthorizationPolicy,
    AuthorizationSnapshot,
) {
    let spec = EffectiveSpec::root(vec![
        SpecDeclaration::new("objective", "feature_transform").unwrap(),
        SpecDeclaration::new("input.semantic", "dense_features").unwrap(),
        SpecDeclaration::new("planner.note", "agent_selects_runtime_graph").unwrap(),
    ])
    .unwrap();

    let subject = spec.approval_subject().unwrap();
    let mut review = SubjectBoundReviewSession::new("intent-runtime-bridge", subject).unwrap();
    review.submit("owner").unwrap();
    let approval = review.approve("owner").unwrap();

    let approved = ApprovedEffectiveSpec::bind_root(spec, approval.clone()).unwrap();
    let policy = AuthorizationPolicy::new("runtime-policy", 3, "owner", vec![]).unwrap();
    let authorization = policy.authorize(&approval).unwrap();

    (approved, policy, authorization)
}

#[test]
fn bridge_contract_is_artifact_discoverable_and_non_planning() {
    let contract: serde_json::Value =
        serde_json::from_str(&resolution_runtime_bridge_capabilities()).unwrap();

    assert_eq!(
        contract["schema_id"],
        "burn-research.resolution-runtime-bridge.v1"
    );
    assert_eq!(contract["state_ownership"], "none");
    assert_eq!(
        contract["native_projection"]["interpretation_policy"],
        "EffectiveSpec fields are transported as opaque semantic facts with provenance; the bridge never chooses AgentLayerSpec constructors or graph topology."
    );
}

#[test]
fn authorized_effective_spec_projects_exact_existing_identity_tuple() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

    assert_eq!(projection.intent_id, "intent-runtime-bridge");
    assert_eq!(projection.approval_id, approved.approval_id());
    assert_eq!(projection.subject_kind, "effective-spec");
    assert_eq!(projection.subject_identity, approved.spec.identity);
    assert_eq!(projection.effective_spec_identity, approved.spec.identity);
    assert_eq!(projection.authorization_policy_id, "runtime-policy");
    assert_eq!(projection.authorization_policy_revision, 3);
    assert!(!projection.authorization_is_revision);
    assert_eq!(projection.fields.len(), 3);

    let projection_json: serde_json::Value =
        serde_json::from_str(&projection.to_json()).unwrap();
    assert_eq!(
        projection_json["planner_policy"],
        "opaque_fields_agent_planner_required"
    );
    assert!(projection_json["effective_spec"]["fields"].is_array());
    assert!(projection_json.get("runtime_subject_id").is_none());
}

#[test]
fn stale_policy_fails_closed_before_runtime_projection() {
    let (approved, _policy, authorization) = fixture();
    let current_policy = AuthorizationPolicy::new("runtime-policy", 4, "owner", vec![]).unwrap();

    let error =
        RuntimeSubjectProjection::from_authorized(&approved, &current_policy, &authorization)
            .unwrap_err();

    assert!(
        error.contains("stale")
            || error.contains("policy")
            || error.contains("authorization snapshot")
    );
}

#[test]
fn manual_runtime_binding_validation_fails_without_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let before = workspace.snapshot();

    let error = workspace_bind_runtime_subject(
        &mut workspace,
        "".into(),
        1,
        "approval".into(),
        "effective-spec".into(),
        "identity".into(),
        "policy".into(),
        1,
        false,
    )
    .unwrap_err();

    assert!(error.contains("intent_id"));
    assert_eq!(workspace.snapshot(), before);
    assert!(workspace_runtime_subject(&workspace).contains("\"status\":\"unbound\""));
}

#[test]
fn bound_subject_is_visible_in_workspace_and_graph_without_touching_execution() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    let steps_before = builder.num_steps();
    let params_before = registry.total_params();

    bind_runtime_subject_projection(&mut workspace, &projection);

    let workspace_view: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    let graph_view: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

    assert_eq!(
        workspace_view["runtime_subject"]["subject_identity"],
        approved.spec.identity
    );
    assert_eq!(
        graph_view["runtime_subject"]["approval_id"],
        approved.approval_id()
    );
    assert_eq!(
        workspace_view["runtime_subject"]["authorization_policy_id"],
        "runtime-policy"
    );

    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(registry.total_params(), params_before);
}

#[test]
fn burn_backed_receipt_preserves_exact_bound_runtime_subject_context() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

    let mut workspace = AgentWorkspace::new(2).unwrap();
    bind_runtime_subject_projection(&mut workspace, &projection);

    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let mut registry = LayerRegistry::new();
    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let output_slot = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    let graph = workspace_compile(&builder, &registry, output_slot).unwrap();

    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &[0.0, 3.0],
            0.0,
            0.0,
            "bridge-graph-proof".into(),
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(receipt["authority"], "wasm_verifier");
    assert_eq!(
        receipt["reference_authority"],
        "burn_compiled_graph"
    );
    assert_eq!(
        receipt["runtime_subject"]["intent_id"],
        "intent-runtime-bridge"
    );
    assert_eq!(
        receipt["runtime_subject"]["approval_id"],
        approved.approval_id()
    );
    assert_eq!(
        receipt["runtime_subject"]["subject_identity"],
        approved.spec.identity
    );
    assert_eq!(
        receipt["runtime_subject"]["authorization_policy_revision"],
        3
    );
    assert_eq!(receipt["result"]["passed"], true);
}

#[test]
fn clearing_runtime_subject_returns_to_explicit_unbound_execution_context() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();
    let mut workspace = AgentWorkspace::new(2).unwrap();

    bind_runtime_subject_projection(&mut workspace, &projection);
    assert!(workspace_runtime_subject(&workspace).contains("\"status\":\"bound\""));
    assert!(workspace_clear_runtime_subject(&mut workspace));
    assert!(workspace_runtime_subject(&workspace).contains("\"status\":\"unbound\""));
}
