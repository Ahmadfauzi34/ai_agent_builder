use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::authorization::{AuthorizationPolicy, AuthorizationSnapshot};
use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration,
};
use burn_research::introspection::{describe_graph, describe_workspace};
use burn_research::proof_provenance::{
    workspace_record_attestation, workspace_verify_graph_receipt,
};
use burn_research::registry::LayerRegistry;
use burn_research::resolution_runtime_bridge::{
    bind_runtime_subject_projection, resolution_runtime_bridge_capabilities,
    workspace_bind_runtime_subject, workspace_runtime_program_binding,
    workspace_runtime_subject, RuntimeSubjectProjection,
};
use burn_research::resolution_subject::SubjectBoundReviewSession;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{
    workspace_compile, workspace_compile_for_runtime_subject, workspace_init_unary,
};
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

    assert!(bind_runtime_subject_projection(&mut workspace, &projection).unwrap());

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
    assert!(bind_runtime_subject_projection(&mut workspace, &projection).unwrap());

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
    let graph =
        workspace_compile_for_runtime_subject(&mut workspace, &builder, &registry, output_slot)
            .unwrap();
    let binding: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace, &graph)).unwrap();
    assert_eq!(binding["program_bound"], true);
    assert_eq!(binding["binding_count"], 1);

    let duplicate =
        workspace_compile_for_runtime_subject(&mut workspace, &builder, &registry, output_slot)
            .unwrap();
    assert_eq!(duplicate.program_identity(), graph.program_identity());
    let binding: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace, &graph)).unwrap();
    assert_eq!(binding["binding_count"], 1);

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
fn runtime_subject_binding_is_idempotent_but_not_replaceable() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();
    let mut workspace = AgentWorkspace::new(2).unwrap();

    assert!(bind_runtime_subject_projection(&mut workspace, &projection).unwrap());
    assert!(!bind_runtime_subject_projection(&mut workspace, &projection).unwrap());

    let error = workspace_bind_runtime_subject(
        &mut workspace,
        "different-intent".into(),
        projection.workflow_revision,
        projection.approval_id.clone(),
        projection.subject_kind.clone(),
        projection.subject_identity.clone(),
        projection.authorization_policy_id.clone(),
        projection.authorization_policy_revision,
        projection.authorization_is_revision,
    )
    .unwrap_err();

    assert!(error.contains("immutable"));
    assert!(workspace_runtime_subject(&workspace)
        .contains("\"intent_id\":\"intent-runtime-bridge\""));
}

#[test]
fn late_binding_after_runtime_reservation_is_rejected_without_mutation() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let registry = LayerRegistry::new();

    workspace
        .reserve_layer_id(&registry, "pre-bind-reservation".into())
        .unwrap();
    let before = workspace.snapshot();

    let error = bind_runtime_subject_projection(&mut workspace, &projection).unwrap_err();
    assert!(error.contains("before reserving runtime layers or slots"));
    assert_eq!(workspace.snapshot(), before);
    assert!(workspace_runtime_subject(&workspace).contains("\"status\":\"unbound\""));
}

#[test]
fn binding_after_proof_state_is_rejected_without_relabeling_evidence() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();
    let mut workspace = AgentWorkspace::new(2).unwrap();

    workspace_record_attestation(
        &mut workspace,
        "pre-bind-claim".into(),
        true,
        "legacy unbound context".into(),
    )
    .unwrap();
    let before = workspace.snapshot();

    let error = bind_runtime_subject_projection(&mut workspace, &projection).unwrap_err();
    assert!(error.contains("recording proof evidence"));
    assert_eq!(workspace.snapshot(), before);
    assert!(workspace_runtime_subject(&workspace).contains("\"status\":\"unbound\""));
}

#[test]
fn bound_subject_rejects_legacy_compiled_graph_until_exact_identity_is_bound() {
    let (approved, policy, authorization) = fixture();
    let projection =
        RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

    let mut workspace = AgentWorkspace::new(2).unwrap();
    assert!(bind_runtime_subject_projection(&mut workspace, &projection).unwrap());
    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let mut registry = LayerRegistry::new();
    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();

    let graph = workspace_compile(&builder, &registry, out).unwrap();
    let status: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace, &graph)).unwrap();
    assert_eq!(status["program_bound"], false);

    let before = workspace.snapshot();
    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let error = workspace_verify_graph_receipt(
        &mut workspace,
        &graph,
        &registry,
        &input,
        &[0.0, 3.0],
        0.0,
        0.0,
        "legacy-under-bound-subject".into(),
    )
    .unwrap_err();

    assert!(error.contains("workspaceCompileForRuntimeSubject"));
    assert_eq!(workspace.snapshot(), before);
}

#[test]
fn cross_context_graph_relabeling_is_rejected_without_receipt_mutation() {
    let mut workspace_a = AgentWorkspace::new(2).unwrap();
    let mut workspace_b = AgentWorkspace::new(2).unwrap();

    assert!(workspace_bind_runtime_subject(
        &mut workspace_a,
        "intent-a".into(),
        1,
        "approval-a".into(),
        "effective-spec".into(),
        "spec-A".into(),
        "policy".into(),
        1,
        false,
    )
    .unwrap());
    assert!(workspace_bind_runtime_subject(
        &mut workspace_b,
        "intent-b".into(),
        1,
        "approval-b".into(),
        "effective-spec".into(),
        "spec-B".into(),
        "policy".into(),
        1,
        false,
    )
    .unwrap());

    let mut builder_b = AgentGraphBuilder::new(2).unwrap();
    let mut registry_b = LayerRegistry::new();
    let layer_id = workspace_b
        .reserve_layer_id(&registry_b, "relu-b".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let out = workspace_init_unary(
        &mut workspace_b,
        &mut builder_b,
        &mut registry_b,
        &spec,
        0,
        "relu-b".into(),
    )
    .unwrap();
    let graph_b = workspace_compile_for_runtime_subject(
        &mut workspace_b,
        &builder_b,
        &registry_b,
        out,
    )
    .unwrap();

    let status_a: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace_a, &graph_b)).unwrap();
    let status_b: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace_b, &graph_b)).unwrap();
    assert_eq!(status_a["program_bound"], false);
    assert_eq!(status_b["program_bound"], true);

    let before_a = workspace_a.snapshot();
    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let error = workspace_verify_graph_receipt(
        &mut workspace_a,
        &graph_b,
        &registry_b,
        &input,
        &[0.0, 3.0],
        0.0,
        0.0,
        "cross-context-attack".into(),
    )
    .unwrap_err();

    assert!(error.contains("not bound to this runtime subject"));
    assert_eq!(workspace_a.snapshot(), before_a);

    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_graph_receipt(
            &mut workspace_b,
            &graph_b,
            &registry_b,
            &input,
            &[0.0, 3.0],
            0.0,
            0.0,
            "correct-context".into(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["runtime_subject"]["subject_identity"], "spec-B");
    assert_eq!(receipt["result"]["passed"], true);
}

#[test]
fn unbound_legacy_graph_receipt_remains_supported() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let mut registry = LayerRegistry::new();
    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    let graph = workspace_compile(&builder, &registry, out).unwrap();

    let status: serde_json::Value =
        serde_json::from_str(&workspace_runtime_program_binding(&workspace, &graph)).unwrap();
    assert_eq!(status["runtime_subject_bound"], false);
    assert_eq!(status["program_bound"], false);
    assert_eq!(
        status["receipt_policy"],
        "legacy_unbound_graph_receipt_allowed"
    );

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
            "legacy-unbound".into(),
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(receipt["runtime_subject"]["status"], "unbound");
    assert_eq!(receipt["result"]["passed"], true);
}
