use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_contract::{
    input_contract_compatibility, input_contract_capabilities, workspace_bind_input_contract,
    workspace_clear_input_contract, workspace_input_contract,
};
use burn_research::interaction_fault::interaction_check_init_unary;
use burn_research::introspection::{describe_graph, describe_workspace};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn input_contract_is_optional_and_artifact_discoverable() {
    let caps: serde_json::Value =
        serde_json::from_str(&input_contract_capabilities()).unwrap();
    assert_eq!(caps["schema_id"], "burn-research.agent-input-contract.v1");
    assert_eq!(caps["scope"]["slot"], 0);
    assert_eq!(caps["scope"]["dtype"], "f32");
    assert_eq!(caps["scope"]["optional"], true);

    let workspace = AgentWorkspace::new(3).unwrap();
    let unbound: serde_json::Value =
        serde_json::from_str(&workspace_input_contract(&workspace)).unwrap();
    assert_eq!(unbound["status"], "unbound");
    assert_eq!(unbound["policy"], "defer_to_runtime");
}

#[test]
fn shape_proven_mismatch_rejects_before_canonical_init_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_contract(
        &mut workspace,
        1,
        2,
        2,
        1,
        "unknown".into(),
        "external-features".into(),
    )
    .unwrap();

    let layer_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let spec = AgentLayerSpec::linear(layer_id, 2, 1, false).unwrap();

    let compatibility: serde_json::Value =
        serde_json::from_str(&input_contract_compatibility(&workspace, &spec)).unwrap();
    assert_eq!(compatibility["status"], "incompatible");
    assert_eq!(compatibility["compatible"], false);

    let preflight: serde_json::Value = serde_json::from_str(&interaction_check_init_unary(
        &workspace,
        &builder,
        &registry,
        &spec,
        0,
        "linear".into(),
    ))
    .unwrap();
    assert_eq!(preflight["status"], "fault");
    assert_eq!(preflight["fault"]["code"], "E_LAYOUT_PREFLIGHT");

    let workspace_before = workspace.snapshot();
    let steps_before = builder.num_steps();
    let params_before = registry.total_params();

    let err = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "linear".into(),
    )
    .unwrap_err();
    assert!(err.contains("violates layout feature_axis1_singleton"));

    assert_eq!(workspace.snapshot(), workspace_before);
    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(registry.total_params(), params_before);
    assert!(!registry.layer_exists(spec.layer_type(), spec.layer_id()));
}

#[test]
fn valid_shape_unknown_layout_remains_allowed_but_not_overclaimed() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_contract(
        &mut workspace,
        1,
        2,
        1,
        1,
        "unknown".into(),
        "external-features".into(),
    )
    .unwrap();

    let layer_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let spec = AgentLayerSpec::linear(layer_id, 2, 1, false).unwrap();

    let compatibility: serde_json::Value =
        serde_json::from_str(&input_contract_compatibility(&workspace, &spec)).unwrap();
    assert_eq!(
        compatibility["status"],
        "shape_compatible_layout_unknown"
    );
    assert!(compatibility["compatible"].is_null());

    let output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "linear".into(),
    )
    .unwrap();
    assert_eq!(output, 1);
}

#[test]
fn clearing_contract_restores_backward_compatible_defer_to_runtime_behavior() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_contract(
        &mut workspace,
        1,
        2,
        2,
        1,
        "unknown".into(),
        "".into(),
    )
    .unwrap();
    assert!(workspace_clear_input_contract(&mut workspace));

    let layer_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let spec = AgentLayerSpec::linear(layer_id, 2, 1, false).unwrap();

    let compatibility: serde_json::Value =
        serde_json::from_str(&input_contract_compatibility(&workspace, &spec)).unwrap();
    assert_eq!(compatibility["status"], "unbound");

    assert!(workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "linear".into(),
    )
    .is_ok());
}

#[test]
fn introspection_exposes_bound_input_contract_without_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    workspace_bind_input_contract(
        &mut workspace,
        2,
        8,
        1,
        1,
        "feature_axis1_singleton".into(),
        "batch-features".into(),
    )
    .unwrap();

    let before = workspace.snapshot();
    let workspace_view: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    let graph_view: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

    assert_eq!(
        workspace_view["external_input_contract"]["layout"],
        "feature_axis1_singleton"
    );
    assert_eq!(
        graph_view["external_input_contract"]["shape"],
        serde_json::json!([2, 8, 1, 1])
    );
    assert_eq!(workspace.snapshot(), before);
}

#[test]
fn declaration_rejects_self_inconsistent_layout_shape_without_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let before = workspace.snapshot();

    let err = workspace_bind_input_contract(
        &mut workspace,
        1,
        4,
        8,
        2,
        "channel_first_singleton_width".into(),
        "".into(),
    )
    .unwrap_err();
    assert!(err.contains("violates layout channel_first_singleton_width"));
    assert_eq!(workspace.snapshot(), before);
}
