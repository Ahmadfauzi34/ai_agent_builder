use burn_research::agent::AgentGraphBuilder;
use burn_research::input_contract::{workspace_bind_input_contract, workspace_input_contract};
use burn_research::input_port::{
    input_port_capabilities, workspace_bind_input_port_metadata,
    workspace_clear_input_port_metadata, workspace_input_port_metadata,
};
use burn_research::introspection::{describe_graph, describe_workspace};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;

#[test]
fn semantic_input_port_is_independent_from_shape_layout_contract_and_execution_state() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    workspace_bind_input_contract(
        &mut workspace,
        1,
        8,
        1,
        1,
        "feature_axis1_singleton".into(),
        "normalized market features".into(),
    )
    .unwrap();

    let before_steps = builder.num_steps();
    let before_params = registry.total_params();

    assert!(workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "fnv1a64:abcd".into(),
    )
    .unwrap());

    assert_eq!(builder.num_steps(), before_steps);
    assert_eq!(registry.total_params(), before_params);

    let metadata: serde_json::Value =
        serde_json::from_str(&workspace_input_port_metadata(&workspace)).unwrap();
    assert_eq!(metadata["status"], "bound");
    assert_eq!(metadata["metadata"]["role"], "observation");
    assert_eq!(metadata["metadata"]["provenance"]["source"], "market-feed");
    assert_eq!(metadata["metadata"]["provenance"]["revision"], 18);

    let input_contract: serde_json::Value =
        serde_json::from_str(&workspace_input_contract(&workspace)).unwrap();
    assert_eq!(input_contract["status"], "bound");

    assert!(workspace_clear_input_port_metadata(&mut workspace));
    let input_contract_after: serde_json::Value =
        serde_json::from_str(&workspace_input_contract(&workspace)).unwrap();
    assert_eq!(input_contract_after["status"], "bound");
}

#[test]
fn introspection_exposes_semantic_input_port_without_inference() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    let unbound_workspace: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    assert_eq!(unbound_workspace["external_input_port"]["status"], "unbound");

    workspace_bind_input_port_metadata(
        &mut workspace,
        "x-market-regime".into(),
        "regime-classifier".into(),
        7,
        String::new(),
    )
    .unwrap();

    let workspace_json: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    assert_eq!(workspace_json["external_input_port"]["role"], "x-market-regime");
    assert!(workspace_json["external_input_port"]["provenance"]["fingerprint"].is_null());

    let graph_json: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();
    assert_eq!(graph_json["external_input_port"]["role"], "x-market-regime");
}

#[test]
fn invalid_role_fails_without_mutating_existing_metadata() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    workspace_bind_input_port_metadata(
        &mut workspace,
        "state".into(),
        "agent-state".into(),
        2,
        "state:2".into(),
    )
    .unwrap();

    let before = workspace_input_port_metadata(&workspace);
    let result = workspace_bind_input_port_metadata(
        &mut workspace,
        "arbitrary role".into(),
        "agent-state".into(),
        3,
        "state:3".into(),
    );
    assert!(result.is_err());
    assert_eq!(workspace_input_port_metadata(&workspace), before);
}

#[test]
fn capability_keeps_extension_path_and_execution_boundary_explicit() {
    let caps: serde_json::Value = serde_json::from_str(&input_port_capabilities()).unwrap();
    assert_eq!(caps["role"], "optional_semantic_input_port");
    assert_eq!(caps["scope"]["execution_effect"], "none");
    assert_eq!(
        caps["extension_role_policy"],
        "custom roles must use x- namespace with lowercase ascii token syntax"
    );
}
