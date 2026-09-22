use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_routing::{
    input_port_consumer_edge_compatibility, input_port_routing_capabilities,
    interaction_valid_actions_for_input_consumer,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn valid_actions_overlay_keeps_canonical_candidates_visible() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "fnv1a64:abcd".into(),
    )
    .unwrap();

    let consumer = InputPortConsumerSpec::new(
        "reward-updater".into(),
        "[\"reward\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();

    let before_workspace = workspace.snapshot();
    let before_steps = builder.num_steps();
    let before_params = registry.total_params();

    let projection: serde_json::Value = serde_json::from_str(
        &interaction_valid_actions_for_input_consumer(
            &workspace,
            &builder,
            &registry,
            &consumer,
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(
        projection["semantic_overlay"]["compatibility"]["status"],
        "incompatible"
    );
    assert_eq!(projection["semantic_overlay"]["candidate_retained"], true);
    assert_eq!(projection["execution_authorized"], false);
    assert_eq!(projection["decision_authority"], "agent");

    assert_eq!(
        projection["canonical_valid_actions"]["schema_id"],
        "burn-research.agent-interaction.v1"
    );
    assert_eq!(
        projection["canonical_valid_actions"]["projection_only"],
        true
    );
    let canonical = projection["canonical_valid_actions"]["actions"]
        .as_array()
        .unwrap();
    let unary = canonical
        .iter()
        .find(|entry| entry["operation"] == "workspaceInitUnary")
        .unwrap();
    assert!(unary["input_slots"]
        .as_array()
        .unwrap()
        .iter()
        .any(|slot| slot == 0));

    assert_eq!(workspace.snapshot(), before_workspace);
    assert_eq!(builder.num_steps(), before_steps);
    assert_eq!(registry.total_params(), before_params);
}

#[test]
fn edge_projection_applies_only_to_steps_that_consume_external_slot_zero() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "fnv1a64:abcd".into(),
    )
    .unwrap();

    let first_id = workspace
        .reserve_layer_id(&registry, "relu-a".into())
        .unwrap();
    let first = AgentLayerSpec::relu(first_id);
    let first_output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &first,
        0,
        "relu-a".into(),
    )
    .unwrap();

    let second_id = workspace
        .reserve_layer_id(&registry, "relu-b".into())
        .unwrap();
    let second = AgentLayerSpec::relu(second_id);
    workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &second,
        first_output,
        "relu-b".into(),
    )
    .unwrap();

    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();

    let before_workspace = workspace.snapshot();
    let before_steps = builder.num_steps();
    let before_params = registry.total_params();

    let external: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            0,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(external["status"], "applicable");
    assert_eq!(external["edge"]["external_input_positions"][0], "input");
    assert_eq!(external["compatibility"]["status"], "compatible");

    let internal: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            1,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(internal["status"], "not_applicable");
    assert!(internal["compatibility"].is_null());

    assert_eq!(workspace.snapshot(), before_workspace);
    assert_eq!(builder.num_steps(), before_steps);
    assert_eq!(registry.total_params(), before_params);
}

#[test]
fn routing_capability_keeps_selection_and_execution_outside_projection() {
    let caps: serde_json::Value =
        serde_json::from_str(&input_port_routing_capabilities()).unwrap();

    assert_eq!(caps["role"], "read_only_semantic_routing_projection");
    assert_eq!(caps["scope"]["state_ownership"], "none");
    assert_eq!(caps["scope"]["execution_effect"], "none");
    assert_eq!(caps["scope"]["selection_effect"], "none");
    assert_eq!(
        caps["valid_actions_overlay"]["candidate_retained"],
        true
    );
}
