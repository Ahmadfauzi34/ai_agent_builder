use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::{
    bind_input_port_consumer_edge, input_port_consumer_edge_binding,
    input_port_edge_binding_capabilities, semantic_graph_identity,
};
use burn_research::introspection::describe_graph;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn persistent_edge_binding_changes_semantic_identity_not_program_identity() {
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

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();

    let program_before = builder
        .compile_with_output(&registry, output)
        .unwrap()
        .program_identity();
    let semantic_before: serde_json::Value =
        serde_json::from_str(&semantic_graph_identity(&builder)).unwrap();
    let workspace_before = workspace.snapshot();
    let steps_before = builder.num_steps();
    let params_before = registry.total_params();

    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();

    assert!(bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &consumer,
        0,
    )
    .unwrap());
    assert!(!bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &consumer,
        0,
    )
    .unwrap());

    let program_after = builder
        .compile_with_output(&registry, output)
        .unwrap()
        .program_identity();
    let semantic_after: serde_json::Value =
        serde_json::from_str(&semantic_graph_identity(&builder)).unwrap();

    assert_eq!(program_before, program_after);
    assert_ne!(
        semantic_before["fingerprint"],
        semantic_after["fingerprint"]
    );
    assert_eq!(semantic_after["semantic_binding_count"], 1);
    assert_eq!(workspace.snapshot(), workspace_before);
    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(registry.total_params(), params_before);

    let graph: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();
    assert_eq!(
        graph["semantic_graph_identity"]["fingerprint"],
        semantic_after["fingerprint"]
    );
    assert_eq!(
        graph["steps"][0]["semantic_input_edge_binding"]["consumer"]["consumer_id"],
        "feature-extractor"
    );
}

#[test]
fn conflicting_rebind_fails_closed_without_rewriting_first_binding() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "feed".into(),
        2,
        "fp".into(),
    )
    .unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();

    let first = InputPortConsumerSpec::new(
        "first".into(),
        "[\"observation\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();
    let second = InputPortConsumerSpec::new(
        "second".into(),
        "[\"observation\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();

    bind_input_port_consumer_edge(&workspace, &mut builder, &first, 0).unwrap();
    let before = input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap();

    let err = bind_input_port_consumer_edge(&workspace, &mut builder, &second, 0)
        .unwrap_err();
    assert!(err.contains("different immutable semantic binding"));

    let after = input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap();
    assert_eq!(before, after);
}

#[test]
fn provenance_drift_is_reported_without_mutating_binding_history() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "feed-a".into(),
        4,
        "fp:a".into(),
    )
    .unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();

    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        1,
    )
    .unwrap();
    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    let current: serde_json::Value =
        serde_json::from_str(&input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap())
            .unwrap();
    let original_fingerprint = current["binding"]["binding_fingerprint"].clone();
    assert_eq!(current["binding_state"], "current");

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "feed-b".into(),
        5,
        "fp:b".into(),
    )
    .unwrap();

    let compatible_drift: serde_json::Value =
        serde_json::from_str(&input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap())
            .unwrap();
    assert_eq!(compatible_drift["binding_state"], "drifted_compatible");
    assert_eq!(compatible_drift["current_compatible"], true);
    assert_eq!(
        compatible_drift["binding"]["binding_fingerprint"],
        original_fingerprint
    );

    workspace_bind_input_port_metadata(
        &mut workspace,
        "reward".into(),
        "reward-feed".into(),
        6,
        "fp:r".into(),
    )
    .unwrap();

    let incompatible_drift: serde_json::Value =
        serde_json::from_str(&input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap())
            .unwrap();
    assert_eq!(
        incompatible_drift["binding_state"],
        "drifted_incompatible"
    );
    assert_eq!(incompatible_drift["current_compatible"], false);
    assert_eq!(
        incompatible_drift["binding"]["binding_fingerprint"],
        original_fingerprint
    );
}

#[test]
fn non_external_edge_is_rejected_but_incompatible_consumer_is_recorded_as_provenance() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "feed".into(),
        1,
        "fp".into(),
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
    let second_output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &second,
        first_output,
        "relu-b".into(),
    )
    .unwrap();

    let observation_consumer = InputPortConsumerSpec::new(
        "feature".into(),
        "[\"observation\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();
    assert!(
        bind_input_port_consumer_edge(&workspace, &mut builder, &observation_consumer, 1)
            .unwrap_err()
            .contains("does not consume external slot 0")
    );
    let internal_status: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_binding(&workspace, &builder, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(internal_status["status"], "not_applicable");

    let reward_consumer = InputPortConsumerSpec::new(
        "reward-only".into(),
        "[\"reward\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();

    let program_before = builder
        .compile_with_output(&registry, second_output)
        .unwrap()
        .program_identity();

    assert!(bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &reward_consumer,
        0,
    )
    .unwrap());

    let bound: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(bound["binding"]["compatibility_at_bind"], "incompatible");
    assert_eq!(bound["current_compatible"], false);

    let program_after = builder
        .compile_with_output(&registry, second_output)
        .unwrap()
        .program_identity();
    assert_eq!(program_before, program_after);
}

#[test]
fn capability_contract_keeps_binding_outside_execution_authority() {
    let caps: serde_json::Value =
        serde_json::from_str(&input_port_edge_binding_capabilities()).unwrap();

    assert_eq!(caps["scope"]["state_owner"], "AgentGraphBuilder");
    assert_eq!(caps["scope"]["execution_plan_effect"], "none");
    assert_eq!(caps["scope"]["numerical_effect"], "none");
    assert_eq!(caps["scope"]["decision_authority"], "agent");
    assert_eq!(
        caps["binding"]["compatibility_policy"],
        "advisory: compatible and incompatible consumer choices may both be persisted; compatibility_at_bind records the fact"
    );
}
