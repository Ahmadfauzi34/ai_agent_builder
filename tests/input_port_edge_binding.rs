use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::{
    bind_input_port_consumer_edge, input_port_consumer_edge_binding,
    input_port_edge_binding_capabilities,
};
use burn_research::introspection::describe_graph;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

fn prepared_graph() -> (AgentWorkspace, AgentGraphBuilder, LayerRegistry, u8) {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
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

    (workspace, builder, registry, output)
}

#[test]
fn persistent_binding_does_not_change_canonical_program_identity() {
    let (workspace, mut builder, registry, output) = prepared_graph();

    let before = builder.compile_with_output(&registry, output).unwrap();
    let before_plan = before.program_plan();
    let before_identity = before.program_identity();

    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();

    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    let after = builder.compile_with_output(&registry, output).unwrap();
    assert_eq!(after.program_plan(), before_plan);
    assert_eq!(after.program_identity(), before_identity);
}

#[test]
fn binding_is_visible_in_graph_introspection_and_is_immutable() {
    let (workspace, mut builder, registry, _) = prepared_graph();

    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();

    let first: serde_json::Value = serde_json::from_str(
        &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(first["created"], true);
    assert_eq!(first["binding"]["compatibility_at_bind"], "compatible");

    let second: serde_json::Value = serde_json::from_str(
        &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(second["created"], false);

    let graph: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();
    assert_eq!(graph["semantic_edge_binding_count"], 1);
    assert_eq!(
        graph["steps"][0]["semantic_input_consumer_binding"]["consumer"]["consumer_id"],
        "feature-extractor"
    );

    let different = InputPortConsumerSpec::new(
        "reward-updater".into(),
        "[\"reward\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();
    assert!(bind_input_port_consumer_edge(&workspace, &mut builder, &different, 0)
        .unwrap_err()
        .contains("immutable semantic binding"));
}

#[test]
fn input_metadata_drift_is_observed_but_not_enforced() {
    let (mut workspace, mut builder, registry, output) = prepared_graph();
    let consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();

    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();
    let identity_before = builder
        .compile_with_output(&registry, output)
        .unwrap()
        .program_identity();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        19,
        "fnv1a64:next".into(),
    )
    .unwrap();

    let status: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap(),
    )
    .unwrap();

    assert_eq!(status["current"]["status"], "drifted");
    assert_eq!(status["current"]["input_snapshot_match"], false);
    assert_eq!(status["current"]["current_compatibility"], "compatible");
    assert_eq!(status["execution_authorized"], false);
    assert_eq!(status["decision_authority"], "agent");

    let identity_after = builder
        .compile_with_output(&registry, output)
        .unwrap()
        .program_identity();
    assert_eq!(identity_after, identity_before);
}

#[test]
fn incompatible_consumer_can_be_persisted_as_agent_provenance() {
    let (workspace, mut builder, _, _) = prepared_graph();
    let consumer = InputPortConsumerSpec::new(
        "reward-updater".into(),
        "[\"reward\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();

    let bound: serde_json::Value = serde_json::from_str(
        &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
    )
    .unwrap();

    assert_eq!(bound["binding"]["compatibility_at_bind"], "incompatible");
    assert_eq!(bound["binding"]["execution_authorized"], false);
    assert_eq!(bound["binding"]["decision_authority"], "agent");
}

#[test]
fn binding_capability_explicitly_excludes_execution_and_selection_authority() {
    let caps: serde_json::Value =
        serde_json::from_str(&input_port_edge_binding_capabilities()).unwrap();

    assert_eq!(caps["role"], "persistent_semantic_graph_provenance");
    assert_eq!(caps["scope"]["execution_effect"], "none");
    assert_eq!(caps["scope"]["numerical_effect"], "none");
    assert_eq!(caps["scope"]["selection_effect"], "none");
}
