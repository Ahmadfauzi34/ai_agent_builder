use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::bind_input_port_consumer_edge;
use burn_research::introspection::describe_graph;
use burn_research::registry::LayerRegistry;
use burn_research::semantic_lifecycle::{
    bind_semantic_lifecycle_transition, semantic_lifecycle_identity,
    semantic_lifecycle_projection, semantic_lifecycle_transition, SemanticTransitionSpec,
};
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;
use burn_research::WasmTensor;

fn two_step_graph() -> (
    AgentWorkspace,
    AgentGraphBuilder,
    LayerRegistry,
    u8,
) {
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
    let second_output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &second,
        first_output,
        "relu-b".into(),
    )
    .unwrap();

    let consumer = InputPortConsumerSpec::new(
        "feature-ingress".into(),
        "[\"observation\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();
    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    (workspace, builder, registry, second_output)
}

#[test]
fn lifecycle_chain_changes_semantic_identity_not_program_or_numeric_output() {
    let (workspace, mut builder, registry, output_slot) = two_step_graph();

    let graph_before = builder
        .compile_with_output(&registry, output_slot)
        .unwrap();
    let program_before = graph_before.program_identity();
    let input = WasmTensor::new(&[-2.0, -0.5, 1.25, 3.0], &[1, 4, 1, 1]);
    let output_before = graph_before.run(&registry, &input).unwrap().to_array();
    let lifecycle_before: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_identity(&builder)).unwrap();

    let first = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &first,
        0,
    )
    .unwrap());

    let second = SemanticTransitionSpec::new(
        "feature-to-candidate".into(),
        "[\"feature\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &second,
        1,
    )
    .unwrap());

    let graph_after = builder
        .compile_with_output(&registry, output_slot)
        .unwrap();
    let program_after = graph_after.program_identity();
    let output_after = graph_after.run(&registry, &input).unwrap().to_array();
    let lifecycle_after: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_identity(&builder)).unwrap();

    assert_eq!(program_before, program_after);
    assert_eq!(output_before, output_after);
    assert_ne!(
        lifecycle_before["fingerprint"],
        lifecycle_after["fingerprint"]
    );
    assert_eq!(lifecycle_after["transition_count"], 2);

    let second_status: serde_json::Value = serde_json::from_str(
        &semantic_lifecycle_transition(&builder, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        second_status["transition"]["inputs"][0]["source_kind"],
        "prior_transition"
    );
    assert_eq!(
        second_status["transition"]["inputs"][0]["source_step_index"],
        0
    );
    assert_eq!(
        second_status["transition"]["inputs"][0]["role"],
        "feature"
    );
    assert_eq!(
        second_status["transition"]["output"]["role"],
        "candidate"
    );

    let projection: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_projection(&builder)).unwrap();
    assert_eq!(projection["coverage_complete"], true);
    assert_eq!(projection["unbound_step_indices"].as_array().unwrap().len(), 0);

    let description: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();
    assert_eq!(
        description["steps"][0]["semantic_lifecycle_transition"]["output"]["role"],
        "feature"
    );
    assert_eq!(
        description["steps"][1]["semantic_lifecycle_transition"]["output"]["role"],
        "candidate"
    );
    assert_eq!(
        description["semantic_lifecycle_identity"]["fingerprint"],
        lifecycle_after["fingerprint"]
    );
}

#[test]
fn lifecycle_binding_is_idempotent_and_conflicting_rebind_fails_closed() {
    let (workspace, mut builder, _registry, _output_slot) = two_step_graph();

    let first = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();

    assert!(bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &first,
        0,
    )
    .unwrap());
    assert!(!bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &first,
        0,
    )
    .unwrap());

    let before = semantic_lifecycle_transition(&builder, 0).unwrap();
    let conflicting = SemanticTransitionSpec::new(
        "observation-to-state".into(),
        "[\"observation\"]".into(),
        "state".into(),
    )
    .unwrap();
    let err = bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &conflicting,
        0,
    )
    .unwrap_err();
    assert!(err.contains("different immutable semantic lifecycle transition"));
    assert_eq!(semantic_lifecycle_transition(&builder, 0).unwrap(), before);
}

#[test]
fn external_provenance_drift_blocks_lifecycle_binding() {
    let (mut workspace, mut builder, _registry, _output_slot) = two_step_graph();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed-v2".into(),
        19,
        "fnv1a64:efgh".into(),
    )
    .unwrap();

    let transition = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    let err = bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &transition,
        0,
    )
    .unwrap_err();

    assert!(err.contains("provenance drift"));
    let status: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_transition(&builder, 0).unwrap()).unwrap();
    assert_eq!(status["status"], "unbound");
}

#[test]
fn internal_lineage_requires_prior_transition_and_exact_role_match() {
    let (workspace, mut builder, _registry, _output_slot) = two_step_graph();

    let missing = SemanticTransitionSpec::new(
        "feature-to-candidate".into(),
        "[\"feature\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    let err = bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &missing,
        1,
    )
    .unwrap_err();
    assert!(err.contains("producer step 0 has no semantic lifecycle transition"));

    let first = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &first,
        0,
    )
    .unwrap();

    let mismatch = SemanticTransitionSpec::new(
        "wrong-role".into(),
        "[\"state\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    let err = bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &mismatch,
        1,
    )
    .unwrap_err();
    assert!(err.contains("declared state, lineage resolves to feature"));
}
