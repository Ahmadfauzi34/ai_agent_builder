use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::bind_input_port_consumer_edge;
use burn_research::input_port_routing::input_port_consumer_edge_compatibility;
use burn_research::registry::LayerRegistry;
use burn_research::semantic_lifecycle::{
    bind_semantic_lifecycle_transition, semantic_lifecycle_transition, SemanticTransitionSpec,
};
use burn_research::workspace::AgentWorkspace;
use burn_research::WasmTensor;

#[test]
fn overwritten_slot_zero_becomes_internal_lineage_across_routing_binding_and_lifecycle() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "sensor".into(),
        1,
        "fp:external".into(),
    )
    .unwrap();

    let first = AgentLayerSpec::relu(1);
    let second = AgentLayerSpec::relu(2);
    registry.init_agent_layer(&first).unwrap();
    registry.init_agent_layer(&second).unwrap();

    // Step 0 consumes the original external value and overwrites slot 0.
    builder.add_unary(&first, 0, 0).unwrap();
    // Step 1 still names slot 0, but at execution time that value now comes from step 0.
    builder.add_unary(&second, 0, 1).unwrap();
    builder.set_output(1).unwrap();

    let consumer = InputPortConsumerSpec::new(
        "observation-consumer".into(),
        "[\"observation\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();

    let step0_route: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            0,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(step0_route["status"], "applicable");
    assert_eq!(
        step0_route["edge"]["external_input_positions"][0],
        "input"
    );

    assert!(bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &consumer,
        0,
    )
    .unwrap());

    let first_transition = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &first_transition,
        0,
    )
    .unwrap());

    let step1_route: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            1,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(step1_route["status"], "not_applicable");
    assert_eq!(
        step1_route["reason"],
        "step_does_not_consume_external_slot_0"
    );

    let edge_error =
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 1).unwrap_err();
    assert!(edge_error.contains("does not consume external slot 0"));

    // The lifecycle resolver must now follow the step-0 producer, despite the slot index being 0.
    let second_transition = SemanticTransitionSpec::new(
        "feature-to-candidate".into(),
        "[\"feature\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &workspace,
        &mut builder,
        &second_transition,
        1,
    )
    .unwrap());

    let lineage: serde_json::Value = serde_json::from_str(
        &semantic_lifecycle_transition(&builder, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        lineage["transition"]["inputs"][0]["slot"],
        0
    );
    assert_eq!(
        lineage["transition"]["inputs"][0]["role"],
        "feature"
    );
    assert_eq!(
        lineage["transition"]["inputs"][0]["source_kind"],
        "prior_transition"
    );
    assert_eq!(
        lineage["transition"]["inputs"][0]["source_step_index"],
        0
    );

    // Executable semantics agree with the topology-origin classification.
    let graph = builder.compile(&registry).unwrap();
    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let output = graph.run(&registry, &input).unwrap().to_array();
    assert_eq!(output, vec![0.0, 3.0]);
}
