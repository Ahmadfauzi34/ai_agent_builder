use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::{
    bind_input_port_consumer_edge, input_port_consumer_edge_binding,
};
use burn_research::input_port_routing::input_port_consumer_edge_compatibility;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::WasmTensor;

#[test]
fn slot_zero_read_after_prior_write_is_internal_not_external() {
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

    // Step 0 consumes the original external input and overwrites slot 0.
    builder.add_unary(&first, 0, 0).unwrap();
    // Step 1 reads slot 0 again, but now its live value comes from step 0.
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

    let step0: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            0,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(step0["status"], "applicable");
    assert_eq!(step0["edge"]["external_input_positions"][0], "input");

    assert!(bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &consumer,
        0,
    )
    .unwrap());

    let step1: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_compatibility(
            &workspace,
            &builder,
            &consumer,
            1,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(step1["status"], "not_applicable");
    assert_eq!(
        step1["reason"],
        "step_does_not_consume_external_slot_0"
    );

    let bind_error =
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 1).unwrap_err();
    assert!(bind_error.contains("does not consume external slot 0"));

    let binding_status: serde_json::Value = serde_json::from_str(
        &input_port_consumer_edge_binding(&workspace, &builder, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(binding_status["status"], "not_applicable");

    // The executable graph itself confirms that step 1 reads the overwritten value.
    let graph = builder.compile(&registry).unwrap();
    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let output = graph.run(&registry, &input).unwrap().to_array();
    assert_eq!(output, vec![0.0, 3.0]);
}
