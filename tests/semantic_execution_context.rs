use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::bind_input_port_consumer_edge;
use burn_research::proof_provenance::{
    workspace_proof_ledger, workspace_verify_semantic_graph_receipt,
};
use burn_research::registry::LayerRegistry;
use burn_research::semantic_execution_context::semantic_execution_context;
use burn_research::semantic_lifecycle::{
    bind_semantic_lifecycle_transition, SemanticTransitionSpec,
};
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;
use burn_research::WasmTensor;

#[test]
fn semantic_context_enriches_graph_proof_without_changing_program_identity() {
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

    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
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

    let graph = builder.compile_with_output(&registry, output).unwrap();
    let program_identity_before = graph.program_identity();

    let context_before: serde_json::Value =
        serde_json::from_str(&semantic_execution_context(&builder, &graph, &registry).unwrap())
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

    let transition = SemanticTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    bind_semantic_lifecycle_transition(&workspace, &mut builder, &transition, 0).unwrap();

    let context_after: serde_json::Value =
        serde_json::from_str(&semantic_execution_context(&builder, &graph, &registry).unwrap())
            .unwrap();

    assert_eq!(graph.program_identity(), program_identity_before);
    assert_eq!(
        context_before["program_identity"],
        context_after["program_identity"]
    );
    assert_ne!(
        context_before["context_fingerprint"],
        context_after["context_fingerprint"]
    );
    assert_eq!(context_after["lifecycle_coverage_complete"], true);
    assert_eq!(context_after["program_identity_effect"], "none");
    assert_eq!(context_after["execution_effect"], "none");

    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let expected = graph.run(&registry, &input).unwrap().to_array();

    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_semantic_graph_receipt(
            &mut workspace,
            &builder,
            &graph,
            &registry,
            &input,
            &expected,
            0.0,
            0.0,
            "semantic-graph-check".into(),
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(receipt["authority"], "wasm_verifier");
    assert_eq!(receipt["verifier"], "CompiledGraph.verifyFlat");
    assert_eq!(receipt["reference_authority"], "burn_compiled_graph");
    assert_eq!(receipt["result"]["passed"], true);
    assert_eq!(
        receipt["semantic_execution_context"]["context_fingerprint"],
        receipt["semantic_context_fingerprint"]
    );
    assert_eq!(
        receipt["semantic_execution_context"]["program_identity"],
        receipt["program_identity"]
    );
    assert_eq!(
        receipt["semantic_execution_context"]["lifecycle_coverage_complete"],
        true
    );

    let ledger = workspace_proof_ledger(&workspace);
    assert!(ledger.contains("semantic_context_fingerprint"));
    assert!(ledger.contains("CompiledGraph.verifyFlat"));
}

#[test]
fn foreign_builder_fails_before_receipt_allocation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();

    let first = AgentLayerSpec::relu(1);
    let second = AgentLayerSpec::relu(2);
    registry.init_agent_layer(&first).unwrap();
    registry.init_agent_layer(&second).unwrap();

    let mut original = AgentGraphBuilder::new(3).unwrap();
    original.add_unary(&first, 0, 1).unwrap();
    let graph = original.compile_with_output(&registry, 1).unwrap();

    let mut foreign = AgentGraphBuilder::new(3).unwrap();
    foreign.add_unary(&first, 0, 1).unwrap();
    foreign.add_unary(&second, 1, 2).unwrap();

    let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
    let expected = graph.run(&registry, &input).unwrap().to_array();

    let err = workspace_verify_semantic_graph_receipt(
        &mut workspace,
        &foreign,
        &graph,
        &registry,
        &input,
        &expected,
        0.0,
        0.0,
        "foreign".into(),
    )
    .unwrap_err();
    assert!(err.contains("executable identity does not match"));

    let receipt = workspace_verify_semantic_graph_receipt(
        &mut workspace,
        &original,
        &graph,
        &registry,
        &input,
        &expected,
        0.0,
        0.0,
        "exact".into(),
    )
    .unwrap();

    assert!(receipt.contains("\"receipt_id\":1"));
}
