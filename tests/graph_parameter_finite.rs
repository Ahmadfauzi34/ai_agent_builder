use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::protocol::LAYER_LINEAR;
use burn_research::registry::LayerRegistry;

fn init(registry: &mut LayerRegistry, spec: &AgentLayerSpec) {
    registry.init_agent_layer(spec).unwrap();
}

fn two_linear_graph() -> (LayerRegistry, burn_research::graph::CompiledGraph) {
    let mut registry = LayerRegistry::new();
    let first = AgentLayerSpec::linear(11, 2, 2, true).unwrap();
    let relu = AgentLayerSpec::relu(12);
    let second = AgentLayerSpec::linear(13, 2, 1, true).unwrap();
    init(&mut registry, &first);
    init(&mut registry, &relu);
    init(&mut registry, &second);

    let mut builder = AgentGraphBuilder::new(4).unwrap();
    builder.add_unary(&first, 0, 1).unwrap();
    builder.add_unary(&relu, 1, 2).unwrap();
    builder.add_unary(&second, 2, 3).unwrap();
    builder.set_output(3).unwrap();
    let graph = builder.compile(&registry).unwrap();
    (registry, graph)
}

fn assert_rejected_without_mutation(index_kind: &str, value: f32) {
    let (mut registry, graph) = two_linear_graph();
    let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
    assert_eq!(binding.owners().len(), 2);

    let index = match index_kind {
        "first" => binding.owners()[0].offset(),
        "later" => binding.owners()[1].offset(),
        other => panic!("unknown index kind: {other}"),
    };

    let before_first = registry.get_weights_flat(11, LAYER_LINEAR).unwrap();
    let before_second = registry.get_weights_flat(13, LAYER_LINEAR).unwrap();
    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();
    let layout = binding.layout_json();

    let mut candidate = vec![0.25; binding.total_len()];
    candidate[index] = value;

    let err = binding
        .apply_flat(&graph, &mut registry, &candidate)
        .unwrap_err();
    assert!(err.contains("non-finite"), "unexpected error: {err}");
    assert!(
        err.contains(&format!("index {index}")),
        "error should identify the non-finite coordinate: {err}"
    );

    assert_eq!(
        registry.get_weights_flat(11, LAYER_LINEAR).unwrap(),
        before_first,
        "rejected candidate mutated the first owner"
    );
    assert_eq!(
        registry.get_weights_flat(13, LAYER_LINEAR).unwrap(),
        before_second,
        "rejected candidate mutated the second owner"
    );
    assert_eq!(graph.program_identity(), program_identity);
    assert_eq!(binding.identity_json(), binding_identity);
    assert_eq!(binding.layout_json(), layout);

    let rebuilt = GraphParameterBinding::build(&graph, &registry).unwrap();
    assert_eq!(rebuilt.identity_json(), binding_identity);
    assert_eq!(rebuilt.layout_json(), layout);

    let retry = vec![0.5; binding.total_len()];
    binding
        .apply_flat(&graph, &mut registry, &retry)
        .expect("valid finite retry must succeed after non-finite rejection");
    assert_eq!(binding.read_flat(&graph, &registry).unwrap(), retry);
}

#[test]
fn nan_in_first_owner_rejects_before_any_mutation() {
    assert_rejected_without_mutation("first", f32::NAN);
}

#[test]
fn nan_in_later_owner_rejects_before_earlier_owner_mutation() {
    assert_rejected_without_mutation("later", f32::NAN);
}

#[test]
fn positive_infinity_rejects_before_any_mutation() {
    assert_rejected_without_mutation("later", f32::INFINITY);
}

#[test]
fn negative_infinity_rejects_before_any_mutation() {
    assert_rejected_without_mutation("first", f32::NEG_INFINITY);
}
