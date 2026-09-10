use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::layers::activation::WasmActivation;
use burn_research::protocol::{ACT_RELU, LAYER_ACTIVATION, OP_INIT, PacketHeader};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{
    workspace_capabilities, workspace_compile, workspace_init_unary, workspace_wire_unary,
};
use burn_research::WasmTensor;

fn relu_input() -> WasmTensor {
    WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1])
}

fn assert_relu_output(output: WasmTensor) {
    assert_eq!(output.to_array(), vec![0.0, 3.0]);
}

fn raw_relu_init(registry: &mut LayerRegistry, layer_id: u32) {
    let payload = layer_id.to_le_bytes().to_vec();
    let header = PacketHeader {
        opcode: OP_INIT,
        layer_type: LAYER_ACTIVATION,
        variant: ACT_RELU,
        flags: 0,
        payload_len: payload.len() as u32,
    };
    registry.init_layer(&header, &payload).unwrap();
}

fn raw_unary_plan(layer_type: u8, layer_id: u32) -> Vec<u8> {
    let mut plan = Vec::new();
    plan.extend_from_slice(&1u32.to_le_bytes()); // num_steps
    plan.extend_from_slice(&2u32.to_le_bytes()); // num_slots
    plan.push(1); // unary arity
    plan.push(layer_type);
    plan.extend_from_slice(&layer_id.to_le_bytes());
    plan.push(0); // input slot
    plan.push(0); // unused second input mirrors input
    plan.push(1); // output slot
    plan.push(1); // graph output slot
    plan
}

#[test]
fn level4_workspace_canonical_path_executes() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(2).unwrap();

    let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(id);
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    let graph = workspace_compile(&builder, &registry, out).unwrap();
    assert_relu_output(graph.run(&registry, &relu_input()).unwrap());
}

#[test]
fn level3_typed_builder_path_executes_without_workspace() {
    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::relu(31);
    registry.init_agent_layer(&spec).unwrap();

    let mut builder = AgentGraphBuilder::new(2).unwrap();
    builder.add_unary(&spec, 0, 1).unwrap();
    builder.set_output(1).unwrap();
    let graph = builder.compile(&registry).unwrap();
    assert_relu_output(graph.run(&registry, &relu_input()).unwrap());
}

#[test]
fn level2_registry_graph_path_executes_without_builder() {
    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::relu(32);
    registry.init_agent_layer(&spec).unwrap();

    let graph = registry
        .compile_graph(&raw_unary_plan(LAYER_ACTIVATION, 32))
        .unwrap();
    assert_relu_output(graph.run(&registry, &relu_input()).unwrap());
}

#[test]
fn level1_raw_protocol_path_executes_without_typed_spec() {
    let mut registry = LayerRegistry::new();
    raw_relu_init(&mut registry, 33);

    let graph = registry
        .compile_graph(&raw_unary_plan(LAYER_ACTIVATION, 33))
        .unwrap();
    assert_relu_output(graph.run(&registry, &relu_input()).unwrap());
}

#[test]
fn raw_initialized_layer_can_rejoin_canonical_workspace() {
    let mut registry = LayerRegistry::new();
    raw_relu_init(&mut registry, 42);

    let mut workspace = AgentWorkspace::new(2).unwrap();
    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let matching_spec = AgentLayerSpec::relu(42);

    let out = workspace_wire_unary(
        &mut workspace,
        &mut builder,
        &registry,
        &matching_spec,
        0,
        "raw-rejoined".into(),
    )
    .unwrap();
    let graph = workspace_compile(&builder, &registry, out).unwrap();
    assert_relu_output(graph.run(&registry, &relu_input()).unwrap());
    assert!(workspace
        .get("_layers".into(), "42".into())
        .contains("raw-rejoined"));
}

#[test]
fn direct_wasm_layer_wrapper_remains_burn_adjacent_escape_hatch() {
    let relu = WasmActivation::new_relu();
    assert_relu_output(relu.forward(&relu_input()));
}

#[test]
fn workspace_capabilities_keep_lower_level_escape_hatches_explicit() {
    let capabilities = workspace_capabilities();
    for required in [
        "AgentLayerSpec",
        "AgentGraphBuilder",
        "LayerRegistry",
        "raw_protocol",
    ] {
        assert!(
            capabilities.contains(required),
            "missing progressive-disclosure escape hatch: {required}"
        );
    }
}
