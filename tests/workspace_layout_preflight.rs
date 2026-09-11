use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::protocol::LAYER_NORM;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{
    workspace_compile, workspace_init_unary, workspace_wire_unary,
};
use burn_research::WasmTensor;

#[test]
fn canonical_init_rejects_known_layout_mismatch_before_mutation() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(4).unwrap();

    let linear_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let linear = AgentLayerSpec::linear(linear_id, 4, 4, true).unwrap();
    let linear_out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &linear,
        0,
        "linear".into(),
    )
    .unwrap();

    let norm_id = workspace.reserve_layer_id(&registry, "layernorm".into()).unwrap();
    let layer_norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
    let before = workspace.snapshot();
    let steps_before = builder.num_steps();

    let err = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &layer_norm,
        linear_out,
        "layernorm".into(),
    )
    .expect_err("known Linear -> LayerNorm layout mismatch must fail before execution");

    assert!(err.contains("layout"), "unexpected error: {err}");
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(builder.num_steps(), steps_before);
    assert!(!registry.layer_exists(LAYER_NORM, norm_id));
}

#[test]
fn canonical_wire_rejects_known_layout_mismatch_without_metadata_or_slot_mutation() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(4).unwrap();

    let linear_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let linear = AgentLayerSpec::linear(linear_id, 4, 4, true).unwrap();
    let linear_out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &linear,
        0,
        "linear".into(),
    )
    .unwrap();

    let layer_norm = AgentLayerSpec::layer_norm(99, 4, None).unwrap();
    registry.init_agent_layer(&layer_norm).unwrap();
    let before = workspace.snapshot();
    let steps_before = builder.num_steps();

    let err = workspace_wire_unary(
        &mut workspace,
        &mut builder,
        &registry,
        &layer_norm,
        linear_out,
        "manual-layernorm".into(),
    )
    .expect_err("known Linear -> LayerNorm layout mismatch must fail before wire mutation");

    assert!(err.contains("layout"), "unexpected error: {err}");
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(workspace.get("_layers".into(), "99".into()), "null");
}

#[test]
fn unknown_preserve_layout_still_defers_to_runtime() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(4).unwrap();

    let relu_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let relu = AgentLayerSpec::relu(relu_id);
    let relu_out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &relu,
        0,
        "relu".into(),
    )
    .unwrap();

    let norm_id = workspace.reserve_layer_id(&registry, "layernorm".into()).unwrap();
    let layer_norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &layer_norm,
        relu_out,
        "layernorm".into(),
    )
    .expect("preserve/unknown producer layout must defer rather than overclaim incompatibility");

    assert_eq!(builder.num_steps(), 2);
    assert!(registry.layer_exists(LAYER_NORM, norm_id));
    assert_eq!(out, 2);
}

#[test]
fn known_compatible_linear_to_batchnorm_remains_executable() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(4).unwrap();

    let linear_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
    let linear = AgentLayerSpec::linear(linear_id, 4, 4, true).unwrap();
    let linear_out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &linear,
        0,
        "linear".into(),
    )
    .unwrap();

    let norm_id = workspace.reserve_layer_id(&registry, "batchnorm".into()).unwrap();
    let batch_norm = AgentLayerSpec::batch_norm(norm_id, 4, None).unwrap();
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &batch_norm,
        linear_out,
        "batchnorm".into(),
    )
    .unwrap();

    let graph = workspace_compile(&builder, &registry, out).unwrap();
    let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
    let output = graph.run(&registry, &input).unwrap();
    assert_eq!(output.shape(), vec![1, 4, 1, 1]);
}