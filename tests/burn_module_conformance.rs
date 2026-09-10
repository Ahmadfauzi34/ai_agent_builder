use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::layers::linear::WasmLinear;
use burn_research::protocol::{
    LAYER_ACTIVATION, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM,
    LAYER_POOL, LAYER_SEBLOCK,
};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn expect_err_without_debug<T, E>(result: Result<T, E>) -> E {
    match result {
        Ok(_) => panic!("expected operation to fail"),
        Err(err) => err,
    }
}

#[test]
fn direct_burn_record_registry_and_compiled_graph_preserve_linear_semantics() {
    let direct = WasmLinear::new(3, 2, true);
    let direct_state = direct.get_state().unwrap();
    let input = WasmTensor::new(&[1.0, -2.0, 0.5, 3.0, 1.0, -1.0], &[2, 3, 1, 1]);
    let direct_output = direct.forward(&input).to_array();

    let spec = AgentLayerSpec::linear(11, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();
    registry
        .load_layer_state(spec.layer_id(), LAYER_LINEAR, &direct_state)
        .unwrap();

    let registry_output = registry
        .forward_layer(spec.layer_id(), LAYER_LINEAR, &input)
        .unwrap()
        .to_array();
    assert_close(&registry_output, &direct_output, 1e-6);

    let mut builder = AgentGraphBuilder::new(2).unwrap();
    builder.add_unary(&spec, 0, 1).unwrap();
    let graph = builder.compile_with_output(&registry, 1).unwrap();
    let graph_output = graph.run(&registry, &input).unwrap().to_array();
    assert_close(&graph_output, &direct_output, 1e-6);
}

#[test]
fn malformed_record_load_is_no_mutation_and_registry_remains_retryable() {
    let spec = AgentLayerSpec::linear(12, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let state_before = registry.get_layer_state(12, LAYER_LINEAR).unwrap();
    let weights_before = registry.get_weights_flat(12, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();

    assert!(registry
        .load_layer_state(12, LAYER_LINEAR, &[0xde, 0xad, 0xbe, 0xef])
        .is_err());
    assert_eq!(registry.get_layer_state(12, LAYER_LINEAR).unwrap(), state_before);
    assert_eq!(registry.get_weights_flat(12, LAYER_LINEAR).unwrap(), weights_before);
    assert_eq!(registry.total_params(), params_before);

    registry
        .load_layer_state(12, LAYER_LINEAR, &state_before)
        .unwrap();
    let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    assert!(registry.forward_layer(12, LAYER_LINEAR, &input).is_ok());
}

#[test]
fn malformed_record_bytes_fail_closed_for_every_stateful_layer_family() {
    let cases = vec![
        (AgentLayerSpec::linear(40, 3, 2, true).unwrap(), LAYER_LINEAR),
        (AgentLayerSpec::batch_norm(41, 2, None).unwrap(), LAYER_NORM),
        (AgentLayerSpec::conv1d(42, 1, 2, 3, None, None).unwrap(), LAYER_CONV),
        (AgentLayerSpec::embedding(43, 4, 3).unwrap(), LAYER_EMBEDDING),
        (AgentLayerSpec::prelu(44, 1, 0.25).unwrap(), LAYER_ACTIVATION),
        (
            AgentLayerSpec::ghost(45, 2, 4, 1, 1, 2, None, None, None, None).unwrap(),
            LAYER_GHOST,
        ),
        (AgentLayerSpec::se_block(46, 4, 2).unwrap(), LAYER_SEBLOCK),
    ];

    let mut registry = LayerRegistry::new();
    for (spec, layer_type) in cases {
        registry.init_agent_layer(&spec).unwrap();
        let layer_id = spec.layer_id();
        let state_before = registry.get_layer_state(layer_id, layer_type).unwrap();
        let params_before = registry.total_params();

        let error = registry
            .load_layer_state(layer_id, layer_type, &[0xde, 0xad, 0xbe, 0xef])
            .expect_err("malformed Burn record must be rejected");
        assert!(
            error.contains("invalid Burn bincode record"),
            "unexpected malformed-state error for type 0x{layer_type:02X}: {error}"
        );
        assert_eq!(
            registry.get_layer_state(layer_id, layer_type).unwrap(),
            state_before,
            "malformed state mutated layer type 0x{layer_type:02X} id {layer_id}"
        );
        assert_eq!(registry.total_params(), params_before);

        registry
            .load_layer_state(layer_id, layer_type, &state_before)
            .unwrap();
    }
}

#[test]
fn wrong_length_weight_update_is_no_mutation_and_retryable() {
    let spec = AgentLayerSpec::linear(13, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let state_before = registry.get_layer_state(13, LAYER_LINEAR).unwrap();
    let weights_before = registry.get_weights_flat(13, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();
    let wrong = vec![0.0; weights_before.len() - 1];

    assert!(registry.set_weights_flat(13, LAYER_LINEAR, &wrong).is_err());
    assert_eq!(registry.get_layer_state(13, LAYER_LINEAR).unwrap(), state_before);
    assert_eq!(registry.get_weights_flat(13, LAYER_LINEAR).unwrap(), weights_before);
    assert_eq!(registry.total_params(), params_before);

    let replacement = weights_before
        .iter()
        .map(|value| value + 0.125)
        .collect::<Vec<_>>();
    registry
        .set_weights_flat(13, LAYER_LINEAR, &replacement)
        .unwrap();
    assert_eq!(registry.get_weights_flat(13, LAYER_LINEAR).unwrap(), replacement);
    assert_eq!(registry.total_params(), params_before);
}

#[test]
fn batch_norm_optimizer_bridge_exposes_only_trainable_state_and_forward_is_non_mutating() {
    let spec = AgentLayerSpec::batch_norm(21, 2, None).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let layout = registry.weight_layout(21, LAYER_NORM).unwrap();
    assert!(layout.contains("gamma"));
    assert!(layout.contains("beta"));
    assert!(!layout.contains("running_mean"));
    assert!(!layout.contains("running_var"));

    let weights = registry.get_weights_flat(21, LAYER_NORM).unwrap();
    assert_eq!(weights.len(), 4, "BatchNorm(2) optimizer bridge must be gamma(2)+beta(2)");

    let state_before_bad_update = registry.get_layer_state(21, LAYER_NORM).unwrap();
    assert!(registry.set_weights_flat(21, LAYER_NORM, &[1.0, 1.0, 0.0]).is_err());
    assert_eq!(
        registry.get_layer_state(21, LAYER_NORM).unwrap(),
        state_before_bad_update
    );

    registry
        .set_weights_flat(21, LAYER_NORM, &[2.0, 3.0, 0.5, -0.25])
        .unwrap();
    let state_after_weight_update = registry.get_layer_state(21, LAYER_NORM).unwrap();
    let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
    registry.forward_layer(21, LAYER_NORM, &input).unwrap();
    registry.forward_layer(21, LAYER_NORM, &input).unwrap();
    assert_eq!(
        registry.get_layer_state(21, LAYER_NORM).unwrap(),
        state_after_weight_update,
        "non-AD NdArray BatchNorm forward must not mutate serialized running state"
    );
}

#[test]
fn linear_shape_adapter_rejects_hidden_spatial_data_without_state_mutation_then_retries() {
    let spec = AgentLayerSpec::linear(31, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();
    let state_before = registry.get_layer_state(31, LAYER_LINEAR).unwrap();

    let invalid = WasmTensor::new(&[1.0; 6], &[1, 3, 2, 1]);
    let err = expect_err_without_debug(registry.forward_layer(31, LAYER_LINEAR, &invalid));
    assert!(err.contains("axis 2"));
    assert_eq!(registry.get_layer_state(31, LAYER_LINEAR).unwrap(), state_before);

    let valid = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    assert!(registry.forward_layer(31, LAYER_LINEAR, &valid).is_ok());
}

#[test]
fn one_dimensional_conv_and_pool_reject_hidden_width_then_accept_canonical_rank4_adapter() {
    let conv = AgentLayerSpec::conv1d(32, 1, 2, 3, None, None).unwrap();
    let pool = AgentLayerSpec::max_pool1d(33, 2, None, None).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&conv).unwrap();
    registry.init_agent_layer(&pool).unwrap();

    let conv_state_before = registry.get_layer_state(32, LAYER_CONV).unwrap();
    let invalid = WasmTensor::new(&[1.0; 10], &[1, 1, 5, 2]);
    let conv_err = expect_err_without_debug(registry.forward_layer(32, LAYER_CONV, &invalid));
    assert!(conv_err.contains("axis 3"));
    assert_eq!(registry.get_layer_state(32, LAYER_CONV).unwrap(), conv_state_before);

    let pool_err = expect_err_without_debug(registry.forward_layer(33, LAYER_POOL, &invalid));
    assert!(pool_err.contains("axis 3"));

    let valid = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5, 1]);
    let conv_out = registry.forward_layer(32, LAYER_CONV, &valid).unwrap();
    assert_eq!(conv_out.shape()[3], 1);
    let pool_out = registry.forward_layer(33, LAYER_POOL, &valid).unwrap();
    assert_eq!(pool_out.shape()[3], 1);
}

#[test]
fn conv2d_adapter_preserves_canonical_rank4_spatial_semantics() {
    let spec = AgentLayerSpec::conv2d(
        35,
        1,
        2,
        3,
        3,
        None,
        None,
        Some(1),
        Some(1),
    )
    .unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let input = WasmTensor::new(&[1.0; 16], &[1, 1, 4, 4]);
    let output = registry.forward_layer(35, LAYER_CONV, &input).unwrap();
    assert_eq!(output.shape(), vec![1, 2, 4, 4]);
}

#[test]
fn embedding_validates_float_bridge_indices_before_burn_integer_conversion_and_retries() {
    let spec = AgentLayerSpec::embedding(34, 4, 3).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();
    let state_before = registry.get_layer_state(34, LAYER_EMBEDDING).unwrap();

    for invalid_values in [
        vec![0.0, 1.5],
        vec![-1.0, 1.0],
        vec![0.0, 4.0],
        vec![0.0, f32::NAN],
    ] {
        let invalid = WasmTensor::new(&invalid_values, &[1, 2, 1, 1]);
        assert!(registry.forward_layer(34, LAYER_EMBEDDING, &invalid).is_err());
        assert_eq!(registry.get_layer_state(34, LAYER_EMBEDDING).unwrap(), state_before);
    }

    let valid = WasmTensor::new(&[0.0, 1.0, 3.0], &[1, 3, 1, 1]);
    let output = registry.forward_layer(34, LAYER_EMBEDDING, &valid).unwrap();
    assert_eq!(output.shape(), vec![1, 3, 3, 1]);
}
