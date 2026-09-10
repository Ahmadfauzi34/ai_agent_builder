use burn_research::agent::AgentLayerSpec;
use burn_research::layers::linear::WasmLinear;
use burn_research::protocol::{
    LAYER_BINARY, LAYER_LINEAR, LAYER_POOL, LAYER_SHIFT,
};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;

#[test]
fn cross_config_linear_state_load_must_not_change_structural_identity() {
    let foreign = WasmLinear::new(4, 2, true);
    let foreign_state = foreign.get_state().unwrap();

    let spec = AgentLayerSpec::linear(70, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let fingerprint_before = registry
        .layer_init_fingerprint(LAYER_LINEAR, 70)
        .unwrap();
    let state_before = registry.get_layer_state(70, LAYER_LINEAR).unwrap();
    let layout_before = registry.weight_layout(70, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();

    let result = registry.load_layer_state(70, LAYER_LINEAR, &foreign_state);
    assert!(
        result.is_err(),
        "state from Linear(4->2) must not load into a live Linear(3->2)"
    );
    assert_eq!(
        registry.layer_init_fingerprint(LAYER_LINEAR, 70).unwrap(),
        fingerprint_before,
        "failed state load must preserve init identity"
    );
    assert_eq!(
        registry.get_layer_state(70, LAYER_LINEAR).unwrap(),
        state_before,
        "failed cross-config load must preserve exact module state"
    );
    assert_eq!(
        registry.weight_layout(70, LAYER_LINEAR).unwrap(),
        layout_before,
        "failed cross-config load must preserve parameter layout"
    );
    assert_eq!(registry.total_params(), params_before);

    registry
        .load_layer_state(70, LAYER_LINEAR, &state_before)
        .unwrap();
    let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    assert!(registry.forward_layer(70, LAYER_LINEAR, &input).is_ok());
}

#[test]
fn same_config_linear_state_load_preserves_init_identity_and_structure() {
    let source = WasmLinear::new(3, 2, true);
    let source_state = source.get_state().unwrap();

    let spec = AgentLayerSpec::linear(71, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let fingerprint_before = registry
        .layer_init_fingerprint(LAYER_LINEAR, 71)
        .unwrap();
    let layout_before = registry.weight_layout(71, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();

    registry
        .load_layer_state(71, LAYER_LINEAR, &source_state)
        .unwrap();

    assert_eq!(
        registry.layer_init_fingerprint(LAYER_LINEAR, 71).unwrap(),
        fingerprint_before
    );
    assert_eq!(registry.weight_layout(71, LAYER_LINEAR).unwrap(), layout_before);
    assert_eq!(registry.total_params(), params_before);

    let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    assert!(registry.forward_layer(71, LAYER_LINEAR, &input).is_ok());
}

#[test]
fn stateless_layers_reject_nonempty_state_payload_and_accept_empty_state() {
    let cases = vec![
        (
            AgentLayerSpec::max_pool1d(80, 2, None, None).unwrap(),
            LAYER_POOL,
        ),
        (AgentLayerSpec::shift_up(81, 1), LAYER_SHIFT),
        (AgentLayerSpec::add(82), LAYER_BINARY),
    ];

    let mut registry = LayerRegistry::new();
    for (spec, layer_type) in cases {
        registry.init_agent_layer(&spec).unwrap();
        let layer_id = spec.layer_id();
        let fingerprint_before = registry
            .layer_init_fingerprint(layer_type, layer_id)
            .unwrap();

        assert_eq!(
            registry.get_layer_state(layer_id, layer_type).unwrap(),
            Vec::<u8>::new(),
            "stateless layer must serialize as empty state"
        );
        assert!(
            registry
                .load_layer_state(layer_id, layer_type, &[1, 2, 3])
                .is_err(),
            "non-empty state must not be silently accepted for stateless type 0x{layer_type:02X}"
        );
        assert_eq!(
            registry
                .layer_init_fingerprint(layer_type, layer_id)
                .unwrap(),
            fingerprint_before
        );
        assert!(registry.layer_exists(layer_type, layer_id));
        registry
            .load_layer_state(layer_id, layer_type, &[])
            .unwrap();
    }
}

#[test]
fn valid_same_length_weight_mutation_preserves_structural_identity() {
    let spec = AgentLayerSpec::linear(90, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let fingerprint_before = registry
        .layer_init_fingerprint(LAYER_LINEAR, 90)
        .unwrap();
    let layout_before = registry.weight_layout(90, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();
    let weights_before = registry.get_weights_flat(90, LAYER_LINEAR).unwrap();
    let replacement = weights_before
        .iter()
        .enumerate()
        .map(|(index, value)| value + (index as f32 + 1.0) * 0.01)
        .collect::<Vec<_>>();

    registry
        .set_weights_flat(90, LAYER_LINEAR, &replacement)
        .unwrap();

    assert_eq!(
        registry.layer_init_fingerprint(LAYER_LINEAR, 90).unwrap(),
        fingerprint_before
    );
    assert_eq!(registry.weight_layout(90, LAYER_LINEAR).unwrap(), layout_before);
    assert_eq!(registry.total_params(), params_before);
    assert_eq!(registry.get_weights_flat(90, LAYER_LINEAR).unwrap(), replacement);
}
