use burn_research::agent::AgentLayerSpec;
use burn_research::layers::activation::WasmActivation;
use burn_research::layers::conv::WasmConv;
use burn_research::layers::custom::ghost::WasmGhostModule;
use burn_research::layers::custom::seblock::WasmSeBlock;
use burn_research::layers::embedding::WasmEmbedding;
use burn_research::layers::linear::WasmLinear;
use burn_research::layers::norm::WasmNorm;
use burn_research::protocol::{
    LAYER_BINARY, LAYER_LINEAR, LAYER_NORM, LAYER_POOL, LAYER_SHIFT,
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
fn cross_variant_norm_state_must_fail_closed_without_panicking_or_mutating() {
    let foreign = WasmNorm::new_batch_norm(2, None);
    let foreign_state = foreign.get_state().unwrap();

    let spec = AgentLayerSpec::layer_norm(72, 2, None).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let fingerprint_before = registry.layer_init_fingerprint(LAYER_NORM, 72).unwrap();
    let state_before = registry.get_layer_state(72, LAYER_NORM).unwrap();
    let layout_before = registry.weight_layout(72, LAYER_NORM).unwrap();
    let params_before = registry.total_params();

    let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        registry.load_layer_state(72, LAYER_NORM, &foreign_state)
    }));
    assert!(call.is_ok(), "foreign valid Norm record must not panic");
    assert!(
        call.unwrap().is_err(),
        "BatchNorm state must not load into a live LayerNorm"
    );
    assert_eq!(
        registry.layer_init_fingerprint(LAYER_NORM, 72).unwrap(),
        fingerprint_before
    );
    assert_eq!(registry.get_layer_state(72, LAYER_NORM).unwrap(), state_before);
    assert_eq!(registry.weight_layout(72, LAYER_NORM).unwrap(), layout_before);
    assert_eq!(registry.total_params(), params_before);

    registry
        .load_layer_state(72, LAYER_NORM, &state_before)
        .unwrap();
}

#[test]
fn state_decoder_rejects_trailing_bytes_without_mutating_live_layer() {
    let spec = AgentLayerSpec::linear(73, 3, 2, true).unwrap();
    let mut registry = LayerRegistry::new();
    registry.init_agent_layer(&spec).unwrap();

    let fingerprint_before = registry
        .layer_init_fingerprint(LAYER_LINEAR, 73)
        .unwrap();
    let state_before = registry.get_layer_state(73, LAYER_LINEAR).unwrap();
    let layout_before = registry.weight_layout(73, LAYER_LINEAR).unwrap();
    let params_before = registry.total_params();

    let mut tainted = state_before.clone();
    tainted.extend_from_slice(&[0xAA, 0x55, 0x01]);

    assert!(
        registry
            .load_layer_state(73, LAYER_LINEAR, &tainted)
            .is_err(),
        "a canonical Burn state followed by trailing bytes must be rejected"
    );
    assert_eq!(
        registry.layer_init_fingerprint(LAYER_LINEAR, 73).unwrap(),
        fingerprint_before
    );
    assert_eq!(registry.get_layer_state(73, LAYER_LINEAR).unwrap(), state_before);
    assert_eq!(registry.weight_layout(73, LAYER_LINEAR).unwrap(), layout_before);
    assert_eq!(registry.total_params(), params_before);

    registry
        .load_layer_state(73, LAYER_LINEAR, &state_before)
        .unwrap();
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

#[test]
fn direct_conv_rejects_cross_variant_state_without_panicking_or_mutating() {
    let foreign = WasmConv::new_conv2d(3, 4, 3, 3, None, None, None, None);
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmConv::new_conv1d(3, 4, 3, None, None);
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        target.load_state(&foreign_state)
    }));
    assert!(call.is_ok(), "foreign valid Conv record must not panic");
    assert!(call.unwrap().is_err(), "Conv2d state must not load into Conv1d");
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_conv_rejects_same_variant_shape_mismatch_without_mutating() {
    let foreign = WasmConv::new_conv2d(5, 4, 3, 3, None, None, None, None);
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmConv::new_conv2d(3, 4, 3, 3, None, None, None, None);
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    assert!(target.load_state(&foreign_state).is_err());
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_embedding_rejects_shape_mismatch_without_mutating() {
    let foreign = WasmEmbedding::new(11, 4);
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmEmbedding::new(10, 4);
    let before = target.get_state().unwrap();
    let dims_before = target.weight_dims();
    let params_before = target.num_params();

    assert!(target.load_state(&foreign_state).is_err());
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.weight_dims(), dims_before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_activation_rejects_cross_variant_state_without_panicking_or_mutating() {
    let foreign = WasmActivation::new_relu();
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmActivation::new_prelu(Some(3), None);
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        target.load_state(&foreign_state)
    }));
    assert!(call.is_ok(), "foreign valid Activation record must not panic");
    assert!(call.unwrap().is_err(), "Relu state must not load into PRelu");
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_activation_rejects_prelu_shape_mismatch_without_mutating() {
    let foreign = WasmActivation::new_prelu(Some(4), None);
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmActivation::new_prelu(Some(3), None);
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    assert!(target.load_state(&foreign_state).is_err());
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_ghost_rejects_structural_state_mismatch_without_mutating() {
    let foreign = WasmGhostModule::new(6, 8, 3, 3, Some(2), None, None, None, None);
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmGhostModule::new(4, 8, 3, 3, Some(2), None, None, None, None);
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    assert!(target.load_state(&foreign_state).is_err());
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}

#[test]
fn direct_seblock_rejects_structural_state_mismatch_without_mutating() {
    let foreign = WasmSeBlock::new(20, Some(4));
    let foreign_state = foreign.get_state().unwrap();
    let mut target = WasmSeBlock::new(16, Some(4));
    let before = target.get_state().unwrap();
    let params_before = target.num_params();

    assert!(target.load_state(&foreign_state).is_err());
    assert_eq!(target.get_state().unwrap(), before);
    assert_eq!(target.num_params(), params_before);
    target.load_state(&before).unwrap();
}
