use burn::tensor::backend::Backend;
use burn_research::agent::AgentLayerSpec;
use burn_research::protocol::LAYER_NORM;
use burn_research::registry::LayerRegistry;
use burn_research::{WasmBackend, WasmTensor};

#[test]
fn ndarray_backend_is_explicitly_non_autodiff() {
    let device = <WasmBackend as Backend>::Device::default();
    assert!(
        !<WasmBackend as Backend>::ad_enabled(&device),
        "NdArray baseline must remain non-autodiff; enabling AD changes Burn module semantics and requires a contract audit"
    );
}

#[test]
fn batch_norm_forward_keeps_running_state_unchanged_on_ndarray_baseline() {
    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::batch_norm(7, 2, None).unwrap();
    registry.init_agent_layer(&spec).unwrap();

    let state_before = registry.get_layer_state(7, LAYER_NORM).unwrap();
    let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);

    registry.forward_layer(7, LAYER_NORM, &input).unwrap();
    registry.forward_layer(7, LAYER_NORM, &input).unwrap();

    let state_after = registry.get_layer_state(7, LAYER_NORM).unwrap();
    assert_eq!(
        state_after, state_before,
        "Burn BatchNorm must stay on inference semantics for the non-AD NdArray baseline"
    );
}

#[test]
fn rejected_batch_norm_shape_does_not_mutate_module_state() {
    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::batch_norm(8, 2, None).unwrap();
    registry.init_agent_layer(&spec).unwrap();

    let state_before = registry.get_layer_state(8, LAYER_NORM).unwrap();
    let wrong_channels = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);

    let err = registry
        .forward_layer(8, LAYER_NORM, &wrong_channels)
        .unwrap_err();
    assert!(err.contains("expected axis 1 size 2"));

    let state_after = registry.get_layer_state(8, LAYER_NORM).unwrap();
    assert_eq!(state_after, state_before);
}
