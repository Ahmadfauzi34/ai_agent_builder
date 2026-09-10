use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::{
    finalize_initialized_binary, finalize_initialized_unary, workspace_init_binary,
    workspace_init_unary,
};

#[test]
fn unary_failure_after_workspace_sync_restores_exact_state_and_registry_accounting() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut too_small_builder = AgentGraphBuilder::new(1).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "linear-reservation".into())
        .unwrap();
    let spec = AgentLayerSpec::linear(layer_id, 2, 2, true).unwrap();
    let snapshot_before = workspace.snapshot();
    let layer_before = workspace.get("_layers".into(), layer_id.to_string());
    let checkpoint = workspace.clone();

    // Reproduce the state immediately after Registry init but before final graph commit.
    // output slot 1 is valid in Workspace but intentionally invalid in this 1-slot Builder,
    // so finalize_initialized_unary reaches syncLayer and then fails in addUnary.
    let output_slot = workspace
        .reserve_slot(format!("layer:{layer_id}"))
        .unwrap();
    assert_eq!(output_slot, 1);
    registry.init_agent_layer(&spec).unwrap();
    assert!(registry.layer_exists(spec.layer_type(), layer_id));
    assert!(registry.total_params() > 0);

    let err = finalize_initialized_unary(
        &mut workspace,
        &checkpoint,
        &mut too_small_builder,
        &mut registry,
        &spec,
        0,
        output_slot,
        "linear-runtime".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside num_slots 1"));
    assert!(err.contains("transaction rolled back to pre-call state"));
    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(
        workspace.get("_layers".into(), layer_id.to_string()),
        layer_before
    );
    assert!(workspace
        .get("_slots".into(), output_slot.to_string())
        .contains("\"state\":\"free\""));
    assert!(!registry.layer_exists(spec.layer_type(), layer_id));
    assert!(registry
        .layer_init_fingerprint(spec.layer_type(), layer_id)
        .is_err());
    assert_eq!(registry.total_params(), 0);
    assert_eq!(too_small_builder.num_steps(), 0);

    let mut compatible_builder = AgentGraphBuilder::new(3).unwrap();
    let retried = workspace_init_unary(
        &mut workspace,
        &mut compatible_builder,
        &mut registry,
        &spec,
        0,
        "linear-retry".into(),
    )
    .unwrap();
    assert_eq!(retried, output_slot);
    assert!(registry.layer_exists(spec.layer_type(), layer_id));
    assert_eq!(compatible_builder.num_steps(), 1);
}

#[test]
fn binary_failure_after_workspace_sync_restores_exact_state_and_is_retriable() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut too_small_builder = AgentGraphBuilder::new(1).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "add-reservation".into())
        .unwrap();
    let spec = AgentLayerSpec::add(layer_id);
    let snapshot_before = workspace.snapshot();
    let layer_before = workspace.get("_layers".into(), layer_id.to_string());
    let checkpoint = workspace.clone();

    let output_slot = workspace
        .reserve_slot(format!("layer:{layer_id}"))
        .unwrap();
    assert_eq!(output_slot, 1);
    registry.init_agent_layer(&spec).unwrap();
    assert!(registry.layer_exists(spec.layer_type(), layer_id));

    let err = finalize_initialized_binary(
        &mut workspace,
        &checkpoint,
        &mut too_small_builder,
        &mut registry,
        &spec,
        0,
        0,
        output_slot,
        "add-runtime".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside num_slots 1"));
    assert!(err.contains("transaction rolled back to pre-call state"));
    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(
        workspace.get("_layers".into(), layer_id.to_string()),
        layer_before
    );
    assert!(workspace
        .get("_slots".into(), output_slot.to_string())
        .contains("\"state\":\"free\""));
    assert!(!registry.layer_exists(spec.layer_type(), layer_id));
    assert!(registry
        .layer_init_fingerprint(spec.layer_type(), layer_id)
        .is_err());
    assert_eq!(too_small_builder.num_steps(), 0);

    let mut compatible_builder = AgentGraphBuilder::new(3).unwrap();
    let retried = workspace_init_binary(
        &mut workspace,
        &mut compatible_builder,
        &mut registry,
        &spec,
        0,
        0,
        "add-retry".into(),
    )
    .unwrap();
    assert_eq!(retried, output_slot);
    assert!(registry.layer_exists(spec.layer_type(), layer_id));
    assert_eq!(compatible_builder.num_steps(), 1);
}
