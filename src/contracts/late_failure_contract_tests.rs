use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::protocol::LAYER_ACTIVATION;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::{workspace_init_unary, workspace_wire_unary};

fn fill_workspace_to_row_limit(workspace: &mut AgentWorkspace) {
    let mut reached_limit = false;
    for index in 0..2048u32 {
        let result = workspace.put(
            "late_failure_fill".into(),
            format!("row-{index}"),
            "probe".into(),
            "filled".into(),
            "x".into(),
        );
        if result.is_err() {
            reached_limit = true;
            break;
        }
    }
    assert!(reached_limit, "workspace row limit was not reached by adversarial fill");
}

#[test]
fn output_slot_exhaustion_fails_before_registry_initialization_and_retry_succeeds() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(2).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu-reservation".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let occupied = workspace.reserve_slot("occupied".into()).unwrap();

    let snapshot_before = workspace.snapshot();
    let layer_before = workspace.get("_layers".into(), layer_id.to_string());
    let steps_before = builder.num_steps();

    let err = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap_err();

    assert!(err.contains("no free slot"));
    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(workspace.get("_layers".into(), layer_id.to_string()), layer_before);
    assert_eq!(builder.num_steps(), steps_before);
    assert!(!registry.layer_exists(LAYER_ACTIVATION, layer_id));

    workspace.release_slot(occupied).unwrap();
    let output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    assert_eq!(output, occupied);
    assert!(registry.layer_exists(LAYER_ACTIVATION, layer_id));
    assert_eq!(builder.num_steps(), steps_before + 1);
}

#[test]
fn builder_workspace_slot_mismatch_rolls_back_output_reservation_before_registry_init() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut too_small_builder = AgentGraphBuilder::new(1).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu-reservation".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);

    let snapshot_before = workspace.snapshot();
    let layer_before = workspace.get("_layers".into(), layer_id.to_string());

    let err = workspace_init_unary(
        &mut workspace,
        &mut too_small_builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside builder num_slots"));
    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(workspace.get("_layers".into(), layer_id.to_string()), layer_before);
    assert_eq!(too_small_builder.num_steps(), 0);
    assert!(!registry.layer_exists(LAYER_ACTIVATION, layer_id));

    let mut compatible_builder = AgentGraphBuilder::new(3).unwrap();
    let output = workspace_init_unary(
        &mut workspace,
        &mut compatible_builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    assert_eq!(output, 1);
    assert_eq!(compatible_builder.num_steps(), 1);
}

#[test]
fn full_workspace_and_max_helper_label_do_not_create_a_post_init_sync_failure() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(3).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "reserved-before-fill".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);

    fill_workspace_to_row_limit(&mut workspace);
    let full_snapshot = workspace.snapshot();
    assert!(full_snapshot.contains("\"rows\":1024"));

    // P2 declares the helper label boundary as 4000 bytes. syncLayer rewrites the
    // already-existing _layers row, so neither the row cap nor the maximal label
    // should introduce a late failure after Registry initialization.
    let label = "x".repeat(4000);
    let output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        label,
    )
    .unwrap();

    assert_eq!(output, 1);
    assert!(registry.layer_exists(LAYER_ACTIVATION, layer_id));
    assert!(workspace
        .get("_layers".into(), layer_id.to_string())
        .contains("initialized"));
    assert_eq!(builder.num_steps(), 1);
    assert!(workspace.snapshot().contains("\"rows\":1024"));
}

#[test]
fn registry_destroy_is_a_complete_post_init_abort_primitive_for_parametric_layers() {
    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::linear(77, 2, 3, true).unwrap();

    registry.init_agent_layer(&spec).unwrap();
    assert!(registry.layer_exists(spec.layer_type(), spec.layer_id()));
    assert!(registry.total_params() > 0);
    assert!(registry
        .layer_init_fingerprint(spec.layer_type(), spec.layer_id())
        .is_ok());

    assert!(registry.destroy_layer(spec.layer_id(), spec.layer_type()));
    assert!(!registry.layer_exists(spec.layer_type(), spec.layer_id()));
    assert_eq!(registry.total_params(), 0);
    assert!(registry
        .layer_init_fingerprint(spec.layer_type(), spec.layer_id())
        .is_err());
}

#[test]
fn simulated_abort_immediately_after_registry_init_can_restore_exact_retriable_control_state() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(3).unwrap();

    let layer_id = workspace
        .reserve_layer_id(&registry, "retryable-reservation".into())
        .unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let snapshot_before = workspace.snapshot();
    let layer_before = workspace.get("_layers".into(), layer_id.to_string());

    // Reproduce the mutation order used by workspaceInit*: reserve output, then
    // initialize Registry. Abort before workspace metadata or graph commit.
    let output_slot = workspace.reserve_slot(format!("layer:{layer_id}")).unwrap();
    registry.init_agent_layer(&spec).unwrap();
    assert!(registry.layer_exists(LAYER_ACTIVATION, layer_id));

    assert!(registry.destroy_layer(layer_id, LAYER_ACTIVATION));
    workspace.release_slot(output_slot).unwrap();

    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(workspace.get("_layers".into(), layer_id.to_string()), layer_before);
    assert!(!registry.layer_exists(LAYER_ACTIVATION, layer_id));
    assert_eq!(builder.num_steps(), 0);

    let retried_output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "retry".into(),
    )
    .unwrap();
    assert_eq!(retried_output, output_slot);
    assert!(registry.layer_exists(LAYER_ACTIVATION, layer_id));
    assert_eq!(builder.num_steps(), 1);
}

#[test]
fn manual_wire_identity_failure_remains_precommit_and_recoverable() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(3).unwrap();

    let live = AgentLayerSpec::relu(90);
    registry.init_agent_layer(&live).unwrap();
    let wrong = AgentLayerSpec::gelu(90);

    let snapshot_before = workspace.snapshot();
    let err = workspace_wire_unary(
        &mut workspace,
        &mut builder,
        &registry,
        &wrong,
        0,
        "wrong".into(),
    )
    .unwrap_err();
    assert!(err.contains("identity mismatch"));
    assert_eq!(workspace.snapshot(), snapshot_before);
    assert_eq!(builder.num_steps(), 0);

    let output = workspace_wire_unary(
        &mut workspace,
        &mut builder,
        &registry,
        &live,
        0,
        "correct".into(),
    )
    .unwrap();
    assert_eq!(output, 1);
    assert_eq!(builder.num_steps(), 1);
}
