use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::interaction_fault::{
    interaction_check_compile, interaction_check_init_unary, interaction_check_release_slot,
    interaction_check_reserve_slot, interaction_fault_capabilities,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn fault_surface_is_read_only_and_machine_actionable() {
    let workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    let caps = interaction_fault_capabilities();
    assert!(caps.contains("\"legacy_errors_unchanged\":true"));
    assert!(caps.contains("\"read_only_preflight_companion\""));

    let workspace_before = workspace.snapshot();
    let release = interaction_check_release_slot(&workspace, 1);
    assert!(release.contains("\"code\":\"E_SLOT_TRANSITION\""));
    assert!(release.contains("\"predicate\":\"slot.state_reserved\""));
    assert!(release.contains("\"recoverable\":true"));
    assert_eq!(workspace.snapshot(), workspace_before);

    let reserve = interaction_check_reserve_slot(&workspace, "probe".into());
    assert!(reserve.contains("\"status\":\"ok\""));
    assert_eq!(workspace.snapshot(), workspace_before);

    let unreserved = AgentLayerSpec::relu(99);
    let init = interaction_check_init_unary(
        &workspace,
        &builder,
        &registry,
        &unreserved,
        0,
        "relu".into(),
    );
    assert!(init.contains("\"code\":\"E_LAYER_NOT_RESERVED\""));
    assert!(init.contains("\"suggested_actions\":[\"reserveLayerId\"]"));
}

#[test]
fn compile_fault_preflight_tracks_real_graph_state_without_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();

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

    let before_workspace = workspace.snapshot();
    let before_steps = builder.num_steps();
    let before_params = registry.total_params();

    let good = interaction_check_compile(&builder, &registry, output);
    assert!(good.contains("\"status\":\"ok\""));

    let wrong_output = interaction_check_compile(&builder, &registry, 2);
    assert!(wrong_output.contains("\"code\":\"E_OUTPUT_NOT_WRITTEN\""));
    assert!(wrong_output.contains("\"predicate\":\"builder.output_written\""));

    assert_eq!(workspace.snapshot(), before_workspace);
    assert_eq!(builder.num_steps(), before_steps);
    assert_eq!(registry.total_params(), before_params);
}
