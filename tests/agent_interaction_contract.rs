use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::interaction::{
    interaction_capabilities, interaction_snapshot, interaction_valid_actions,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn public_interaction_surface_is_projection_only_and_state_aware() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();

    let capabilities = interaction_capabilities();
    assert!(capabilities.contains("\"role\":\"projection_only\""));
    assert!(capabilities.contains("\"state_ownership\":\"none\""));

    let workspace_before = workspace.snapshot();
    let steps_before = builder.num_steps();
    let params_before = registry.total_params();

    let initial = interaction_snapshot(&workspace, &builder, &registry).unwrap();
    let actions = interaction_valid_actions(&workspace, &builder, &registry).unwrap();

    assert!(initial.contains("\"phase\":\"workspace_ready\""));
    assert!(actions.contains("\"operation\":\"reserveLayerId\""));
    assert_eq!(workspace.snapshot(), workspace_before);
    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(registry.total_params(), params_before);

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

    let ready = interaction_snapshot(&workspace, &builder, &registry).unwrap();
    assert!(ready.contains("\"phase\":\"graph_ready\""));
    assert!(ready.contains(&format!("\"compile_candidate_slots\":[{output}]")));
    assert!(ready.contains(
        "\"operation\":\"workspaceCompile\",\"class\":\"canonical\",\"available\":true"
    ));
}

#[test]
fn interaction_projection_rejects_mismatched_control_domains() {
    let workspace = AgentWorkspace::new(2).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    let err = interaction_snapshot(&workspace, &builder, &registry).unwrap_err();
    assert!(err.contains("workspace num_slots 2 does not match builder num_slots 3"));
}
