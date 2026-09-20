use std::collections::BTreeSet;

use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::introspection::{
    agent_layer_catalog, describe_graph, describe_workspace, introspection_capabilities,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::workspace_init_unary;

#[test]
fn public_introspection_contract_is_embedded_and_machine_readable() {
    let contract: serde_json::Value =
        serde_json::from_str(&introspection_capabilities()).unwrap();
    assert_eq!(contract["schema_id"], "burn-research.agent-introspection.v1");
    assert_eq!(contract["role"], "read_only_semantic_projection");
    assert_eq!(contract["state_ownership"], "none");
}

#[test]
fn public_layer_catalog_is_complete_and_machine_readable() {
    let catalog: serde_json::Value = serde_json::from_str(&agent_layer_catalog()).unwrap();
    let constructors = catalog["constructors"].as_object().unwrap();

    assert_eq!(constructors.len(), 41);
    for required in [
        "relu",
        "prelu",
        "swiGlu",
        "linear",
        "groupNorm",
        "convTranspose2d",
        "embedding",
        "adaptiveAvgPool2d",
        "featureNorm",
        "ghost",
        "seBlock",
        "matmul",
        "concat",
    ] {
        let entry = &constructors[required];
        assert!(entry["signature"].is_string(), "missing signature for {required}");
        assert!(entry["parameters"].is_array(), "missing parameters for {required}");
        assert!(entry["arity"].is_number(), "missing arity for {required}");
        assert_eq!(entry["layout_contract_key"], required);
    }
}

#[test]
fn workspace_and_graph_descriptions_are_read_only_semantic_views() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();

    workspace
        .put(
            "experiments".into(),
            "candidate-a".into(),
            "candidate".into(),
            "ready".into(),
            "source=agent".into(),
        )
        .unwrap();

    let layer_id = workspace.reserve_layer_id(&registry, "relu-main".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let output = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu-main".into(),
    )
    .unwrap();

    let workspace_before = workspace.snapshot();
    let steps_before = builder.num_steps();
    let params_before = registry.total_params();

    let workspace_description: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    let graph_description: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

    assert_eq!(workspace_description["projection_only"], true);
    assert_eq!(workspace_description["layers"][0]["constructor"], "relu");
    assert_eq!(workspace_description["layers"][0]["registry_present"], true);
    assert!(workspace_description["custom_tables"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value == "experiments"));

    assert_eq!(graph_description["projection_only"], true);
    assert_eq!(graph_description["num_steps"], 1);
    assert_eq!(graph_description["steps"][0]["constructor"], "relu");
    assert_eq!(graph_description["steps"][0]["input_slots"][0], 0);
    assert_eq!(graph_description["steps"][0]["output_slot"], output);
    assert_eq!(graph_description["steps"][0]["registry_present"], true);

    assert_eq!(workspace.snapshot(), workspace_before);
    assert_eq!(builder.num_steps(), steps_before);
    assert_eq!(registry.total_params(), params_before);
}

#[test]
fn lower_level_graph_does_not_invent_missing_workspace_semantics() {
    let workspace = AgentWorkspace::new(2).unwrap();
    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let mut registry = LayerRegistry::new();

    let spec = AgentLayerSpec::relu(77);
    registry.init_agent_layer(&spec).unwrap();
    builder.add_unary(&spec, 0, 1).unwrap();

    let description: serde_json::Value =
        serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

    assert_eq!(description["unknown_metadata_policy"], "report_null_do_not_infer");
    assert_eq!(description["steps"][0]["workspace_metadata"], false);
    assert!(description["steps"][0]["variant"].is_null());
    assert!(description["steps"][0]["constructor"].is_null());
    assert_eq!(description["steps"][0]["registry_present"], true);
}

#[test]
fn catalog_constructor_names_remain_unique() {
    let catalog: serde_json::Value = serde_json::from_str(&agent_layer_catalog()).unwrap();
    let names = catalog["constructors"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let unique = names.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(names.len(), unique.len());
}
