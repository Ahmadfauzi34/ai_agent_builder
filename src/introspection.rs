pub use crate::facade::introspection::{
    agent_layer_catalog, describe_graph, describe_workspace, introspection_capabilities,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{agent_layer_catalog, describe_graph, describe_workspace};
    use crate::agent::{capability_manifest, AgentGraphBuilder, AgentLayerSpec};
    use crate::contracts::agent_layout_contract;
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn layer_catalog_covers_every_advertised_constructor_and_layout_key() {
        let catalog: serde_json::Value = serde_json::from_str(&agent_layer_catalog()).unwrap();
        let capabilities: serde_json::Value = serde_json::from_str(&capability_manifest()).unwrap();
        let layout: serde_json::Value = serde_json::from_str(&agent_layout_contract()).unwrap();

        let catalog_keys = catalog["constructors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let advertised = capabilities["agent_facade"]["constructors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect::<BTreeSet<_>>();
        let layout_keys = layout["constructors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        assert_eq!(catalog_keys.len(), 41);
        assert_eq!(catalog_keys, advertised);
        assert_eq!(catalog_keys, layout_keys);
    }

    #[test]
    fn workspace_description_is_read_only_and_reports_live_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let id = workspace
            .reserve_layer_id(&registry, "relu".into())
            .unwrap();
        let spec = AgentLayerSpec::relu(id);
        registry.init_agent_layer(&spec).unwrap();
        workspace
            .sync_layer(&registry, &spec, "relu".into())
            .unwrap();

        let before = workspace.snapshot();
        let description: serde_json::Value =
            serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();

        assert_eq!(description["layers"][0]["constructor"], "relu");
        assert_eq!(description["layers"][0]["registry_present"], true);
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn descriptions_expose_semantic_input_port_without_mutating_execution_state() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let registry = LayerRegistry::new();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap();

        let before = workspace.snapshot();
        let workspace_json: serde_json::Value =
            serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
        assert_eq!(workspace_json["external_input_port"]["role"], "observation");
        assert_eq!(
            workspace_json["external_input_port"]["provenance"]["source"],
            "market-feed"
        );
        assert_eq!(workspace.snapshot(), before);

        let builder = AgentGraphBuilder::new(3).unwrap();
        let graph_json: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap())
                .unwrap();
        assert_eq!(graph_json["external_input_port"]["role"], "observation");
        assert_eq!(
            graph_json["external_input_port"]["provenance"]["revision"],
            18
        );
    }

    #[test]
    fn graph_description_exposes_canonical_topology_and_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let id = workspace
            .reserve_layer_id(&registry, "relu".into())
            .unwrap();
        let spec = AgentLayerSpec::relu(id);
        let out = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu".into(),
        )
        .unwrap();

        let description: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap())
                .unwrap();

        assert_eq!(description["steps"][0]["constructor"], "relu");
        assert_eq!(description["steps"][0]["input_slots"][0], 0);
        assert_eq!(description["steps"][0]["output_slot"], out);
        assert_eq!(description["steps"][0]["registry_present"], true);
    }

    #[test]
    fn lower_level_graph_missing_workspace_variant_is_not_guessed() {
        let workspace = AgentWorkspace::new(2).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(77);
        registry.init_agent_layer(&spec).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();

        let description: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap())
                .unwrap();

        assert!(description["steps"][0]["constructor"].is_null());
        assert!(description["steps"][0]["variant"].is_null());
        assert_eq!(description["steps"][0]["workspace_metadata"], false);
        assert_eq!(description["steps"][0]["registry_present"], true);
    }
}
