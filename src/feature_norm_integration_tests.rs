#[cfg(test)]
mod tests {
    use crate::agent::{agent_capabilities, AgentGraphBuilder, AgentLayerSpec};
    use crate::contracts::{agent_layout_compatibility, agent_spec_layout};
    use crate::program_bundle::{export_program_bundle, import_program_bundle};
    use crate::protocol::LAYER_FEATURE_NORM;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    fn compiled_feature_norm(
        registry: &mut LayerRegistry,
        layer_id: u32,
        epsilon: Option<f64>,
    ) -> (
        AgentLayerSpec,
        AgentGraphBuilder,
        crate::graph::CompiledGraph,
    ) {
        let spec = AgentLayerSpec::feature_norm(layer_id, epsilon).unwrap();
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph = builder.compile(registry).unwrap();
        (spec, builder, graph)
    }

    #[test]
    fn typed_feature_norm_is_stateless_parameter_free_and_executable() {
        assert!(AgentLayerSpec::feature_norm(1, Some(0.0)).is_err());
        assert!(AgentLayerSpec::feature_norm(1, Some(f64::NAN)).is_err());

        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::feature_norm(7, None).unwrap();
        registry.init_agent_layer(&spec).unwrap();
        assert_eq!(spec.layer_type(), LAYER_FEATURE_NORM);
        assert!(registry.layer_exists(LAYER_FEATURE_NORM, 7));
        assert_eq!(registry.total_params(), 0);
        assert_eq!(
            registry.get_layer_state(7, LAYER_FEATURE_NORM).unwrap(),
            Vec::<u8>::new()
        );
        assert!(registry
            .load_layer_state(7, LAYER_FEATURE_NORM, &[1])
            .is_err());
        assert!(registry
            .load_layer_state(7, LAYER_FEATURE_NORM, &[])
            .is_ok());

        let input = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        let output = registry
            .forward_layer(7, LAYER_FEATURE_NORM, &input)
            .unwrap();
        assert_close(&output.to_array(), &[0.6, 0.8]);
        let zero = WasmTensor::new(&[0.0, 0.0], &[1, 2, 1, 1]);
        assert_eq!(
            registry
                .forward_layer(7, LAYER_FEATURE_NORM, &zero)
                .unwrap()
                .to_array(),
            vec![0.0, 0.0]
        );

        assert!(registry.destroy_layer(7, LAYER_FEATURE_NORM));
        assert!(!registry.layer_exists(LAYER_FEATURE_NORM, 7));
        assert!(registry.get_layer_state(7, LAYER_FEATURE_NORM).is_err());
    }

    #[test]
    fn feature_norm_identity_binds_epsilon_and_requires_recompile_after_change() {
        let mut registry = LayerRegistry::new();
        let (_spec, _builder, graph) = compiled_feature_norm(&mut registry, 11, None);
        let plan = graph.program_plan();
        let old_identity = graph.program_identity();

        let replacement = AgentLayerSpec::feature_norm(11, Some(1e-6)).unwrap();
        registry.init_agent_layer(&replacement).unwrap();
        assert!(graph.validate_registry_binding(&registry).is_err());

        let rebound = registry.compile_graph(&plan).unwrap();
        assert_ne!(rebound.program_identity(), old_identity);
        assert!(rebound.validate_registry_binding(&registry).is_ok());
    }

    #[test]
    fn feature_norm_program_bundle_round_trip_preserves_identity_and_output() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = compiled_feature_norm(&mut source, 17, None);
        let input = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        let expected = graph.run(&source, &input).unwrap().to_array();
        let identity = graph.program_identity();
        let bundle = export_program_bundle(&graph, &source, true).unwrap();

        let mut target = LayerRegistry::new();
        let replay = import_program_bundle(&mut target, &bundle).unwrap();
        assert_eq!(replay.program_identity(), identity);
        assert_eq!(replay.program_plan(), graph.program_plan());
        assert_eq!(
            target.get_layer_state(17, LAYER_FEATURE_NORM).unwrap(),
            Vec::<u8>::new()
        );
        assert_close(&replay.run(&target, &input).unwrap().to_array(), &expected);
        assert!(replay.validate_registry_binding(&target).is_ok());
    }

    #[test]
    fn feature_norm_layout_and_capability_are_machine_discoverable() {
        let feature = AgentLayerSpec::feature_norm(21, None).unwrap();
        let linear = AgentLayerSpec::linear(22, 2, 2, false).unwrap();
        assert_eq!(
            agent_spec_layout(&feature),
            "{\"input\":\"feature_axis1_singleton\",\"output\":\"feature_axis1_singleton\"}"
        );
        assert_eq!(agent_layout_compatibility(&linear, &feature), "compatible");

        let caps: serde_json::Value = serde_json::from_str(&agent_capabilities()).unwrap();
        assert!(caps["agent_facade"]["constructors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str() == Some("featureNorm")));
        assert_eq!(
            caps["layers"]["feature_norm"]["code"].as_u64(),
            Some(u64::from(LAYER_FEATURE_NORM))
        );
    }
}
