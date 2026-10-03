pub use crate::facade::ingress::{
    input_port_consumer_edge_compatibility, input_port_routing_capabilities,
    interaction_valid_actions_for_input_consumer,
};
use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::input_port_consumer::{input_port_consumer_compatibility, InputPortConsumerSpec};
use crate::interaction::{valid_actions_json, validate_projection_inputs};
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;

pub(crate) const INPUT_PORT_ROUTING_V1: &str =
    include_str!("../../docs/contracts/agent-input-port-routing.v1.json");

pub(crate) fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

pub(crate) fn positions_json(positions: &[&str]) -> String {
    let body = positions
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

#[cfg(test)]
mod tests {
    use super::{
        input_port_consumer_edge_compatibility, input_port_routing_capabilities,
        interaction_valid_actions_for_input_consumer,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn valid_actions_overlay_is_advisory_and_keeps_canonical_candidate() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let builder = AgentGraphBuilder::new(3).unwrap();
        let registry = LayerRegistry::new();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap();
        let consumer = InputPortConsumerSpec::new(
            "reward-updater".into(),
            "[\"reward\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();

        let before = workspace.snapshot();
        let result: serde_json::Value = serde_json::from_str(
            &interaction_valid_actions_for_input_consumer(
                &workspace, &builder, &registry, &consumer,
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            result["semantic_overlay"]["compatibility"]["status"],
            "incompatible"
        );
        assert_eq!(result["semantic_overlay"]["candidate_retained"], true);
        assert_eq!(result["execution_authorized"], false);
        assert_eq!(result["decision_authority"], "agent");
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn edge_projection_targets_exact_step_and_external_input_position() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();

        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap();

        let layer_id = workspace
            .reserve_layer_id(&registry, "relu".into())
            .unwrap();
        let spec = AgentLayerSpec::relu(layer_id);
        workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu-input".into(),
        )
        .unwrap();

        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            true,
            10,
        )
        .unwrap();

        let before_workspace = workspace.snapshot();
        let before_steps = builder.num_steps();
        let before_params = registry.total_params();

        let result: serde_json::Value = serde_json::from_str(
            &input_port_consumer_edge_compatibility(&workspace, &builder, &consumer, 0).unwrap(),
        )
        .unwrap();

        assert_eq!(result["status"], "applicable");
        assert_eq!(result["edge"]["step_index"], 0);
        assert_eq!(result["edge"]["external_input_positions"][0], "input");
        assert_eq!(result["compatibility"]["status"], "compatible");
        assert_eq!(workspace.snapshot(), before_workspace);
        assert_eq!(builder.num_steps(), before_steps);
        assert_eq!(registry.total_params(), before_params);
    }

    #[test]
    fn internal_edge_is_not_applicable_not_rejected() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let mut registry = LayerRegistry::new();

        let first_id = workspace
            .reserve_layer_id(&registry, "relu-a".into())
            .unwrap();
        let first = AgentLayerSpec::relu(first_id);
        let first_output = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &first,
            0,
            "relu-a".into(),
        )
        .unwrap();

        let second_id = workspace
            .reserve_layer_id(&registry, "relu-b".into())
            .unwrap();
        let second = AgentLayerSpec::relu(second_id);
        workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &second,
            first_output,
            "relu-b".into(),
        )
        .unwrap();

        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();

        let result: serde_json::Value = serde_json::from_str(
            &input_port_consumer_edge_compatibility(&workspace, &builder, &consumer, 1).unwrap(),
        )
        .unwrap();

        assert_eq!(result["status"], "not_applicable");
        assert!(result["compatibility"].is_null());
    }

    #[test]
    fn capability_keeps_projection_boundary_explicit() {
        let caps: serde_json::Value =
            serde_json::from_str(&input_port_routing_capabilities()).unwrap();
        assert_eq!(caps["role"], "read_only_semantic_routing_projection");
        assert_eq!(caps["scope"]["state_ownership"], "none");
        assert_eq!(caps["scope"]["execution_effect"], "none");
        assert_eq!(caps["scope"]["selection_effect"], "none");
    }
}
