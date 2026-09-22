use wasm_bindgen::prelude::*;

use crate::agent::{AgentGraphBuilder, AgentGraphInputPortBinding};
use crate::input_port_consumer::{
    evaluate_consumer_requirements, InputPortConsumerSpec,
};
use crate::workspace::AgentWorkspace;

const INPUT_PORT_EDGE_BINDING_V1: &str =
    include_str!("../docs/agent-input-port-edge-binding.v1.json");

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn bool_json(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn string_array_json(values: &[String]) -> String {
    let body = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn nullable_string_json(value: &str) -> String {
    if value.is_empty() {
        "null".to_string()
    } else {
        format!("\"{}\"", json_escape(value))
    }
}

fn fnv1a64(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn external_positions(
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<Vec<String>, String> {
    let steps = builder.introspection_steps();
    let index = usize::try_from(step_index)
        .map_err(|_| "inputPortEdgeBinding: step index conversion failed".to_string())?;
    let Some((arity, _, _, in_slot, in_slot2, _)) = steps.get(index).copied() else {
        return Err(format!(
            "inputPortEdgeBinding: step index {step_index} is outside num_steps {}",
            steps.len()
        ));
    };

    let mut positions = Vec::new();
    if arity == 1 {
        if in_slot == 0 {
            positions.push("input".to_string());
        }
    } else {
        if in_slot == 0 {
            positions.push("left".to_string());
        }
        if in_slot2 == 0 {
            positions.push("right".to_string());
        }
    }
    Ok(positions)
}

fn canonical_binding_identity(
    step_index: u32,
    consumer: &InputPortConsumerSpec,
    input_role: &str,
    input_source: &str,
    input_revision: u64,
    input_fingerprint: &str,
    compatibility_at_bind: &str,
    positions: &[String],
) -> String {
    let roles = consumer.snapshot_accepted_roles().join(",");
    let positions = positions.join(",");
    fnv1a64(&format!(
        concat!(
            "v1|step={step_index}|positions={positions}|consumer={}|roles={roles}|",
            "extension={}|fingerprint_required={}|minimum_revision={}|",
            "input_role={input_role}|input_source={input_source}|input_revision={input_revision}|",
            "input_fingerprint={input_fingerprint}|compatibility={compatibility_at_bind}"
        ),
        consumer.snapshot_consumer_id(),
        consumer.snapshot_allow_extension_roles(),
        consumer.snapshot_require_fingerprint(),
        consumer.snapshot_minimum_revision(),
    ))
}

pub(crate) fn binding_json(binding: &AgentGraphInputPortBinding) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-edge-binding-record.v1\",",
            "\"binding_identity\":\"{}\",",
            "\"immutable\":true,",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\",",
            "\"consumer\":{{",
                "\"consumer_id\":\"{}\",",
                "\"accepted_roles\":{},",
                "\"allow_extension_roles\":{},",
                "\"require_fingerprint\":{},",
                "\"minimum_revision\":{}",
            "}},",
            "\"input_port_snapshot\":{{",
                "\"role\":\"{}\",",
                "\"source\":\"{}\",",
                "\"revision\":{},",
                "\"fingerprint\":{}",
            "}},",
            "\"compatibility_at_bind\":\"{}\",",
            "\"external_input_positions\":{}",
            "}}"
        ),
        json_escape(&binding.binding_identity),
        json_escape(&binding.consumer_id),
        string_array_json(&binding.accepted_roles),
        bool_json(binding.allow_extension_roles),
        bool_json(binding.require_fingerprint),
        binding.minimum_revision,
        json_escape(&binding.input_role),
        json_escape(&binding.input_source),
        binding.input_revision,
        nullable_string_json(&binding.input_fingerprint),
        json_escape(&binding.compatibility_at_bind),
        string_array_json(&binding.external_input_positions),
    )
}

fn current_status_json(
    workspace: &AgentWorkspace,
    binding: &AgentGraphInputPortBinding,
) -> String {
    let Some(metadata) = workspace.input_port_metadata() else {
        return concat!(
            "{",
            "\"status\":\"unknown\",",
            "\"input_snapshot_match\":null,",
            "\"current_compatibility\":null,",
            "\"reason\":\"semantic_port_unbound\"",
            "}"
        )
        .to_string();
    };

    let snapshot_match = metadata.role == binding.input_role
        && metadata.source == binding.input_source
        && metadata.revision == binding.input_revision
        && metadata.fingerprint == binding.input_fingerprint;

    let evaluation = evaluate_consumer_requirements(
        &binding.accepted_roles,
        binding.allow_extension_roles,
        binding.require_fingerprint,
        binding.minimum_revision,
        &metadata.role,
        metadata.revision,
        !metadata.fingerprint.is_empty(),
    );

    format!(
        concat!(
            "{{",
            "\"status\":\"{}\",",
            "\"input_snapshot_match\":{},",
            "\"current_compatibility\":\"{}\",",
            "\"current_input\":{{",
                "\"role\":\"{}\",",
                "\"source\":\"{}\",",
                "\"revision\":{},",
                "\"fingerprint\":{}",
            "}},",
            "\"predicates\":{{",
                "\"role_match\":{},",
                "\"fingerprint_ok\":{},",
                "\"revision_ok\":{}",
            "}}",
            "}}"
        ),
        if snapshot_match { "exact" } else { "drifted" },
        bool_json(snapshot_match),
        if evaluation.compatible() {
            "compatible"
        } else {
            "incompatible"
        },
        json_escape(&metadata.role),
        json_escape(&metadata.source),
        metadata.revision,
        nullable_string_json(&metadata.fingerprint),
        bool_json(evaluation.role_match),
        bool_json(evaluation.fingerprint_ok),
        bool_json(evaluation.revision_ok),
    )
}

#[wasm_bindgen(js_name = inputPortEdgeBindingCapabilities)]
pub fn input_port_edge_binding_capabilities() -> String {
    INPUT_PORT_EDGE_BINDING_V1.to_string()
}

#[wasm_bindgen(js_name = bindInputPortConsumerEdge)]
pub fn bind_input_port_consumer_edge(
    workspace: &AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    consumer: &InputPortConsumerSpec,
    step_index: u32,
) -> Result<String, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "bindInputPortConsumerEdge: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let positions = external_positions(builder, step_index)?;
    if positions.is_empty() {
        return Err(format!(
            "bindInputPortConsumerEdge: step index {step_index} does not consume external slot 0"
        ));
    }

    let metadata = workspace.input_port_metadata().ok_or_else(|| {
        "bindInputPortConsumerEdge: semantic input-port metadata must be bound before persistent edge binding"
            .to_string()
    })?;

    let evaluation = consumer.evaluate_metadata(
        &metadata.role,
        metadata.revision,
        !metadata.fingerprint.is_empty(),
    );
    let compatibility_at_bind = if evaluation.compatible() {
        "compatible"
    } else {
        "incompatible"
    };

    let binding_identity = canonical_binding_identity(
        step_index,
        consumer,
        &metadata.role,
        &metadata.source,
        metadata.revision,
        &metadata.fingerprint,
        compatibility_at_bind,
        &positions,
    );

    let binding = AgentGraphInputPortBinding {
        binding_identity,
        consumer_id: consumer.snapshot_consumer_id().to_string(),
        accepted_roles: consumer.snapshot_accepted_roles().to_vec(),
        allow_extension_roles: consumer.snapshot_allow_extension_roles(),
        require_fingerprint: consumer.snapshot_require_fingerprint(),
        minimum_revision: consumer.snapshot_minimum_revision(),
        input_role: metadata.role.clone(),
        input_source: metadata.source.clone(),
        input_revision: metadata.revision,
        input_fingerprint: metadata.fingerprint.clone(),
        compatibility_at_bind: compatibility_at_bind.to_string(),
        external_input_positions: positions,
    };

    let created = builder.set_input_port_binding(step_index, binding)?;
    let stored = builder
        .input_port_binding(step_index)
        .ok_or_else(|| "bindInputPortConsumerEdge: binding disappeared after write".to_string())?;

    Ok(format!(
        "{{\"status\":\"bound\",\"created\":{},\"step_index\":{},\"binding\":{}}}",
        bool_json(created),
        step_index,
        binding_json(stored),
    ))
}

#[wasm_bindgen(js_name = inputPortConsumerEdgeBinding)]
pub fn input_port_consumer_edge_binding(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<String, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "inputPortConsumerEdgeBinding: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let positions = external_positions(builder, step_index)?;
    let Some(binding) = builder.input_port_binding(step_index) else {
        return Ok(format!(
            "{{\"status\":\"unbound\",\"step_index\":{},\"external_input_positions\":{}}}",
            step_index,
            string_array_json(&positions),
        ));
    };

    Ok(format!(
        concat!(
            "{{",
            "\"status\":\"bound\",",
            "\"step_index\":{},",
            "\"binding\":{},",
            "\"current\":{},",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\"",
            "}}"
        ),
        step_index,
        binding_json(binding),
        current_status_json(workspace, binding),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        bind_input_port_consumer_edge, input_port_consumer_edge_binding,
        input_port_edge_binding_capabilities,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    fn graph_with_external_edge() -> (AgentWorkspace, AgentGraphBuilder, LayerRegistry) {
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
            "relu".into(),
        )
        .unwrap();

        (workspace, builder, registry)
    }

    #[test]
    fn first_bind_persists_and_identical_bind_is_idempotent() {
        let (workspace, mut builder, _) = graph_with_external_edge();
        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            true,
            10,
        )
        .unwrap();

        let first: serde_json::Value = serde_json::from_str(
            &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
        )
        .unwrap();
        assert_eq!(first["created"], true);
        assert_eq!(first["binding"]["compatibility_at_bind"], "compatible");

        let second: serde_json::Value = serde_json::from_str(
            &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
        )
        .unwrap();
        assert_eq!(second["created"], false);
        assert_eq!(builder.input_port_binding_count(), 1);
    }

    #[test]
    fn conflicting_rebind_is_rejected() {
        let (workspace, mut builder, _) = graph_with_external_edge();
        let first = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &first, 0).unwrap();

        let second = InputPortConsumerSpec::new(
            "reward-updater".into(),
            "[\"reward\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        let err = bind_input_port_consumer_edge(&workspace, &mut builder, &second, 0)
            .unwrap_err();
        assert!(err.contains("already has an immutable semantic binding"));
    }

    #[test]
    fn incompatible_decision_can_be_persisted_without_execution_authority() {
        let (workspace, mut builder, _) = graph_with_external_edge();
        let consumer = InputPortConsumerSpec::new(
            "reward-updater".into(),
            "[\"reward\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();

        let bound: serde_json::Value = serde_json::from_str(
            &bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap(),
        )
        .unwrap();
        assert_eq!(bound["binding"]["compatibility_at_bind"], "incompatible");
        assert_eq!(bound["binding"]["execution_authorized"], false);
        assert_eq!(bound["binding"]["decision_authority"], "agent");
    }

    #[test]
    fn current_status_reports_metadata_drift_without_mutating_graph() {
        let (mut workspace, mut builder, _) = graph_with_external_edge();
        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            true,
            10,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

        let before_steps = builder.num_steps();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            19,
            "fnv1a64:next".into(),
        )
        .unwrap();

        let status: serde_json::Value = serde_json::from_str(
            &input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap(),
        )
        .unwrap();

        assert_eq!(status["current"]["status"], "drifted");
        assert_eq!(status["current"]["input_snapshot_match"], false);
        assert_eq!(status["current"]["current_compatibility"], "compatible");
        assert_eq!(builder.num_steps(), before_steps);
    }

    #[test]
    fn binding_requires_external_edge_and_bound_semantic_input() {
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

        assert!(bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0)
            .unwrap_err()
            .contains("metadata must be bound"));

        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            1,
            String::new(),
        )
        .unwrap();

        assert!(bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 1)
            .unwrap_err()
            .contains("does not consume external slot 0"));
    }

    #[test]
    fn capability_contract_keeps_provenance_non_authoritative() {
        let caps: serde_json::Value =
            serde_json::from_str(&input_port_edge_binding_capabilities()).unwrap();
        assert_eq!(caps["role"], "persistent_semantic_graph_provenance");
        assert_eq!(caps["scope"]["execution_effect"], "none");
        assert_eq!(caps["scope"]["selection_effect"], "none");
    }
}
