use wasm_bindgen::prelude::*;

use crate::agent::{
    AgentGraphBuilder, AgentGraphSemanticEdgeBinding,
};
use crate::input_port_consumer::InputPortConsumerSpec;
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

fn fnv1a64(value: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn external_positions(
    builder: &AgentGraphBuilder,
    step_index: u32,
    context: &str,
) -> Result<Vec<String>, String> {
    builder
        .external_input_positions_for_step(step_index)
        .map(|positions| positions.into_iter().map(str::to_string).collect())
        .map_err(|error| format!("{context}: {error}"))
}

fn binding_fingerprint(
    step_index: u32,
    positions: &[String],
    consumer: &InputPortConsumerSpec,
    role: &str,
    source: &str,
    revision: u64,
    fingerprint: &str,
) -> String {
    let canonical = format!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.input-port-edge-binding-fingerprint.v1\",",
            "\"step_index\":{},",
            "\"positions\":{},",
            "\"consumer\":{},",
            "\"input\":{{",
                "\"role\":\"{}\",",
                "\"source\":\"{}\",",
                "\"revision\":{},",
                "\"fingerprint\":\"{}\"" ,
            "}}",
            "}}"
        ),
        step_index,
        string_array_json(positions),
        consumer.json(),
        json_escape(role),
        json_escape(source),
        revision,
        json_escape(fingerprint),
    );
    fnv1a64(canonical.as_bytes())
}

fn binding_snapshot_compatible(binding: &AgentGraphSemanticEdgeBinding) -> bool {
    let role_match = binding
        .accepted_roles
        .iter()
        .any(|role| role == &binding.bound_role)
        || (binding.allow_extension_roles && binding.bound_role.starts_with("x-"));
    let fingerprint_ok = !binding.require_fingerprint || !binding.bound_fingerprint.is_empty();
    let revision_ok =
        binding.minimum_revision == 0 || binding.bound_revision >= binding.minimum_revision;
    role_match && fingerprint_ok && revision_ok
}

fn stored_consumer_compatible(
    binding: &AgentGraphSemanticEdgeBinding,
    workspace: &AgentWorkspace,
) -> Option<bool> {
    let metadata = workspace.input_port_metadata()?;
    let role_match = binding
        .accepted_roles
        .iter()
        .any(|role| role == &metadata.role)
        || (binding.allow_extension_roles && metadata.role.starts_with("x-"));
    let fingerprint_ok = !binding.require_fingerprint || !metadata.fingerprint.is_empty();
    let revision_ok =
        binding.minimum_revision == 0 || metadata.revision >= binding.minimum_revision;
    Some(role_match && fingerprint_ok && revision_ok)
}

pub(crate) fn binding_record_json(binding: &AgentGraphSemanticEdgeBinding) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-edge-binding-record.v1\",",
            "\"step_index\":{},",
            "\"external_input_positions\":{},",
            "\"consumer\":{{",
                "\"consumer_id\":\"{}\",",
                "\"accepted_roles\":{},",
                "\"allow_extension_roles\":{},",
                "\"require_fingerprint\":{},",
                "\"minimum_revision\":{}",
            "}},",
            "\"bound_input\":{{",
                "\"role\":\"{}\",",
                "\"source\":\"{}\",",
                "\"revision\":{},",
                "\"fingerprint\":{}",
            "}},",
            "\"binding_fingerprint\":\"{}\",",
            "\"compatibility_at_bind\":\"{}\"" ,
            "}}"
        ),
        binding.step_index,
        string_array_json(&binding.external_input_positions),
        json_escape(&binding.consumer_id),
        string_array_json(&binding.accepted_roles),
        bool_json(binding.allow_extension_roles),
        bool_json(binding.require_fingerprint),
        binding.minimum_revision,
        json_escape(&binding.bound_role),
        json_escape(&binding.bound_source),
        binding.bound_revision,
        if binding.bound_fingerprint.is_empty() {
            "null".to_string()
        } else {
            format!("\"{}\"", json_escape(&binding.bound_fingerprint))
        },
        json_escape(&binding.binding_fingerprint),
        if binding_snapshot_compatible(binding) {
            "compatible"
        } else {
            "incompatible"
        },
    )
}

pub(crate) fn semantic_graph_identity_json(builder: &AgentGraphBuilder) -> String {
    let mut canonical = format!("v1|slots={}|", builder.num_slots());
    for (index, (arity, layer_type, layer_id, in_slot, in_slot2, out_slot)) in
        builder.introspection_steps().iter().copied().enumerate()
    {
        canonical.push_str(&format!(
            "step={index}:{arity}:{layer_type}:{layer_id}:{in_slot}:{in_slot2}:{out_slot}|"
        ));
    }
    match builder.introspection_output_slot() {
        Some(slot) => canonical.push_str(&format!("output={slot}|")),
        None => canonical.push_str("output=null|"),
    }
    for binding in builder.semantic_edge_bindings() {
        canonical.push_str(&format!(
            "binding={}:{}|",
            binding.step_index, binding.binding_fingerprint
        ));
    }

    let fingerprint = fnv1a64(canonical.as_bytes());
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-graph-identity.v1\",",
            "\"topology_authority\":\"AgentGraphBuilder\",",
            "\"semantic_binding_count\":{},",
            "\"execution_program_identity_effect\":\"none\",",
            "\"fingerprint\":\"{}\"",
            "}}"
        ),
        builder.semantic_edge_bindings().len(),
        fingerprint,
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
) -> Result<bool, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "bindInputPortConsumerEdge: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let positions = external_positions(builder, step_index, "bindInputPortConsumerEdge")?;
    if positions.is_empty() {
        return Err(format!(
            "bindInputPortConsumerEdge: step {step_index} does not consume external slot 0"
        ));
    }

    if consumer.is_compatible_with_workspace(workspace).is_none() {
        return Err(
            "bindInputPortConsumerEdge: semantic input port is unbound; compatibility is unknown"
                .to_string(),
        );
    }

    let metadata = workspace
        .input_port_metadata()
        .ok_or_else(|| "bindInputPortConsumerEdge: semantic input metadata disappeared".to_string())?;
    let fingerprint = binding_fingerprint(
        step_index,
        &positions,
        consumer,
        &metadata.role,
        &metadata.source,
        metadata.revision,
        &metadata.fingerprint,
    );

    builder.bind_semantic_edge(AgentGraphSemanticEdgeBinding {
        step_index,
        external_input_positions: positions,
        consumer_id: consumer.consumer_id_ref().to_string(),
        accepted_roles: consumer.accepted_roles_slice().to_vec(),
        allow_extension_roles: consumer.allow_extension_roles_value(),
        require_fingerprint: consumer.require_fingerprint_value(),
        minimum_revision: consumer.minimum_revision_value(),
        bound_role: metadata.role.clone(),
        bound_source: metadata.source.clone(),
        bound_revision: metadata.revision,
        bound_fingerprint: metadata.fingerprint.clone(),
        binding_fingerprint: fingerprint,
    })
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
    let positions =
        external_positions(builder, step_index, "inputPortConsumerEdgeBinding")?;
    if positions.is_empty() {
        return Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",\"status\":\"not_applicable\",\"step_index\":{step_index},\"reason\":\"step_does_not_consume_external_slot_0\"}}"
        ));
    }

    let Some(binding) = builder.semantic_edge_binding(step_index) else {
        return Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",\"status\":\"unbound\",\"step_index\":{step_index}}}"
        ));
    };

    let (state, current_compatible, snapshot_match) = match workspace.input_port_metadata() {
        None => ("input_unbound", None, false),
        Some(metadata) => {
            let compatible = stored_consumer_compatible(binding, workspace).unwrap_or(false);
            let snapshot_match = metadata.role == binding.bound_role
                && metadata.source == binding.bound_source
                && metadata.revision == binding.bound_revision
                && metadata.fingerprint == binding.bound_fingerprint;
            let state = if snapshot_match {
                "current"
            } else if compatible {
                "drifted_compatible"
            } else {
                "drifted_incompatible"
            };
            (state, Some(compatible), snapshot_match)
        }
    };

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",",
            "\"status\":\"bound\",",
            "\"binding_state\":\"{}\",",
            "\"current_compatible\":{},",
            "\"input_snapshot_match\":{},",
            "\"binding\":{},",
            "\"semantic_graph_identity\":{}",
            "}}"
        ),
        state,
        current_compatible
            .map(bool_json)
            .unwrap_or("null"),
        bool_json(snapshot_match),
        binding_record_json(binding),
        semantic_graph_identity_json(builder),
    ))
}

#[wasm_bindgen(js_name = semanticGraphIdentity)]
pub fn semantic_graph_identity(builder: &AgentGraphBuilder) -> String {
    semantic_graph_identity_json(builder)
}

#[cfg(test)]
mod tests {
    use super::{
        bind_input_port_consumer_edge, input_port_consumer_edge_binding,
        input_port_edge_binding_capabilities, semantic_graph_identity,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn binding_is_persistent_idempotent_and_conflict_fails_closed() {
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

        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            true,
            10,
        )
        .unwrap();
        assert!(bind_input_port_consumer_edge(
            &workspace,
            &mut builder,
            &consumer,
            0,
        )
        .unwrap());
        assert!(!bind_input_port_consumer_edge(
            &workspace,
            &mut builder,
            &consumer,
            0,
        )
        .unwrap());

        let other = InputPortConsumerSpec::new(
            "other".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        assert!(bind_input_port_consumer_edge(&workspace, &mut builder, &other, 0).is_err());
    }

    #[test]
    fn binding_reports_provenance_drift_without_rewriting_history() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "feed-a".into(),
            4,
            "fp:a".into(),
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

        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            true,
            1,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();
        let before: serde_json::Value =
            serde_json::from_str(&input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap())
                .unwrap();
        let binding_fingerprint = before["binding"]["binding_fingerprint"].clone();

        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "feed-b".into(),
            5,
            "fp:b".into(),
        )
        .unwrap();

        let after: serde_json::Value =
            serde_json::from_str(&input_port_consumer_edge_binding(&workspace, &builder, 0).unwrap())
                .unwrap();
        assert_eq!(after["binding_state"], "drifted_compatible");
        assert_eq!(after["input_snapshot_match"], false);
        assert_eq!(after["binding"]["binding_fingerprint"], binding_fingerprint);
    }

    #[test]
    fn semantic_identity_changes_while_execution_plan_remains_outside_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "feed".into(),
            1,
            "fp".into(),
        )
        .unwrap();
        let layer_id = workspace
            .reserve_layer_id(&registry, "relu".into())
            .unwrap();
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

        let program_before = builder
            .compile_with_output(&registry, output)
            .unwrap()
            .program_identity();
        let semantic_before = semantic_graph_identity(&builder);

        let consumer = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

        let program_after = builder
            .compile_with_output(&registry, output)
            .unwrap()
            .program_identity();
        let semantic_after = semantic_graph_identity(&builder);

        assert_eq!(program_before, program_after);
        assert_ne!(semantic_before, semantic_after);
    }

    #[test]
    fn capability_keeps_execution_identity_separate() {
        let caps: serde_json::Value =
            serde_json::from_str(&input_port_edge_binding_capabilities()).unwrap();
        assert_eq!(caps["scope"]["execution_plan_effect"], "none");
        assert_eq!(caps["scope"]["decision_authority"], "agent");
        assert_eq!(
            caps["identity"]["explicit_non_identity"],
            "Burn programIdentity remains based on executable plan bytes and is not changed by semantic edge binding"
        );
    }
}
