use wasm_bindgen::prelude::*;

use crate::agent::{
    AgentGraphBuilder, AgentGraphInputOrigin, AgentGraphSemanticLifecycleInput,
    AgentGraphSemanticLifecycleTransition,
};
use crate::input_port::role_valid;
use crate::input_port_edge_binding::semantic_graph_identity_json;

const SEMANTIC_LIFECYCLE_V1: &str =
    include_str!("../docs/agent-semantic-lifecycle.v1.json");
const MAX_TRANSITION_ID_BYTES: usize = 128;

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

fn parse_roles(value: &str) -> Result<Vec<String>, String> {
    let trimmed = value.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(
            "SemanticLifecycleTransitionSpec.expected_input_roles_json: expected JSON array of role strings"
                .to_string(),
        );
    }

    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return Err(
            "SemanticLifecycleTransitionSpec.expected_input_roles_json: at least one role is required"
                .to_string(),
        );
    }

    let items = inner.split(',').collect::<Vec<_>>();
    if !(1..=2).contains(&items.len()) {
        return Err(format!(
            "SemanticLifecycleTransitionSpec.expected_input_roles_json: {} roles provided; graph steps support 1 or 2 inputs",
            items.len()
        ));
    }

    let mut roles = Vec::with_capacity(items.len());
    for item in items {
        let token = item.trim();
        if token.len() < 2 || !token.starts_with('"') || !token.ends_with('"') {
            return Err(
                "SemanticLifecycleTransitionSpec.expected_input_roles_json: every entry must be a quoted string"
                    .to_string(),
            );
        }
        let role = &token[1..token.len() - 1];
        if role.contains('"') || role.contains('\\') {
            return Err(
                "SemanticLifecycleTransitionSpec.expected_input_roles_json: escaped role strings are not supported"
                    .to_string(),
            );
        }
        if !role_valid(role) {
            return Err(format!(
                "SemanticLifecycleTransitionSpec.expected_input_roles_json: invalid role {role}"
            ));
        }
        roles.push(role.to_string());
    }
    Ok(roles)
}

#[wasm_bindgen]
pub struct SemanticLifecycleTransitionSpec {
    transition_id: String,
    expected_input_roles: Vec<String>,
    output_role: String,
}

impl SemanticLifecycleTransitionSpec {
    pub(crate) fn transition_id_ref(&self) -> &str {
        &self.transition_id
    }

    pub(crate) fn expected_input_roles_slice(&self) -> &[String] {
        &self.expected_input_roles
    }

    pub(crate) fn output_role_ref(&self) -> &str {
        &self.output_role
    }

    pub(crate) fn json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.semantic-lifecycle-transition-spec.v1\",",
                "\"transition_id\":\"{}\",",
                "\"expected_input_roles\":{},",
                "\"output_role\":\"{}\"",
                "}}"
            ),
            json_escape(&self.transition_id),
            string_array_json(&self.expected_input_roles),
            json_escape(&self.output_role),
        )
    }
}

#[wasm_bindgen]
impl SemanticLifecycleTransitionSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(
        transition_id: String,
        expected_input_roles_json: String,
        output_role: String,
    ) -> Result<SemanticLifecycleTransitionSpec, String> {
        if transition_id.is_empty() {
            return Err(
                "SemanticLifecycleTransitionSpec.transition_id: value must be non-empty".to_string(),
            );
        }
        if transition_id.len() > MAX_TRANSITION_ID_BYTES {
            return Err(format!(
                "SemanticLifecycleTransitionSpec.transition_id: {} bytes exceeds limit {MAX_TRANSITION_ID_BYTES}",
                transition_id.len()
            ));
        }
        let expected_input_roles = parse_roles(&expected_input_roles_json)?;
        if !role_valid(&output_role) {
            return Err(format!(
                "SemanticLifecycleTransitionSpec.output_role: invalid role {output_role}"
            ));
        }

        Ok(Self {
            transition_id,
            expected_input_roles,
            output_role,
        })
    }

    #[wasm_bindgen(js_name = transitionId)]
    pub fn transition_id(&self) -> String {
        self.transition_id.clone()
    }

    #[wasm_bindgen(js_name = expectedInputRoles)]
    pub fn expected_input_roles(&self) -> String {
        string_array_json(&self.expected_input_roles)
    }

    #[wasm_bindgen(js_name = outputRole)]
    pub fn output_role(&self) -> String {
        self.output_role.clone()
    }

    #[wasm_bindgen(js_name = describe)]
    pub fn describe(&self) -> String {
        self.json()
    }
}

fn resolve_inputs(
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<Vec<AgentGraphSemanticLifecycleInput>, String> {
    let origins = builder
        .input_origins_for_step(step_index)
        .map_err(|error| format!("bindSemanticLifecycleTransition: {error}"))?;

    origins
        .into_iter()
        .map(|record| match record.origin {
            AgentGraphInputOrigin::ExternalInput => {
                let binding = builder.semantic_edge_binding(step_index).ok_or_else(|| {
                    format!(
                        "bindSemanticLifecycleTransition: step {step_index} position {} consumes external input but has no persistent input-port edge binding",
                        record.position
                    )
                })?;
                if !binding
                    .external_input_positions
                    .iter()
                    .any(|position| position == record.position)
                {
                    return Err(format!(
                        "bindSemanticLifecycleTransition: step {step_index} position {} is external by topology but missing from persistent edge binding",
                        record.position
                    ));
                }
                Ok(AgentGraphSemanticLifecycleInput {
                    position: record.position.to_string(),
                    slot: record.slot,
                    role: binding.bound_role.clone(),
                    origin_kind: "external_input_binding".to_string(),
                    origin_step_index: None,
                    origin_fingerprint: binding.binding_fingerprint.clone(),
                })
            }
            AgentGraphInputOrigin::StepOutput {
                step_index: source_step,
            } => {
                let upstream = builder
                    .semantic_lifecycle_transition(source_step)
                    .ok_or_else(|| {
                        format!(
                            "bindSemanticLifecycleTransition: step {step_index} position {} consumes slot {} from step {source_step}, but that upstream step has no semantic lifecycle transition",
                            record.position,
                            record.slot
                        )
                    })?;
                Ok(AgentGraphSemanticLifecycleInput {
                    position: record.position.to_string(),
                    slot: record.slot,
                    role: upstream.output_role.clone(),
                    origin_kind: "step_output_transition".to_string(),
                    origin_step_index: Some(source_step),
                    origin_fingerprint: upstream.transition_fingerprint.clone(),
                })
            }
            AgentGraphInputOrigin::Unresolved => Err(format!(
                "bindSemanticLifecycleTransition: step {step_index} position {} consumes unresolved slot {}",
                record.position, record.slot
            )),
        })
        .collect()
}

fn input_record_json(input: &AgentGraphSemanticLifecycleInput) -> String {
    format!(
        concat!(
            "{{",
            "\"position\":\"{}\",",
            "\"slot\":{},",
            "\"role\":\"{}\",",
            "\"origin_kind\":\"{}\",",
            "\"origin_step_index\":{},",
            "\"origin_fingerprint\":\"{}\"",
            "}}"
        ),
        json_escape(&input.position),
        input.slot,
        json_escape(&input.role),
        json_escape(&input.origin_kind),
        input
            .origin_step_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        json_escape(&input.origin_fingerprint),
    )
}

fn inputs_json(inputs: &[AgentGraphSemanticLifecycleInput]) -> String {
    format!(
        "[{}]",
        inputs
            .iter()
            .map(input_record_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn transition_fingerprint(
    step_index: u32,
    output_slot: u8,
    spec: &SemanticLifecycleTransitionSpec,
    inputs: &[AgentGraphSemanticLifecycleInput],
) -> String {
    let canonical = format!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.semantic-lifecycle-transition-fingerprint.v1\",",
            "\"step_index\":{},",
            "\"transition_id\":\"{}\",",
            "\"inputs\":{},",
            "\"output\":{{\"slot\":{},\"role\":\"{}\"}}",
            "}}"
        ),
        step_index,
        json_escape(spec.transition_id_ref()),
        inputs_json(inputs),
        output_slot,
        json_escape(spec.output_role_ref()),
    );
    fnv1a64(canonical.as_bytes())
}

pub(crate) fn transition_record_json(
    transition: &AgentGraphSemanticLifecycleTransition,
) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-transition.v1\",",
            "\"step_index\":{},",
            "\"transition_id\":\"{}\",",
            "\"inputs\":{},",
            "\"output\":{{",
                "\"slot\":{},",
                "\"role\":\"{}\",",
                "\"provenance_fingerprint\":\"{}\"",
            "}},",
            "\"transition_fingerprint\":\"{}\"",
            "}}"
        ),
        transition.step_index,
        json_escape(&transition.transition_id),
        inputs_json(&transition.inputs),
        transition.output_slot,
        json_escape(&transition.output_role),
        json_escape(&transition.transition_fingerprint),
        json_escape(&transition.transition_fingerprint),
    )
}

pub(crate) fn semantic_lifecycle_identity_json(builder: &AgentGraphBuilder) -> String {
    let base_identity = semantic_graph_identity_json(builder);
    let transition_fingerprints = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(|transition| {
            format!(
                "\"{}\"",
                json_escape(&transition.transition_fingerprint)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let canonical = format!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.semantic-lifecycle-identity-canonical.v1\",",
            "\"semantic_graph_identity\":{},",
            "\"transition_fingerprints\":[{}]",
            "}}"
        ),
        base_identity,
        transition_fingerprints,
    );
    let fingerprint = fnv1a64(canonical.as_bytes());

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-identity.v1\",",
            "\"base_semantic_graph_identity\":{},",
            "\"transition_count\":{},",
            "\"execution_program_identity_effect\":\"none\",",
            "\"semantic_graph_identity_effect\":\"none\",",
            "\"fingerprint\":\"{}\"",
            "}}"
        ),
        semantic_graph_identity_json(builder),
        builder.semantic_lifecycle_transitions().len(),
        fingerprint,
    )
}

#[wasm_bindgen(js_name = semanticLifecycleCapabilities)]
pub fn semantic_lifecycle_capabilities() -> String {
    SEMANTIC_LIFECYCLE_V1.to_string()
}

#[wasm_bindgen(js_name = bindSemanticLifecycleTransition)]
pub fn bind_semantic_lifecycle_transition(
    builder: &mut AgentGraphBuilder,
    spec: &SemanticLifecycleTransitionSpec,
    step_index: u32,
) -> Result<bool, String> {
    let index = usize::try_from(step_index).map_err(|_| {
        "bindSemanticLifecycleTransition: step index conversion failed".to_string()
    })?;
    let steps = builder.introspection_steps();
    let Some((arity, _, _, _, _, output_slot)) = steps.get(index).copied() else {
        return Err(format!(
            "bindSemanticLifecycleTransition: step index {step_index} is outside num_steps {}",
            steps.len()
        ));
    };

    let inputs = resolve_inputs(builder, step_index)?;
    if usize::from(arity) != spec.expected_input_roles_slice().len() {
        return Err(format!(
            "bindSemanticLifecycleTransition: step {step_index} arity {arity} does not match {} expected semantic input roles",
            spec.expected_input_roles_slice().len()
        ));
    }

    for (input, expected) in inputs.iter().zip(spec.expected_input_roles_slice()) {
        if &input.role != expected {
            return Err(format!(
                "bindSemanticLifecycleTransition: step {step_index} position {} resolved role {} does not match expected role {}",
                input.position, input.role, expected
            ));
        }
    }

    let fingerprint = transition_fingerprint(
        step_index,
        output_slot,
        spec,
        &inputs,
    );

    builder.bind_semantic_lifecycle_transition(AgentGraphSemanticLifecycleTransition {
        step_index,
        transition_id: spec.transition_id_ref().to_string(),
        inputs,
        output_slot,
        output_role: spec.output_role_ref().to_string(),
        transition_fingerprint: fingerprint,
    })
}

#[wasm_bindgen(js_name = semanticLifecycleTransition)]
pub fn semantic_lifecycle_transition(
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<String, String> {
    builder
        .input_origins_for_step(step_index)
        .map_err(|error| format!("semanticLifecycleTransition: {error}"))?;

    let Some(transition) = builder.semantic_lifecycle_transition(step_index) else {
        return Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",\"status\":\"unbound\",\"step_index\":{step_index}}}"
        ));
    };

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",",
            "\"status\":\"bound\",",
            "\"transition\":{},",
            "\"lifecycle_identity\":{}",
            "}}"
        ),
        transition_record_json(transition),
        semantic_lifecycle_identity_json(builder),
    ))
}

#[wasm_bindgen(js_name = semanticLifecycleTrace)]
pub fn semantic_lifecycle_trace(builder: &AgentGraphBuilder) -> String {
    let transitions = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(transition_record_json)
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-trace.v1\",",
            "\"projection_only\":true,",
            "\"decision_authority\":\"agent\",",
            "\"identity\":{},",
            "\"transitions\":[{}]",
            "}}"
        ),
        semantic_lifecycle_identity_json(builder),
        transitions,
    )
}

#[wasm_bindgen(js_name = semanticLifecycleIdentity)]
pub fn semantic_lifecycle_identity(builder: &AgentGraphBuilder) -> String {
    semantic_lifecycle_identity_json(builder)
}

#[cfg(test)]
mod tests {
    use super::{
        bind_semantic_lifecycle_transition, semantic_lifecycle_identity,
        semantic_lifecycle_trace, SemanticLifecycleTransitionSpec,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::input_port_edge_binding::{
        bind_input_port_consumer_edge, semantic_graph_identity,
    };
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn chained_transitions_preserve_execution_and_semantic_graph_identity() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let mut registry = LayerRegistry::new();

        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "feed".into(),
            1,
            "fp".into(),
        )
        .unwrap();

        let id0 = workspace.reserve_layer_id(&registry, "relu-a".into()).unwrap();
        let layer0 = AgentLayerSpec::relu(id0);
        let slot1 = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &layer0,
            0,
            "relu-a".into(),
        )
        .unwrap();

        let id1 = workspace.reserve_layer_id(&registry, "relu-b".into()).unwrap();
        let layer1 = AgentLayerSpec::relu(id1);
        let output = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &layer1,
            slot1,
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
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

        let program_before = builder
            .compile_with_output(&registry, output)
            .unwrap()
            .program_identity();
        let semantic_graph_before = semantic_graph_identity(&builder);
        let lifecycle_before = semantic_lifecycle_identity(&builder);

        let first = SemanticLifecycleTransitionSpec::new(
            "observe-to-feature".into(),
            "[\"observation\"]".into(),
            "feature".into(),
        )
        .unwrap();
        bind_semantic_lifecycle_transition(&mut builder, &first, 0).unwrap();

        let second = SemanticLifecycleTransitionSpec::new(
            "feature-to-candidate".into(),
            "[\"feature\"]".into(),
            "candidate".into(),
        )
        .unwrap();
        bind_semantic_lifecycle_transition(&mut builder, &second, 1).unwrap();

        let program_after = builder
            .compile_with_output(&registry, output)
            .unwrap()
            .program_identity();
        let semantic_graph_after = semantic_graph_identity(&builder);
        let lifecycle_after = semantic_lifecycle_identity(&builder);

        assert_eq!(program_before, program_after);
        assert_eq!(semantic_graph_before, semantic_graph_after);
        assert_ne!(lifecycle_before, lifecycle_after);

        let trace: serde_json::Value =
            serde_json::from_str(&semantic_lifecycle_trace(&builder)).unwrap();
        assert_eq!(trace["transitions"][0]["inputs"][0]["role"], "observation");
        assert_eq!(trace["transitions"][0]["output"]["role"], "feature");
        assert_eq!(trace["transitions"][1]["inputs"][0]["role"], "feature");
        assert_eq!(trace["transitions"][1]["output"]["role"], "candidate");
    }

    #[test]
    fn downstream_transition_requires_upstream_semantic_provenance() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let mut registry = LayerRegistry::new();

        let id0 = workspace.reserve_layer_id(&registry, "relu-a".into()).unwrap();
        let layer0 = AgentLayerSpec::relu(id0);
        let slot1 = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &layer0,
            0,
            "relu-a".into(),
        )
        .unwrap();

        let id1 = workspace.reserve_layer_id(&registry, "relu-b".into()).unwrap();
        let layer1 = AgentLayerSpec::relu(id1);
        workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &layer1,
            slot1,
            "relu-b".into(),
        )
        .unwrap();

        let second = SemanticLifecycleTransitionSpec::new(
            "feature-to-candidate".into(),
            "[\"feature\"]".into(),
            "candidate".into(),
        )
        .unwrap();
        let error =
            bind_semantic_lifecycle_transition(&mut builder, &second, 1).unwrap_err();
        assert!(error.contains("upstream step has no semantic lifecycle transition"));
    }
}
