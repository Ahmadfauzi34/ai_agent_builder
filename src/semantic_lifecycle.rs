use wasm_bindgen::prelude::*;

use crate::agent::{
    AgentGraphBuilder, AgentGraphSemanticInputLineage, AgentGraphSemanticLifecycleTransition,
};
use crate::input_port::role_valid;
use crate::input_port_edge_binding::semantic_graph_identity_json;
use crate::workspace::AgentWorkspace;

const SEMANTIC_LIFECYCLE_V1: &str =
    include_str!("../docs/agent-semantic-lifecycle.v1.json");
const MAX_TRANSITION_ID_BYTES: usize = 128;
const MAX_ROLE_BYTES: usize = 64;

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

fn parse_roles(value: &str) -> Result<Vec<String>, String> {
    let trimmed = value.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(
            "SemanticTransitionSpec.input_roles_json: expected JSON array of role strings"
                .to_string(),
        );
    }

    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return Err(
            "SemanticTransitionSpec.input_roles_json: at least one input role is required"
                .to_string(),
        );
    }

    let items = inner.split(',').collect::<Vec<_>>();
    if items.len() > 2 {
        return Err(format!(
            "SemanticTransitionSpec.input_roles_json: {} roles exceeds graph arity limit 2",
            items.len()
        ));
    }

    let mut roles = Vec::with_capacity(items.len());
    for item in items {
        let token = item.trim();
        if token.len() < 2 || !token.starts_with('"') || !token.ends_with('"') {
            return Err(
                "SemanticTransitionSpec.input_roles_json: every entry must be a quoted string"
                    .to_string(),
            );
        }
        let role = &token[1..token.len() - 1];
        if role.contains('"') || role.contains('\\') {
            return Err(
                "SemanticTransitionSpec.input_roles_json: escaped role strings are not supported"
                    .to_string(),
            );
        }
        validate_role(role, "SemanticTransitionSpec.input_roles_json")?;
        roles.push(role.to_string());
    }
    Ok(roles)
}

fn validate_role(role: &str, context: &str) -> Result<(), String> {
    if role.is_empty() {
        return Err(format!("{context}: role must be non-empty"));
    }
    if role.len() > MAX_ROLE_BYTES {
        return Err(format!(
            "{context}: role {} bytes exceeds limit {MAX_ROLE_BYTES}",
            role.len()
        ));
    }
    if !role_valid(role) {
        return Err(format!(
            "{context}: unsupported role {role}; use canonical role or x- extension namespace"
        ));
    }
    Ok(())
}

fn fnv1a64(value: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn lineage_json(lineage: &AgentGraphSemanticInputLineage) -> String {
    format!(
        concat!(
            "{{",
            "\"position\":\"{}\",",
            "\"slot\":{},",
            "\"role\":\"{}\",",
            "\"source_kind\":\"{}\",",
            "\"source_step_index\":{},",
            "\"source_fingerprint\":\"{}\"",
            "}}"
        ),
        json_escape(&lineage.position),
        lineage.slot,
        json_escape(&lineage.role),
        json_escape(&lineage.source_kind),
        lineage
            .source_step_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        json_escape(&lineage.source_fingerprint),
    )
}

fn transition_record_json(transition: &AgentGraphSemanticLifecycleTransition) -> String {
    let inputs = transition
        .inputs
        .iter()
        .map(lineage_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-transition-record.v1\",",
            "\"step_index\":{},",
            "\"transition_id\":\"{}\",",
            "\"inputs\":[{}],",
            "\"output\":{{",
                "\"slot\":{},",
                "\"role\":\"{}\"",
            "}},",
            "\"transition_fingerprint\":\"{}\"",
            "}}"
        ),
        transition.step_index,
        json_escape(&transition.transition_id),
        inputs,
        transition.output_slot,
        json_escape(&transition.output_role),
        json_escape(&transition.transition_fingerprint),
    )
}

fn latest_prior_producer(
    builder: &AgentGraphBuilder,
    step_index: u32,
    slot: u8,
) -> Option<u32> {
    let steps = builder.introspection_steps();
    let upper = usize::try_from(step_index).ok()?.min(steps.len());
    (0..upper)
        .rev()
        .find(|index| steps[*index].5 == slot)
        .and_then(|index| u32::try_from(index).ok())
}

fn exact_external_binding_current(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<(String, String), String> {
    let binding = builder.semantic_edge_binding(step_index).ok_or_else(|| {
        format!(
            "bindSemanticLifecycleTransition: step {step_index} reads external slot 0 but has no persistent input edge binding"
        )
    })?;
    let metadata = workspace.input_port_metadata().ok_or_else(|| {
        "bindSemanticLifecycleTransition: external input metadata is unbound".to_string()
    })?;

    if metadata.role != binding.bound_role
        || metadata.source != binding.bound_source
        || metadata.revision != binding.bound_revision
        || metadata.fingerprint != binding.bound_fingerprint
    {
        return Err(format!(
            "bindSemanticLifecycleTransition: persistent input edge binding on step {step_index} has provenance drift; create a new graph revision before binding lifecycle semantics"
        ));
    }

    Ok((binding.bound_role.clone(), binding.binding_fingerprint.clone()))
}

fn resolve_input_lineage(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    step_index: u32,
    position: &str,
    slot: u8,
) -> Result<AgentGraphSemanticInputLineage, String> {
    if slot == 0 {
        let (role, fingerprint) =
            exact_external_binding_current(workspace, builder, step_index)?;
        return Ok(AgentGraphSemanticInputLineage {
            position: position.to_string(),
            slot,
            role,
            source_kind: "external_input_edge_binding".to_string(),
            source_step_index: None,
            source_fingerprint: fingerprint,
        });
    }

    let producer_step = latest_prior_producer(builder, step_index, slot).ok_or_else(|| {
        format!(
            "bindSemanticLifecycleTransition: {position} slot {slot} has no prior topology producer with lifecycle provenance"
        )
    })?;
    let producer = builder
        .semantic_lifecycle_transition(producer_step)
        .ok_or_else(|| {
            format!(
                "bindSemanticLifecycleTransition: {position} slot {slot} producer step {producer_step} has no semantic lifecycle transition"
            )
        })?;

    Ok(AgentGraphSemanticInputLineage {
        position: position.to_string(),
        slot,
        role: producer.output_role.clone(),
        source_kind: "prior_transition".to_string(),
        source_step_index: Some(producer_step),
        source_fingerprint: producer.transition_fingerprint.clone(),
    })
}

fn transition_fingerprint(
    step_index: u32,
    transition_id: &str,
    inputs: &[AgentGraphSemanticInputLineage],
    output_slot: u8,
    output_role: &str,
) -> String {
    let inputs_json = inputs
        .iter()
        .map(lineage_json)
        .collect::<Vec<_>>()
        .join(",");
    let canonical = format!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.semantic-lifecycle-transition-fingerprint.v1\",",
            "\"step_index\":{},",
            "\"transition_id\":\"{}\",",
            "\"inputs\":[{}],",
            "\"output\":{{\"slot\":{},\"role\":\"{}\"}}",
            "}}"
        ),
        step_index,
        json_escape(transition_id),
        inputs_json,
        output_slot,
        json_escape(output_role),
    );
    fnv1a64(canonical.as_bytes())
}

#[wasm_bindgen]
pub struct SemanticTransitionSpec {
    transition_id: String,
    input_roles: Vec<String>,
    output_role: String,
}

#[wasm_bindgen]
impl SemanticTransitionSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(
        transition_id: String,
        input_roles_json: String,
        output_role: String,
    ) -> Result<SemanticTransitionSpec, String> {
        if transition_id.is_empty() {
            return Err(
                "SemanticTransitionSpec.transition_id: value must be non-empty".to_string(),
            );
        }
        if transition_id.len() > MAX_TRANSITION_ID_BYTES {
            return Err(format!(
                "SemanticTransitionSpec.transition_id: {} bytes exceeds limit {MAX_TRANSITION_ID_BYTES}",
                transition_id.len()
            ));
        }
        let input_roles = parse_roles(&input_roles_json)?;
        validate_role(&output_role, "SemanticTransitionSpec.output_role")?;
        Ok(Self {
            transition_id,
            input_roles,
            output_role,
        })
    }

    #[wasm_bindgen(js_name = transitionId)]
    pub fn transition_id(&self) -> String {
        self.transition_id.clone()
    }

    #[wasm_bindgen(js_name = inputRoles)]
    pub fn input_roles(&self) -> String {
        string_array_json(&self.input_roles)
    }

    #[wasm_bindgen(js_name = outputRole)]
    pub fn output_role(&self) -> String {
        self.output_role.clone()
    }

    #[wasm_bindgen(js_name = describe)]
    pub fn describe(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.semantic-transition-spec.v1\",",
                "\"transition_id\":\"{}\",",
                "\"input_roles\":{},",
                "\"output_role\":\"{}\"",
                "}}"
            ),
            json_escape(&self.transition_id),
            string_array_json(&self.input_roles),
            json_escape(&self.output_role),
        )
    }
}

#[wasm_bindgen(js_name = semanticLifecycleCapabilities)]
pub fn semantic_lifecycle_capabilities() -> String {
    SEMANTIC_LIFECYCLE_V1.to_string()
}

#[wasm_bindgen(js_name = bindSemanticLifecycleTransition)]
pub fn bind_semantic_lifecycle_transition(
    workspace: &AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    spec: &SemanticTransitionSpec,
    step_index: u32,
) -> Result<bool, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "bindSemanticLifecycleTransition: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let index = usize::try_from(step_index)
        .map_err(|_| "bindSemanticLifecycleTransition: step index conversion failed".to_string())?;
    let steps = builder.introspection_steps();
    let Some((arity, _, _, in_slot, in_slot2, out_slot)) = steps.get(index).copied() else {
        return Err(format!(
            "bindSemanticLifecycleTransition: step index {step_index} is outside num_steps {}",
            steps.len()
        ));
    };

    if spec.input_roles.len() != usize::from(arity) {
        return Err(format!(
            "bindSemanticLifecycleTransition: spec has {} input roles but step {step_index} arity is {arity}",
            spec.input_roles.len()
        ));
    }

    let mut inputs = Vec::with_capacity(usize::from(arity));
    if arity == 1 {
        inputs.push(resolve_input_lineage(
            workspace,
            builder,
            step_index,
            "input",
            in_slot,
        )?);
    } else {
        inputs.push(resolve_input_lineage(
            workspace,
            builder,
            step_index,
            "left",
            in_slot,
        )?);
        inputs.push(resolve_input_lineage(
            workspace,
            builder,
            step_index,
            "right",
            in_slot2,
        )?);
    }

    for (position, (declared, actual)) in spec
        .input_roles
        .iter()
        .zip(inputs.iter().map(|input| &input.role))
        .enumerate()
    {
        if declared != actual {
            return Err(format!(
                "bindSemanticLifecycleTransition: input role mismatch at position {position}; declared {declared}, lineage resolves to {actual}"
            ));
        }
    }

    let fingerprint = transition_fingerprint(
        step_index,
        &spec.transition_id,
        &inputs,
        out_slot,
        &spec.output_role,
    );

    builder.bind_semantic_lifecycle_transition(AgentGraphSemanticLifecycleTransition {
        step_index,
        transition_id: spec.transition_id.clone(),
        inputs,
        output_slot: out_slot,
        output_role: spec.output_role.clone(),
        transition_fingerprint: fingerprint,
    })
}

pub(crate) fn semantic_lifecycle_identity_json(builder: &AgentGraphBuilder) -> String {
    let base_identity = semantic_graph_identity_json(builder);
    let mut canonical = format!(
        "{{\"base_semantic_graph_identity\":{},\"transitions\":[",
        base_identity
    );
    for (index, transition) in builder
        .semantic_lifecycle_transitions()
        .iter()
        .enumerate()
    {
        if index > 0 {
            canonical.push(',');
        }
        canonical.push_str(&format!(
            "{{\"step_index\":{},\"fingerprint\":\"{}\"}}",
            transition.step_index,
            json_escape(&transition.transition_fingerprint)
        ));
    }
    canonical.push_str("]}");

    let fingerprint = fnv1a64(canonical.as_bytes());
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-identity.v1\",",
            "\"base_semantic_graph_identity\":{},",
            "\"transition_count\":{},",
            "\"execution_program_identity_effect\":\"none\",",
            "\"fingerprint\":\"{}\"",
            "}}"
        ),
        base_identity,
        builder.semantic_lifecycle_transitions().len(),
        fingerprint,
    )
}

#[wasm_bindgen(js_name = semanticLifecycleIdentity)]
pub fn semantic_lifecycle_identity(builder: &AgentGraphBuilder) -> String {
    semantic_lifecycle_identity_json(builder)
}

#[wasm_bindgen(js_name = semanticLifecycleTransition)]
pub fn semantic_lifecycle_transition(
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<String, String> {
    let index = usize::try_from(step_index)
        .map_err(|_| "semanticLifecycleTransition: step index conversion failed".to_string())?;
    if index >= builder.introspection_steps().len() {
        return Err(format!(
            "semanticLifecycleTransition: step index {step_index} is outside num_steps {}",
            builder.num_steps()
        ));
    }

    match builder.semantic_lifecycle_transition(step_index) {
        Some(transition) => Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",\"status\":\"bound\",\"transition\":{}}}",
            transition_record_json(transition)
        )),
        None => Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",\"status\":\"unbound\",\"step_index\":{step_index}}}"
        )),
    }
}

#[wasm_bindgen(js_name = semanticLifecycleProjection)]
pub fn semantic_lifecycle_projection(builder: &AgentGraphBuilder) -> String {
    let transitions = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(transition_record_json)
        .collect::<Vec<_>>()
        .join(",");

    let bound = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(|transition| transition.step_index)
        .collect::<std::collections::BTreeSet<_>>();
    let unbound = (0..builder.num_steps())
        .filter(|step_index| !bound.contains(step_index))
        .map(|step_index| step_index.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-projection.v1\",",
            "\"projection_only\":true,",
            "\"transition_count\":{},",
            "\"coverage_complete\":{},",
            "\"unbound_step_indices\":[{}],",
            "\"identity\":{},",
            "\"transitions\":[{}]",
            "}}"
        ),
        builder.semantic_lifecycle_transitions().len(),
        if builder.semantic_lifecycle_transitions().len() == builder.num_steps() as usize {
            "true"
        } else {
            "false"
        },
        unbound,
        semantic_lifecycle_identity_json(builder),
        transitions,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        bind_semantic_lifecycle_transition, semantic_lifecycle_capabilities,
        semantic_lifecycle_identity, semantic_lifecycle_projection,
        semantic_lifecycle_transition, SemanticTransitionSpec,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::input_port_edge_binding::bind_input_port_consumer_edge;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn lifecycle_chains_external_and_internal_roles_without_layer_inference() {
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

        let first_id = workspace.reserve_layer_id(&registry, "relu-a".into()).unwrap();
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

        let consumer = InputPortConsumerSpec::new(
            "feature-consumer".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

        let first_transition = SemanticTransitionSpec::new(
            "observe-to-feature".into(),
            "[\"observation\"]".into(),
            "feature".into(),
        )
        .unwrap();
        bind_semantic_lifecycle_transition(
            &workspace,
            &mut builder,
            &first_transition,
            0,
        )
        .unwrap();

        let second_id = workspace.reserve_layer_id(&registry, "relu-b".into()).unwrap();
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

        let second_transition = SemanticTransitionSpec::new(
            "feature-to-candidate".into(),
            "[\"feature\"]".into(),
            "candidate".into(),
        )
        .unwrap();
        bind_semantic_lifecycle_transition(
            &workspace,
            &mut builder,
            &second_transition,
            1,
        )
        .unwrap();

        let status: serde_json::Value = serde_json::from_str(
            &semantic_lifecycle_transition(&builder, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(
            status["transition"]["inputs"][0]["source_step_index"],
            0
        );
        assert_eq!(status["transition"]["inputs"][0]["role"], "feature");
        assert_eq!(status["transition"]["output"]["role"], "candidate");
    }

    #[test]
    fn missing_internal_lineage_fails_closed() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let mut registry = LayerRegistry::new();

        let first_id = workspace.reserve_layer_id(&registry, "relu-a".into()).unwrap();
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

        let second_id = workspace.reserve_layer_id(&registry, "relu-b".into()).unwrap();
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

        let transition = SemanticTransitionSpec::new(
            "feature-to-candidate".into(),
            "[\"feature\"]".into(),
            "candidate".into(),
        )
        .unwrap();
        let err = bind_semantic_lifecycle_transition(
            &workspace,
            &mut builder,
            &transition,
            1,
        )
        .unwrap_err();
        assert!(err.contains("producer step 0 has no semantic lifecycle transition"));
    }

    #[test]
    fn lifecycle_identity_changes_without_claiming_execution_identity() {
        let builder = AgentGraphBuilder::new(3).unwrap();
        let identity: serde_json::Value =
            serde_json::from_str(&semantic_lifecycle_identity(&builder)).unwrap();
        assert_eq!(identity["transition_count"], 0);
        assert_eq!(identity["execution_program_identity_effect"], "none");

        let caps: serde_json::Value =
            serde_json::from_str(&semantic_lifecycle_capabilities()).unwrap();
        assert_eq!(caps["scope"]["execution_plan_effect"], "none");
        assert_eq!(caps["scope"]["decision_authority"], "agent");

        let projection: serde_json::Value =
            serde_json::from_str(&semantic_lifecycle_projection(&builder)).unwrap();
        assert_eq!(projection["coverage_complete"], true);
    }
}
