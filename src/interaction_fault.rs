pub use crate::facade::interaction::{
    interaction_check_compile, interaction_check_init_binary, interaction_check_init_unary,
    interaction_check_release_slot, interaction_check_reserve_slot, interaction_fault_capabilities,
};
use wasm_bindgen::prelude::*;

use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::{
    ensure_workspace_layer_reserved, validate_spec_is_new, validate_workspace_input_slot,
    validate_workspace_layout_input, validate_workspace_op_label, workspace_compile,
};

const FAULT_SCHEMA_ID: &str = "burn-research.agent-fault.v1";

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

fn string_array_json(values: &[&str]) -> String {
    let body = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

pub(crate) fn ok(operation: &str) -> String {
    format!(
        "{{\"schema_version\":1,\"schema_id\":\"{FAULT_SCHEMA_ID}\",\"status\":\"ok\",\"operation\":\"{}\",\"mutation\":\"none\"}}",
        json_escape(operation)
    )
}

struct Fault<'a> {
    code: &'a str,
    class: &'a str,
    operation: &'a str,
    predicate: &'a str,
    expected: String,
    actual: String,
    message: String,
    recoverable: bool,
    suggested_actions: Vec<&'a str>,
}

impl Fault<'_> {
    fn json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"{}\",",
                "\"status\":\"fault\",",
                "\"fault\":{{",
                "\"code\":\"{}\",",
                "\"class\":\"{}\",",
                "\"operation\":\"{}\",",
                "\"predicate\":\"{}\",",
                "\"expected\":\"{}\",",
                "\"actual\":\"{}\",",
                "\"mutation\":\"none\",",
                "\"recoverable\":{},",
                "\"suggested_actions\":{},",
                "\"message\":\"{}\"",
                "}}",
                "}}"
            ),
            FAULT_SCHEMA_ID,
            json_escape(self.code),
            json_escape(self.class),
            json_escape(self.operation),
            json_escape(self.predicate),
            json_escape(&self.expected),
            json_escape(&self.actual),
            if self.recoverable { "true" } else { "false" },
            string_array_json(&self.suggested_actions),
            json_escape(&self.message),
        )
    }
}

pub(crate) fn fault(
    code: &'static str,
    class: &'static str,
    operation: &'static str,
    predicate: &'static str,
    expected: impl Into<String>,
    actual: impl Into<String>,
    message: impl Into<String>,
    recoverable: bool,
    suggested_actions: Vec<&'static str>,
) -> String {
    Fault {
        code,
        class,
        operation,
        predicate,
        expected: expected.into(),
        actual: actual.into(),
        message: message.into(),
        recoverable,
        suggested_actions,
    }
    .json()
}

pub(crate) fn control_domain_fault(
    operation: &'static str,
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
) -> Option<String> {
    if workspace.interaction_num_slots() == builder.num_slots() {
        None
    } else {
        Some(fault(
            "E_CONTROL_DOMAIN_MISMATCH",
            "control_precondition",
            operation,
            "workspace.num_slots == builder.num_slots",
            builder.num_slots().to_string(),
            workspace.interaction_num_slots().to_string(),
            format!(
                "{operation}: workspace num_slots {} does not match builder num_slots {}",
                workspace.interaction_num_slots(),
                builder.num_slots()
            ),
            true,
            vec!["recreate_workspace_or_builder_with_matching_num_slots"],
        ))
    }
}

pub(crate) fn input_slot_fault(
    operation: &'static str,
    side: &'static str,
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    slot: u8,
) -> Option<String> {
    if u32::from(slot) >= builder.num_slots() {
        return Some(fault(
            "E_SLOT_OUT_OF_RANGE",
            "control_precondition",
            operation,
            "slot.in_builder_range",
            format!("0..{}", builder.num_slots()),
            slot.to_string(),
            format!(
                "{operation}.{side}: slot {slot} is outside builder num_slots {}",
                builder.num_slots()
            ),
            true,
            vec!["choose_in_range_slot", "interactionSnapshot"],
        ));
    }

    match workspace.interaction_slot_state(slot) {
        Some("input" | "reserved") => None,
        Some(state) => Some(fault(
            "E_SLOT_NOT_READABLE",
            "control_precondition",
            operation,
            "slot.readable",
            "input|reserved",
            state,
            format!("{operation}.{side}: slot {slot} is not readable while state is {state}"),
            true,
            vec!["interactionValidActions", "reserveSlot"],
        )),
        None => Some(fault(
            "E_SLOT_NOT_FOUND",
            "control_integrity",
            operation,
            "slot.exists",
            "workspace slot row",
            "missing",
            format!("{operation}.{side}: slot {slot} is not present in workspace metadata"),
            false,
            vec!["recreate_workspace"],
        )),
    }
}

pub(crate) fn free_output_fault(
    operation: &'static str,
    workspace: &AgentWorkspace,
) -> Option<String> {
    if workspace.interaction_free_slots().is_empty() {
        Some(fault(
            "E_NO_FREE_SLOT",
            "resource_precondition",
            operation,
            "slot.free_exists",
            "at least one free slot",
            "none",
            format!("{operation}: no free output slot is available"),
            true,
            vec!["releaseSlot", "create_larger_workspace"],
        ))
    } else {
        None
    }
}

pub(crate) fn missing_registry_layers(
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> Vec<(u8, u32)> {
    builder
        .interaction_referenced_layers()
        .into_iter()
        .filter(|(layer_type, layer_id)| !registry.layer_exists(*layer_type, *layer_id))
        .collect()
}

pub(crate) fn missing_registry_layers_string(values: &[(u8, u32)]) -> String {
    values
        .iter()
        .map(|(layer_type, layer_id)| format!("type=0x{layer_type:02X},id={layer_id}"))
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use super::{
        interaction_check_compile, interaction_check_init_unary, interaction_check_release_slot,
        interaction_fault_capabilities,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn capabilities_state_fault_channel_is_companion_not_replacement() {
        let caps = interaction_fault_capabilities();
        assert!(caps.contains("\"legacy_errors_unchanged\":true"));
        assert!(caps.contains("\"role\":\"read_only_preflight_companion\""));
    }

    #[test]
    fn illegal_release_is_machine_readable_without_mutation() {
        let workspace = AgentWorkspace::new(3).unwrap();
        let before = workspace.snapshot();

        let result = interaction_check_release_slot(&workspace, 1);
        assert!(result.contains("\"status\":\"fault\""));
        assert!(result.contains("\"code\":\"E_SLOT_TRANSITION\""));
        assert!(result.contains("\"predicate\":\"slot.state_reserved\""));
        assert!(result.contains("\"actual\":\"free\""));
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn unreserved_layer_is_explained_before_init() {
        let workspace = AgentWorkspace::new(3).unwrap();
        let builder = AgentGraphBuilder::new(3).unwrap();
        let registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(77);

        let result =
            interaction_check_init_unary(&workspace, &builder, &registry, &spec, 0, "relu".into());
        assert!(result.contains("\"code\":\"E_LAYER_NOT_RESERVED\""));
        assert!(result.contains("\"suggested_actions\":[\"reserveLayerId\"]"));
    }

    #[test]
    fn compile_check_tracks_real_graph_without_mutating_it() {
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
        let before_steps = builder.num_steps();
        let before_params = registry.total_params();

        let ok = interaction_check_compile(&builder, &registry, out);
        assert!(ok.contains("\"status\":\"ok\""));
        assert_eq!(builder.num_steps(), before_steps);
        assert_eq!(registry.total_params(), before_params);

        let bad = interaction_check_compile(&builder, &registry, 2);
        assert!(bad.contains("\"code\":\"E_OUTPUT_NOT_WRITTEN\""));
    }
}
