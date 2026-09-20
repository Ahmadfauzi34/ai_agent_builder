use wasm_bindgen::prelude::*;

use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::{
    validate_spec_is_new, validate_workspace_input_slot, validate_workspace_layout_input,
    validate_workspace_op_label, workspace_compile,
};

const FAULT_SCHEMA_ID: &str = "burn-research.agent-fault.v1";

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\""),
            '\\' => out.push_str("\\\"),
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

fn ok(operation: &str) -> String {
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

fn fault(
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

fn control_domain_fault(
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

fn input_slot_fault(
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

fn free_output_fault(
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

fn missing_registry_layers(
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> Vec<(u8, u32)> {
    builder
        .interaction_referenced_layers()
        .into_iter()
        .filter(|(layer_type, layer_id)| !registry.layer_exists(*layer_type, *layer_id))
        .collect()
}

fn missing_registry_layers_string(values: &[(u8, u32)]) -> String {
    values
        .iter()
        .map(|(layer_type, layer_id)| format!("type=0x{layer_type:02X},id={layer_id}"))
        .collect::<Vec<_>>()
        .join(";")
}

/// Machine-readable description of the non-mutating action-preflight fault surface.
#[wasm_bindgen(js_name = interactionFaultCapabilities)]
pub fn interaction_fault_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"schema_id\":\"burn-research.agent-fault.v1\",",
        "\"role\":\"read_only_preflight_companion\",",
        "\"legacy_errors_unchanged\":true,",
        "\"mutation\":\"none\",",
        "\"scope\":[\"reserveSlot\",\"releaseSlot\",\"workspaceInitUnary\",\"workspaceInitBinary\",\"workspaceCompile\"],",
        "\"classes\":[\"control_precondition\",\"control_integrity\",\"resource_precondition\",\"semantic_precondition\",\"registry_precondition\",\"compile_precondition\"],",
        "\"deferred\":[\"Burn execution faults\",\"tensor runtime shape faults\",\"raw protocol faults\",\"workspaceWire identity diagnostics\"],",
        "\"contract\":\"fault codes derive from checked predicates rather than parsing legacy error strings\"",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = interactionCheckReserveSlot)]
pub fn interaction_check_reserve_slot(
    workspace: &AgentWorkspace,
    owner: String,
) -> String {
    if owner.is_empty() {
        return fault(
            "E_OWNER_EMPTY",
            "control_precondition",
            "reserveSlot",
            "owner.non_empty",
            "non-empty owner",
            "empty",
            "AgentWorkspace.reserveSlot.owner: value must be non-empty",
            true,
            vec!["provide_owner"],
        );
    }

    if workspace.interaction_free_slots().is_empty() {
        return fault(
            "E_NO_FREE_SLOT",
            "resource_precondition",
            "reserveSlot",
            "slot.free_exists",
            "at least one free slot",
            "none",
            "AgentWorkspace.reserveSlot: no free slot available",
            true,
            vec!["releaseSlot", "create_larger_workspace"],
        );
    }

    let mut probe = workspace.clone();
    match probe.reserve_slot(owner) {
        Ok(_) => ok("reserveSlot"),
        Err(message) => fault(
            "E_RESERVE_SLOT_REJECTED",
            "control_precondition",
            "reserveSlot",
            "reserveSlot.remaining_validation",
            "valid owner and available capacity",
            "rejected",
            message,
            true,
            vec!["inspect_workspace_limits"],
        ),
    }
}

#[wasm_bindgen(js_name = interactionCheckReleaseSlot)]
pub fn interaction_check_release_slot(
    workspace: &AgentWorkspace,
    slot: u8,
) -> String {
    if slot == 0 || u32::from(slot) >= workspace.interaction_num_slots() {
        return fault(
            "E_SLOT_OUT_OF_RANGE",
            "control_precondition",
            "releaseSlot",
            "slot.releasable_range",
            format!("1..{}", workspace.interaction_num_slots()),
            slot.to_string(),
            format!(
                "AgentWorkspace.releaseSlot: slot {slot} must be in 1..{}",
                workspace.interaction_num_slots()
            ),
            true,
            vec!["interactionSnapshot"],
        );
    }

    match workspace.interaction_slot_state(slot) {
        Some("reserved") => ok("releaseSlot"),
        Some(state) => fault(
            "E_SLOT_TRANSITION",
            "control_precondition",
            "releaseSlot",
            "slot.state_reserved",
            "reserved",
            state,
            format!(
                "AgentWorkspace.releaseSlot: slot {slot} cannot transition {state}->free; expected reserved->free"
            ),
            true,
            vec!["interactionSnapshot", "reserveSlot"],
        ),
        None => fault(
            "E_SLOT_NOT_FOUND",
            "control_integrity",
            "releaseSlot",
            "slot.exists",
            "workspace slot row",
            "missing",
            format!("AgentWorkspace.releaseSlot: slot {slot} not found"),
            false,
            vec!["recreate_workspace"],
        ),
    }
}

#[wasm_bindgen(js_name = interactionCheckInitUnary)]
pub fn interaction_check_init_unary(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    input_slot: u8,
    label: String,
) -> String {
    const OP: &str = "workspaceInitUnary";

    if let Some(problem) = control_domain_fault(OP, workspace, builder) {
        return problem;
    }

    if spec.layer_type() == LAYER_BINARY {
        return fault(
            "E_SPEC_ARITY",
            "control_precondition",
            OP,
            "spec.unary",
            "unary AgentLayerSpec",
            "binary AgentLayerSpec",
            "workspaceInitUnary: binary spec requires workspaceInitBinary",
            true,
            vec!["workspaceInitBinary"],
        );
    }

    if let Some(problem) = input_slot_fault(OP, "input", workspace, builder, input_slot) {
        return problem;
    }

    if let Err(message) = validate_workspace_layout_input(workspace, input_slot, spec, OP) {
        return fault(
            "E_LAYOUT_PREFLIGHT",
            "semantic_precondition",
            OP,
            "layout.edge_not_known_incompatible",
            "compatible|unknown",
            "known_incompatible_or_invalid_provenance",
            message,
            true,
            vec!["choose_compatible_layer", "inspect_layout_contract"],
        );
    }

    if let Err(message) = validate_spec_is_new(registry, spec, OP) {
        return fault(
            "E_LAYER_ALREADY_EXISTS",
            "registry_precondition",
            OP,
            "registry.layer_absent",
            "absent",
            "present",
            message,
            true,
            vec!["workspaceWireUnary", "reserveLayerId"],
        );
    }

    match workspace.interaction_layer_state(spec.layer_id()) {
        Some("reserved") => {}
        Some(state) => {
            return fault(
                "E_LAYER_NOT_RESERVED",
                "control_precondition",
                OP,
                "workspace.layer_reserved",
                "reserved",
                state,
                format!(
                    "{OP}: layer id {} must be reserved through AgentWorkspace.reserveLayerId before initialization",
                    spec.layer_id()
                ),
                true,
                vec!["reserveLayerId"],
            )
        }
        None => {
            return fault(
                "E_LAYER_NOT_RESERVED",
                "control_precondition",
                OP,
                "workspace.layer_reserved",
                "reserved",
                "missing",
                format!(
                    "{OP}: layer id {} must be reserved through AgentWorkspace.reserveLayerId before initialization",
                    spec.layer_id()
                ),
                true,
                vec!["reserveLayerId"],
            )
        }
    }

    if let Err(message) = validate_workspace_op_label(&label, OP) {
        return fault(
            "E_LABEL_LIMIT",
            "control_precondition",
            OP,
            "label.within_limit",
            "label within helper limit",
            format!("{} bytes", label.len()),
            message,
            true,
            vec!["shorten_label"],
        );
    }

    if let Some(problem) = free_output_fault(OP, workspace) {
        return problem;
    }

    if let Err(message) = validate_workspace_input_slot(workspace, builder, input_slot, OP) {
        return fault(
            "E_INPUT_PRECONDITION",
            "control_precondition",
            OP,
            "slot.readable",
            "readable slot",
            "rejected",
            message,
            true,
            vec!["interactionSnapshot"],
        );
    }

    ok(OP)
}

#[wasm_bindgen(js_name = interactionCheckInitBinary)]
pub fn interaction_check_init_binary(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    left_slot: u8,
    right_slot: u8,
    label: String,
) -> String {
    const OP: &str = "workspaceInitBinary";

    if let Some(problem) = control_domain_fault(OP, workspace, builder) {
        return problem;
    }

    if spec.layer_type() != LAYER_BINARY {
        return fault(
            "E_SPEC_ARITY",
            "control_precondition",
            OP,
            "spec.binary",
            "binary AgentLayerSpec",
            "unary AgentLayerSpec",
            "workspaceInitBinary: spec is not binary",
            true,
            vec!["workspaceInitUnary"],
        );
    }

    if let Some(problem) = input_slot_fault(OP, "left", workspace, builder, left_slot) {
        return problem;
    }
    if let Some(problem) = input_slot_fault(OP, "right", workspace, builder, right_slot) {
        return problem;
    }

    if let Err(message) = validate_workspace_layout_input(workspace, left_slot, spec, "workspaceInitBinary.left") {
        return fault(
            "E_LAYOUT_PREFLIGHT",
            "semantic_precondition",
            OP,
            "layout.left_edge_not_known_incompatible",
            "compatible|unknown",
            "known_incompatible_or_invalid_provenance",
            message,
            true,
            vec!["choose_compatible_layer", "inspect_layout_contract"],
        );
    }
    if let Err(message) = validate_workspace_layout_input(workspace, right_slot, spec, "workspaceInitBinary.right") {
        return fault(
            "E_LAYOUT_PREFLIGHT",
            "semantic_precondition",
            OP,
            "layout.right_edge_not_known_incompatible",
            "compatible|unknown",
            "known_incompatible_or_invalid_provenance",
            message,
            true,
            vec!["choose_compatible_layer", "inspect_layout_contract"],
        );
    }

    if let Err(message) = validate_spec_is_new(registry, spec, OP) {
        return fault(
            "E_LAYER_ALREADY_EXISTS",
            "registry_precondition",
            OP,
            "registry.layer_absent",
            "absent",
            "present",
            message,
            true,
            vec!["workspaceWireBinary", "reserveLayerId"],
        );
    }

    if workspace.interaction_layer_state(spec.layer_id()) != Some("reserved") {
        return fault(
            "E_LAYER_NOT_RESERVED",
            "control_precondition",
            OP,
            "workspace.layer_reserved",
            "reserved",
            workspace.interaction_layer_state(spec.layer_id()).unwrap_or("missing"),
            format!(
                "{OP}: layer id {} must be reserved through AgentWorkspace.reserveLayerId before initialization",
                spec.layer_id()
            ),
            true,
            vec!["reserveLayerId"],
        );
    }

    if let Err(message) = validate_workspace_op_label(&label, OP) {
        return fault(
            "E_LABEL_LIMIT",
            "control_precondition",
            OP,
            "label.within_limit",
            "label within helper limit",
            format!("{} bytes", label.len()),
            message,
            true,
            vec!["shorten_label"],
        );
    }

    if let Some(problem) = free_output_fault(OP, workspace) {
        return problem;
    }

    ok(OP)
}

#[wasm_bindgen(js_name = interactionCheckCompile)]
pub fn interaction_check_compile(
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> String {
    const OP: &str = "workspaceCompile";

    if u32::from(output_slot) >= builder.num_slots() {
        return fault(
            "E_SLOT_OUT_OF_RANGE",
            "compile_precondition",
            OP,
            "builder.output_in_range",
            format!("0..{}", builder.num_slots()),
            output_slot.to_string(),
            format!(
                "workspaceCompile: slot {output_slot} is outside builder num_slots {}",
                builder.num_slots()
            ),
            true,
            vec!["interactionSnapshot"],
        );
    }

    let written = builder.interaction_written_slots();
    if !written.contains(&output_slot) {
        return fault(
            "E_OUTPUT_NOT_WRITTEN",
            "compile_precondition",
            OP,
            "builder.output_written",
            "slot written by at least one graph step",
            output_slot.to_string(),
            format!("AgentGraphBuilder.compile: output slot {output_slot} is never written"),
            true,
            vec!["choose_written_output_slot", "interactionSnapshot"],
        );
    }

    let missing = missing_registry_layers(builder, registry);
    if !missing.is_empty() {
        return fault(
            "E_REGISTRY_BINDING",
            "registry_precondition",
            OP,
            "registry.contains_all_graph_layers",
            "all referenced layers initialized",
            missing_registry_layers_string(&missing),
            "workspaceCompile: graph references one or more layers absent from LayerRegistry",
            true,
            vec!["initialize_missing_layers", "workspaceWireUnary", "workspaceWireBinary"],
        );
    }

    match workspace_compile(builder, registry, output_slot) {
        Ok(_) => ok(OP),
        Err(message) => fault(
            "E_COMPILE_REJECTED",
            "compile_precondition",
            OP,
            "builder.compile_with_output",
            "compilable graph",
            "rejected",
            message,
            true,
            vec!["inspect_graph", "interactionSnapshot"],
        ),
    }
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

        let result = interaction_check_init_unary(
            &workspace,
            &builder,
            &registry,
            &spec,
            0,
            "relu".into(),
        );
        assert!(result.contains("\"code\":\"E_LAYER_NOT_RESERVED\""));
        assert!(result.contains("\"suggested_actions\":[\"reserveLayerId\"]"));
    }

    #[test]
    fn compile_check_tracks_real_graph_without_mutating_it() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
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
