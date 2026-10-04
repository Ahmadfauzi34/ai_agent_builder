//! Fasad WASM tunggal — domain `interaction` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::interaction::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentLayerSpec;
use crate::interaction::actions_json;
use crate::interaction::bool_json;
use crate::interaction::compile_candidates;
use crate::interaction::phase;
use crate::interaction::registry_complete;
use crate::interaction::u32_array_json;
use crate::interaction::u8_array_json;
use crate::interaction::valid_actions_json;
use crate::interaction::validate_projection_inputs;
use crate::interaction::INTERACTION_SCHEMA_ID;
use crate::interaction_fault::control_domain_fault;
use crate::interaction_fault::fault;
use crate::interaction_fault::free_output_fault;
use crate::interaction_fault::input_slot_fault;
use crate::interaction_fault::missing_registry_layers;
use crate::interaction_fault::missing_registry_layers_string;
use crate::interaction_fault::ok;
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::ensure_workspace_layer_reserved;
use crate::workspace_ops::validate_spec_is_new;
use crate::workspace_ops::validate_workspace_input_slot;
use crate::workspace_ops::validate_workspace_layout_input;
use crate::workspace_ops::validate_workspace_op_label;
use crate::workspace_ops::workspace_compile;

/// Describe the projection-only interaction surface layered above existing
/// workspace, builder, and registry authorities.
#[wasm_bindgen(js_name = interactionCapabilities)]
pub fn interaction_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"schema_id\":\"burn-research.agent-interaction.v1\",",
        "\"role\":\"projection_only\",",
        "\"state_ownership\":\"none\",",
        "\"authorities\":{",
        "\"workspace\":\"AgentWorkspace metadata_and_control\",",
        "\"graph\":\"AgentGraphBuilder plan\",",
        "\"execution\":\"LayerRegistry\",",
        "\"numerics\":\"Burn reference machine\"",
        "},",
        "\"availability_semantics\":\"at_least_one_argument_assignment_satisfies_known_state_predicates; per-spec and runtime predicates still apply\",",
        "\"canonical_actions\":[\"reserveSlot\",\"releaseSlot\",\"workspaceInitUnary\",\"workspaceInitBinary\",\"workspaceCompileForRuntimeSubject when subject-bound\"],",
        "\"setup_actions\":[\"workspaceBindRuntimeSubject (optional semantic context)\",\"reserveLayerId\",\"AgentLayerSpec constructors\"],",
        "\"conditional_rejoin_actions\":[\"workspaceWireUnary\",\"workspaceWireBinary\"],",
        "\"introspection\":[\"introspectionCapabilities\",\"agentLayerCatalog\",\"describeWorkspace\",\"describeGraph\"],",
        "\"input_contract\":\"inputContractCapabilities\",",
        "\"input_port\":\"inputPortCapabilities\",",
        "\"input_port_consumer\":\"inputPortConsumerCapabilities\",",
        "\"input_port_routing\":\"inputPortRoutingCapabilities\",",
        "\"input_port_edge_binding\":\"inputPortEdgeBindingCapabilities\",",
        "\"semantic_ingress_manifest\":\"semanticIngressManifestCapabilities\",",
        "\"semantic_lifecycle\":\"semanticLifecycleCapabilities\",",
        "\"semantic_execution_context\":\"semanticExecutionContextCapabilities\",",
        "\"proof_provenance\":\"proofProvenanceCapabilities\",",
        "\"resolution_runtime_bridge\":\"resolutionRuntimeBridgeCapabilities\",",
        "\"runtime_resolution_evidence\":\"runtimeResolutionEvidenceCapabilities\",",
        "\"math_interaction\":\"mathInteractionCapabilities\",",
        "\"escape_hatches\":[\"workspaceCompile (unbound/legacy)\",\"AgentLayerSpec\",\"AgentGraphBuilder\",\"LayerRegistry\",\"raw_protocol\"],",
        "\"read_only_guarantee\":\"snapshot_and_valid_actions_do_not_mutate_inputs\"",
        "}"
    )
    .to_string()
}

/// Return the currently state-admissible canonical interaction families.
///
/// This is a read-only projection. Availability means the current state admits
/// at least one valid argument assignment; operation-specific validation remains
/// authoritative at the actual call boundary.
#[wasm_bindgen(js_name = interactionValidActions)]
pub fn interaction_valid_actions(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> Result<String, String> {
    validate_projection_inputs(workspace, builder)?;
    Ok(valid_actions_json(workspace, builder, registry))
}

/// Produce a compact point-in-time projection for agent planning.
///
/// No new source of truth is introduced: every field is derived from the
/// supplied workspace, graph builder, and registry.
#[wasm_bindgen(js_name = interactionSnapshot)]
pub fn interaction_snapshot(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> Result<String, String> {
    validate_projection_inputs(workspace, builder)?;

    let free_slots = workspace.interaction_free_slots();
    let readable_slots = workspace.interaction_readable_slots();
    let releasable_slots = workspace.interaction_releasable_slots();
    let reserved_layers = workspace.interaction_reserved_layer_ids();
    let initialized_layers = workspace.interaction_initialized_layer_ids();
    let written_slots = builder.interaction_written_slots();
    let complete = registry_complete(builder, registry);
    let compile_slots = compile_candidates(builder, registry);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"projection_only\":true,",
            "\"phase\":\"{}\",",
            "\"workspace\":{{",
            "\"num_slots\":{},",
            "\"runtime_subject\":{},",
            "\"free_slots\":{},",
            "\"readable_slots\":{},",
            "\"releasable_slots\":{},",
            "\"reserved_layer_ids\":{},",
            "\"initialized_layer_ids\":{}",
            "}},",
            "\"graph\":{{",
            "\"num_steps\":{},",
            "\"written_slots\":{},",
            "\"compile_candidate_slots\":{}",
            "}},",
            "\"registry\":{{",
            "\"referenced_layers_complete\":{}",
            "}},",
            "\"valid_actions\":{}",
            "}}"
        ),
        INTERACTION_SCHEMA_ID,
        phase(workspace, builder, registry),
        workspace.interaction_num_slots(),
        runtime_subject_binding_json(workspace),
        u8_array_json(&free_slots),
        u8_array_json(&readable_slots),
        u8_array_json(&releasable_slots),
        u32_array_json(&reserved_layers),
        u32_array_json(&initialized_layers),
        builder.num_steps(),
        u8_array_json(&written_slots),
        u8_array_json(&compile_slots),
        bool_json(complete),
        actions_json(workspace, builder, registry),
    ))
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
pub fn interaction_check_reserve_slot(workspace: &AgentWorkspace, owner: String) -> String {
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
pub fn interaction_check_release_slot(workspace: &AgentWorkspace, slot: u8) -> String {
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

    if let Err(message) = ensure_workspace_layer_reserved(workspace, spec, OP) {
        return fault(
            "E_LAYER_NOT_RESERVED",
            "control_precondition",
            OP,
            "workspace.layer_reserved",
            "reserved",
            workspace
                .interaction_layer_state(spec.layer_id())
                .unwrap_or("missing"),
            message,
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

    if let Err(message) =
        validate_workspace_layout_input(workspace, left_slot, spec, "workspaceInitBinary.left")
    {
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
    if let Err(message) =
        validate_workspace_layout_input(workspace, right_slot, spec, "workspaceInitBinary.right")
    {
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

    if let Err(message) = ensure_workspace_layer_reserved(workspace, spec, OP) {
        return fault(
            "E_LAYER_NOT_RESERVED",
            "control_precondition",
            OP,
            "workspace.layer_reserved",
            "reserved",
            workspace
                .interaction_layer_state(spec.layer_id())
                .unwrap_or("missing"),
            message,
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

    if let Err(message) =
        validate_workspace_input_slot(workspace, builder, left_slot, "workspaceInitBinary.left")
    {
        return fault(
            "E_INPUT_PRECONDITION",
            "control_precondition",
            OP,
            "slot.left_readable",
            "readable slot",
            "rejected",
            message,
            true,
            vec!["interactionSnapshot"],
        );
    }
    if let Err(message) =
        validate_workspace_input_slot(workspace, builder, right_slot, "workspaceInitBinary.right")
    {
        return fault(
            "E_INPUT_PRECONDITION",
            "control_precondition",
            OP,
            "slot.right_readable",
            "readable slot",
            "rejected",
            message,
            true,
            vec!["interactionSnapshot"],
        );
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
            vec![
                "initialize_missing_layers",
                "workspaceWireUnary",
                "workspaceWireBinary",
            ],
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
