//! Fasad WASM tunggal — domain `workspace` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::workspace::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentLayerSpec;
use crate::graph::CompiledGraph;
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::ensure_registry_matches_spec;
use crate::workspace_ops::ensure_workspace_layer_reserved;
use crate::workspace_ops::finalize_initialized_binary;
use crate::workspace_ops::finalize_initialized_unary;
use crate::workspace_ops::reserve_workspace_output_slot;
use crate::workspace_ops::rollback_init_transaction;
use crate::workspace_ops::validate_builder_slot;
use crate::workspace_ops::validate_spec_is_new;
use crate::workspace_ops::validate_workspace_input_slot;
use crate::workspace_ops::validate_workspace_layout_input;
use crate::workspace_ops::validate_workspace_op_label;

/// Discover the canonical agent-facing workspace/control-plane API.
#[wasm_bindgen(js_name = workspaceCapabilities)]
pub fn workspace_capabilities() -> String {
    concat!(
        "{",
        "\"state\":\"AgentWorkspace\",",
        "\"ownership\":\"metadata_only\",",
        "\"execution_truth\":\"LayerRegistry\",",
        "\"graph\":\"AgentGraphBuilder\",",
        "\"provenance\":{\"wire_identity\":\"exact_validated_init_fingerprint\",\"syncLayer\":\"exact_identity_metadata_only_not_canonical_orchestration\",\"layout_preflight\":\"canonical_slot_owner_to_layer_type_variant\"},",
        "\"atomicity\":{\"compile\":\"non_mutating_output_override\",\"subject_bound_compile\":\"bind_exact_program_identity_only_after_successful_compile\",\"workspace_init\":\"transactional_post_registry_rollback\"},",
        "\"slot_lifecycle\":{\"states\":[\"input\",\"free\",\"reserved\"],\"readable\":[\"input\",\"reserved\"],\"reserve\":\"free->reserved\",\"release\":\"reserved->free\",\"invalid_transition\":\"error_no_mutation\"},",
        "\"layout_policy\":{\"known_incompatible\":\"reject_before_mutation\",\"unknown\":\"defer_to_runtime\",\"implicit_relayout\":\"forbidden\"},",
        "\"ops\":[\"workspaceInitUnary\",\"workspaceInitBinary\",\"workspaceWireUnary\",\"workspaceWireBinary\",\"workspaceCompile\",\"workspaceCompileForRuntimeSubject\"],",
        "\"workspace_methods\":[\"reserveLayerId\",\"reserveSlot\",\"releaseSlot\",\"syncLayer\",\"forgetLayer\",\"recordProof\",\"recordEvent\",\"put\",\"get\",\"query\",\"remove\",\"tableNames\",\"snapshot\",\"limits\"],",
        "\"escape_hatches\":[\"workspaceCompile\",\"AgentLayerSpec\",\"AgentGraphBuilder\",\"LayerRegistry\",\"raw_protocol\"],",
        "\"recommended_flow\":[\"bind_runtime_subject_if_used\",\"reserve_layer\",\"construct_spec\",\"init_or_wire\",\"subject_bound_compile_if_bound\",\"run\",\"verify\"]",
        "}"
    )
    .to_string()
}

/// Reconcile an already initialized unary layer into workspace metadata and graph wiring.
/// The supplied spec must exactly match the live registry layer's validated init identity.
#[wasm_bindgen(js_name = workspaceWireUnary)]
pub fn workspace_wire_unary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    input_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() == LAYER_BINARY {
        return Err("workspaceWireUnary: binary spec requires workspaceWireBinary".into());
    }
    validate_workspace_input_slot(workspace, builder, input_slot, "workspaceWireUnary")?;
    validate_workspace_layout_input(workspace, input_slot, spec, "workspaceWireUnary")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireUnary")?;
    validate_workspace_op_label(&label, "workspaceWireUnary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireUnary",
    )?;
    // Metadata reconciliation happens only after identity proof and output reservation succeed.
    if let Err(err) = workspace.sync_layer(registry, spec, label) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    if let Err(err) = builder.add_unary(spec, input_slot, output_slot) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    Ok(output_slot)
}

/// Reconcile an already initialized binary layer into workspace metadata and graph wiring.
/// The supplied spec must exactly match the live registry layer's validated init identity.
#[wasm_bindgen(js_name = workspaceWireBinary)]
pub fn workspace_wire_binary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    left_slot: u8,
    right_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() != LAYER_BINARY {
        return Err("workspaceWireBinary: spec is not binary".into());
    }
    validate_workspace_input_slot(workspace, builder, left_slot, "workspaceWireBinary.left")?;
    validate_workspace_input_slot(workspace, builder, right_slot, "workspaceWireBinary.right")?;
    validate_workspace_layout_input(workspace, left_slot, spec, "workspaceWireBinary.left")?;
    validate_workspace_layout_input(workspace, right_slot, spec, "workspaceWireBinary.right")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireBinary")?;
    validate_workspace_op_label(&label, "workspaceWireBinary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireBinary",
    )?;
    if let Err(err) = workspace.sync_layer(registry, spec, label) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    if let Err(err) = builder.add_binary(spec, left_slot, right_slot, output_slot) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    Ok(output_slot)
}

/// Initialize a reserved unary layer and wire it into the graph.
/// A workspace checkpoint protects the entire control state until graph commit succeeds.
#[wasm_bindgen(js_name = workspaceInitUnary)]
pub fn workspace_init_unary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &mut LayerRegistry,
    spec: &AgentLayerSpec,
    input_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() == LAYER_BINARY {
        return Err("workspaceInitUnary: binary spec requires workspaceInitBinary".into());
    }
    validate_workspace_input_slot(workspace, builder, input_slot, "workspaceInitUnary")?;
    validate_workspace_layout_input(workspace, input_slot, spec, "workspaceInitUnary")?;
    validate_spec_is_new(registry, spec, "workspaceInitUnary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitUnary")?;
    validate_workspace_op_label(&label, "workspaceInitUnary")?;

    let workspace_checkpoint = workspace.clone();
    let output_slot = match reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitUnary",
    ) {
        Ok(slot) => slot,
        Err(err) => {
            *workspace = workspace_checkpoint;
            return Err(err);
        }
    };

    if let Err(err) = registry.init_agent_layer(spec) {
        return Err(rollback_init_transaction(
            workspace,
            &workspace_checkpoint,
            registry,
            spec,
            "workspaceInitUnary",
            err,
        ));
    }

    finalize_initialized_unary(
        workspace,
        &workspace_checkpoint,
        builder,
        registry,
        spec,
        input_slot,
        output_slot,
        label,
    )
}

/// Initialize a reserved binary layer and wire it into the graph.
/// A workspace checkpoint protects the entire control state until graph commit succeeds.
#[wasm_bindgen(js_name = workspaceInitBinary)]
pub fn workspace_init_binary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &mut LayerRegistry,
    spec: &AgentLayerSpec,
    left_slot: u8,
    right_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() != LAYER_BINARY {
        return Err("workspaceInitBinary: spec is not binary".into());
    }
    validate_workspace_input_slot(workspace, builder, left_slot, "workspaceInitBinary.left")?;
    validate_workspace_input_slot(workspace, builder, right_slot, "workspaceInitBinary.right")?;
    validate_workspace_layout_input(workspace, left_slot, spec, "workspaceInitBinary.left")?;
    validate_workspace_layout_input(workspace, right_slot, spec, "workspaceInitBinary.right")?;
    validate_spec_is_new(registry, spec, "workspaceInitBinary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitBinary")?;
    validate_workspace_op_label(&label, "workspaceInitBinary")?;

    let workspace_checkpoint = workspace.clone();
    let output_slot = match reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitBinary",
    ) {
        Ok(slot) => slot,
        Err(err) => {
            *workspace = workspace_checkpoint;
            return Err(err);
        }
    };

    if let Err(err) = registry.init_agent_layer(spec) {
        return Err(rollback_init_transaction(
            workspace,
            &workspace_checkpoint,
            registry,
            spec,
            "workspaceInitBinary",
            err,
        ));
    }

    finalize_initialized_binary(
        workspace,
        &workspace_checkpoint,
        builder,
        registry,
        spec,
        left_slot,
        right_slot,
        output_slot,
        label,
    )
}

/// Compile using a temporary output selection without mutating builder state.
#[wasm_bindgen(js_name = workspaceCompile)]
pub fn workspace_compile(
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    validate_builder_slot(builder, output_slot, "workspaceCompile")?;
    builder.compile_with_output(registry, output_slot)
}

/// Compile and bind the exact CompiledGraph.programIdentity to the immutable runtime subject.
///
/// This is the canonical compile path for subject-bound verification. The historical
/// workspaceCompile surface remains available as an explicit unbound/legacy escape hatch.
#[wasm_bindgen(js_name = workspaceCompileForRuntimeSubject)]
pub fn workspace_compile_for_runtime_subject(
    workspace: &mut AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceCompileForRuntimeSubject: workspace has no bound runtime subject".to_string(),
        );
    }
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "workspaceCompileForRuntimeSubject: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    validate_builder_slot(builder, output_slot, "workspaceCompileForRuntimeSubject")?;
    let graph = builder.compile_with_output(registry, output_slot)?;
    workspace.bind_runtime_program_identity(graph.program_identity())?;
    Ok(graph)
}
