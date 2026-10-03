pub use crate::facade::interaction::{
    interaction_capabilities, interaction_snapshot, interaction_valid_actions,
};
use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::workspace::AgentWorkspace;

pub(crate) const INTERACTION_SCHEMA_ID: &str = "burn-research.agent-interaction.v1";

pub(crate) fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

pub(crate) fn u8_array_json(values: &[u8]) -> String {
    let body = values
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

pub(crate) fn u32_array_json(values: &[u32]) -> String {
    let body = values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

pub(crate) fn validate_projection_inputs(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
) -> Result<(), String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "interaction: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }
    Ok(())
}

pub(crate) fn registry_complete(builder: &AgentGraphBuilder, registry: &LayerRegistry) -> bool {
    builder
        .interaction_referenced_layers()
        .into_iter()
        .all(|(layer_type, layer_id)| registry.layer_exists(layer_type, layer_id))
}

pub(crate) fn compile_candidates(builder: &AgentGraphBuilder, registry: &LayerRegistry) -> Vec<u8> {
    if builder.num_steps() == 0 || !registry_complete(builder, registry) {
        Vec::new()
    } else {
        builder.interaction_written_slots()
    }
}

pub(crate) fn phase(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> &'static str {
    if builder.num_steps() > 0 {
        if registry_complete(builder, registry) {
            "graph_ready"
        } else {
            "graph_incomplete"
        }
    } else if !workspace.interaction_reserved_layer_ids().is_empty() {
        "spec_construction"
    } else {
        "workspace_ready"
    }
}

pub(crate) fn actions_json(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> String {
    let free_slots = workspace.interaction_free_slots();
    let readable_slots = workspace.interaction_readable_slots();
    let releasable_slots = workspace.interaction_releasable_slots();
    let reserved_layers = workspace.interaction_reserved_layer_ids();
    let compile_slots = compile_candidates(builder, registry);

    let runtime_subject_bound = workspace.runtime_subject_binding().is_some();
    let runtime_subject_bind_available = workspace.runtime_subject_bind_available();
    let reserve_layer_available = workspace.interaction_row_capacity_available();
    let reserve_slot_available = !free_slots.is_empty();
    let init_available =
        !reserved_layers.is_empty() && !free_slots.is_empty() && !readable_slots.is_empty();
    let release_available = !releasable_slots.is_empty();
    let compile_available = !compile_slots.is_empty();

    format!(
        concat!(
            "[",
            "{{\"operation\":\"workspaceBindRuntimeSubject\",\"class\":\"optional_semantic_context\",\"available\":{},",
            "\"authority\":\"ResolutionRuntimeBridge host projection\",\"note\":\"bind before runtime reservation; optional for legacy workflows\"}},",
            "{{\"operation\":\"reserveLayerId\",\"class\":\"setup\",\"available\":{},",
            "\"authority\":\"AgentWorkspace\",\"note\":\"allocates metadata identity only\"}},",
            "{{\"operation\":\"reserveSlot\",\"class\":\"contracted_direct\",\"available\":{},",
            "\"candidate_slots\":{}}},",
            "{{\"operation\":\"releaseSlot\",\"class\":\"contracted_direct\",\"available\":{},",
            "\"candidate_slots\":{},\"candidate_policy\":\"manual_reservations_only\"}},",
            "{{\"operation\":\"workspaceInitUnary\",\"class\":\"canonical\",\"available\":{},",
            "\"candidate_layer_ids\":{},\"input_slots\":{},\"requires\":[\"matching_unary_AgentLayerSpec\",\"free_output_slot\"]}},",
            "{{\"operation\":\"workspaceInitBinary\",\"class\":\"canonical\",\"available\":{},",
            "\"candidate_layer_ids\":{},\"input_slots\":{},\"requires\":[\"matching_binary_AgentLayerSpec\",\"free_output_slot\"]}},",
            "{{\"operation\":\"workspaceCompileForRuntimeSubject\",\"class\":\"canonical_subject_bound\",\"available\":{},",
            "\"candidate_output_slots\":{},\"requires\":[\"runtime_subject_bound\",\"registry_complete_for_builder\"]}},",
            "{{\"operation\":\"workspaceCompile\",\"class\":\"legacy_escape_hatch\",\"available\":{},",
            "\"candidate_output_slots\":{},\"note\":\"does not bind programIdentity; bound-subject graph receipts require a previously exact-bound identity\"}}",
            "]"
        ),
        bool_json(runtime_subject_bind_available),
        bool_json(reserve_layer_available),
        bool_json(reserve_slot_available),
        u8_array_json(&free_slots),
        bool_json(release_available),
        u8_array_json(&releasable_slots),
        bool_json(init_available),
        u32_array_json(&reserved_layers),
        u8_array_json(&readable_slots),
        bool_json(init_available),
        u32_array_json(&reserved_layers),
        u8_array_json(&readable_slots),
        bool_json(compile_available && runtime_subject_bound),
        u8_array_json(&compile_slots),
        bool_json(compile_available),
        u8_array_json(&compile_slots),
    )
}

pub(crate) fn valid_actions_json(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> String {
    format!(
        "{{\"schema_version\":1,\"schema_id\":\"{INTERACTION_SCHEMA_ID}\",\"projection_only\":true,\"actions\":{}}}",
        actions_json(workspace, builder, registry)
    )
}

#[cfg(test)]
mod tests {
    use super::{interaction_capabilities, interaction_snapshot, interaction_valid_actions};
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn capabilities_make_projection_only_authority_explicit() {
        let capabilities = interaction_capabilities();
        assert!(capabilities.contains("\"role\":\"projection_only\""));
        assert!(capabilities.contains("\"state_ownership\":\"none\""));
        assert!(capabilities.contains("\"execution\":\"LayerRegistry\""));
        assert!(capabilities.contains("introspectionCapabilities"));
        assert!(capabilities.contains("describeWorkspace"));
        assert!(capabilities.contains("describeGraph"));
        assert!(capabilities.contains("inputContractCapabilities"));
        assert!(capabilities.contains("proofProvenanceCapabilities"));
        assert!(capabilities.contains("resolutionRuntimeBridgeCapabilities"));
        assert!(capabilities.contains("runtimeResolutionEvidenceCapabilities"));
        assert!(capabilities.contains("mathInteractionCapabilities"));
        assert!(capabilities.contains("raw_protocol"));
    }

    #[test]
    fn initial_snapshot_exposes_setup_without_inventing_execution_state() {
        let workspace = AgentWorkspace::new(3).unwrap();
        let builder = AgentGraphBuilder::new(3).unwrap();
        let registry = LayerRegistry::new();

        let snapshot = interaction_snapshot(&workspace, &builder, &registry).unwrap();
        assert!(snapshot.contains("\"phase\":\"workspace_ready\""));
        assert!(snapshot.contains("\"runtime_subject\":{\"status\":\"unbound\"}"));
        assert!(snapshot.contains("\"operation\":\"workspaceBindRuntimeSubject\",\"class\":\"optional_semantic_context\",\"available\":true"));
        assert!(snapshot
            .contains("\"operation\":\"reserveLayerId\",\"class\":\"setup\",\"available\":true"));
        assert!(snapshot.contains("\"operation\":\"workspaceCompileForRuntimeSubject\",\"class\":\"canonical_subject_bound\",\"available\":false"));
        assert!(snapshot.contains("\"operation\":\"workspaceCompile\",\"class\":\"legacy_escape_hatch\",\"available\":false"));
    }

    #[test]
    fn projection_is_read_only_and_tracks_canonical_progress() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();

        let workspace_before = workspace.snapshot();
        let steps_before = builder.num_steps();
        let params_before = registry.total_params();

        let _ = interaction_valid_actions(&workspace, &builder, &registry).unwrap();
        let _ = interaction_snapshot(&workspace, &builder, &registry).unwrap();

        assert_eq!(workspace.snapshot(), workspace_before);
        assert_eq!(builder.num_steps(), steps_before);
        assert_eq!(registry.total_params(), params_before);

        let layer_id = workspace
            .reserve_layer_id(&registry, "relu".into())
            .unwrap();
        let reserved = interaction_snapshot(&workspace, &builder, &registry).unwrap();
        assert!(reserved.contains("\"phase\":\"spec_construction\""));
        assert!(reserved.contains(&format!("\"reserved_layer_ids\":[{layer_id}]")));

        let relu = AgentLayerSpec::relu(layer_id);
        let out = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &relu,
            0,
            "relu".into(),
        )
        .unwrap();

        let ready = interaction_snapshot(&workspace, &builder, &registry).unwrap();
        assert!(ready.contains("\"phase\":\"graph_ready\""));
        assert!(ready.contains(&format!("\"compile_candidate_slots\":[{out}]")));
        assert!(ready.contains("\"operation\":\"workspaceCompileForRuntimeSubject\",\"class\":\"canonical_subject_bound\",\"available\":false"));
        assert!(ready.contains("\"operation\":\"workspaceCompile\",\"class\":\"legacy_escape_hatch\",\"available\":true"));
    }

    #[test]
    fn slot_count_mismatch_is_rejected_without_mutation() {
        let workspace = AgentWorkspace::new(2).unwrap();
        let builder = AgentGraphBuilder::new(3).unwrap();
        let registry = LayerRegistry::new();
        let before = workspace.snapshot();

        let err = interaction_snapshot(&workspace, &builder, &registry).unwrap_err();
        assert!(err.contains("does not match"));
        assert_eq!(workspace.snapshot(), before);
    }
}
