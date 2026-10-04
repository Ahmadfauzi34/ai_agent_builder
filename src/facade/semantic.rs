//! Fasad WASM tunggal — domain `semantic` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::semantic::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentGraphSemanticLifecycleTransition;
use crate::graph::CompiledGraph;
use crate::registry::LayerRegistry;
use crate::semantic::semantic_execution_context::semantic_execution_context_for;
use crate::semantic::semantic_execution_context::SEMANTIC_EXECUTION_CONTEXT_V1;
use crate::semantic::semantic_ingress_manifest::bool_json;
use crate::semantic::semantic_ingress_manifest::json_escape;
use crate::semantic::semantic_ingress_manifest::workspace_port_status;
use crate::semantic::semantic_ingress_manifest::RuntimeBacking;
use crate::semantic::semantic_ingress_manifest::SemanticIngressManifest;
use crate::semantic::semantic_ingress_manifest::SEMANTIC_INGRESS_MANIFEST_V1;
use crate::semantic::semantic_ingress_manifest_v2::CONTRACT;
use crate::semantic::semantic_lifecycle::resolve_input_lineage;
use crate::semantic::semantic_lifecycle::semantic_lifecycle_identity_json;
use crate::semantic::semantic_lifecycle::transition_fingerprint;
use crate::semantic::semantic_lifecycle::transition_record_json;
use crate::semantic::semantic_lifecycle::SemanticTransitionSpec;
use crate::semantic::semantic_lifecycle::SEMANTIC_LIFECYCLE_V1;
use crate::workspace::AgentWorkspace;

#[wasm_bindgen(js_name = semanticExecutionContextCapabilities)]
pub fn semantic_execution_context_capabilities() -> String {
    SEMANTIC_EXECUTION_CONTEXT_V1.to_string()
}

#[wasm_bindgen(js_name = semanticExecutionContext)]
pub fn semantic_execution_context(
    builder: &AgentGraphBuilder,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    semantic_execution_context_for(builder, graph, registry).map(|context| context.json())
}

#[wasm_bindgen(js_name = semanticIngressManifestCapabilities)]
pub fn semantic_ingress_manifest_capabilities() -> String {
    SEMANTIC_INGRESS_MANIFEST_V1.to_string()
}

#[wasm_bindgen(js_name = semanticIngressManifestStatus)]
pub fn semantic_ingress_manifest_status(
    workspace: &AgentWorkspace,
    manifest: &SemanticIngressManifest,
) -> String {
    let mut required_uncovered = 0usize;
    let mut runtime_backed_count = 0usize;
    let mut deferred_count = 0usize;

    let port_status = manifest
        .ports
        .iter()
        .map(|port| {
            let status = workspace_port_status(workspace, port);
            if port.runtime_backing == RuntimeBacking::Slot0 {
                runtime_backed_count += 1;
            } else {
                deferred_count += 1;
            }
            if port.required && status != "runtime_backing_current" {
                required_uncovered += 1;
            }
            format!(
                concat!(
                    "{{",
                    "\"logical_port_id\":\"{}\",",
                    "\"role\":\"{}\",",
                    "\"required\":{},",
                    "\"runtime_backing\":\"{}\",",
                    "\"status\":\"{}\"",
                    "}}"
                ),
                json_escape(&port.logical_port_id),
                json_escape(&port.role),
                bool_json(port.required),
                port.runtime_backing.as_str(),
                status,
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-ingress-manifest-status.v1\",",
            "\"manifest_fingerprint\":\"{}\",",
            "\"port_count\":{},",
            "\"runtime_backed_port_count\":{},",
            "\"deferred_port_count\":{},",
            "\"required_uncovered_count\":{},",
            "\"runtime_coverage_complete\":{},",
            "\"ports\":[{}],",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\"",
            "}}"
        ),
        manifest.manifest_fingerprint_internal(),
        manifest.ports.len(),
        runtime_backed_count,
        deferred_count,
        required_uncovered,
        bool_json(required_uncovered == 0),
        port_status,
    )
}

#[wasm_bindgen(js_name = semanticIngressManifestV2Capabilities)]
pub fn semantic_ingress_manifest_v2_capabilities() -> String {
    CONTRACT.to_string()
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
            workspace, builder, step_index, "input", in_slot,
        )?);
    } else {
        inputs.push(resolve_input_lineage(
            workspace, builder, step_index, "left", in_slot,
        )?);
        inputs.push(resolve_input_lineage(
            workspace, builder, step_index, "right", in_slot2,
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
