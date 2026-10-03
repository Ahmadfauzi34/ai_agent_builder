use wasm_bindgen::prelude::*;

use crate::graph::CompiledGraph;
use crate::graph_parameters::GraphParameterBinding;
use crate::registry::LayerRegistry;

// These helpers are intentionally private to the packaged same-call adapter.
// GraphParameterBinding::build has just validated the graph/registry structure,
// owner support, lengths, layouts, and fingerprints. WASM free-function calls are
// synchronous, so the host cannot mutate the registry between that successful build
// and the immediate read/apply below. Public core read_flat/apply_flat keep their
// full validate_current path for bindings retained across time.
fn read_fresh_binding(
    binding: GraphParameterBinding,
    registry: &LayerRegistry,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(binding.total_len());
    for owner in binding.owners() {
        let weights = registry
            .get_weights_flat(owner.layer_id(), owner.layer_type())
            .map_err(|error| format!("graph parameter fresh read: {error}"))?;
        if weights.len() != owner.len() {
            return Err(format!(
                "graph parameter fresh read: layer type 0x{:02X} id {} length changed from {} to {} inside one packaged call",
                owner.layer_type(),
                owner.layer_id(),
                owner.len(),
                weights.len()
            ));
        }
        out.extend_from_slice(&weights);
    }
    if out.len() != binding.total_len() {
        return Err(format!(
            "graph parameter fresh read: internal total length mismatch: expected {}, got {}",
            binding.total_len(),
            out.len()
        ));
    }
    Ok(out)
}

fn apply_fresh_binding(
    binding: GraphParameterBinding,
    registry: &mut LayerRegistry,
    candidate: &[f32],
) -> Result<(), String> {
    if candidate.len() != binding.total_len() {
        return Err(format!(
            "graph parameter apply: expected {} floats, got {}",
            binding.total_len(),
            candidate.len()
        ));
    }

    if let Some((index, _)) = candidate
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "graph parameter apply: candidate contains non-finite value at index {index}"
        ));
    }

    // Validate every candidate slice boundary before the first setter. Structural
    // setter preconditions were already proven by the immediately preceding build.
    for owner in binding.owners() {
        let end = owner
            .offset()
            .checked_add(owner.len())
            .ok_or_else(|| "graph parameter apply: slice boundary overflow".to_string())?;
        if end > candidate.len() {
            return Err(format!(
                "graph parameter apply: layer type 0x{:02X} id {} slice {}..{} exceeds candidate length {}",
                owner.layer_type(),
                owner.layer_id(),
                owner.offset(),
                end,
                candidate.len()
            ));
        }
    }

    for owner in binding.owners() {
        let end = owner.offset() + owner.len();
        registry
            .set_weights_flat(
                owner.layer_id(),
                owner.layer_type(),
                &candidate[owner.offset()..end],
            )
            .map_err(|error| {
                format!(
                    "graph parameter apply: fresh prevalidated setter failed for layer type 0x{:02X} id {}: {error}",
                    owner.layer_type(),
                    owner.layer_id()
                )
            })?;
    }
    Ok(())
}

#[wasm_bindgen(js_name = graphParameterCapabilities)]
pub fn graph_parameter_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"schema\":\"burn-research.graph-parameter-binding.v1\",",
        "\"layout_schema\":\"burn-research.graph-parameter-layout.v1\",",
        "\"ordering\":\"unique_first_use_graph_plan\",",
        "\"supported_flat_owner_types\":[\"linear\",\"conv\",\"embedding\",\"norm\"],",
        "\"parameterized_owner_without_flat_bridge\":\"fail_closed\",",
        "\"stateless_owner_coordinates\":0,",
        "\"apply_atomicity\":\"full_prevalidation_before_first_mutation\",",
        "\"mutable_parameter_values_in_identity\":false,",
        "\"program_bundle_is_candidate_format\":false,",
        "\"optimizer_state_in_binding\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = graphParameterLayout)]
pub fn graph_parameter_layout(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    Ok(GraphParameterBinding::build(graph, registry)?.layout_json())
}

#[wasm_bindgen(js_name = graphParameterIdentity)]
pub fn graph_parameter_identity(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    Ok(GraphParameterBinding::build(graph, registry)?.identity_json())
}

#[wasm_bindgen(js_name = getGraphParametersFlat)]
pub fn get_graph_parameters_flat(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<Vec<f32>, String> {
    let binding = GraphParameterBinding::build(graph, registry)?;
    read_fresh_binding(binding, registry)
}

#[wasm_bindgen(js_name = setGraphParametersFlat)]
pub fn set_graph_parameters_flat(
    graph: &CompiledGraph,
    registry: &mut LayerRegistry,
    candidate: &[f32],
) -> Result<(), String> {
    let binding = GraphParameterBinding::build(graph, registry)?;
    apply_fresh_binding(binding, registry, candidate)
}
