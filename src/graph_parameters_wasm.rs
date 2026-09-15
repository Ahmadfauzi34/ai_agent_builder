use wasm_bindgen::prelude::*;

use crate::graph::CompiledGraph;
use crate::graph_parameters::GraphParameterBinding;
use crate::registry::LayerRegistry;

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
    binding.read_flat(graph, registry)
}

#[wasm_bindgen(js_name = setGraphParametersFlat)]
pub fn set_graph_parameters_flat(
    graph: &CompiledGraph,
    registry: &mut LayerRegistry,
    candidate: &[f32],
) -> Result<(), String> {
    let binding = GraphParameterBinding::build(graph, registry)?;
    binding.apply_flat(graph, registry, candidate)
}
