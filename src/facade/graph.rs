//! Fasad WASM tunggal — domain `graph` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::graph::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::graph::graph_mutation_transaction::MAX_BRANCH_CANDIDATE_BYTES;
use crate::graph::graph_mutation_transaction::MAX_BRANCH_SNAPSHOT_BYTES;
use crate::graph::graph_mutation_transaction::MAX_CHECKPOINT_BRANCHES;
use crate::graph::graph_mutation_transaction::MAX_STATE_DIFF_RANGES;
use crate::graph::graph_parameters_wasm::apply_fresh_binding;
use crate::graph::graph_parameters_wasm::read_fresh_binding;
use crate::graph::multi_input_graph::multi_input_graph_capabilities as multi_input_graph_capabilities_json;
use crate::graph::CompiledGraph;
use crate::graph_parameters::GraphParameterBinding;
use crate::registry::LayerRegistry;

#[wasm_bindgen(js_name = programCapabilities)]
pub fn program_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"identity_schema\":\"burn-research.program-identity.v1\",",
        "\"entry\":\"CompiledGraph\",",
        "\"plan\":\"programPlan\",",
        "\"identity\":\"programIdentity\",",
        "\"binding_validation\":\"validateRegistryBinding\",",
        "\"execution_binding\":\"required\",",
        "\"identity_scope\":\"graph_plan_plus_layer_init_identity\",",
        "\"mutable_state_in_identity\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = multiInputGraphCapabilities)]
pub fn multi_input_graph_capabilities() -> String {
    multi_input_graph_capabilities_json()
}

#[wasm_bindgen(js_name = checkpointBranchCapabilities)]
pub fn checkpoint_branch_capabilities() -> String {
    format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.checkpoint-branch-capabilities.v1\",\"implementation\":\"wasm_burn_reference_machine\",\"contract\":\"transactional-graph-checkpoint-branch.v1\",\"checkpoint_format\":\"burn-research.multi-input-program-bundle.v1\",\"fork_source\":\"one validated exact stateful checkpoint snapshot\",\"branch_limit\":{MAX_CHECKPOINT_BRANCHES},\"baseline_snapshot_budget_bytes\":{MAX_BRANCH_SNAPSHOT_BYTES},\"candidate_bundle_budget_bytes\":{MAX_BRANCH_CANDIDATE_BYTES},\"state_diff_ranges_limit\":{MAX_STATE_DIFF_RANGES},\"promotion\":\"one branch per set; explicit caller authorization and that branch's passing receipt required\",\"state_diff_semantics\":\"exact checkpoint bundle byte difference with digests; does not assert semantic state equivalence\",\"output_comparison\":\"MultiInputVerificationCases Burn receipt included in branch receipt\",\"durable_host_ledger\":false}}")
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
