use burn::prelude::*;
use burn::tensor::TensorData;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

pub mod agent;
// --- domain modules (Opsi A): flat modules grouped without API change ---
pub mod dispatch;
pub mod evidence;
pub mod graph;
pub mod ingress;
pub mod resolution;
pub mod semantic;
// --- root re-exports: old `burn_research::<module>::` paths keep working ---
pub use dispatch::{
    agent_response_intent, response_dispatch_executor_preflight, response_dispatch_request,
    response_dispatch_requirements, response_intent_execution_gate,
    revision_dispatch_execution_adapter, revision_execution_evidence_rejoin,
};
pub use evidence::{
    program_bundle, proof_provenance, runtime_evidence_interpretation, runtime_resolution_evidence,
};
pub use graph::{
    graph_candidate_verification, graph_execution_trace, graph_mutation_transaction,
    graph_parameters, graph_parameters_wasm, graph_plan, graph_plan_explain,
    graph_reverify_execution_adapter, graph_reverify_runtime_binding, multi_input_graph,
};
pub use ingress::{
    input_contract, input_port, input_port_consumer, input_port_edge_binding, input_port_routing,
};
pub use resolution::{
    resolution_review, resolution_revision, resolution_runtime_bridge, resolution_subject,
};
pub use semantic::{
    semantic_execution_context, semantic_ingress_manifest, semantic_ingress_manifest_v2,
    semantic_lifecycle,
};
// --- root re-exports: items moved to the WASM facade (Opsi C, Fase 1) ---
// `burn_research::<item>` root paths keep working; JS surface unchanged.
pub use facade::tensor::{TensorView, WasmTensor};
pub use facade::wasm_types::{
    WasmActivation, WasmBinary, WasmComparison, WasmConv, WasmEmbedding, WasmFeatureNorm,
    WasmGhostModule, WasmIndexSource, WasmLinear, WasmLinearAlgebra, WasmMathProgram,
    WasmMathProgramBuilder, WasmMathProgramV4, WasmMathProgramV4Builder, WasmMathProgramV5,
    WasmMathProgramV5Builder, WasmMathProgramV6, WasmMathProgramV6Builder, WasmMathProgramV7,
    WasmMathProgramV7Builder, WasmMathProgramV8, WasmMathProgramV8Builder, WasmMathProgramV9,
    WasmMathProgramV9Builder, WasmNorm, WasmNumericKernel, WasmPool, WasmProbability,
    WasmReduction, WasmSeBlock, WasmShift, WasmStatistics, WasmTensorTransform,
};
// --- root re-exports: #[wasm_bindgen] free functions moved to the facade (Opsi C, Fase 1) ---
pub use facade::agent::agent_capabilities;
pub use facade::contracts::{
    agent_contract_schema, agent_contract_schema_version, agent_layout_compatibility,
    agent_layout_contract, agent_layout_contract_version, agent_spec_layout,
    validate_agent_layout_edge,
};
pub use facade::coprocessor::math_verify_vectors;
pub use facade::es::es_capabilities;
pub use facade::evidence::{
    export_multi_input_program_bundle, export_program_bundle, import_multi_input_program_bundle,
    import_program_bundle, math_proof_capabilities, multi_input_program_bundle_capabilities,
    program_bundle_capabilities, proof_provenance_capabilities,
    runtime_resolution_evidence_capabilities, workspace_bind_runtime_direct_math_operation,
    workspace_bind_runtime_math_program_plan, workspace_proof_ledger, workspace_record_attestation,
    workspace_verify_direct_math_1_receipt, workspace_verify_direct_math_2_receipt,
    workspace_verify_graph_receipt, workspace_verify_math_program_1_receipt,
    workspace_verify_math_program_2_receipt, workspace_verify_semantic_graph_receipt,
    workspace_verify_vector_receipt,
};
pub use facade::graph::{
    checkpoint_branch_capabilities, get_graph_parameters_flat, graph_parameter_capabilities,
    graph_parameter_identity, graph_parameter_layout, multi_input_graph_capabilities,
    program_capabilities, set_graph_parameters_flat,
};
pub use facade::ingress::{
    bind_input_port_consumer_edge, input_contract_capabilities, input_contract_compatibility,
    input_port_capabilities, input_port_consumer_capabilities, input_port_consumer_compatibility,
    input_port_consumer_edge_binding, input_port_consumer_edge_compatibility,
    input_port_edge_binding_capabilities, input_port_routing_capabilities,
    interaction_valid_actions_for_input_consumer, semantic_graph_identity,
    workspace_bind_input_contract, workspace_bind_input_port_metadata,
    workspace_clear_input_contract, workspace_clear_input_port_metadata, workspace_input_contract,
    workspace_input_port_metadata,
};
pub use facade::interaction::{
    interaction_capabilities, interaction_check_compile, interaction_check_init_binary,
    interaction_check_init_unary, interaction_check_release_slot, interaction_check_reserve_slot,
    interaction_fault_capabilities, interaction_snapshot, interaction_valid_actions,
};
pub use facade::introspection::{
    agent_layer_catalog, describe_graph, describe_workspace, introspection_capabilities,
};
pub use facade::math::{
    linear_algebra_capabilities, math_check_operation, math_describe_operation,
    math_interaction_capabilities, math_operation_catalog, math_plan_binding,
    math_valid_operations, numeric_kernel_capabilities, probability_capabilities,
    statistics_capabilities, tensor_transform_capabilities, wasm_comparison_capabilities,
    wasm_index_source_capabilities, wasm_math_program_capabilities,
    wasm_math_program_v4_capabilities, wasm_math_program_v5_capabilities,
    wasm_math_program_v6_capabilities, wasm_math_program_v7_capabilities,
    wasm_math_program_v8_capabilities, wasm_math_program_v9_capabilities,
    wasm_reduction_capabilities,
};
pub use facade::registry::{
    layer_registry_inventory_capabilities, layer_registry_operation_binding_capabilities,
};
pub use facade::resolution::{
    bind_runtime_subject_projection, resolution_runtime_bridge_capabilities,
    workspace_bind_runtime_subject, workspace_runtime_program_binding, workspace_runtime_subject,
};
pub use facade::semantic::{
    bind_semantic_lifecycle_transition, semantic_execution_context,
    semantic_execution_context_capabilities, semantic_ingress_manifest_capabilities,
    semantic_ingress_manifest_status, semantic_ingress_manifest_v2_capabilities,
    semantic_lifecycle_capabilities, semantic_lifecycle_identity, semantic_lifecycle_projection,
    semantic_lifecycle_transition,
};
pub use facade::workspace::{
    workspace_capabilities, workspace_compile, workspace_compile_for_runtime_subject,
    workspace_init_binary, workspace_init_unary, workspace_wire_binary, workspace_wire_unary,
};
pub mod authorization;
#[cfg(test)]
mod contract_matrix_tests;
pub mod contracts;
pub mod coprocessor;
pub mod effective_spec;
pub mod effective_spec_inherit_remainder;
pub mod es;
pub mod facade;
#[cfg(test)]
mod feature_norm_integration_tests;
#[cfg(test)]
mod hardening_tests;
pub mod interaction;
pub mod interaction_fault;
pub mod introspection;
pub mod layers;
pub mod math;
pub mod protocol;
pub mod registry;
#[cfg(test)]
mod stateless_lifecycle_tests;
#[cfg(test)]
mod tests;
pub mod workspace;
pub mod workspace_ops;

pub type WasmBackend = burn_ndarray::NdArray<f32>;

fn boundary_fail(message: String) -> ! {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

fn normalize_rank4_shape(shape: &[usize], context: &str) -> Result<[usize; 4], String> {
    if shape.len() > 4 {
        return Err(format!(
            "{context}: rank {} exceeds the 4D tensor bridge",
            shape.len()
        ));
    }

    let mut dims = [1usize; 4];
    for (index, &dim) in shape.iter().enumerate() {
        dims[index] = dim;
    }

    checked_element_count(dims, context)?;
    Ok(dims)
}

fn checked_element_count(dims: [usize; 4], context: &str) -> Result<usize, String> {
    dims.into_iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("{context}: shape element count overflow for {dims:?}"))
    })
}

fn validate_element_count(dims: [usize; 4], actual: usize, context: &str) -> Result<(), String> {
    let expected = checked_element_count(dims, context)?;
    if actual != expected {
        return Err(format!(
            "{context}: shape {dims:?} requires {expected} elements, got {actual}"
        ));
    }
    Ok(())
}

fn checked_sab_byte_length(total_elements: usize) -> Result<u32, String> {
    let bytes = total_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "TensorView: byte length overflow".to_string())?;
    u32::try_from(bytes).map_err(|_| {
        format!(
            "TensorView: {} elements require {} bytes, exceeding SharedArrayBuffer u32 length",
            total_elements, bytes
        )
    })
}

#[cfg(test)]
mod tensor_boundary_tests {
    use super::{
        checked_element_count, checked_sab_byte_length, normalize_rank4_shape,
        validate_element_count,
    };

    #[test]
    fn rank4_shape_pads_short_shapes_with_singletons() {
        assert_eq!(
            normalize_rank4_shape(&[2, 3], "test").unwrap(),
            [2, 3, 1, 1]
        );
    }

    #[test]
    fn rank4_shape_rejects_hidden_fifth_axis() {
        assert!(normalize_rank4_shape(&[1, 2, 3, 4, 5], "test").is_err());
    }

    #[test]
    fn element_count_rejects_shape_overflow() {
        assert!(checked_element_count([usize::MAX, 2, 1, 1], "test").is_err());
    }

    #[test]
    fn element_count_rejects_buffer_mismatch() {
        let err = validate_element_count([2, 3, 1, 1], 5, "test").unwrap_err();
        assert!(err.contains("requires 6 elements, got 5"));
    }

    #[test]
    fn sab_byte_length_rejects_u32_overflow() {
        let too_many = (u32::MAX as usize / std::mem::size_of::<f32>()) + 1;
        assert!(checked_sab_byte_length(too_many).is_err());
    }
}
