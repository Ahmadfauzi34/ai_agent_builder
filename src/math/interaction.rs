use wasm_bindgen::prelude::*;

const MATH_INTERACTION_SCHEMA_ID: &str = "burn-research.math-interaction.v1";
const MATH_OPERATION_CATALOG_SCHEMA_ID: &str = "burn-research.math-operation-catalog.v1";

#[derive(Clone, Copy)]
struct MathOperationDescriptor {
    id: &'static str,
    family: &'static str,
    arity: u8,
    direct_surface: &'static str,
    source_contract: &'static str,
    program_builder: &'static str,
    minimum_program_generation: &'static str,
    shape_rule: &'static str,
}

const OPERATIONS: &[MathOperationDescriptor] = &[
    MathOperationDescriptor { id: "numeric.add", family: "numeric", arity: 2, direct_surface: "WasmNumericKernel.add", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_ADD)", minimum_program_generation: "v1", shape_rule: "exact_shape_match_no_broadcast" },
    MathOperationDescriptor { id: "numeric.sub", family: "numeric", arity: 2, direct_surface: "WasmNumericKernel.sub", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_SUB)", minimum_program_generation: "v1", shape_rule: "exact_shape_match_no_broadcast" },
    MathOperationDescriptor { id: "numeric.mul", family: "numeric", arity: 2, direct_surface: "WasmNumericKernel.mul", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_MUL)", minimum_program_generation: "v1", shape_rule: "exact_shape_match_no_broadcast" },
    MathOperationDescriptor { id: "numeric.div", family: "numeric", arity: 2, direct_surface: "WasmNumericKernel.div", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_DIV)", minimum_program_generation: "v1", shape_rule: "exact_shape_match_no_broadcast" },
    MathOperationDescriptor { id: "numeric.abs", family: "numeric", arity: 1, direct_surface: "WasmNumericKernel.abs", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_ABS)", minimum_program_generation: "v1", shape_rule: "shape_preserved" },
    MathOperationDescriptor { id: "numeric.sqrt", family: "numeric", arity: 1, direct_surface: "WasmNumericKernel.sqrt", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_SQRT)", minimum_program_generation: "v1", shape_rule: "shape_preserved" },
    MathOperationDescriptor { id: "numeric.exp", family: "numeric", arity: 1, direct_surface: "WasmNumericKernel.exp", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_EXP)", minimum_program_generation: "v1", shape_rule: "shape_preserved" },
    MathOperationDescriptor { id: "numeric.log", family: "numeric", arity: 1, direct_surface: "WasmNumericKernel.log", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_LOG)", minimum_program_generation: "v1", shape_rule: "shape_preserved" },
    MathOperationDescriptor { id: "numeric.clamp", family: "numeric", arity: 1, direct_surface: "WasmNumericKernel.clamp", source_contract: "numericKernelCapabilities", program_builder: "MathProgramBuilder.addClamp", minimum_program_generation: "v2", shape_rule: "shape_preserved" },

    MathOperationDescriptor { id: "tensor.transpose", family: "tensor_transform", arity: 1, direct_surface: "WasmTensorTransform.transpose", source_contract: "tensorTransformCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_TRANSPOSE)", minimum_program_generation: "v1", shape_rule: "swap_axes_2_3" },
    MathOperationDescriptor { id: "tensor.reshape", family: "tensor_transform", arity: 1, direct_surface: "WasmTensorTransform.reshape", source_contract: "tensorTransformCapabilities", program_builder: "MathProgramBuilder.addReshape", minimum_program_generation: "v3", shape_rule: "rank4_equal_element_count" },
    MathOperationDescriptor { id: "tensor.permute", family: "tensor_transform", arity: 1, direct_surface: "WasmTensorTransform.permute", source_contract: "tensorTransformCapabilities", program_builder: "MathProgramBuilder.addPermute", minimum_program_generation: "v3", shape_rule: "complete_unique_rank4_axes" },
    MathOperationDescriptor { id: "tensor.slice", family: "tensor_transform", arity: 1, direct_surface: "WasmTensorTransform.slice", source_contract: "tensorTransformCapabilities", program_builder: "MathProgramBuilder.addSlice", minimum_program_generation: "v3", shape_rule: "bounded_nonempty_rank4_ranges" },
    MathOperationDescriptor { id: "tensor.select_axis", family: "tensor_transform", arity: 1, direct_surface: "WasmTensorTransform.selectAxis", source_contract: "tensorTransformCapabilities", program_builder: "MathProgramV4Builder.addSelectAxis", minimum_program_generation: "v4", shape_rule: "validated_axis_and_indices" },

    MathOperationDescriptor { id: "linalg.matmul", family: "linear_algebra", arity: 2, direct_surface: "WasmLinearAlgebra.matmul", source_contract: "linearAlgebraCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_MATMUL)", minimum_program_generation: "v1", shape_rule: "[B,G,M,K]@[B,G,K,N]" },
    MathOperationDescriptor { id: "linalg.dot", family: "linear_algebra", arity: 2, direct_surface: "WasmLinearAlgebra.dot", source_contract: "linearAlgebraCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_DOT)", minimum_program_generation: "v1", shape_rule: "paired_[B,F,1,1]_exact_match" },
    MathOperationDescriptor { id: "linalg.l2_norm", family: "linear_algebra", arity: 1, direct_surface: "WasmLinearAlgebra.l2Norm", source_contract: "linearAlgebraCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_L2_NORM)", minimum_program_generation: "v1", shape_rule: "[B,F,1,1]_to_[B,1,1,1]" },
    MathOperationDescriptor { id: "linalg.cosine_similarity", family: "linear_algebra", arity: 2, direct_surface: "WasmLinearAlgebra.cosineSimilarity", source_contract: "linearAlgebraCapabilities", program_builder: "MathProgramBuilder.addCosineSimilarity", minimum_program_generation: "v2", shape_rule: "paired_[B,F,1,1]_exact_match" },
    MathOperationDescriptor { id: "linalg.l2_distance", family: "linear_algebra", arity: 2, direct_surface: "WasmLinearAlgebra.l2Distance", source_contract: "linearAlgebraCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_L2_DISTANCE)", minimum_program_generation: "v1", shape_rule: "paired_[B,F,1,1]_exact_match" },

    MathOperationDescriptor { id: "statistics.sum", family: "statistics", arity: 1, direct_surface: "WasmStatistics.sum", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_SUM)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },
    MathOperationDescriptor { id: "statistics.mean", family: "statistics", arity: 1, direct_surface: "WasmStatistics.mean", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_MEAN)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },
    MathOperationDescriptor { id: "statistics.variance_population", family: "statistics", arity: 1, direct_surface: "WasmStatistics.variancePopulation", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_VARIANCE_POPULATION)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },
    MathOperationDescriptor { id: "statistics.std_population", family: "statistics", arity: 1, direct_surface: "WasmStatistics.stdPopulation", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_STD_POPULATION)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },
    MathOperationDescriptor { id: "statistics.min", family: "statistics", arity: 1, direct_surface: "WasmStatistics.min", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_MIN)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },
    MathOperationDescriptor { id: "statistics.max", family: "statistics", arity: 1, direct_surface: "WasmStatistics.max", source_contract: "statisticsCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_MAX)", minimum_program_generation: "v1", shape_rule: "global_reduction_rank4_scalar_tensor" },

    MathOperationDescriptor { id: "probability.normalize", family: "probability", arity: 1, direct_surface: "WasmProbability.normalize", source_contract: "probabilityCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_NORMALIZE)", minimum_program_generation: "v1", shape_rule: "source_contract_defined" },
    MathOperationDescriptor { id: "probability.entropy", family: "probability", arity: 1, direct_surface: "WasmProbability.entropy", source_contract: "probabilityCapabilities", program_builder: "MathProgramBuilder.addUnary(OP_ENTROPY)", minimum_program_generation: "v1", shape_rule: "source_contract_defined" },
    MathOperationDescriptor { id: "probability.cross_entropy", family: "probability", arity: 2, direct_surface: "WasmProbability.crossEntropy", source_contract: "probabilityCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_CROSS_ENTROPY)", minimum_program_generation: "v1", shape_rule: "source_contract_defined" },
    MathOperationDescriptor { id: "probability.kl_divergence", family: "probability", arity: 2, direct_surface: "WasmProbability.klDivergence", source_contract: "probabilityCapabilities", program_builder: "MathProgramBuilder.addBinary(OP_KL_DIVERGENCE)", minimum_program_generation: "v1", shape_rule: "source_contract_defined" },

    MathOperationDescriptor { id: "reduction.sum_axis", family: "reduction", arity: 1, direct_surface: "WasmReduction.sumAxis", source_contract: "reductionCapabilities", program_builder: "MathProgramV8Builder.addSumAxis", minimum_program_generation: "v8", shape_rule: "axis_reduction_keep_rank4" },
    MathOperationDescriptor { id: "reduction.mean_axis", family: "reduction", arity: 1, direct_surface: "WasmReduction.meanAxis", source_contract: "reductionCapabilities", program_builder: "MathProgramV8Builder.addMeanAxis", minimum_program_generation: "v8", shape_rule: "axis_reduction_keep_rank4" },
    MathOperationDescriptor { id: "reduction.min_axis", family: "reduction", arity: 1, direct_surface: "WasmReduction.minAxis", source_contract: "reductionCapabilities", program_builder: "MathProgramV8Builder.addMinAxis", minimum_program_generation: "v8", shape_rule: "axis_reduction_keep_rank4" },
    MathOperationDescriptor { id: "reduction.max_axis", family: "reduction", arity: 1, direct_surface: "WasmReduction.maxAxis", source_contract: "reductionCapabilities", program_builder: "MathProgramV8Builder.addMaxAxis", minimum_program_generation: "v8", shape_rule: "axis_reduction_keep_rank4" },

    MathOperationDescriptor { id: "comparison.less_equal_01", family: "comparison", arity: 2, direct_surface: "WasmComparison.lessEqual01", source_contract: "comparisonCapabilities", program_builder: "MathProgramV9Builder.addLessEqual01", minimum_program_generation: "v9", shape_rule: "exact_shape_match_numeric_predicate_0_or_1" },
    MathOperationDescriptor { id: "index.indices_like", family: "index_source", arity: 1, direct_surface: "WasmIndexSource.indicesLike", source_contract: "indexSourceCapabilities", program_builder: "MathProgramV9Builder.addIndicesLike", minimum_program_generation: "v9", shape_rule: "reference_shape_preserved_axis_coordinates" },
];

/// Machine-readable authority and lifecycle description for the unified math vocabulary.
///
/// This layer is discovery/projection only. It owns no tensor, program, execution,
/// verification, or Resolution state.
#[wasm_bindgen(js_name = mathInteractionCapabilities)]
pub fn math_interaction_capabilities() -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"role\":\"unified_math_vocabulary\",",
            "\"state_ownership\":\"none\",",
            "\"execution\":\"none\",",
            "\"operation_selection\":\"agent_authority\",",
            "\"tensor_contract\":{{\"dtype\":\"f32\",\"rank\":4}},",
            "\"operation_count\":{},",
            "\"families\":[\"numeric\",\"tensor_transform\",\"linear_algebra\",\"statistics\",\"probability\",\"reduction\",\"comparison\",\"index_source\"],",
            "\"authorities\":{{",
                "\"operation_semantics\":\"existing per-family capability contracts\",",
                "\"program_structure\":\"MathProgram canonical plan and programIdentity\",",
                "\"numerics\":\"Burn-backed existing math surfaces\",",
                "\"verification\":\"existing verifier receipt surfaces\",",
                "\"resolution\":\"Resolution remains external; this contract grants no semantic authorization\"",
            "}},",
            "\"canonical_id_policy\":\"stable_across_direct_and_program_backends\",",
            "\"program_generation_policy\":\"minimum compatible generation is binding metadata, not operation identity\",",
            "\"discovery\":[\"mathInteractionCapabilities\",\"mathOperationCatalog\"],",
            "\"deferred\":[\"operand-aware preflight\",\"valid-operation projection\",\"execution facade\"],",
            "\"read_only_guarantee\":\"discovery calls allocate no persistent state and execute no tensor operations\"",
            "}}"
        ),
        MATH_INTERACTION_SCHEMA_ID,
        OPERATIONS.len()
    )
}

/// Return canonical math operation IDs and their bindings to existing direct/program surfaces.
///
/// The catalog intentionally delegates detailed numerical/domain constraints to the existing
/// per-family capability contracts instead of becoming a second semantic authority.
#[wasm_bindgen(js_name = mathOperationCatalog)]
pub fn math_operation_catalog() -> String {
    let operations = OPERATIONS
        .iter()
        .map(|op| {
            format!(
                concat!(
                    "{{",
                    "\"id\":\"{}\",",
                    "\"family\":\"{}\",",
                    "\"arity\":{},",
                    "\"direct_surface\":\"{}\",",
                    "\"source_contract\":\"{}\",",
                    "\"program\":{{",
                        "\"supported\":true,",
                        "\"minimum_generation\":\"{}\",",
                        "\"builder\":\"{}\"",
                    "}},",
                    "\"shape_rule\":\"{}\",",
                    "\"authority\":\"binding_projection_only\"",
                    "}}"
                ),
                op.id,
                op.family,
                op.arity,
                op.direct_surface,
                op.source_contract,
                op.minimum_program_generation,
                op.program_builder,
                op.shape_rule,
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"interaction_schema\":\"{}\",",
            "\"projection_only\":true,",
            "\"operation_count\":{},",
            "\"operations\":[{}]",
            "}}"
        ),
        MATH_OPERATION_CATALOG_SCHEMA_ID,
        MATH_INTERACTION_SCHEMA_ID,
        OPERATIONS.len(),
        operations
    )
}

#[cfg(test)]
mod tests {
    use super::{math_interaction_capabilities, math_operation_catalog, OPERATIONS};
    use std::collections::HashSet;

    #[test]
    fn capabilities_are_projection_only_and_keep_authorities_separate() {
        let caps: serde_json::Value =
            serde_json::from_str(&math_interaction_capabilities()).unwrap();
        assert_eq!(caps["schema_id"], "burn-research.math-interaction.v1");
        assert_eq!(caps["state_ownership"], "none");
        assert_eq!(caps["execution"], "none");
        assert_eq!(caps["operation_selection"], "agent_authority");
        assert_eq!(caps["operation_count"], 35);
        assert_eq!(
            caps["authorities"]["program_structure"],
            "MathProgram canonical plan and programIdentity"
        );
        assert_eq!(
            caps["authorities"]["numerics"],
            "Burn-backed existing math surfaces"
        );
    }

    #[test]
    fn operation_ids_are_unique_and_catalog_is_valid_json() {
        let catalog: serde_json::Value =
            serde_json::from_str(&math_operation_catalog()).unwrap();
        let operations = catalog["operations"].as_array().unwrap();
        assert_eq!(operations.len(), OPERATIONS.len());
        assert_eq!(catalog["operation_count"], 35);

        let mut ids = HashSet::new();
        for operation in operations {
            assert!(ids.insert(operation["id"].as_str().unwrap()));
            assert_eq!(operation["authority"], "binding_projection_only");
            assert_eq!(operation["program"]["supported"], true);
        }
    }

    #[test]
    fn versioned_program_bindings_do_not_replace_canonical_operation_ids() {
        let catalog = math_operation_catalog();
        assert!(catalog.contains("\"id\":\"numeric.clamp\""));
        assert!(catalog.contains("\"minimum_generation\":\"v2\""));
        assert!(catalog.contains("\"id\":\"tensor.select_axis\""));
        assert!(catalog.contains("\"minimum_generation\":\"v4\""));
        assert!(catalog.contains("\"id\":\"reduction.mean_axis\""));
        assert!(catalog.contains("\"minimum_generation\":\"v8\""));
        assert!(catalog.contains("\"id\":\"comparison.less_equal_01\""));
        assert!(catalog.contains("\"minimum_generation\":\"v9\""));
        assert!(catalog.contains("\"id\":\"index.indices_like\""));
    }

    #[test]
    fn discovery_is_deterministic_and_has_no_runtime_inputs() {
        assert_eq!(math_interaction_capabilities(), math_interaction_capabilities());
        assert_eq!(math_operation_catalog(), math_operation_catalog());
    }
}
