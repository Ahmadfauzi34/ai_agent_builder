pub use crate::facade::math::{
    math_check_operation, math_describe_operation, math_interaction_capabilities,
    math_operation_catalog, math_plan_binding, math_valid_operations,
};
use wasm_bindgen::prelude::*;

use crate::math::index_source::MAX_INDICES_LIKE_AXIS_LENGTH;

pub(crate) const MATH_INTERACTION_SCHEMA_ID: &str = "burn-research.math-interaction.v1";
pub(crate) const MATH_OPERATION_CATALOG_SCHEMA_ID: &str = "burn-research.math-operation-catalog.v1";

#[derive(Clone, Copy)]
pub(crate) struct MathOperationDescriptor {
    pub(crate) id: &'static str,
    pub(crate) family: &'static str,
    pub(crate) arity: u8,
    pub(crate) direct_surface: &'static str,
    pub(crate) source_contract: &'static str,
    pub(crate) program_builder: &'static str,
    pub(crate) minimum_program_generation: &'static str,
    pub(crate) shape_rule: &'static str,
}

pub(crate) const OPERATIONS: &[MathOperationDescriptor] = &[
    MathOperationDescriptor {
        id: "numeric.add",
        family: "numeric",
        arity: 2,
        direct_surface: "WasmNumericKernel.add",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_ADD)",
        minimum_program_generation: "v1",
        shape_rule: "exact_shape_match_no_broadcast",
    },
    MathOperationDescriptor {
        id: "numeric.sub",
        family: "numeric",
        arity: 2,
        direct_surface: "WasmNumericKernel.sub",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_SUB)",
        minimum_program_generation: "v1",
        shape_rule: "exact_shape_match_no_broadcast",
    },
    MathOperationDescriptor {
        id: "numeric.mul",
        family: "numeric",
        arity: 2,
        direct_surface: "WasmNumericKernel.mul",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_MUL)",
        minimum_program_generation: "v1",
        shape_rule: "exact_shape_match_no_broadcast",
    },
    MathOperationDescriptor {
        id: "numeric.div",
        family: "numeric",
        arity: 2,
        direct_surface: "WasmNumericKernel.div",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_DIV)",
        minimum_program_generation: "v1",
        shape_rule: "exact_shape_match_no_broadcast",
    },
    MathOperationDescriptor {
        id: "numeric.abs",
        family: "numeric",
        arity: 1,
        direct_surface: "WasmNumericKernel.abs",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_ABS)",
        minimum_program_generation: "v1",
        shape_rule: "shape_preserved",
    },
    MathOperationDescriptor {
        id: "numeric.sqrt",
        family: "numeric",
        arity: 1,
        direct_surface: "WasmNumericKernel.sqrt",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_SQRT)",
        minimum_program_generation: "v1",
        shape_rule: "shape_preserved",
    },
    MathOperationDescriptor {
        id: "numeric.exp",
        family: "numeric",
        arity: 1,
        direct_surface: "WasmNumericKernel.exp",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_EXP)",
        minimum_program_generation: "v1",
        shape_rule: "shape_preserved",
    },
    MathOperationDescriptor {
        id: "numeric.log",
        family: "numeric",
        arity: 1,
        direct_surface: "WasmNumericKernel.log",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_LOG)",
        minimum_program_generation: "v1",
        shape_rule: "shape_preserved",
    },
    MathOperationDescriptor {
        id: "numeric.clamp",
        family: "numeric",
        arity: 1,
        direct_surface: "WasmNumericKernel.clamp",
        source_contract: "numericKernelCapabilities",
        program_builder: "MathProgramBuilder.addClamp",
        minimum_program_generation: "v2",
        shape_rule: "shape_preserved",
    },
    MathOperationDescriptor {
        id: "tensor.transpose",
        family: "tensor_transform",
        arity: 1,
        direct_surface: "WasmTensorTransform.transpose",
        source_contract: "tensorTransformCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_TRANSPOSE)",
        minimum_program_generation: "v1",
        shape_rule: "swap_axes_2_3",
    },
    MathOperationDescriptor {
        id: "tensor.reshape",
        family: "tensor_transform",
        arity: 1,
        direct_surface: "WasmTensorTransform.reshape",
        source_contract: "tensorTransformCapabilities",
        program_builder: "MathProgramBuilder.addReshape",
        minimum_program_generation: "v3",
        shape_rule: "rank4_equal_element_count",
    },
    MathOperationDescriptor {
        id: "tensor.permute",
        family: "tensor_transform",
        arity: 1,
        direct_surface: "WasmTensorTransform.permute",
        source_contract: "tensorTransformCapabilities",
        program_builder: "MathProgramBuilder.addPermute",
        minimum_program_generation: "v3",
        shape_rule: "complete_unique_rank4_axes",
    },
    MathOperationDescriptor {
        id: "tensor.slice",
        family: "tensor_transform",
        arity: 1,
        direct_surface: "WasmTensorTransform.slice",
        source_contract: "tensorTransformCapabilities",
        program_builder: "MathProgramBuilder.addSlice",
        minimum_program_generation: "v3",
        shape_rule: "bounded_nonempty_rank4_ranges",
    },
    MathOperationDescriptor {
        id: "tensor.select_axis",
        family: "tensor_transform",
        arity: 1,
        direct_surface: "WasmTensorTransform.selectAxis",
        source_contract: "tensorTransformCapabilities",
        program_builder: "MathProgramV4Builder.addSelectAxis",
        minimum_program_generation: "v4",
        shape_rule: "validated_axis_and_indices",
    },
    MathOperationDescriptor {
        id: "linalg.matmul",
        family: "linear_algebra",
        arity: 2,
        direct_surface: "WasmLinearAlgebra.matmul",
        source_contract: "linearAlgebraCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_MATMUL)",
        minimum_program_generation: "v1",
        shape_rule: "[B,G,M,K]@[B,G,K,N]",
    },
    MathOperationDescriptor {
        id: "linalg.dot",
        family: "linear_algebra",
        arity: 2,
        direct_surface: "WasmLinearAlgebra.dot",
        source_contract: "linearAlgebraCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_DOT)",
        minimum_program_generation: "v1",
        shape_rule: "paired_[B,F,1,1]_exact_match",
    },
    MathOperationDescriptor {
        id: "linalg.l2_norm",
        family: "linear_algebra",
        arity: 1,
        direct_surface: "WasmLinearAlgebra.l2Norm",
        source_contract: "linearAlgebraCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_L2_NORM)",
        minimum_program_generation: "v1",
        shape_rule: "[B,F,1,1]_to_[B,1,1,1]",
    },
    MathOperationDescriptor {
        id: "linalg.cosine_similarity",
        family: "linear_algebra",
        arity: 2,
        direct_surface: "WasmLinearAlgebra.cosineSimilarity",
        source_contract: "linearAlgebraCapabilities",
        program_builder: "MathProgramBuilder.addCosineSimilarity",
        minimum_program_generation: "v2",
        shape_rule: "paired_[B,F,1,1]_exact_match",
    },
    MathOperationDescriptor {
        id: "linalg.l2_distance",
        family: "linear_algebra",
        arity: 2,
        direct_surface: "WasmLinearAlgebra.l2Distance",
        source_contract: "linearAlgebraCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_L2_DISTANCE)",
        minimum_program_generation: "v1",
        shape_rule: "paired_[B,F,1,1]_exact_match",
    },
    MathOperationDescriptor {
        id: "statistics.sum",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.sum",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_SUM)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "statistics.mean",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.mean",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_MEAN)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "statistics.variance_population",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.variancePopulation",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_VARIANCE_POPULATION)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "statistics.std_population",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.stdPopulation",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_STD_POPULATION)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "statistics.min",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.min",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_MIN)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "statistics.max",
        family: "statistics",
        arity: 1,
        direct_surface: "WasmStatistics.max",
        source_contract: "statisticsCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_MAX)",
        minimum_program_generation: "v1",
        shape_rule: "global_reduction_rank4_scalar_tensor",
    },
    MathOperationDescriptor {
        id: "probability.normalize",
        family: "probability",
        arity: 1,
        direct_surface: "WasmProbability.normalize",
        source_contract: "probabilityCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_NORMALIZE)",
        minimum_program_generation: "v1",
        shape_rule: "source_contract_defined",
    },
    MathOperationDescriptor {
        id: "probability.entropy",
        family: "probability",
        arity: 1,
        direct_surface: "WasmProbability.entropy",
        source_contract: "probabilityCapabilities",
        program_builder: "MathProgramBuilder.addUnary(OP_ENTROPY)",
        minimum_program_generation: "v1",
        shape_rule: "source_contract_defined",
    },
    MathOperationDescriptor {
        id: "probability.cross_entropy",
        family: "probability",
        arity: 2,
        direct_surface: "WasmProbability.crossEntropy",
        source_contract: "probabilityCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_CROSS_ENTROPY)",
        minimum_program_generation: "v1",
        shape_rule: "source_contract_defined",
    },
    MathOperationDescriptor {
        id: "probability.kl_divergence",
        family: "probability",
        arity: 2,
        direct_surface: "WasmProbability.klDivergence",
        source_contract: "probabilityCapabilities",
        program_builder: "MathProgramBuilder.addBinary(OP_KL_DIVERGENCE)",
        minimum_program_generation: "v1",
        shape_rule: "source_contract_defined",
    },
    MathOperationDescriptor {
        id: "reduction.sum_axis",
        family: "reduction",
        arity: 1,
        direct_surface: "WasmReduction.sumAxis",
        source_contract: "reductionCapabilities",
        program_builder: "MathProgramV8Builder.addSumAxis",
        minimum_program_generation: "v8",
        shape_rule: "axis_reduction_keep_rank4",
    },
    MathOperationDescriptor {
        id: "reduction.mean_axis",
        family: "reduction",
        arity: 1,
        direct_surface: "WasmReduction.meanAxis",
        source_contract: "reductionCapabilities",
        program_builder: "MathProgramV8Builder.addMeanAxis",
        minimum_program_generation: "v8",
        shape_rule: "axis_reduction_keep_rank4",
    },
    MathOperationDescriptor {
        id: "reduction.min_axis",
        family: "reduction",
        arity: 1,
        direct_surface: "WasmReduction.minAxis",
        source_contract: "reductionCapabilities",
        program_builder: "MathProgramV8Builder.addMinAxis",
        minimum_program_generation: "v8",
        shape_rule: "axis_reduction_keep_rank4",
    },
    MathOperationDescriptor {
        id: "reduction.max_axis",
        family: "reduction",
        arity: 1,
        direct_surface: "WasmReduction.maxAxis",
        source_contract: "reductionCapabilities",
        program_builder: "MathProgramV8Builder.addMaxAxis",
        minimum_program_generation: "v8",
        shape_rule: "axis_reduction_keep_rank4",
    },
    MathOperationDescriptor {
        id: "comparison.less_equal_01",
        family: "comparison",
        arity: 2,
        direct_surface: "WasmComparison.lessEqual01",
        source_contract: "comparisonCapabilities",
        program_builder: "MathProgramV9Builder.addLessEqual01",
        minimum_program_generation: "v9",
        shape_rule: "exact_shape_match_numeric_predicate_0_or_1",
    },
    MathOperationDescriptor {
        id: "index.indices_like",
        family: "index_source",
        arity: 1,
        direct_surface: "WasmIndexSource.indicesLike",
        source_contract: "indexSourceCapabilities",
        program_builder: "MathProgramV9Builder.addIndicesLike",
        minimum_program_generation: "v9",
        shape_rule: "reference_shape_preserved_axis_coordinates",
    },
];

pub(crate) const MATH_OPERATION_DESCRIPTION_SCHEMA_ID: &str =
    "burn-research.math-operation-description.v1";
const MATH_PREFLIGHT_SCHEMA_ID: &str = "burn-research.math-preflight.v1";

pub(crate) fn operation_descriptor(operation_id: &str) -> Option<&'static MathOperationDescriptor> {
    OPERATIONS
        .iter()
        .find(|operation| operation.id == operation_id)
}

pub(crate) fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

pub(crate) fn shape_json(shape: [u32; 4]) -> String {
    format!("[{},{},{},{}]", shape[0], shape[1], shape[2], shape[3])
}

pub(crate) fn parse_shape4(shape: &[u32], role: &str) -> Result<[u32; 4], String> {
    if shape.len() != 4 {
        return Err(format!(
            "{role}.rank4: expected exactly 4 dimensions, got {}",
            shape.len()
        ));
    }
    Ok([shape[0], shape[1], shape[2], shape[3]])
}

pub(crate) fn checked_nonzero_product(shape: [u32; 4], role: &str) -> Result<u64, String> {
    shape.into_iter().try_fold(1u64, |count, dim| {
        if dim == 0 {
            return Err(format!(
                "{role}.nonzero_dimensions: zero-sized dimensions are not allowed"
            ));
        }
        count
            .checked_mul(u64::from(dim))
            .ok_or_else(|| format!("{role}.element_count: overflow"))
    })
}

pub(crate) fn feature_shape(shape: [u32; 4], role: &str) -> Result<(), String> {
    if shape[0] == 0 {
        return Err(format!("{role}.batch_nonempty: batch axis must be > 0"));
    }
    if shape[1] == 0 {
        return Err(format!("{role}.feature_nonempty: feature axis must be > 0"));
    }
    if shape[2] != 1 || shape[3] != 1 {
        return Err(format!(
            "{role}.feature_layout: expected [B,F,1,1], got {}",
            shape_json(shape)
        ));
    }
    Ok(())
}

pub(crate) fn first_error_predicate(error: &str) -> &str {
    error.split(':').next().unwrap_or("metadata.valid")
}

pub(crate) fn rejected(
    operation_id: &str,
    predicate: &str,
    expected: &str,
    actual: &str,
) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"operation_id\":\"{}\",",
            "\"status\":\"rejected\",",
            "\"metadata_admissible\":false,",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"recoverable\":true,",
            "\"failure\":{{",
            "\"predicate\":\"{}\",",
            "\"expected\":\"{}\",",
            "\"actual\":\"{}\"",
            "}},",
            "\"output_shape\":null,",
            "\"deferred_value_checks\":[]",
            "}}"
        ),
        MATH_PREFLIGHT_SCHEMA_ID,
        json_escape(operation_id),
        json_escape(predicate),
        json_escape(expected),
        json_escape(actual),
    )
}

fn deferred_checks_json(checks: &[&str]) -> String {
    checks
        .iter()
        .map(|check| format!("\"{}\"", json_escape(check)))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn admissible(
    operation_id: &str,
    output_shape: [u32; 4],
    deferred_value_checks: &[&str],
    program_binding_deferred: bool,
) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"operation_id\":\"{}\",",
            "\"status\":\"admissible\",",
            "\"metadata_admissible\":true,",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"recoverable\":true,",
            "\"output_shape\":{},",
            "\"deferred_value_checks\":[{}],",
            "\"program_binding_deferred\":{}",
            "}}"
        ),
        MATH_PREFLIGHT_SCHEMA_ID,
        json_escape(operation_id),
        shape_json(output_shape),
        deferred_checks_json(deferred_value_checks),
        if program_binding_deferred {
            "true"
        } else {
            "false"
        },
    )
}

pub(crate) fn reject_error(operation_id: &str, error: String) -> String {
    let predicate = first_error_predicate(&error).to_string();
    rejected(
        operation_id,
        &predicate,
        "source_contract_predicate_satisfied",
        &error,
    )
}

pub(crate) fn ensure_no_params(
    operation_id: &str,
    u32_params: &[u32],
    f32_params: &[f32],
) -> Option<String> {
    if !u32_params.is_empty() || !f32_params.is_empty() {
        return Some(rejected(
            operation_id,
            "parameters.empty",
            "u32_params=[] and f32_params=[]",
            &format!(
                "u32_params_len={},f32_params_len={}",
                u32_params.len(),
                f32_params.len()
            ),
        ));
    }
    None
}

pub(crate) fn ensure_arity_shapes(
    operation: &MathOperationDescriptor,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
) -> Result<([u32; 4], Option<[u32; 4]>), String> {
    let lhs = parse_shape4(lhs_shape, "lhs")?;
    match operation.arity {
        1 => {
            if !rhs_shape.is_empty() {
                return Err(format!(
                    "arity.unary: rhs_shape must be empty, got {} dimensions",
                    rhs_shape.len()
                ));
            }
            Ok((lhs, None))
        }
        2 => Ok((lhs, Some(parse_shape4(rhs_shape, "rhs")?))),
        other => Err(format!("arity.supported: unsupported arity {other}")),
    }
}

pub(crate) fn parameter_contract(operation_id: &str) -> (&'static str, &'static str) {
    match operation_id {
        "numeric.clamp" => ("none", "[min,max] finite with min<=max"),
        "linalg.cosine_similarity" => (
            "none",
            "[] uses direct-surface default epsilon but leaves MathProgram binding deferred; [epsilon] requires finite epsilon>0",
        ),
        "tensor.reshape" => ("[d0,d1,d2,d3]", "none"),
        "tensor.permute" => ("[axis0,axis1,axis2,axis3]", "none"),
        "tensor.slice" => ("[start0,start1,start2,start3,end0,end1,end2,end3]", "none"),
        "tensor.select_axis" => ("[axis,index0,...] with at least one index", "none"),
        "reduction.sum_axis"
        | "reduction.mean_axis"
        | "reduction.min_axis"
        | "reduction.max_axis"
        | "index.indices_like" => ("[axis]", "none"),
        _ => ("none", "none"),
    }
}

pub(crate) const MATH_VALID_OPERATIONS_SCHEMA_ID: &str = "burn-research.math-valid-operations.v1";

pub(crate) fn operation_requires_explicit_parameters(operation_id: &str) -> bool {
    matches!(
        operation_id,
        "numeric.clamp"
            | "tensor.reshape"
            | "tensor.permute"
            | "tensor.slice"
            | "tensor.select_axis"
            | "reduction.sum_axis"
            | "reduction.mean_axis"
            | "reduction.min_axis"
            | "reduction.max_axis"
            | "index.indices_like"
    )
}

pub(crate) fn parameterized_operation_globally_possible(
    operation_id: &str,
    lhs: [u32; 4],
) -> Result<(), String> {
    match operation_id {
        "tensor.reshape" => {
            checked_nonzero_product(lhs, "reshape.source")?;
            Ok(())
        }
        "tensor.slice"
        | "reduction.sum_axis"
        | "reduction.mean_axis"
        | "reduction.min_axis"
        | "reduction.max_axis"
        | "index.indices_like" => {
            if lhs.iter().any(|dim| *dim == 0) {
                Err(format!(
                    "{operation_id}.nonzero_dimensions: all input dimensions must be > 0"
                ))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

pub(crate) fn candidate_rejected_json(
    operation: &MathOperationDescriptor,
    predicate: &str,
    expected: &str,
    actual: &str,
) -> String {
    format!(
        concat!(
            "{{",
            "\"id\":\"{}\",",
            "\"family\":\"{}\",",
            "\"arity\":{},",
            "\"status\":\"metadata_rejected\",",
            "\"candidate\":false,",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"reason\":{{",
            "\"predicate\":\"{}\",",
            "\"expected\":\"{}\",",
            "\"actual\":\"{}\"",
            "}}",
            "}}"
        ),
        operation.id,
        operation.family,
        operation.arity,
        json_escape(predicate),
        json_escape(expected),
        json_escape(actual),
    )
}

pub(crate) fn candidate_requires_parameters_json(operation: &MathOperationDescriptor) -> String {
    let (u32_params, f32_params) = parameter_contract(operation.id);
    format!(
        concat!(
            "{{",
            "\"id\":\"{}\",",
            "\"family\":\"{}\",",
            "\"arity\":{},",
            "\"status\":\"requires_parameters\",",
            "\"candidate\":true,",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"parameter_contract\":{{",
            "\"u32_params\":\"{}\",",
            "\"f32_params\":\"{}\"",
            "}},",
            "\"next_validation_surface\":\"mathCheckOperation\"",
            "}}"
        ),
        operation.id,
        operation.family,
        operation.arity,
        json_escape(u32_params),
        json_escape(f32_params),
    )
}

pub(crate) fn candidate_from_preflight_json(
    operation: &MathOperationDescriptor,
    preflight: String,
) -> String {
    let status = if preflight.contains("\"status\":\"admissible\"") {
        "metadata_admissible"
    } else {
        "metadata_rejected"
    };
    let candidate = status == "metadata_admissible";
    format!(
        concat!(
            "{{",
            "\"id\":\"{}\",",
            "\"family\":\"{}\",",
            "\"arity\":{},",
            "\"status\":\"{}\",",
            "\"candidate\":{},",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"preflight\":{}",
            "}}"
        ),
        operation.id,
        operation.family,
        operation.arity,
        status,
        if candidate { "true" } else { "false" },
        preflight,
    )
}

pub(crate) const MATH_BINDING_PLAN_SCHEMA_ID: &str = "burn-research.math-binding-plan.v1";

pub(crate) fn program_binding_method(
    operation_id: &str,
) -> (&'static str, Option<&'static str>, &'static str) {
    match operation_id {
        "numeric.abs" => ("addUnary", Some("OP_ABS"), "opcode,input_slot,output_slot"),
        "numeric.sqrt" => ("addUnary", Some("OP_SQRT"), "opcode,input_slot,output_slot"),
        "numeric.exp" => ("addUnary", Some("OP_EXP"), "opcode,input_slot,output_slot"),
        "numeric.log" => ("addUnary", Some("OP_LOG"), "opcode,input_slot,output_slot"),
        "numeric.add" => (
            "addBinary",
            Some("OP_ADD"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "numeric.sub" => (
            "addBinary",
            Some("OP_SUB"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "numeric.mul" => (
            "addBinary",
            Some("OP_MUL"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "numeric.div" => (
            "addBinary",
            Some("OP_DIV"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "numeric.clamp" => ("addClamp", None, "input_slot,output_slot,min,max"),
        "tensor.transpose" => (
            "addUnary",
            Some("OP_TRANSPOSE"),
            "opcode,input_slot,output_slot",
        ),
        "tensor.reshape" => ("addReshape", None, "input_slot,output_slot,shape"),
        "tensor.permute" => ("addPermute", None, "input_slot,output_slot,axes"),
        "tensor.slice" => ("addSlice", None, "input_slot,output_slot,starts,ends"),
        "tensor.select_axis" => ("addSelectAxis", None, "input_slot,output_slot,axis,indices"),
        "linalg.matmul" => (
            "addBinary",
            Some("OP_MATMUL"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "linalg.dot" => (
            "addBinary",
            Some("OP_DOT"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "linalg.l2_norm" => (
            "addUnary",
            Some("OP_L2_NORM"),
            "opcode,input_slot,output_slot",
        ),
        "linalg.cosine_similarity" => (
            "addCosineSimilarity",
            None,
            "lhs_slot,rhs_slot,output_slot,epsilon",
        ),
        "linalg.l2_distance" => (
            "addBinary",
            Some("OP_L2_DISTANCE"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "statistics.sum" => ("addUnary", Some("OP_SUM"), "opcode,input_slot,output_slot"),
        "statistics.mean" => ("addUnary", Some("OP_MEAN"), "opcode,input_slot,output_slot"),
        "statistics.variance_population" => (
            "addUnary",
            Some("OP_VARIANCE_POPULATION"),
            "opcode,input_slot,output_slot",
        ),
        "statistics.std_population" => (
            "addUnary",
            Some("OP_STD_POPULATION"),
            "opcode,input_slot,output_slot",
        ),
        "statistics.min" => ("addUnary", Some("OP_MIN"), "opcode,input_slot,output_slot"),
        "statistics.max" => ("addUnary", Some("OP_MAX"), "opcode,input_slot,output_slot"),
        "probability.normalize" => (
            "addUnary",
            Some("OP_NORMALIZE"),
            "opcode,input_slot,output_slot",
        ),
        "probability.entropy" => (
            "addUnary",
            Some("OP_ENTROPY"),
            "opcode,input_slot,output_slot",
        ),
        "probability.cross_entropy" => (
            "addBinary",
            Some("OP_CROSS_ENTROPY"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "probability.kl_divergence" => (
            "addBinary",
            Some("OP_KL_DIVERGENCE"),
            "opcode,lhs_slot,rhs_slot,output_slot",
        ),
        "reduction.sum_axis" => ("addSumAxis", None, "input_slot,output_slot,axis"),
        "reduction.mean_axis" => ("addMeanAxis", None, "input_slot,output_slot,axis"),
        "reduction.min_axis" => ("addMinAxis", None, "input_slot,output_slot,axis"),
        "reduction.max_axis" => ("addMaxAxis", None, "input_slot,output_slot,axis"),
        "comparison.less_equal_01" => ("addLessEqual01", None, "lhs_slot,rhs_slot,output_slot"),
        "index.indices_like" => ("addIndicesLike", None, "reference_slot,output_slot,axis"),
        _ => ("unsupported", None, "none"),
    }
}

pub(crate) fn program_minimum_builder_class(minimum_generation: &str) -> &'static str {
    match minimum_generation {
        "v4" => "WasmMathProgramV4Builder",
        "v8" => "WasmMathProgramV8Builder",
        "v9" => "WasmMathProgramV9Builder",
        _ => "WasmMathProgramBuilder",
    }
}

pub(crate) fn parameter_layout(operation_id: &str) -> &'static str {
    match operation_id {
        "numeric.clamp" => "clamp_min_max",
        "linalg.cosine_similarity" => "epsilon",
        "tensor.reshape" => "shape4",
        "tensor.permute" => "axes4",
        "tensor.slice" => "slice_start4_end4",
        "tensor.select_axis" => "select_axis_indices",
        "reduction.sum_axis"
        | "reduction.mean_axis"
        | "reduction.min_axis"
        | "reduction.max_axis"
        | "index.indices_like" => "axis",
        _ => "none",
    }
}

pub(crate) fn u32_array_json(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn f32_array_json(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn binding_rejected(operation_id: &str, target: &str, preflight: String) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"operation_id\":\"{}\",",
            "\"target\":\"{}\",",
            "\"target_selected_by\":\"agent\",",
            "\"status\":\"rejected\",",
            "\"execution\":\"none\",",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"binding_authority\":\"projection_only\",",
            "\"binding\":null,",
            "\"preflight\":{}",
            "}}"
        ),
        MATH_BINDING_PLAN_SCHEMA_ID,
        json_escape(operation_id),
        json_escape(target),
        preflight,
    )
}

pub(crate) fn binding_requires_parameters(
    operation: &MathOperationDescriptor,
    target: &str,
    preflight: String,
) -> String {
    let (u32_params, f32_params) = parameter_contract(operation.id);
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"operation_id\":\"{}\",",
            "\"target\":\"{}\",",
            "\"target_selected_by\":\"agent\",",
            "\"status\":\"requires_parameters\",",
            "\"execution\":\"none\",",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"binding_authority\":\"projection_only\",",
            "\"required_parameters\":{{",
            "\"u32_params\":\"{}\",",
            "\"f32_params\":\"{}\"",
            "}},",
            "\"binding\":null,",
            "\"preflight\":{}",
            "}}"
        ),
        MATH_BINDING_PLAN_SCHEMA_ID,
        operation.id,
        target,
        json_escape(u32_params),
        json_escape(f32_params),
        preflight,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        math_check_operation, math_describe_operation, math_interaction_capabilities,
        math_operation_catalog, math_plan_binding, math_valid_operations, OPERATIONS,
    };
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
        assert_eq!(
            caps["direct_verifier"]["verifier"],
            "DirectMath.verifyAgainstMathProgramV9"
        );
        assert_eq!(
            caps["direct_verifier"]["candidate_authority"],
            "burn_direct_math"
        );
        assert_eq!(
            caps["direct_verifier"]["reference_authority"],
            "burn_math_program"
        );
        assert!(caps["deferred"]
            .as_array()
            .is_some_and(|items| items.is_empty()));
    }

    #[test]
    fn operation_ids_are_unique_and_catalog_is_valid_json() {
        let catalog: serde_json::Value = serde_json::from_str(&math_operation_catalog()).unwrap();
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
        assert_eq!(
            math_interaction_capabilities(),
            math_interaction_capabilities()
        );
        assert_eq!(math_operation_catalog(), math_operation_catalog());
    }

    #[test]
    fn every_catalog_operation_is_describable_without_execution() {
        for operation in OPERATIONS {
            let description: serde_json::Value =
                serde_json::from_str(&math_describe_operation(operation.id.to_string()).unwrap())
                    .unwrap();
            assert_eq!(description["operation"]["id"], operation.id);
            assert_eq!(description["authority"], "introspection_projection_only");
            assert_eq!(description["execution"], "none");
        }
    }

    #[test]
    fn matmul_preflight_infers_output_and_rejects_incompatible_inner_dimension() {
        let valid: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "linalg.matmul".into(),
                &[1, 2, 3, 4],
                &[1, 2, 4, 5],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(valid["status"], "admissible");
        assert_eq!(valid["output_shape"], serde_json::json!([1, 2, 3, 5]));
        assert_eq!(valid["execution_authorized"], false);

        let invalid: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "linalg.matmul".into(),
                &[1, 2, 3, 4],
                &[1, 2, 3, 5],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(invalid["status"], "rejected");
        assert_eq!(invalid["failure"]["predicate"], "matmul.compatible_shapes");
        assert_eq!(invalid["mutation"], "none");
    }

    #[test]
    fn reshape_preflight_enforces_exact_element_count() {
        let valid: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "tensor.reshape".into(),
                &[1, 2, 1, 3],
                &[],
                &[1, 1, 3, 2],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(valid["status"], "admissible");
        assert_eq!(valid["output_shape"], serde_json::json!([1, 1, 3, 2]));

        let invalid: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "tensor.reshape".into(),
                &[1, 2, 1, 3],
                &[],
                &[1, 1, 2, 2],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(invalid["status"], "rejected");
        assert_eq!(invalid["failure"]["predicate"], "reshape.element_count");
    }

    #[test]
    fn value_domain_checks_are_deferred_instead_of_guessed() {
        let div: serde_json::Value = serde_json::from_str(
            &math_check_operation("numeric.div".into(), &[1, 2, 1, 1], &[1, 2, 1, 1], &[], &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(div["status"], "admissible");
        assert!(div["deferred_value_checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "rhs_values_nonzero"));

        let probability: serde_json::Value = serde_json::from_str(
            &math_check_operation("probability.entropy".into(), &[2, 4, 1, 1], &[], &[], &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(probability["status"], "admissible");
        assert!(probability["deferred_value_checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "input_distribution_normalized_and_nonnegative"));
    }

    #[test]
    fn cosine_default_is_directly_admissible_but_program_binding_remains_deferred() {
        let default_epsilon: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "linalg.cosine_similarity".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(default_epsilon["status"], "admissible");
        assert_eq!(default_epsilon["program_binding_deferred"], true);

        let explicit_epsilon: serde_json::Value = serde_json::from_str(
            &math_check_operation(
                "linalg.cosine_similarity".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[1e-6],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(explicit_epsilon["status"], "admissible");
        assert_eq!(explicit_epsilon["program_binding_deferred"], false);
    }

    #[test]
    fn malformed_metadata_fails_closed_without_execution_authority() {
        let bad_axis: serde_json::Value = serde_json::from_str(
            &math_check_operation("reduction.mean_axis".into(), &[1, 2, 3, 1], &[], &[4], &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(bad_axis["status"], "rejected");
        assert_eq!(bad_axis["failure"]["predicate"], "reduction.axis");
        assert_eq!(bad_axis["execution_authorized"], false);

        assert!(
            math_check_operation("unknown.operation".into(), &[1, 1, 1, 1], &[], &[], &[],)
                .is_err()
        );
    }

    #[test]
    fn valid_operation_projection_never_selects_or_ranks_for_the_agent() {
        let projection: serde_json::Value =
            serde_json::from_str(&math_valid_operations(&[1, 3, 1, 1], &[1, 3, 1, 1]).unwrap())
                .unwrap();

        assert_eq!(
            projection["schema_id"],
            "burn-research.math-valid-operations.v1"
        );
        assert_eq!(projection["operation_count"], 35);
        assert_eq!(projection["selection_authority"], "agent");
        assert!(projection["selected_operation"].is_null());
        assert!(projection["ranking"].is_null());
        assert!(projection["recommendation"].is_null());
        assert_eq!(projection["execution_authorized"], false);
        assert_eq!(projection["mutation"], "none");

        let operations = projection["operations"].as_array().unwrap();
        assert_eq!(operations.len(), 35);
        assert!(operations
            .iter()
            .all(|operation| operation["execution_authorized"] == false));
        assert!(operations
            .iter()
            .all(|operation| operation["mutation"] == "none"));
    }

    #[test]
    fn valid_operation_projection_preserves_parameterized_candidates() {
        let projection: serde_json::Value =
            serde_json::from_str(&math_valid_operations(&[1, 3, 1, 1], &[1, 3, 1, 1]).unwrap())
                .unwrap();

        let operations = projection["operations"].as_array().unwrap();
        let find = |id: &str| {
            operations
                .iter()
                .find(|operation| operation["id"] == id)
                .unwrap()
        };

        assert_eq!(find("numeric.add")["status"], "metadata_admissible");
        assert_eq!(find("linalg.dot")["status"], "metadata_admissible");
        assert_eq!(
            find("linalg.cosine_similarity")["status"],
            "metadata_admissible"
        );
        assert_eq!(
            find("linalg.cosine_similarity")["preflight"]["program_binding_deferred"],
            true
        );

        assert_eq!(find("numeric.clamp")["status"], "requires_parameters");
        assert_eq!(find("tensor.reshape")["status"], "requires_parameters");
        assert_eq!(find("reduction.mean_axis")["status"], "requires_parameters");
        assert_eq!(find("index.indices_like")["status"], "requires_parameters");

        let candidate_ids = projection["candidate_operation_ids"].as_array().unwrap();
        assert!(candidate_ids.iter().any(|value| value == "numeric.clamp"));
        assert!(candidate_ids
            .iter()
            .any(|value| value == "reduction.mean_axis"));
    }

    #[test]
    fn unary_candidates_do_not_disappear_when_rhs_is_available() {
        let projection: serde_json::Value =
            serde_json::from_str(&math_valid_operations(&[1, 3, 1, 1], &[1, 1, 3, 1]).unwrap())
                .unwrap();

        let operations = projection["operations"].as_array().unwrap();
        let abs = operations
            .iter()
            .find(|operation| operation["id"] == "numeric.abs")
            .unwrap();
        let add = operations
            .iter()
            .find(|operation| operation["id"] == "numeric.add")
            .unwrap();

        assert_eq!(abs["status"], "metadata_admissible");
        assert_eq!(abs["candidate"], true);
        assert_eq!(add["status"], "metadata_rejected");
        assert_eq!(add["candidate"], false);
    }

    #[test]
    fn binary_candidates_fail_closed_when_rhs_is_absent() {
        let projection: serde_json::Value =
            serde_json::from_str(&math_valid_operations(&[1, 3, 1, 1], &[]).unwrap()).unwrap();

        let operations = projection["operations"].as_array().unwrap();
        let dot = operations
            .iter()
            .find(|operation| operation["id"] == "linalg.dot")
            .unwrap();
        let abs = operations
            .iter()
            .find(|operation| operation["id"] == "numeric.abs")
            .unwrap();

        assert_eq!(dot["status"], "metadata_rejected");
        assert_eq!(dot["reason"]["predicate"], "operand.rhs_available");
        assert_eq!(abs["status"], "metadata_admissible");
        assert_eq!(abs["candidate"], true);
    }

    #[test]
    fn globally_impossible_parameterized_shapes_are_not_candidates() {
        let projection: serde_json::Value =
            serde_json::from_str(&math_valid_operations(&[1, 0, 1, 1], &[]).unwrap()).unwrap();

        let operations = projection["operations"].as_array().unwrap();
        let reduction = operations
            .iter()
            .find(|operation| operation["id"] == "reduction.mean_axis")
            .unwrap();
        let reshape = operations
            .iter()
            .find(|operation| operation["id"] == "tensor.reshape")
            .unwrap();

        assert_eq!(reduction["status"], "metadata_rejected");
        assert_eq!(reduction["candidate"], false);
        assert_eq!(reshape["status"], "metadata_rejected");
        assert_eq!(reshape["candidate"], false);
    }

    #[test]
    fn valid_operation_projection_is_deterministic() {
        assert_eq!(
            math_valid_operations(&[1, 3, 1, 1], &[1, 3, 1, 1]).unwrap(),
            math_valid_operations(&[1, 3, 1, 1], &[1, 3, 1, 1]).unwrap()
        );
    }

    #[test]
    fn direct_binding_plan_maps_canonical_operation_without_calling_it() {
        let plan: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "numeric.add".into(),
                "direct".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(plan["status"], "bound");
        assert_eq!(plan["target_selected_by"], "agent");
        assert_eq!(plan["execution"], "none");
        assert_eq!(plan["execution_authorized"], false);
        assert_eq!(plan["mutation"], "none");
        assert_eq!(plan["binding"]["kind"], "direct");
        assert_eq!(plan["binding"]["class"], "WasmNumericKernel");
        assert_eq!(plan["binding"]["method"], "add");
        assert_eq!(plan["binding"]["call_performed"], false);
    }

    #[test]
    fn program_binding_plan_exposes_minimum_builder_without_selecting_generation() {
        let add: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "numeric.add".into(),
                "program".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(add["status"], "bound");
        assert_eq!(add["binding"]["minimum_generation"], "v1");
        assert_eq!(
            add["binding"]["minimum_builder_class"],
            "WasmMathProgramBuilder"
        );
        assert_eq!(
            add["binding"]["program_generation_selected"],
            serde_json::Value::Null
        );
        assert_eq!(add["binding"]["builder_method"], "addBinary");
        assert_eq!(add["binding"]["opcode_symbol"], "OP_ADD");
        assert_eq!(add["binding"]["builder_mutation_performed"], false);

        let select: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "tensor.select_axis".into(),
                "program".into(),
                &[1, 3, 1, 1],
                &[],
                &[1, 2, 0],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(select["status"], "bound");
        assert_eq!(select["binding"]["minimum_generation"], "v4");
        assert_eq!(
            select["binding"]["minimum_builder_class"],
            "WasmMathProgramV4Builder"
        );
        assert_eq!(select["binding"]["builder_method"], "addSelectAxis");

        let reduction: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "reduction.mean_axis".into(),
                "program".into(),
                &[1, 3, 2, 1],
                &[],
                &[2],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(reduction["binding"]["minimum_generation"], "v8");
        assert_eq!(
            reduction["binding"]["minimum_builder_class"],
            "WasmMathProgramV8Builder"
        );
        assert_eq!(reduction["binding"]["builder_method"], "addMeanAxis");

        let comparison: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "comparison.less_equal_01".into(),
                "program".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(comparison["binding"]["minimum_generation"], "v9");
        assert_eq!(
            comparison["binding"]["minimum_builder_class"],
            "WasmMathProgramV9Builder"
        );
        assert_eq!(comparison["binding"]["builder_method"], "addLessEqual01");
    }

    #[test]
    fn program_binding_requires_explicit_cosine_epsilon_while_direct_can_use_default() {
        let direct: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "linalg.cosine_similarity".into(),
                "direct".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(direct["status"], "bound");
        assert_eq!(direct["binding"]["method"], "cosineSimilarity");
        assert_eq!(direct["preflight"]["program_binding_deferred"], true);

        let program_missing: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "linalg.cosine_similarity".into(),
                "program".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(program_missing["status"], "requires_parameters");
        assert!(program_missing["binding"].is_null());
        assert_eq!(program_missing["execution_authorized"], false);

        let program_explicit: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "linalg.cosine_similarity".into(),
                "program".into(),
                &[1, 3, 1, 1],
                &[1, 3, 1, 1],
                &[],
                &[1e-6],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(program_explicit["status"], "bound");
        assert_eq!(
            program_explicit["binding"]["builder_method"],
            "addCosineSimilarity"
        );
        assert_eq!(
            program_explicit["binding"]["parameter_projection"]["layout"],
            "epsilon"
        );
        assert_eq!(
            program_explicit["binding"]["builder_mutation_performed"],
            false
        );
    }

    #[test]
    fn binding_plan_propagates_preflight_rejection_without_binding() {
        let rejected: serde_json::Value = serde_json::from_str(
            &math_plan_binding(
                "linalg.matmul".into(),
                "program".into(),
                &[1, 2, 3, 4],
                &[1, 2, 3, 5],
                &[],
                &[],
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(rejected["status"], "rejected");
        assert!(rejected["binding"].is_null());
        assert_eq!(
            rejected["preflight"]["failure"]["predicate"],
            "matmul.compatible_shapes"
        );
        assert_eq!(rejected["execution_authorized"], false);
        assert_eq!(rejected["mutation"], "none");
    }

    #[test]
    fn binding_plan_rejects_implicit_target_selection() {
        assert!(math_plan_binding(
            "numeric.abs".into(),
            "auto".into(),
            &[1, 3, 1, 1],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn binding_plan_is_deterministic_and_never_executes() {
        let first = math_plan_binding(
            "tensor.reshape".into(),
            "program".into(),
            &[1, 2, 1, 3],
            &[],
            &[1, 1, 3, 2],
            &[],
        )
        .unwrap();
        let second = math_plan_binding(
            "tensor.reshape".into(),
            "program".into(),
            &[1, 2, 1, 3],
            &[],
            &[1, 1, 3, 2],
            &[],
        )
        .unwrap();
        assert_eq!(first, second);

        let value: serde_json::Value = serde_json::from_str(&first).unwrap();
        assert_eq!(value["execution"], "none");
        assert_eq!(value["execution_authorized"], false);
        assert_eq!(value["mutation"], "none");
        assert_eq!(value["binding"]["builder_mutation_performed"], false);
    }
}
