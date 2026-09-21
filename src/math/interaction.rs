use wasm_bindgen::prelude::*;

use crate::math::index_source::MAX_INDICES_LIKE_AXIS_LENGTH;

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


const MATH_OPERATION_DESCRIPTION_SCHEMA_ID: &str =
    "burn-research.math-operation-description.v1";
const MATH_PREFLIGHT_SCHEMA_ID: &str = "burn-research.math-preflight.v1";

fn operation_descriptor(operation_id: &str) -> Option<&'static MathOperationDescriptor> {
    OPERATIONS.iter().find(|operation| operation.id == operation_id)
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn shape_json(shape: [u32; 4]) -> String {
    format!("[{},{},{},{}]", shape[0], shape[1], shape[2], shape[3])
}

fn parse_shape4(shape: &[u32], role: &str) -> Result<[u32; 4], String> {
    if shape.len() != 4 {
        return Err(format!(
            "{role}.rank4: expected exactly 4 dimensions, got {}",
            shape.len()
        ));
    }
    Ok([shape[0], shape[1], shape[2], shape[3]])
}

fn checked_nonzero_product(shape: [u32; 4], role: &str) -> Result<u64, String> {
    shape.into_iter().try_fold(1u64, |count, dim| {
        if dim == 0 {
            return Err(format!("{role}.nonzero_dimensions: zero-sized dimensions are not allowed"));
        }
        count
            .checked_mul(u64::from(dim))
            .ok_or_else(|| format!("{role}.element_count: overflow"))
    })
}

fn feature_shape(shape: [u32; 4], role: &str) -> Result<(), String> {
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

fn first_error_predicate(error: &str) -> &str {
    error.split(':').next().unwrap_or("metadata.valid")
}

fn rejected(
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

fn admissible(
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
        if program_binding_deferred { "true" } else { "false" },
    )
}

fn reject_error(operation_id: &str, error: String) -> String {
    let predicate = first_error_predicate(&error).to_string();
    rejected(operation_id, &predicate, "source_contract_predicate_satisfied", &error)
}

fn ensure_no_params(
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

fn ensure_arity_shapes(
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

fn parameter_contract(operation_id: &str) -> (&'static str, &'static str) {
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

/// Describe one canonical operation without executing it or selecting it for the agent.
#[wasm_bindgen(js_name = mathDescribeOperation)]
pub fn math_describe_operation(operation_id: String) -> Result<String, String> {
    let operation = operation_descriptor(&operation_id)
        .ok_or_else(|| format!("mathDescribeOperation: unknown operation_id {operation_id}"))?;
    let (u32_params, f32_params) = parameter_contract(operation.id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"interaction_schema\":\"{}\",",
            "\"operation\":{{",
                "\"id\":\"{}\",",
                "\"family\":\"{}\",",
                "\"arity\":{},",
                "\"direct_surface\":\"{}\",",
                "\"source_contract\":\"{}\",",
                "\"shape_rule\":\"{}\",",
                "\"program\":{{",
                    "\"minimum_generation\":\"{}\",",
                    "\"builder\":\"{}\"",
                "}}",
            "}},",
            "\"preflight_metadata\":{{",
                "\"lhs_shape\":\"required_exact_rank4\",",
                "\"rhs_shape\":\"{}\",",
                "\"u32_params\":\"{}\",",
                "\"f32_params\":\"{}\"",
            "}},",
            "\"authority\":\"introspection_projection_only\",",
            "\"execution\":\"none\"",
            "}}"
        ),
        MATH_OPERATION_DESCRIPTION_SCHEMA_ID,
        MATH_INTERACTION_SCHEMA_ID,
        operation.id,
        operation.family,
        operation.arity,
        operation.direct_surface,
        operation.source_contract,
        operation.shape_rule,
        operation.minimum_program_generation,
        operation.program_builder,
        if operation.arity == 2 {
            "required_exact_rank4"
        } else {
            "must_be_empty"
        },
        u32_params,
        f32_params,
    ))
}

/// Validate only operation metadata that can be proven without reading tensor values.
///
/// u32_params carries shape/axis/index metadata and f32_params carries scalar
/// operation parameters. Value-domain predicates remain explicitly deferred to the
/// authoritative direct/program execution surface.
#[wasm_bindgen(js_name = mathCheckOperation)]
pub fn math_check_operation(
    operation_id: String,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<String, String> {
    let operation = operation_descriptor(&operation_id)
        .ok_or_else(|| format!("mathCheckOperation: unknown operation_id {operation_id}"))?;

    let (lhs, rhs) = match ensure_arity_shapes(operation, lhs_shape, rhs_shape) {
        Ok(value) => value,
        Err(error) => return Ok(reject_error(operation.id, error)),
    };

    let no_params = || ensure_no_params(operation.id, u32_params, f32_params);

    let result = match operation.id {
        "numeric.add" | "numeric.sub" | "numeric.mul" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if lhs != rhs {
                    rejected(
                        operation.id,
                        "shape.exact_match",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(
                        operation.id,
                        lhs,
                        &["finite_input_values", "finite_output_values"],
                        false,
                    )
                }
            }
        }
        "numeric.div" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if lhs != rhs {
                    rejected(
                        operation.id,
                        "shape.exact_match",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(
                        operation.id,
                        lhs,
                        &[
                            "finite_input_values",
                            "rhs_values_nonzero",
                            "finite_output_values",
                        ],
                        false,
                    )
                }
            }
        }
        "numeric.abs" | "numeric.exp" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                admissible(
                    operation.id,
                    lhs,
                    &["finite_input_values", "finite_output_values"],
                    false,
                )
            }
        }
        "numeric.sqrt" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                admissible(
                    operation.id,
                    lhs,
                    &["input_values_finite_and_gte_zero", "finite_output_values"],
                    false,
                )
            }
        }
        "numeric.log" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                admissible(
                    operation.id,
                    lhs,
                    &["input_values_finite_and_gt_zero", "finite_output_values"],
                    false,
                )
            }
        }
        "numeric.clamp" => {
            if !u32_params.is_empty() || f32_params.len() != 2 {
                rejected(
                    operation.id,
                    "parameters.clamp",
                    "u32_params=[] and f32_params=[min,max]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let min = f32_params[0];
                let max = f32_params[1];
                if !min.is_finite() || !max.is_finite() || min > max {
                    rejected(
                        operation.id,
                        "clamp.bounds",
                        "finite min <= max",
                        &format!("min={min},max={max}"),
                    )
                } else {
                    admissible(
                        operation.id,
                        lhs,
                        &["finite_input_values", "finite_output_values"],
                        false,
                    )
                }
            }
        }

        "tensor.transpose" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                admissible(
                    operation.id,
                    [lhs[0], lhs[1], lhs[3], lhs[2]],
                    &[],
                    false,
                )
            }
        }
        "tensor.reshape" => {
            if !f32_params.is_empty() || u32_params.len() != 4 {
                rejected(
                    operation.id,
                    "parameters.reshape",
                    "u32_params=[d0,d1,d2,d3] and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let target = [u32_params[0], u32_params[1], u32_params[2], u32_params[3]];
                match (
                    checked_nonzero_product(lhs, "reshape.source"),
                    checked_nonzero_product(target, "reshape.target"),
                ) {
                    (Ok(source_count), Ok(target_count)) if source_count == target_count => {
                        admissible(operation.id, target, &[], false)
                    }
                    (Ok(source_count), Ok(target_count)) => rejected(
                        operation.id,
                        "reshape.element_count",
                        "source element count == target element count",
                        &format!("{source_count} vs {target_count}"),
                    ),
                    (Err(error), _) | (_, Err(error)) => reject_error(operation.id, error),
                }
            }
        }
        "tensor.permute" => {
            if !f32_params.is_empty() || u32_params.len() != 4 {
                rejected(
                    operation.id,
                    "parameters.permute",
                    "u32_params=[axis0,axis1,axis2,axis3] and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let mut seen = [false; 4];
                let mut invalid = None;
                for (position, axis) in u32_params.iter().copied().enumerate() {
                    if axis >= 4 {
                        invalid = Some(format!("axis {axis} at position {position} is out of 0..4"));
                        break;
                    }
                    if seen[axis as usize] {
                        invalid = Some(format!("duplicate axis {axis} at position {position}"));
                        break;
                    }
                    seen[axis as usize] = true;
                }
                if let Some(actual) = invalid {
                    rejected(
                        operation.id,
                        "permute.complete_unique_axes",
                        "a permutation of [0,1,2,3]",
                        &actual,
                    )
                } else {
                    admissible(
                        operation.id,
                        [
                            lhs[u32_params[0] as usize],
                            lhs[u32_params[1] as usize],
                            lhs[u32_params[2] as usize],
                            lhs[u32_params[3] as usize],
                        ],
                        &[],
                        false,
                    )
                }
            }
        }
        "tensor.slice" => {
            if !f32_params.is_empty() || u32_params.len() != 8 {
                rejected(
                    operation.id,
                    "parameters.slice",
                    "u32_params=[start0..start3,end0..end3] and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let starts = [u32_params[0], u32_params[1], u32_params[2], u32_params[3]];
                let ends = [u32_params[4], u32_params[5], u32_params[6], u32_params[7]];
                let mut failure = None;
                for axis in 0..4 {
                    if starts[axis] >= ends[axis] {
                        failure = Some(format!(
                            "axis {axis} requires start < end, got {}..{}",
                            starts[axis], ends[axis]
                        ));
                        break;
                    }
                    if ends[axis] > lhs[axis] {
                        failure = Some(format!(
                            "axis {axis} end {} exceeds dimension {}",
                            ends[axis], lhs[axis]
                        ));
                        break;
                    }
                }
                if let Some(actual) = failure {
                    rejected(
                        operation.id,
                        "slice.bounded_nonempty_ranges",
                        "0 <= start < end <= input_dim for every axis",
                        &actual,
                    )
                } else {
                    admissible(
                        operation.id,
                        [
                            ends[0] - starts[0],
                            ends[1] - starts[1],
                            ends[2] - starts[2],
                            ends[3] - starts[3],
                        ],
                        &[],
                        false,
                    )
                }
            }
        }
        "tensor.select_axis" => {
            if !f32_params.is_empty() || u32_params.len() < 2 {
                rejected(
                    operation.id,
                    "parameters.select_axis",
                    "u32_params=[axis,index0,...] with at least one index and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let axis = u32_params[0];
                if axis >= 4 {
                    rejected(
                        operation.id,
                        "select_axis.axis",
                        "axis in 0..4",
                        &axis.to_string(),
                    )
                } else if let Some((position, index)) = u32_params[1..]
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, index)| *index >= lhs[axis as usize])
                {
                    rejected(
                        operation.id,
                        "select_axis.index_bounds",
                        "each index < input dimension on selected axis",
                        &format!(
                            "index at position {position} is {index}, dimension={}",
                            lhs[axis as usize]
                        ),
                    )
                } else {
                    let mut output = lhs;
                    output[axis as usize] = (u32_params.len() - 1) as u32;
                    admissible(operation.id, output, &[], false)
                }
            }
        }

        "linalg.matmul" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if lhs.iter().chain(rhs.iter()).any(|dim| *dim == 0) {
                    rejected(
                        operation.id,
                        "matmul.nonzero_dimensions",
                        "all dimensions > 0",
                        &format!("{} @ {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else if lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[3] != rhs[2] {
                    rejected(
                        operation.id,
                        "matmul.compatible_shapes",
                        "[B,G,M,K] @ [B,G,K,N]",
                        &format!("{} @ {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(
                        operation.id,
                        [lhs[0], lhs[1], lhs[2], rhs[3]],
                        &["finite_input_values", "finite_output_values"],
                        false,
                    )
                }
            }
        }
        "linalg.dot" | "linalg.l2_distance" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if let Err(error) = feature_shape(lhs, "lhs") {
                    reject_error(operation.id, error)
                } else if let Err(error) = feature_shape(rhs, "rhs") {
                    reject_error(operation.id, error)
                } else if lhs != rhs {
                    rejected(
                        operation.id,
                        "feature_pair.exact_shape",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(
                        operation.id,
                        [lhs[0], 1, 1, 1],
                        &["finite_input_values", "finite_output_values"],
                        false,
                    )
                }
            }
        }
        "linalg.l2_norm" => {
            if let Some(rejection) = no_params() {
                rejection
            } else if let Err(error) = feature_shape(lhs, "lhs") {
                reject_error(operation.id, error)
            } else {
                admissible(
                    operation.id,
                    [lhs[0], 1, 1, 1],
                    &["finite_input_values", "finite_output_values"],
                    false,
                )
            }
        }
        "linalg.cosine_similarity" => {
            if !u32_params.is_empty() || f32_params.len() > 1 {
                rejected(
                    operation.id,
                    "parameters.cosine_similarity",
                    "u32_params=[] and f32_params=[]|[epsilon]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if let Err(error) = feature_shape(lhs, "lhs") {
                    reject_error(operation.id, error)
                } else if let Err(error) = feature_shape(rhs, "rhs") {
                    reject_error(operation.id, error)
                } else if lhs != rhs {
                    rejected(
                        operation.id,
                        "feature_pair.exact_shape",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else if let Some(epsilon) = f32_params.first().copied() {
                    if !epsilon.is_finite() || epsilon <= 0.0 {
                        rejected(
                            operation.id,
                            "cosine.epsilon",
                            "finite epsilon > 0",
                            &epsilon.to_string(),
                        )
                    } else {
                        admissible(
                            operation.id,
                            [lhs[0], 1, 1, 1],
                            &["finite_input_values", "finite_output_values"],
                            false,
                        )
                    }
                } else {
                    admissible(
                        operation.id,
                        [lhs[0], 1, 1, 1],
                        &["finite_input_values", "finite_output_values"],
                        true,
                    )
                }
            }
        }

        "statistics.sum"
        | "statistics.mean"
        | "statistics.variance_population"
        | "statistics.std_population"
        | "statistics.min"
        | "statistics.max" => {
            if let Some(rejection) = no_params() {
                rejection
            } else if let Err(error) = feature_shape(lhs, "lhs") {
                reject_error(operation.id, error)
            } else {
                admissible(
                    operation.id,
                    [lhs[0], 1, 1, 1],
                    &["finite_input_values", "finite_output_values"],
                    false,
                )
            }
        }

        "probability.normalize" => {
            if let Some(rejection) = no_params() {
                rejection
            } else if let Err(error) = feature_shape(lhs, "lhs") {
                reject_error(operation.id, error)
            } else {
                admissible(
                    operation.id,
                    lhs,
                    &[
                        "input_values_finite_and_nonnegative",
                        "per_batch_mass_strictly_positive",
                        "normalized_output_tolerance",
                    ],
                    false,
                )
            }
        }
        "probability.entropy" => {
            if let Some(rejection) = no_params() {
                rejection
            } else if let Err(error) = feature_shape(lhs, "lhs") {
                reject_error(operation.id, error)
            } else {
                admissible(
                    operation.id,
                    [lhs[0], 1, 1, 1],
                    &["input_distribution_normalized_and_nonnegative", "finite_output_values"],
                    false,
                )
            }
        }
        "probability.cross_entropy" | "probability.kl_divergence" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if let Err(error) = feature_shape(lhs, "lhs") {
                    reject_error(operation.id, error)
                } else if let Err(error) = feature_shape(rhs, "rhs") {
                    reject_error(operation.id, error)
                } else if lhs != rhs {
                    rejected(
                        operation.id,
                        "probability_pair.exact_shape",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(
                        operation.id,
                        [lhs[0], 1, 1, 1],
                        &[
                            "both_distributions_normalized_and_nonnegative",
                            "rhs_support_covers_positive_lhs",
                            "finite_output_values",
                        ],
                        false,
                    )
                }
            }
        }

        "reduction.sum_axis"
        | "reduction.mean_axis"
        | "reduction.min_axis"
        | "reduction.max_axis" => {
            if !f32_params.is_empty() || u32_params.len() != 1 {
                rejected(
                    operation.id,
                    "parameters.reduction_axis",
                    "u32_params=[axis] and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else if lhs.iter().any(|dim| *dim == 0) {
                rejected(
                    operation.id,
                    "reduction.nonzero_dimensions",
                    "all input dimensions > 0",
                    &shape_json(lhs),
                )
            } else {
                let axis = u32_params[0];
                if axis >= 4 {
                    rejected(
                        operation.id,
                        "reduction.axis",
                        "axis in 0..4",
                        &axis.to_string(),
                    )
                } else {
                    let mut output = lhs;
                    output[axis as usize] = 1;
                    admissible(
                        operation.id,
                        output,
                        &["finite_input_values", "finite_output_values"],
                        false,
                    )
                }
            }
        }

        "comparison.less_equal_01" => {
            if let Some(rejection) = no_params() {
                rejection
            } else {
                let rhs = rhs.expect("binary descriptor invariant");
                if lhs != rhs {
                    rejected(
                        operation.id,
                        "comparison.exact_shape",
                        "lhs_shape == rhs_shape",
                        &format!("{} vs {}", shape_json(lhs), shape_json(rhs)),
                    )
                } else {
                    admissible(operation.id, lhs, &["finite_input_values"], false)
                }
            }
        }

        "index.indices_like" => {
            if !f32_params.is_empty() || u32_params.len() != 1 {
                rejected(
                    operation.id,
                    "parameters.index_axis",
                    "u32_params=[axis] and f32_params=[]",
                    &format!(
                        "u32_params_len={},f32_params_len={}",
                        u32_params.len(),
                        f32_params.len()
                    ),
                )
            } else if lhs.iter().any(|dim| *dim == 0) {
                rejected(
                    operation.id,
                    "index.nonzero_dimensions",
                    "all input dimensions > 0",
                    &shape_json(lhs),
                )
            } else {
                let axis = u32_params[0];
                if axis >= 4 {
                    rejected(
                        operation.id,
                        "index.axis",
                        "axis in 0..4",
                        &axis.to_string(),
                    )
                } else if u64::from(lhs[axis as usize]) > MAX_INDICES_LIKE_AXIS_LENGTH as u64 {
                    rejected(
                        operation.id,
                        "index.exact_f32_coordinate_bound",
                        "axis_length <= 16777217",
                        &lhs[axis as usize].to_string(),
                    )
                } else {
                    admissible(operation.id, lhs, &[], false)
                }
            }
        }

        _ => {
            return Err(format!(
                "mathCheckOperation: operation {} has no v1 metadata preflight implementation",
                operation.id
            ));
        }
    };

    Ok(result)
}


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
            "\"discovery\":[\"mathInteractionCapabilities\",\"mathOperationCatalog\",\"mathDescribeOperation\",\"mathCheckOperation\"],",
            "\"deferred\":[\"valid-operation projection\",\"execution facade\"],",
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
    use super::{
        math_check_operation, math_describe_operation, math_interaction_capabilities,
        math_operation_catalog, OPERATIONS,
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
            &math_check_operation(
                "numeric.div".into(),
                &[1, 2, 1, 1],
                &[1, 2, 1, 1],
                &[],
                &[],
            )
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
            &math_check_operation(
                "probability.entropy".into(),
                &[2, 4, 1, 1],
                &[],
                &[],
                &[],
            )
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
            &math_check_operation(
                "reduction.mean_axis".into(),
                &[1, 2, 3, 1],
                &[],
                &[4],
                &[],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(bad_axis["status"], "rejected");
        assert_eq!(bad_axis["failure"]["predicate"], "reduction.axis");
        assert_eq!(bad_axis["execution_authorized"], false);

        assert!(math_check_operation(
            "unknown.operation".into(),
            &[1, 1, 1, 1],
            &[],
            &[],
            &[],
        )
        .is_err());
    }
}
