//! Fasad WASM tunggal — domain `math` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::math::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::math::comparison_capabilities;
use crate::math::index_source::MAX_INDICES_LIKE_AXIS_LENGTH;
use crate::math::index_source_capabilities;
use crate::math::interaction::admissible;
use crate::math::interaction::binding_rejected;
use crate::math::interaction::binding_requires_parameters;
use crate::math::interaction::candidate_from_preflight_json;
use crate::math::interaction::candidate_rejected_json;
use crate::math::interaction::candidate_requires_parameters_json;
use crate::math::interaction::checked_nonzero_product;
use crate::math::interaction::ensure_arity_shapes;
use crate::math::interaction::ensure_no_params;
use crate::math::interaction::f32_array_json;
use crate::math::interaction::feature_shape;
use crate::math::interaction::first_error_predicate;
use crate::math::interaction::json_escape;
use crate::math::interaction::operation_descriptor;
use crate::math::interaction::operation_requires_explicit_parameters;
use crate::math::interaction::parameter_contract;
use crate::math::interaction::parameter_layout;
use crate::math::interaction::parameterized_operation_globally_possible;
use crate::math::interaction::parse_shape4;
use crate::math::interaction::program_binding_method;
use crate::math::interaction::program_minimum_builder_class;
use crate::math::interaction::reject_error;
use crate::math::interaction::rejected;
use crate::math::interaction::shape_json;
use crate::math::interaction::u32_array_json;
use crate::math::interaction::MATH_BINDING_PLAN_SCHEMA_ID;
use crate::math::interaction::MATH_INTERACTION_SCHEMA_ID;
use crate::math::interaction::MATH_OPERATION_CATALOG_SCHEMA_ID;
use crate::math::interaction::MATH_OPERATION_DESCRIPTION_SCHEMA_ID;
use crate::math::interaction::MATH_VALID_OPERATIONS_SCHEMA_ID;
use crate::math::interaction::OPERATIONS;
use crate::math::math_program_capabilities;
use crate::math::math_program_v4_capabilities;
use crate::math::math_program_v5_capabilities;
use crate::math::math_program_v6_capabilities;
use crate::math::math_program_v7_capabilities;
use crate::math::math_program_v8_capabilities;
use crate::math::math_program_v9_capabilities;
use crate::math::reduction_capabilities;

#[wasm_bindgen(js_name = comparisonCapabilities)]
pub fn wasm_comparison_capabilities() -> String {
    comparison_capabilities()
}

#[wasm_bindgen(js_name = indexSourceCapabilities)]
pub fn wasm_index_source_capabilities() -> String {
    index_source_capabilities()
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
                admissible(operation.id, [lhs[0], lhs[1], lhs[3], lhs[2]], &[], false)
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
                        invalid =
                            Some(format!("axis {axis} at position {position} is out of 0..4"));
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
                    &[
                        "input_distribution_normalized_and_nonnegative",
                        "finite_output_values",
                    ],
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

/// Project all canonical math operations against available operand-shape metadata.
///
/// This is a candidate filter, not a chooser. Unary operations are evaluated against
/// lhs independently even when rhs is available. Binary operations require rhs.
/// Operations needing operation-specific parameters are retained as candidates with
/// status requires_parameters rather than being falsely rejected for missing params.
#[wasm_bindgen(js_name = mathValidOperations)]
pub fn math_valid_operations(lhs_shape: &[u32], rhs_shape: &[u32]) -> Result<String, String> {
    let lhs = parse_shape4(lhs_shape, "lhs")?;
    let rhs = if rhs_shape.is_empty() {
        None
    } else {
        Some(parse_shape4(rhs_shape, "rhs")?)
    };

    let mut entries = Vec::with_capacity(OPERATIONS.len());
    let mut candidate_ids = Vec::new();
    let mut metadata_admissible_count = 0usize;
    let mut requires_parameters_count = 0usize;
    let mut metadata_rejected_count = 0usize;

    for operation in OPERATIONS {
        if operation.arity == 2 && rhs.is_none() {
            metadata_rejected_count += 1;
            entries.push(candidate_rejected_json(
                operation,
                "operand.rhs_available",
                "rhs rank-4 operand available",
                "rhs absent",
            ));
            continue;
        }

        if operation_requires_explicit_parameters(operation.id) {
            match parameterized_operation_globally_possible(operation.id, lhs) {
                Ok(()) => {
                    requires_parameters_count += 1;
                    candidate_ids.push(operation.id);
                    entries.push(candidate_requires_parameters_json(operation));
                }
                Err(error) => {
                    metadata_rejected_count += 1;
                    let predicate = first_error_predicate(&error).to_string();
                    entries.push(candidate_rejected_json(
                        operation,
                        &predicate,
                        "operation has at least one metadata-valid parameterization",
                        &error,
                    ));
                }
            }
            continue;
        }

        let preflight_rhs: &[u32] = if operation.arity == 2 { rhs_shape } else { &[] };
        let preflight =
            math_check_operation(operation.id.to_string(), lhs_shape, preflight_rhs, &[], &[])?;
        if preflight.contains("\"status\":\"admissible\"") {
            metadata_admissible_count += 1;
            candidate_ids.push(operation.id);
        } else {
            metadata_rejected_count += 1;
        }
        entries.push(candidate_from_preflight_json(operation, preflight));
    }

    let candidate_ids_json = candidate_ids
        .iter()
        .map(|id| format!("\"{}\"", json_escape(id)))
        .collect::<Vec<_>>()
        .join(",");

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"interaction_schema\":\"{}\",",
            "\"validity_scope\":\"operand_metadata_candidate_projection_only\",",
            "\"selection_authority\":\"agent\",",
            "\"selected_operation\":null,",
            "\"ranking\":null,",
            "\"recommendation\":null,",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"lhs_shape\":{},",
            "\"rhs_shape\":{},",
            "\"operation_count\":{},",
            "\"candidate_count\":{},",
            "\"metadata_admissible_count\":{},",
            "\"requires_parameters_count\":{},",
            "\"metadata_rejected_count\":{},",
            "\"candidate_operation_ids\":[{}],",
            "\"operations\":[{}]",
            "}}"
        ),
        MATH_VALID_OPERATIONS_SCHEMA_ID,
        MATH_INTERACTION_SCHEMA_ID,
        shape_json(lhs),
        rhs.map(shape_json).unwrap_or_else(|| "null".to_string()),
        OPERATIONS.len(),
        candidate_ids.len(),
        metadata_admissible_count,
        requires_parameters_count,
        metadata_rejected_count,
        candidate_ids_json,
        entries.join(","),
    ))
}

/// Translate a canonical operation into an explicit direct or MathProgram binding plan.
///
/// The caller must choose target="direct" or target="program". This function validates
/// metadata through mathCheckOperation, but never calls the direct method, never mutates
/// a MathProgram builder, and never chooses a target or program generation.
#[wasm_bindgen(js_name = mathPlanBinding)]
pub fn math_plan_binding(
    operation_id: String,
    target: String,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<String, String> {
    if target != "direct" && target != "program" {
        return Err(format!(
            "mathPlanBinding: target must be direct|program, got {target}"
        ));
    }
    let operation = operation_descriptor(&operation_id)
        .ok_or_else(|| format!("mathPlanBinding: unknown operation_id {operation_id}"))?;

    let preflight = math_check_operation(
        operation_id.clone(),
        lhs_shape,
        rhs_shape,
        u32_params,
        f32_params,
    )?;
    if !preflight.contains("\"status\":\"admissible\"") {
        return Ok(binding_rejected(&operation_id, &target, preflight));
    }

    if target == "program" && operation.id == "linalg.cosine_similarity" && f32_params.is_empty() {
        return Ok(binding_requires_parameters(operation, &target, preflight));
    }

    let parameter_projection = format!(
        concat!(
            "{{",
            "\"layout\":\"{}\",",
            "\"u32_params\":[{}],",
            "\"f32_params\":[{}]",
            "}}"
        ),
        parameter_layout(operation.id),
        u32_array_json(u32_params),
        f32_array_json(f32_params),
    );

    let binding = if target == "direct" {
        let (class_name, method_name) =
            operation.direct_surface.split_once('.').ok_or_else(|| {
                format!(
                    "mathPlanBinding: malformed direct surface {}",
                    operation.direct_surface
                )
            })?;
        let direct_arguments = match operation.id {
            "numeric.clamp" => "input_tensor,min,max",
            "linalg.cosine_similarity" => "lhs_tensor,rhs_tensor,epsilon_optional",
            "tensor.reshape" => "input_tensor,shape",
            "tensor.permute" => "input_tensor,axes",
            "tensor.slice" => "input_tensor,starts,ends",
            "tensor.select_axis" => "input_tensor,axis,indices",
            "reduction.sum_axis"
            | "reduction.mean_axis"
            | "reduction.min_axis"
            | "reduction.max_axis" => "input_tensor,axis",
            "index.indices_like" => "reference_tensor,axis",
            _ if operation.arity == 1 => "input_tensor",
            _ => "lhs_tensor,rhs_tensor",
        };
        format!(
            concat!(
                "{{",
                "\"kind\":\"direct\",",
                "\"class\":\"{}\",",
                "\"method\":\"{}\",",
                "\"argument_model\":\"{}\",",
                "\"parameter_projection\":{},",
                "\"call_performed\":false",
                "}}"
            ),
            class_name, method_name, direct_arguments, parameter_projection,
        )
    } else {
        let (builder_method, opcode_symbol, argument_model) = program_binding_method(operation.id);
        if builder_method == "unsupported" {
            return Err(format!(
                "mathPlanBinding: no program binding for {}",
                operation.id
            ));
        }
        format!(
            concat!(
                "{{",
                "\"kind\":\"program\",",
                "\"minimum_generation\":\"{}\",",
                "\"minimum_builder_class\":\"{}\",",
                "\"program_generation_selected\":null,",
                "\"builder_method\":\"{}\",",
                "\"opcode_symbol\":{},",
                "\"argument_model\":\"{}\",",
                "\"parameter_projection\":{},",
                "\"builder_mutation_performed\":false",
                "}}"
            ),
            operation.minimum_program_generation,
            program_minimum_builder_class(operation.minimum_program_generation),
            builder_method,
            opcode_symbol
                .map(|value| format!("\"{}\"", value))
                .unwrap_or_else(|| "null".to_string()),
            argument_model,
            parameter_projection,
        )
    };

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"{}\",",
            "\"operation_id\":\"{}\",",
            "\"target\":\"{}\",",
            "\"target_selected_by\":\"agent\",",
            "\"status\":\"bound\",",
            "\"execution\":\"none\",",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\",",
            "\"binding_authority\":\"projection_only\",",
            "\"binding\":{},",
            "\"preflight\":{}",
            "}}"
        ),
        MATH_BINDING_PLAN_SCHEMA_ID, operation.id, target, binding, preflight,
    ))
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
            "\"discovery\":[\"mathInteractionCapabilities\",\"mathOperationCatalog\",\"mathDescribeOperation\",\"mathCheckOperation\",\"mathValidOperations\",\"mathPlanBinding\",\"mathProofCapabilities\"],",
            "\"direct_verifier\":{{",
                "\"surfaces\":[\"workspaceVerifyDirectMath1Receipt\",\"workspaceVerifyDirectMath2Receipt\"],",
                "\"verifier\":\"DirectMath.verifyAgainstMathProgramV9\",",
                "\"candidate_authority\":\"burn_direct_math\",",
                "\"reference_authority\":\"burn_math_program\"",
            "}},",
            "\"deferred\":[],",
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

#[wasm_bindgen(js_name = linearAlgebraCapabilities)]
pub fn linear_algebra_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.linear-algebra.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"matrix_ops\":[\"matmul\"],",
        "\"vector_ops\":[\"dot\",\"l2Norm\",\"cosineSimilarity\",\"l2Distance\"],",
        "\"vector_layout\":\"[B,F,1,1]\",",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"paired_vector_shape\":\"exact_match\",",
        "\"matmul_layout\":\"[B,G,M,K]@[B,G,K,N]\",",
        "\"cosine_zero_vector\":\"stabilized_to_zero\",",
        "\"solve\":\"deferred_v1\"",
        "}",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = numericKernelCapabilities)]
pub fn numeric_kernel_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.numeric-kernel.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"broadcasting\":\"forbidden_v1\",",
        "\"binary_ops\":[\"add\",\"sub\",\"mul\",\"div\"],",
        "\"unary_ops\":[\"abs\",\"sqrt\",\"exp\",\"log\"],",
        "\"bounded_ops\":[\"clamp\"],",
        "\"predicates\":[\"allFinite\"],",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"divisor\":\"finite_nonzero\",",
        "\"sqrt_domain\":\"x>=0\",",
        "\"log_domain\":\"x>0\",",
        "\"clamp_bounds\":\"finite_min_lte_max\"",
        "}",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = probabilityCapabilities)]
pub fn probability_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.probability.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"input_layout\":\"[B,F,1,1]\",",
        "\"scalar_output_layout\":\"[B,1,1,1]\",",
        "\"log_base\":\"e\",",
        "\"ops\":[\"normalize\",\"entropy\",\"crossEntropy\",\"klDivergence\"],",
        "\"contracts\":{",
        "\"non_negative\":true,",
        "\"normalization_tolerance\":1e-5,",
        "\"zero_p_contribution\":\"exact_zero\",",
        "\"q_zero_where_p_positive\":\"controlled_error\",",
        "\"rng_sampling\":\"deferred\"",
        "}",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = mathProgramV5Capabilities)]
pub fn wasm_math_program_v5_capabilities() -> String {
    math_program_v5_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV6Capabilities)]
pub fn wasm_math_program_v6_capabilities() -> String {
    math_program_v6_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV7Capabilities)]
pub fn wasm_math_program_v7_capabilities() -> String {
    math_program_v7_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV8Capabilities)]
pub fn wasm_math_program_v8_capabilities() -> String {
    math_program_v8_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV9Capabilities)]
pub fn wasm_math_program_v9_capabilities() -> String {
    math_program_v9_capabilities()
}

#[wasm_bindgen(js_name = mathProgramCapabilities)]
pub fn wasm_math_program_capabilities() -> String {
    math_program_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV4Capabilities)]
pub fn wasm_math_program_v4_capabilities() -> String {
    math_program_v4_capabilities()
}

#[wasm_bindgen(js_name = reductionCapabilities)]
pub fn wasm_reduction_capabilities() -> String {
    reduction_capabilities()
}

#[wasm_bindgen(js_name = statisticsCapabilities)]
pub fn statistics_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.statistics.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"input_layout\":\"[B,F,1,1]\",",
        "\"output_layout\":\"[B,1,1,1]\",",
        "\"reduction_axis\":1,",
        "\"reducers\":[\"sum\",\"mean\",\"variancePopulation\",\"stdPopulation\",\"min\",\"max\"],",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"non_empty_batch\":true,",
        "\"non_empty_features\":true,",
        "\"variance_semantics\":\"population_no_bessel_correction\",",
        "\"probability_ops\":\"deferred\"",
        "}",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = tensorTransformCapabilities)]
pub fn tensor_transform_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.tensor-transform.v1\",",
        "\"rank\":4,",
        "\"ops\":[\"reshape\",\"transpose\",\"permute\",\"slice\",\"selectAxis\"],",
        "\"contracts\":{",
        "\"reshape\":\"exact_rank4_equal_element_count\",",
        "\"permute\":\"complete_unique_axes\",",
        "\"slice\":\"bounded_nonempty_ranges\",",
        "\"selectAxis\":\"validated_axis_and_indices\"",
        "}",
        "}"
    )
    .to_string()
}
