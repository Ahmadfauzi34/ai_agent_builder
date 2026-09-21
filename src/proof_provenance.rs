use wasm_bindgen::prelude::*;

use crate::coprocessor::verify_vectors_metrics;
use crate::graph::CompiledGraph;
use crate::math::program::{
    OP_ABS, OP_ADD, OP_CROSS_ENTROPY, OP_DIV, OP_DOT, OP_ENTROPY, OP_EXP,
    OP_KL_DIVERGENCE, OP_L2_DISTANCE, OP_L2_NORM, OP_LOG, OP_MATMUL, OP_MAX, OP_MEAN,
    OP_MIN, OP_MUL, OP_NORMALIZE, OP_SQRT, OP_STD_POPULATION, OP_SUB, OP_SUM,
    OP_TRANSPOSE, OP_VARIANCE_POPULATION,
};
use crate::math::{
    math_check_operation, MathProgram, MathProgramV4, MathProgramV5, MathProgramV6,
    MathProgramV7, MathProgramV8, MathProgramV9, MathProgramV9Builder, WasmComparison,
    WasmIndexSource, WasmLinearAlgebra, WasmNumericKernel, WasmProbability, WasmReduction,
    WasmStatistics, WasmTensorTransform,
};
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

const PROOF_PROVENANCE_V1: &str = include_str!("../docs/proof-provenance.v1.json");
const MATH_PROOF_V1: &str = include_str!("../docs/math-proof.v1.json");
const MAX_PROOF_LABEL_BYTES: usize = 256;
const MAX_ATTESTATION_DETAIL_BYTES: usize = 1024;

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn validate_label(label: &str, context: &str) -> Result<(), String> {
    if label.is_empty() {
        return Err(format!("{context}: label must be non-empty"));
    }
    if label.len() > MAX_PROOF_LABEL_BYTES {
        return Err(format!(
            "{context}: label {} bytes exceeds limit {MAX_PROOF_LABEL_BYTES}",
            label.len()
        ));
    }
    Ok(())
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn bytes_fingerprint(bytes: &[u8]) -> String {
    format!("fnv1a64:{:016x}", fnv1a64(bytes.iter().copied()))
}

fn f32_fingerprint(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    bytes_fingerprint(&bytes)
}

fn tensor_fingerprint(tensor: &WasmTensor) -> String {
    let mut bytes = Vec::new();
    for dim in tensor.shape() {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    for value in tensor.to_array() {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes_fingerprint(&bytes)
}


fn tensors_fingerprint(inputs: &[WasmTensor]) -> String {
    let mut bytes = Vec::new();
    for (index, tensor) in inputs.iter().enumerate() {
        bytes.extend_from_slice(&(index as u64).to_le_bytes());
        for dim in tensor.shape() {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for value in tensor.to_array() {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes_fingerprint(&bytes)
}

fn math_program_version(plan: &[u8]) -> Result<u8, String> {
    if plan.len() < 5 {
        return Err("MathProgram proof: plan is truncated before version byte".to_string());
    }
    let version = plan[4];
    if !(1..=9).contains(&version) {
        return Err(format!(
            "MathProgram proof: unsupported canonical plan version {version}; expected 1..=9"
        ));
    }
    Ok(version)
}

fn math_program_declared_inputs(plan: &[u8]) -> Result<usize, String> {
    math_program_version(plan)?;
    if plan.len() < 6 {
        return Err("MathProgram proof: plan is truncated before num_inputs".to_string());
    }
    Ok(plan[5] as usize)
}

fn math_program_identity_from_plan(plan: &[u8]) -> Result<(u8, String), String> {
    let version = math_program_version(plan)?;
    let identity = match version {
        1..=3 => MathProgram::from_plan(plan)?.program_identity(),
        4 => MathProgramV4::from_plan(plan)?.program_identity(),
        5 => MathProgramV5::from_plan(plan)?.program_identity(),
        6 => MathProgramV6::from_plan(plan)?.program_identity(),
        7 => MathProgramV7::from_plan(plan)?.program_identity(),
        8 => MathProgramV8::from_plan(plan)?.program_identity(),
        9 => MathProgramV9::from_plan(plan)?.program_identity(),
        _ => unreachable!("version validated above"),
    };
    Ok((version, identity))
}

fn run_math_program_plan(
    plan: &[u8],
    inputs: &[WasmTensor],
) -> Result<(u8, String, Vec<f32>), String> {
    let version = math_program_version(plan)?;
    let (identity, output) = match version {
        1..=3 => {
            let program = MathProgram::from_plan(plan)?;
            let output = match inputs {
                [input] => program.run1(input)?,
                [a, b] => program.run2(a, b)?,
                _ => {
                    return Err(format!(
                        "MathProgram proof: v1-v3 verifier supports exactly 1 or 2 inputs, got {}",
                        inputs.len()
                    ))
                }
            };
            (program.program_identity(), output)
        }
        4 => {
            let program = MathProgramV4::from_plan(plan)?;
            let output = match inputs {
                [input] => program.run1(input)?,
                [a, b] => program.run2(a, b)?,
                _ => {
                    return Err(format!(
                        "MathProgram proof: v4 verifier supports exactly 1 or 2 inputs, got {}",
                        inputs.len()
                    ))
                }
            };
            (program.program_identity(), output)
        }
        5 => {
            let program = MathProgramV5::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        6 => {
            let program = MathProgramV6::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        7 => {
            let program = MathProgramV7::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        8 => {
            let program = MathProgramV8::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        9 => {
            let program = MathProgramV9::from_plan(plan)?;
            let output = program.run_inputs(inputs)?;
            (program.program_identity(), output)
        }
        _ => unreachable!("version validated above"),
    };
    Ok((version, identity, output.to_array()))
}


fn tensor_shape_u32(tensor: &WasmTensor, context: &str) -> Result<Vec<u32>, String> {
    tensor
        .shape()
        .into_iter()
        .map(|dim| {
            u32::try_from(dim)
                .map_err(|_| format!("{context}: tensor dimension {dim} exceeds u32"))
        })
        .collect()
}

fn direct_math_arity(operation_id: &str) -> Result<usize, String> {
    match operation_id {
        "numeric.abs"
        | "numeric.sqrt"
        | "numeric.exp"
        | "numeric.log"
        | "numeric.clamp"
        | "tensor.transpose"
        | "tensor.reshape"
        | "tensor.permute"
        | "tensor.slice"
        | "tensor.select_axis"
        | "linalg.l2_norm"
        | "statistics.sum"
        | "statistics.mean"
        | "statistics.variance_population"
        | "statistics.std_population"
        | "statistics.min"
        | "statistics.max"
        | "probability.normalize"
        | "probability.entropy"
        | "reduction.sum_axis"
        | "reduction.mean_axis"
        | "reduction.min_axis"
        | "reduction.max_axis"
        | "index.indices_like" => Ok(1),
        "numeric.add"
        | "numeric.sub"
        | "numeric.mul"
        | "numeric.div"
        | "linalg.matmul"
        | "linalg.dot"
        | "linalg.cosine_similarity"
        | "linalg.l2_distance"
        | "probability.cross_entropy"
        | "probability.kl_divergence"
        | "comparison.less_equal_01" => Ok(2),
        _ => Err(format!(
            "DirectMath verifier: unknown canonical operation_id {operation_id}"
        )),
    }
}

fn direct_math_preflight(
    operation_id: &str,
    inputs: &[WasmTensor],
    u32_params: &[u32],
    f32_params: &[f32],
    context: &str,
) -> Result<String, String> {
    let arity = direct_math_arity(operation_id)?;
    if inputs.len() != arity {
        return Err(format!(
            "{context}: operation {operation_id} requires {arity} input(s), got {}",
            inputs.len()
        ));
    }
    if operation_id == "linalg.cosine_similarity" && f32_params.len() != 1 {
        return Err(format!(
            "{context}: cosine verification requires explicit f32_params=[epsilon] so direct and MathProgram reference semantics are identical"
        ));
    }

    let lhs_shape = tensor_shape_u32(&inputs[0], context)?;
    let rhs_shape = if arity == 2 {
        tensor_shape_u32(&inputs[1], context)?
    } else {
        Vec::new()
    };
    let preflight = math_check_operation(
        operation_id.to_string(),
        &lhs_shape,
        &rhs_shape,
        u32_params,
        f32_params,
    )?;
    if !preflight.contains("\"status\":\"admissible\"") {
        return Err(format!(
            "{context}: operation metadata was rejected by mathCheckOperation: {preflight}"
        ));
    }
    Ok(preflight)
}

fn build_direct_math_reference_v9(
    operation_id: &str,
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<MathProgramV9, String> {
    let arity = direct_math_arity(operation_id)? as u8;
    let output = arity;
    let mut builder = MathProgramV9Builder::new(arity, arity + 1)?;

    match operation_id {
        "numeric.abs" => builder.add_unary(OP_ABS, 0, output)?,
        "numeric.sqrt" => builder.add_unary(OP_SQRT, 0, output)?,
        "numeric.exp" => builder.add_unary(OP_EXP, 0, output)?,
        "numeric.log" => builder.add_unary(OP_LOG, 0, output)?,
        "numeric.clamp" => builder.add_clamp(0, output, f32_params[0], f32_params[1])?,
        "numeric.add" => builder.add_binary(OP_ADD, 0, 1, output)?,
        "numeric.sub" => builder.add_binary(OP_SUB, 0, 1, output)?,
        "numeric.mul" => builder.add_binary(OP_MUL, 0, 1, output)?,
        "numeric.div" => builder.add_binary(OP_DIV, 0, 1, output)?,

        "tensor.transpose" => builder.add_unary(OP_TRANSPOSE, 0, output)?,
        "tensor.reshape" => builder.add_reshape(0, output, u32_params)?,
        "tensor.permute" => builder.add_permute(0, output, u32_params)?,
        "tensor.slice" => builder.add_slice(0, output, &u32_params[..4], &u32_params[4..])?,
        "tensor.select_axis" => {
            builder.add_select_axis(0, output, u32_params[0], &u32_params[1..])?
        }

        "linalg.matmul" => builder.add_binary(OP_MATMUL, 0, 1, output)?,
        "linalg.dot" => builder.add_binary(OP_DOT, 0, 1, output)?,
        "linalg.l2_norm" => builder.add_unary(OP_L2_NORM, 0, output)?,
        "linalg.cosine_similarity" => {
            builder.add_cosine_similarity(0, 1, output, f32_params[0])?
        }
        "linalg.l2_distance" => builder.add_binary(OP_L2_DISTANCE, 0, 1, output)?,

        "statistics.sum" => builder.add_unary(OP_SUM, 0, output)?,
        "statistics.mean" => builder.add_unary(OP_MEAN, 0, output)?,
        "statistics.variance_population" => {
            builder.add_unary(OP_VARIANCE_POPULATION, 0, output)?
        }
        "statistics.std_population" => builder.add_unary(OP_STD_POPULATION, 0, output)?,
        "statistics.min" => builder.add_unary(OP_MIN, 0, output)?,
        "statistics.max" => builder.add_unary(OP_MAX, 0, output)?,

        "probability.normalize" => builder.add_unary(OP_NORMALIZE, 0, output)?,
        "probability.entropy" => builder.add_unary(OP_ENTROPY, 0, output)?,
        "probability.cross_entropy" => builder.add_binary(OP_CROSS_ENTROPY, 0, 1, output)?,
        "probability.kl_divergence" => {
            builder.add_binary(OP_KL_DIVERGENCE, 0, 1, output)?
        }

        "reduction.sum_axis" => builder.add_sum_axis(0, output, u32_params[0])?,
        "reduction.mean_axis" => builder.add_mean_axis(0, output, u32_params[0])?,
        "reduction.min_axis" => builder.add_min_axis(0, output, u32_params[0])?,
        "reduction.max_axis" => builder.add_max_axis(0, output, u32_params[0])?,

        "comparison.less_equal_01" => builder.add_less_equal_01(0, 1, output)?,
        "index.indices_like" => builder.add_indices_like(0, output, u32_params[0])?,
        _ => {
            return Err(format!(
                "DirectMath verifier: no V9 reference mapping for {operation_id}"
            ))
        }
    }

    builder.set_output(output)?;
    builder.compile()
}

fn run_direct_math_operation(
    operation_id: &str,
    inputs: &[WasmTensor],
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<WasmTensor, String> {
    let unary = &inputs[0];
    let binary_rhs = || {
        inputs
            .get(1)
            .ok_or_else(|| format!("DirectMath verifier: missing rhs for {operation_id}"))
    };

    match operation_id {
        "numeric.abs" => WasmNumericKernel::new().abs(unary),
        "numeric.sqrt" => WasmNumericKernel::new().sqrt(unary),
        "numeric.exp" => WasmNumericKernel::new().exp(unary),
        "numeric.log" => WasmNumericKernel::new().log(unary),
        "numeric.clamp" => {
            WasmNumericKernel::new().clamp(unary, f32_params[0], f32_params[1])
        }
        "numeric.add" => WasmNumericKernel::new().add(unary, binary_rhs()?),
        "numeric.sub" => WasmNumericKernel::new().sub(unary, binary_rhs()?),
        "numeric.mul" => WasmNumericKernel::new().mul(unary, binary_rhs()?),
        "numeric.div" => WasmNumericKernel::new().div(unary, binary_rhs()?),

        "tensor.transpose" => Ok(WasmTensorTransform::new().transpose(unary)),
        "tensor.reshape" => {
            let shape = u32_params.iter().map(|&value| value as usize).collect::<Vec<_>>();
            WasmTensorTransform::new().reshape(unary, &shape)
        }
        "tensor.permute" => {
            let axes = u32_params.iter().map(|&value| value as usize).collect::<Vec<_>>();
            WasmTensorTransform::new().permute(unary, &axes)
        }
        "tensor.slice" => {
            let starts = u32_params[..4]
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            let ends = u32_params[4..]
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            WasmTensorTransform::new().slice(unary, &starts, &ends)
        }
        "tensor.select_axis" => {
            let indices = u32_params[1..]
                .iter()
                .map(|&value| value as usize)
                .collect::<Vec<_>>();
            WasmTensorTransform::new().select_axis(unary, u32_params[0] as usize, &indices)
        }

        "linalg.matmul" => WasmLinearAlgebra::new().matmul(unary, binary_rhs()?),
        "linalg.dot" => WasmLinearAlgebra::new().dot(unary, binary_rhs()?),
        "linalg.l2_norm" => WasmLinearAlgebra::new().l2_norm(unary),
        "linalg.cosine_similarity" => WasmLinearAlgebra::new().cosine_similarity(
            unary,
            binary_rhs()?,
            Some(f64::from(f32_params[0])),
        ),
        "linalg.l2_distance" => WasmLinearAlgebra::new().l2_distance(unary, binary_rhs()?),

        "statistics.sum" => WasmStatistics::new().sum(unary),
        "statistics.mean" => WasmStatistics::new().mean(unary),
        "statistics.variance_population" => WasmStatistics::new().variance_population(unary),
        "statistics.std_population" => WasmStatistics::new().std_population(unary),
        "statistics.min" => WasmStatistics::new().min(unary),
        "statistics.max" => WasmStatistics::new().max(unary),

        "probability.normalize" => WasmProbability::new().normalize(unary),
        "probability.entropy" => WasmProbability::new().entropy(unary),
        "probability.cross_entropy" => {
            WasmProbability::new().cross_entropy(unary, binary_rhs()?)
        }
        "probability.kl_divergence" => {
            WasmProbability::new().kl_divergence(unary, binary_rhs()?)
        }

        "reduction.sum_axis" => WasmReduction::new().sum_axis(unary, u32_params[0]),
        "reduction.mean_axis" => WasmReduction::new().mean_axis(unary, u32_params[0]),
        "reduction.min_axis" => WasmReduction::new().min_axis(unary, u32_params[0]),
        "reduction.max_axis" => WasmReduction::new().max_axis(unary, u32_params[0]),

        "comparison.less_equal_01" => {
            WasmComparison::new().less_equal_01(unary, binary_rhs()?)
        }
        "index.indices_like" => WasmIndexSource::new().indices_like(unary, u32_params[0]),
        _ => Err(format!(
            "DirectMath verifier: no direct execution mapping for {operation_id}"
        )),
    }
}

fn direct_math_reference_identity(
    operation_id: &str,
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<(MathProgramV9, String), String> {
    let program = build_direct_math_reference_v9(operation_id, u32_params, f32_params)?;
    let identity = program.program_identity();
    Ok((program, identity))
}

fn tolerances_json(abs_tol: f64, rel_tol: f64) -> String {
    format!("{{\"abs\":{abs_tol},\"rel\":{rel_tol}}}")
}

fn ledger_receipt_json(
    receipt_id: u32,
    authority: &str,
    verifier: &str,
    reference_authority: &str,
    label: &str,
    primary_fingerprint_name: &str,
    primary_fingerprint: &str,
    candidate_fingerprint: &str,
    program_identity_fingerprint: Option<&str>,
    runtime_subject_json: Option<&str>,
    abs_tol: f64,
    rel_tol: f64,
    result_json: &str,
) -> String {
    let program = program_identity_fingerprint
        .map(|value| format!(",\"program_identity_fingerprint\":\"{}\"", json_escape(value)))
        .unwrap_or_default();
    let runtime_subject = runtime_subject_json
        .map(|value| format!(",\"runtime_subject\":{value}"))
        .unwrap_or_default();
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"{}\",",
            "\"verifier\":\"{}\",",
            "\"reference_authority\":\"{}\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"{}\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\"",
            "{}",
            "{}",
            ",\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(authority),
        json_escape(verifier),
        json_escape(reference_authority),
        json_escape(label),
        json_escape(primary_fingerprint_name),
        json_escape(primary_fingerprint),
        json_escape(candidate_fingerprint),
        program,
        runtime_subject,
        tolerances_json(abs_tol, rel_tol),
        result_json,
    )
}

/// Return the embedded proof-provenance contract.
#[wasm_bindgen(js_name = proofProvenanceCapabilities)]
pub fn proof_provenance_capabilities() -> String {
    PROOF_PROVENANCE_V1.to_string()
}


/// Return the MathProgram proof/correlation authority contract.
#[wasm_bindgen(js_name = mathProofCapabilities)]
pub fn math_proof_capabilities() -> String {
    MATH_PROOF_V1.to_string()
}

/// Record an explicit caller claim. This never upgrades into verifier authority.
#[wasm_bindgen(js_name = workspaceRecordAttestation)]
pub fn workspace_record_attestation(
    workspace: &mut AgentWorkspace,
    label: String,
    claimed_passed: bool,
    detail: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceRecordAttestation")?;
    if detail.len() > MAX_ATTESTATION_DETAIL_BYTES {
        return Err(format!(
            "workspaceRecordAttestation: detail {} bytes exceeds limit {MAX_ATTESTATION_DETAIL_BYTES}",
            detail.len()
        ));
    }

    let attestation_id =
        workspace.record_attestation_internal(label.clone(), claimed_passed, detail.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.caller-attestation.v1\",",
            "\"attestation_id\":{},",
            "\"authority\":\"caller_attestation\",",
            "\"label\":\"{}\",",
            "\"claimed_passed\":{},",
            "\"detail\":\"{}\"",
            "}}"
        ),
        attestation_id,
        json_escape(&label),
        claimed_passed,
        json_escape(&detail),
    ))
}

/// Verify against a caller-supplied reference and record a lower-authority comparator receipt.
#[wasm_bindgen(js_name = workspaceVerifyVectorReceipt)]
pub fn workspace_verify_vector_receipt(
    workspace: &mut AgentWorkspace,
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceVerifyVectorReceipt")?;

    let report = verify_vectors_metrics(reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();
    let reference_fingerprint = f32_fingerprint(reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_comparator",
        "mathVerifyVectors",
        "caller_supplied",
        &label,
        "reference_fingerprint",
        &reference_fingerprint,
        &candidate_fingerprint,
        None,
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "mathVerifyVectors".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    let runtime_subject = runtime_subject_binding_json(workspace);
    Ok(ledger_receipt_json(
        receipt_id,
        "wasm_comparator",
        "mathVerifyVectors",
        "caller_supplied",
        &label,
        "reference_fingerprint",
        &reference_fingerprint,
        &candidate_fingerprint,
        None,
        Some(&runtime_subject),
        abs_tol,
        rel_tol,
        &result_json,
    ))
}

/// Execute the compiled Burn graph as reference, compare the candidate, and record the receipt.
///
/// The returned receipt includes the exact programIdentity object. The workspace ledger stores a
/// compact receipt keyed by the deterministic program-identity fingerprint.
#[wasm_bindgen(js_name = workspaceVerifyGraphReceipt)]
pub fn workspace_verify_graph_receipt(
    workspace: &mut AgentWorkspace,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    validate_label(&label, "workspaceVerifyGraphReceipt")?;

    let program_identity = graph.program_identity();
    workspace.require_runtime_program_identity_if_bound(
        &program_identity,
        "workspaceVerifyGraphReceipt",
    )?;

    let reference = graph.run(registry, input)?.to_array();
    let report = verify_vectors_metrics(&reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();

    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensor_fingerprint(input);
    let reference_fingerprint = f32_fingerprint(&reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_verifier",
        "CompiledGraph.verifyFlat",
        "burn_compiled_graph",
        &label,
        "input_fingerprint",
        &input_fingerprint,
        &candidate_fingerprint,
        Some(&program_identity_fingerprint),
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "CompiledGraph.verifyFlat".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"wasm_verifier\",",
            "\"verifier\":\"CompiledGraph.verifyFlat\",",
            "\"reference_authority\":\"burn_compiled_graph\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"program_identity\":{},",
            "\"program_identity_fingerprint\":\"{}\",",
            "\"mutable_state_in_program_identity\":false,",
            "\"runtime_subject\":{},",
            "\"input_fingerprint\":\"{}\",",
            "\"reference_fingerprint\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\",",
            "\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(&label),
        program_identity,
        json_escape(&program_identity_fingerprint),
        runtime_subject_binding_json(workspace),
        json_escape(&input_fingerprint),
        json_escape(&reference_fingerprint),
        json_escape(&candidate_fingerprint),
        tolerances_json(abs_tol, rel_tol),
        result_json,
    ))
}



/// Bind a replay-derived MathProgram programIdentity to an already-bound runtime subject.
///
/// Callers supply canonical plan bytes, never a free-form identity string. Replay validation
/// derives the authoritative exact programIdentity before any binding mutation occurs.
#[wasm_bindgen(js_name = workspaceBindRuntimeMathProgramPlan)]
pub fn workspace_bind_runtime_math_program_plan(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
) -> Result<String, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceBindRuntimeMathProgramPlan: runtime subject must be bound first".to_string(),
        );
    }

    let (version, program_identity) = math_program_identity_from_plan(plan)?;
    let newly_bound = workspace.bind_runtime_program_identity(program_identity.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.runtime-math-program-binding.v1\",",
            "\"program_plan_version\":{},",
            "\"program_identity\":{},",
            "\"identity_policy\":\"exact_program_identity\",",
            "\"identity_source\":\"canonical_plan_replay\",",
            "\"newly_bound\":{},",
            "\"binding_count\":{},",
            "\"mutation\":\"runtime_program_binding_metadata_only\"",
            "}}"
        ),
        version,
        program_identity,
        newly_bound,
        workspace.runtime_program_binding_count(),
    ))
}

fn workspace_verify_math_program_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    inputs: &[WasmTensor],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
    context: &str,
) -> Result<String, String> {
    validate_label(&label, context)?;
    if !(1..=2).contains(&inputs.len()) {
        return Err(format!(
            "{context}: verifier supports exactly 1 or 2 external inputs, got {}",
            inputs.len()
        ));
    }

    let declared_inputs = math_program_declared_inputs(plan)?;
    if declared_inputs != inputs.len() {
        return Err(format!(
            "{context}: canonical MathProgram declares {declared_inputs} external inputs but this verifier surface received {}",
            inputs.len()
        ));
    }

    let (identity_version, program_identity) = math_program_identity_from_plan(plan)?;

    if workspace.runtime_subject_binding().is_some()
        && !workspace.runtime_program_identity_bound(&program_identity)
    {
        return Err(format!(
            "{context}: MathProgram programIdentity is not bound to this runtime subject; bind the canonical plan with workspaceBindRuntimeMathProgramPlan first"
        ));
    }

    let (run_version, run_identity, reference) = run_math_program_plan(plan, inputs)?;
    debug_assert_eq!(identity_version, run_version);
    debug_assert_eq!(program_identity, run_identity);

    let report = verify_vectors_metrics(&reference, candidate, abs_tol, rel_tol)?;
    let receipt_id = workspace.next_verifier_receipt_id();

    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensors_fingerprint(inputs);
    let reference_fingerprint = f32_fingerprint(&reference);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_verifier",
        "MathProgram.verifyFlat",
        "burn_math_program",
        &label,
        "input_fingerprint",
        &input_fingerprint,
        &candidate_fingerprint,
        Some(&program_identity_fingerprint),
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "MathProgram.verifyFlat".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"wasm_verifier\",",
            "\"verifier\":\"MathProgram.verifyFlat\",",
            "\"reference_authority\":\"burn_math_program\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"program_plan_version\":{},",
            "\"program_identity\":{},",
            "\"program_identity_fingerprint\":\"{}\",",
            "\"mutable_state_in_program_identity\":false,",
            "\"runtime_subject\":{},",
            "\"input_count\":{},",
            "\"input_fingerprint\":\"{}\",",
            "\"reference_fingerprint\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\",",
            "\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(&label),
        run_version,
        program_identity,
        json_escape(&program_identity_fingerprint),
        runtime_subject_binding_json(workspace),
        inputs.len(),
        json_escape(&input_fingerprint),
        json_escape(&reference_fingerprint),
        json_escape(&candidate_fingerprint),
        tolerances_json(abs_tol, rel_tol),
        result_json,
    ))
}

/// Replay a canonical 1-input MathProgram as the independent numerical reference.
#[wasm_bindgen(js_name = workspaceVerifyMathProgram1Receipt)]
pub fn workspace_verify_math_program_1_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_math_program_receipt(
        workspace,
        plan,
        &[input.clone()],
        candidate,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyMathProgram1Receipt",
    )
}

/// Replay a canonical 2-input MathProgram as the independent numerical reference.
#[wasm_bindgen(js_name = workspaceVerifyMathProgram2Receipt)]
pub fn workspace_verify_math_program_2_receipt(
    workspace: &mut AgentWorkspace,
    plan: &[u8],
    lhs: &WasmTensor,
    rhs: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_math_program_receipt(
        workspace,
        plan,
        &[lhs.clone(), rhs.clone()],
        candidate,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyMathProgram2Receipt",
    )
}


/// Bind the canonical one-step MathProgramV9 reference identity for a direct math operation.
///
/// This does not execute the direct operation. It validates metadata through mathCheckOperation,
/// builds the verifier-owned V9 reference program, and binds only that exact replayable identity
/// to the already-bound runtime subject.
#[wasm_bindgen(js_name = workspaceBindRuntimeDirectMathOperation)]
pub fn workspace_bind_runtime_direct_math_operation(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    lhs_shape: &[u32],
    rhs_shape: &[u32],
    u32_params: &[u32],
    f32_params: &[f32],
) -> Result<String, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceBindRuntimeDirectMathOperation: runtime subject must be bound first"
                .to_string(),
        );
    }

    if operation_id == "linalg.cosine_similarity" && f32_params.len() != 1 {
        return Err(
            "workspaceBindRuntimeDirectMathOperation: cosine verification requires explicit f32_params=[epsilon]"
                .to_string(),
        );
    }

    let preflight = math_check_operation(
        operation_id.clone(),
        lhs_shape,
        rhs_shape,
        u32_params,
        f32_params,
    )?;
    if !preflight.contains("\"status\":\"admissible\"") {
        return Err(format!(
            "workspaceBindRuntimeDirectMathOperation: operation metadata rejected: {preflight}"
        ));
    }

    let (_, program_identity) =
        direct_math_reference_identity(&operation_id, u32_params, f32_params)?;
    let newly_bound = workspace.bind_runtime_program_identity(program_identity.clone())?;

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.runtime-direct-math-binding.v1\",",
            "\"operation_id\":\"{}\",",
            "\"candidate_authority\":\"burn_direct_math\",",
            "\"reference_authority\":\"burn_math_program\",",
            "\"reference_program_generation\":\"v9\",",
            "\"reference_program_identity\":{},",
            "\"identity_source\":\"canonical_operation_to_math_program_v9\",",
            "\"newly_bound\":{},",
            "\"binding_count\":{},",
            "\"mutation\":\"runtime_program_binding_metadata_only\"",
            "}}"
        ),
        json_escape(&operation_id),
        program_identity,
        newly_bound,
        workspace.runtime_program_binding_count(),
    ))
}

fn workspace_verify_direct_math_receipt(
    workspace: &mut AgentWorkspace,
    operation_id: &str,
    inputs: &[WasmTensor],
    u32_params: &[u32],
    f32_params: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
    context: &str,
) -> Result<String, String> {
    validate_label(&label, context)?;
    direct_math_preflight(
        operation_id,
        inputs,
        u32_params,
        f32_params,
        context,
    )?;

    let (reference_program, program_identity) =
        direct_math_reference_identity(operation_id, u32_params, f32_params)?;

    if workspace.runtime_subject_binding().is_some()
        && !workspace.runtime_program_identity_bound(&program_identity)
    {
        return Err(format!(
            "{context}: direct math reference identity is not bound to this runtime subject; bind it with workspaceBindRuntimeDirectMathOperation first"
        ));
    }

    let direct_output =
        run_direct_math_operation(operation_id, inputs, u32_params, f32_params)?.to_array();
    let reference_output = reference_program.run_inputs(inputs)?.to_array();
    let report =
        verify_vectors_metrics(&reference_output, &direct_output, abs_tol, rel_tol)?;

    let receipt_id = workspace.next_verifier_receipt_id();
    let program_identity_fingerprint = bytes_fingerprint(program_identity.as_bytes());
    let input_fingerprint = tensors_fingerprint(inputs);
    let reference_fingerprint = f32_fingerprint(&reference_output);
    let direct_output_fingerprint = f32_fingerprint(&direct_output);
    let result_json = report.to_json();

    let compact = ledger_receipt_json(
        receipt_id,
        "wasm_verifier",
        "DirectMath.verifyAgainstMathProgramV9",
        "burn_math_program",
        &label,
        "input_fingerprint",
        &input_fingerprint,
        &direct_output_fingerprint,
        Some(&program_identity_fingerprint),
        None,
        abs_tol,
        rel_tol,
        &result_json,
    );

    let stored = workspace.record_verifier_receipt_internal(
        "DirectMath.verifyAgainstMathProgramV9".into(),
        report.passed,
        compact,
    )?;
    debug_assert_eq!(stored, receipt_id);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.verifier-receipt.v1\",",
            "\"receipt_id\":{},",
            "\"authority\":\"wasm_verifier\",",
            "\"verifier\":\"DirectMath.verifyAgainstMathProgramV9\",",
            "\"reference_authority\":\"burn_math_program\",",
            "\"candidate_authority\":\"burn_direct_math\",",
            "\"operation_id\":\"{}\",",
            "\"label\":\"{}\",",
            "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
            "\"reference_program_generation\":\"v9\",",
            "\"program_identity\":{},",
            "\"program_identity_fingerprint\":\"{}\",",
            "\"mutable_state_in_program_identity\":false,",
            "\"runtime_subject\":{},",
            "\"input_count\":{},",
            "\"input_fingerprint\":\"{}\",",
            "\"reference_fingerprint\":\"{}\",",
            "\"candidate_fingerprint\":\"{}\",",
            "\"tolerances\":{},",
            "\"result\":{}",
            "}}"
        ),
        receipt_id,
        json_escape(operation_id),
        json_escape(&label),
        program_identity,
        json_escape(&program_identity_fingerprint),
        runtime_subject_binding_json(workspace),
        inputs.len(),
        json_escape(&input_fingerprint),
        json_escape(&reference_fingerprint),
        json_escape(&direct_output_fingerprint),
        tolerances_json(abs_tol, rel_tol),
        result_json,
    ))
}

/// Execute a canonical unary direct math operation and compare it to a verifier-owned V9 reference.
#[wasm_bindgen(js_name = workspaceVerifyDirectMath1Receipt)]
pub fn workspace_verify_direct_math_1_receipt(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    input: &WasmTensor,
    u32_params: &[u32],
    f32_params: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_direct_math_receipt(
        workspace,
        &operation_id,
        &[input.clone()],
        u32_params,
        f32_params,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyDirectMath1Receipt",
    )
}

/// Execute a canonical binary direct math operation and compare it to a verifier-owned V9 reference.
#[wasm_bindgen(js_name = workspaceVerifyDirectMath2Receipt)]
pub fn workspace_verify_direct_math_2_receipt(
    workspace: &mut AgentWorkspace,
    operation_id: String,
    lhs: &WasmTensor,
    rhs: &WasmTensor,
    u32_params: &[u32],
    f32_params: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: String,
) -> Result<String, String> {
    workspace_verify_direct_math_receipt(
        workspace,
        &operation_id,
        &[lhs.clone(), rhs.clone()],
        u32_params,
        f32_params,
        abs_tol,
        rel_tol,
        label,
        "workspaceVerifyDirectMath2Receipt",
    )
}


/// Return proof-related workspace collections without merging their authority classes.
#[wasm_bindgen(js_name = workspaceProofLedger)]
pub fn workspace_proof_ledger(workspace: &AgentWorkspace) -> String {
    let legacy = workspace.query("_proofs".into(), None, None);
    let attestations = workspace.query("_attestations".into(), None, None);
    let receipts = workspace.query("_verifier_receipts".into(), None, None);
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.proof-ledger.v1\",",
            "\"legacy_recordProof_authority\":\"caller_controlled_legacy\",",
            "\"legacy_proofs\":{},",
            "\"attestations\":{},",
            "\"verifier_receipts\":{}",
            "}}"
        ),
        legacy,
        attestations,
        receipts,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        math_proof_capabilities, workspace_bind_runtime_direct_math_operation,
        workspace_bind_runtime_math_program_plan, workspace_proof_ledger,
        workspace_record_attestation, workspace_verify_direct_math_1_receipt,
        workspace_verify_direct_math_2_receipt, workspace_verify_graph_receipt,
        workspace_verify_math_program_1_receipt, workspace_verify_math_program_2_receipt,
        workspace_verify_vector_receipt,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::math::program::OP_ABS;
    use crate::math::{MathProgramBuilder, MathProgramV5Builder, MathProgramV9Builder};
    use crate::registry::LayerRegistry;
    use crate::resolution_runtime_bridge::workspace_bind_runtime_subject;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::{workspace_compile, workspace_init_unary};
    use crate::WasmTensor;

    #[test]
    fn caller_attestation_is_not_verifier_authority() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let value = workspace_record_attestation(
            &mut workspace,
            "self-claim".into(),
            true,
            "not independently checked".into(),
        )
        .unwrap();
        assert!(value.contains("\"authority\":\"caller_attestation\""));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("caller_attestation"));
        assert!(!ledger.contains("\"authority\":\"wasm_verifier\""));
    }

    #[test]
    fn vector_receipt_preserves_caller_reference_authority() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let receipt = workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0, 2.0],
            &[1.0, 2.0],
            0.0,
            0.0,
            "vector-check".into(),
        )
        .unwrap();
        assert!(receipt.contains("\"authority\":\"wasm_comparator\""));
        assert!(receipt.contains("\"reference_authority\":\"caller_supplied\""));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn graph_receipt_is_burn_backed_and_failed_comparison_is_still_recorded() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        let mut registry = LayerRegistry::new();

        let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let spec = AgentLayerSpec::relu(id);
        let out = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu".into(),
        )
        .unwrap();
        let graph = workspace_compile(&builder, &registry, out).unwrap();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);

        let receipt = workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &[0.0, 4.0],
            0.0,
            0.0,
            "graph-check".into(),
        )
        .unwrap();

        assert!(receipt.contains("\"authority\":\"wasm_verifier\""));
        assert!(receipt.contains("\"reference_authority\":\"burn_compiled_graph\""));
        assert!(receipt.contains("\"passed\":false"));
        assert!(receipt.contains("\"program_identity\":{"));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("CompiledGraph.verifyFlat"));
        assert!(ledger.contains("\"state\":\"failed\""));
    }

    #[test]
    fn malformed_verification_does_not_consume_receipt_id() {
        let mut workspace = AgentWorkspace::new(2).unwrap();

        assert!(workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0],
            &[f32::NAN],
            0.0,
            0.0,
            "bad".into(),
        )
        .is_err());

        let good = workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0],
            &[1.0],
            0.0,
            0.0,
            "good".into(),
        )
        .unwrap();
        assert!(good.contains("\"receipt_id\":1"));
    }

    fn abs_plan() -> Vec<u8> {
        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        builder.compile().unwrap().program_plan()
    }

    #[test]
    fn math_program_v1_receipt_uses_program_execution_as_reference() {
        let plan = abs_plan();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
        let mut workspace = AgentWorkspace::new(2).unwrap();

        let passed = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "math-program-pass".into(),
        )
        .unwrap();
        assert!(passed.contains("\"authority\":\"wasm_verifier\""));
        assert!(passed.contains("\"verifier\":\"MathProgram.verifyFlat\""));
        assert!(passed.contains("\"reference_authority\":\"burn_math_program\""));
        assert!(passed.contains("\"program_plan_version\":1"));
        assert!(passed.contains("\"passed\":true"));

        let failed = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 4.0],
            0.0,
            0.0,
            "math-program-fail".into(),
        )
        .unwrap();
        assert!(failed.contains("\"receipt_id\":2"));
        assert!(failed.contains("\"passed\":false"));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("MathProgram.verifyFlat"));
        assert!(ledger.contains("\"state\":\"passed\""));
        assert!(ledger.contains("\"state\":\"failed\""));
    }

    #[test]
    fn math_program_v9_two_input_receipt_dispatches_canonical_plan() {
        let mut builder = MathProgramV9Builder::new(2, 3).unwrap();
        builder.add_less_equal_01(0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let plan = builder.compile().unwrap().program_plan();

        let lhs = WasmTensor::new(&[1.0, 3.0, 2.0], &[1, 3, 1, 1]);
        let rhs = WasmTensor::new(&[1.0, 2.0, 2.0], &[1, 3, 1, 1]);
        let mut workspace = AgentWorkspace::new(3).unwrap();

        let receipt = workspace_verify_math_program_2_receipt(
            &mut workspace,
            &plan,
            &lhs,
            &rhs,
            &[1.0, 0.0, 1.0],
            0.0,
            0.0,
            "math-v9".into(),
        )
        .unwrap();

        assert!(receipt.contains("\"program_plan_version\":9"));
        assert!(receipt.contains("\"input_count\":2"));
        assert!(receipt.contains("\"reference_authority\":\"burn_math_program\""));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn subject_bound_math_program_receipt_requires_exact_replay_derived_binding() {
        let plan = abs_plan();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
        let mut workspace = AgentWorkspace::new(2).unwrap();

        workspace_bind_runtime_subject(
            &mut workspace,
            "intent-math-proof".into(),
            3,
            "approval-math-proof".into(),
            "effective-spec".into(),
            "spec-math-proof".into(),
            "policy-math-proof".into(),
            4,
            false,
        )
        .unwrap();

        let error = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "before-bind".into(),
        )
        .unwrap_err();
        assert!(error.contains("workspaceBindRuntimeMathProgramPlan"));

        let binding =
            workspace_bind_runtime_math_program_plan(&mut workspace, &plan).unwrap();
        assert!(binding.contains("\"identity_source\":\"canonical_plan_replay\""));
        assert!(binding.contains("\"newly_bound\":true"));

        let repeat =
            workspace_bind_runtime_math_program_plan(&mut workspace, &plan).unwrap();
        assert!(repeat.contains("\"newly_bound\":false"));

        let receipt = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "after-bind".into(),
        )
        .unwrap();
        assert!(receipt.contains("\"runtime_subject\":{\"status\":\"bound\""));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn v5_three_input_plan_is_rejected_by_one_input_verifier_without_receipt_allocation() {
        let mut builder = MathProgramV5Builder::new(3, 4).unwrap();
        builder.add_unary(OP_ABS, 0, 3).unwrap();
        builder.set_output(3).unwrap();
        let plan = builder.compile().unwrap().program_plan();

        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
        let mut workspace = AgentWorkspace::new(2).unwrap();

        let error = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "v5-wrong-surface".into(),
        )
        .unwrap_err();
        assert!(error.contains("declares 3 external inputs"));

        let good_plan = abs_plan();
        let good = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &good_plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "after-v5-reject".into(),
        )
        .unwrap();
        assert!(good.contains("\"receipt_id\":1"));
    }

    #[test]
    fn malformed_math_program_verification_does_not_consume_receipt_id() {
        let plan = abs_plan();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
        let mut workspace = AgentWorkspace::new(2).unwrap();

        assert!(workspace_verify_math_program_1_receipt(
            &mut workspace,
            &[0, 1, 2],
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "bad-plan".into(),
        )
        .is_err());

        assert!(workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[f32::NAN, 3.0],
            0.0,
            0.0,
            "bad-candidate".into(),
        )
        .is_err());

        let good = workspace_verify_math_program_1_receipt(
            &mut workspace,
            &plan,
            &input,
            &[2.0, 3.0],
            0.0,
            0.0,
            "good".into(),
        )
        .unwrap();
        assert!(good.contains("\"receipt_id\":1"));
    }

    #[test]
    fn direct_math_receipt_executes_direct_and_v9_reference_without_caller_reference() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let lhs = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let rhs = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);

        let receipt = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "numeric.add".into(),
            &lhs,
            &rhs,
            &[],
            &[],
            0.0,
            0.0,
            "direct-add".into(),
        )
        .unwrap();

        assert!(receipt.contains("\"authority\":\"wasm_verifier\""));
        assert!(receipt.contains("\"verifier\":\"DirectMath.verifyAgainstMathProgramV9\""));
        assert!(receipt.contains("\"reference_authority\":\"burn_math_program\""));
        assert!(receipt.contains("\"candidate_authority\":\"burn_direct_math\""));
        assert!(receipt.contains("\"operation_id\":\"numeric.add\""));
        assert!(receipt.contains("\"reference_program_generation\":\"v9\""));
        assert!(receipt.contains("\"passed\":true"));

        let ledger = workspace_proof_ledger(&workspace);
        assert!(ledger.contains("DirectMath.verifyAgainstMathProgramV9"));
        assert!(ledger.contains("\"state\":\"passed\""));
    }

    #[test]
    fn subject_bound_direct_math_requires_exact_generated_reference_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        workspace_bind_runtime_subject(
            &mut workspace,
            "intent-direct-proof".into(),
            5,
            "approval-direct-proof".into(),
            "effective-spec".into(),
            "spec-direct-proof".into(),
            "policy-direct-proof".into(),
            8,
            false,
        )
        .unwrap();

        let lhs = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let rhs = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);

        let before_error = workspace_proof_ledger(&workspace);
        let error = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "numeric.add".into(),
            &lhs,
            &rhs,
            &[],
            &[],
            0.0,
            0.0,
            "before-bind".into(),
        )
        .unwrap_err();
        assert!(error.contains("workspaceBindRuntimeDirectMathOperation"));
        assert_eq!(workspace_proof_ledger(&workspace), before_error);

        let binding = workspace_bind_runtime_direct_math_operation(
            &mut workspace,
            "numeric.add".into(),
            &[1, 2, 1, 1],
            &[1, 2, 1, 1],
            &[],
            &[],
        )
        .unwrap();
        assert!(binding.contains("\"identity_source\":\"canonical_operation_to_math_program_v9\""));
        assert!(binding.contains("\"newly_bound\":true"));

        let repeat = workspace_bind_runtime_direct_math_operation(
            &mut workspace,
            "numeric.add".into(),
            &[1, 2, 1, 1],
            &[1, 2, 1, 1],
            &[],
            &[],
        )
        .unwrap();
        assert!(repeat.contains("\"newly_bound\":false"));

        let receipt = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "numeric.add".into(),
            &lhs,
            &rhs,
            &[],
            &[],
            0.0,
            0.0,
            "after-bind".into(),
        )
        .unwrap();
        assert!(receipt.contains("\"runtime_subject\":{\"status\":\"bound\""));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn direct_cosine_verification_requires_explicit_epsilon_without_consuming_receipt() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let lhs = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);
        let rhs = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);

        assert!(workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "linalg.cosine_similarity".into(),
            &lhs,
            &rhs,
            &[],
            &[],
            0.0,
            0.0,
            "missing-epsilon".into(),
        )
        .is_err());

        let receipt = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "linalg.cosine_similarity".into(),
            &lhs,
            &rhs,
            &[],
            &[1e-6],
            0.0,
            0.0,
            "explicit-epsilon".into(),
        )
        .unwrap();
        assert!(receipt.contains("\"receipt_id\":1"));
        assert!(receipt.contains("\"passed\":true"));
    }

    #[test]
    fn parameterized_direct_math_paths_cover_v4_v8_v9_semantics() {
        let mut workspace = AgentWorkspace::new(4).unwrap();

        let select_input =
            WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let select = workspace_verify_direct_math_1_receipt(
            &mut workspace,
            "tensor.select_axis".into(),
            &select_input,
            &[1, 2, 0],
            &[],
            0.0,
            0.0,
            "select-axis".into(),
        )
        .unwrap();
        assert!(select.contains("\"passed\":true"));

        let reduction_input =
            WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let reduction = workspace_verify_direct_math_1_receipt(
            &mut workspace,
            "reduction.mean_axis".into(),
            &reduction_input,
            &[2],
            &[],
            0.0,
            0.0,
            "mean-axis".into(),
        )
        .unwrap();
        assert!(reduction.contains("\"passed\":true"));

        let comparison_lhs =
            WasmTensor::new(&[1.0, 3.0, 2.0], &[1, 3, 1, 1]);
        let comparison_rhs =
            WasmTensor::new(&[1.0, 2.0, 2.0], &[1, 3, 1, 1]);
        let comparison = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "comparison.less_equal_01".into(),
            &comparison_lhs,
            &comparison_rhs,
            &[],
            &[],
            0.0,
            0.0,
            "comparison".into(),
        )
        .unwrap();
        assert!(comparison.contains("\"passed\":true"));

        let indices = workspace_verify_direct_math_1_receipt(
            &mut workspace,
            "index.indices_like".into(),
            &select_input,
            &[1],
            &[],
            0.0,
            0.0,
            "indices-like".into(),
        )
        .unwrap();
        assert!(indices.contains("\"passed\":true"));
    }

    #[test]
    fn rejected_direct_metadata_does_not_allocate_receipt() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let lhs = WasmTensor::new(
            &(1..=24).map(|value| value as f32).collect::<Vec<_>>(),
            &[1, 2, 3, 4],
        );
        let rhs_bad = WasmTensor::new(&[1.0; 30], &[1, 2, 3, 5]);

        assert!(workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "linalg.matmul".into(),
            &lhs,
            &rhs_bad,
            &[],
            &[],
            0.0,
            0.0,
            "bad-matmul".into(),
        )
        .is_err());

        let good_lhs = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let good_rhs = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        let good = workspace_verify_direct_math_2_receipt(
            &mut workspace,
            "numeric.add".into(),
            &good_lhs,
            &good_rhs,
            &[],
            &[],
            0.0,
            0.0,
            "good-after-reject".into(),
        )
        .unwrap();
        assert!(good.contains("\"receipt_id\":1"));
    }

    #[test]
    fn math_proof_contract_does_not_upgrade_direct_vector_comparison() {
        let capabilities: serde_json::Value =
            serde_json::from_str(&math_proof_capabilities()).unwrap();
        assert_eq!(
            capabilities["authority_model"]["direct_operation_receipt"],
            "not independently provided by v1; direct target must not be relabeled as MathProgram verifier authority"
        );
        assert_eq!(
            capabilities["direct_target"]["authority_limit"],
            "caller_supplied reference; cannot be described as independent proof that the direct operation itself is correct"
        );
    }
}
