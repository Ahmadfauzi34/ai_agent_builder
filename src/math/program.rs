use crate::math::program_shape_params::{
    FixedShapeParams, PARAM_PERMUTE_RANK4, PARAM_RESHAPE_RANK4, PARAM_SLICE_RANK4,
    SHAPE_PARAM_BYTES,
};
use crate::math::{
    WasmLinearAlgebra, WasmNumericKernel, WasmProbability, WasmStatistics, WasmTensorTransform,
};
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V1: u8 = 1;
const PLAN_VERSION_V2: u8 = 2;
const PLAN_VERSION_V3: u8 = 3;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_STEP_BYTES_V1: usize = 5;
const PLAN_STEP_BYTES_V2: usize = 14;
const PLAN_STEP_BYTES_V3: usize = 5 + 1 + SHAPE_PARAM_BYTES;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_SLOTS: u8 = 64;
const MAX_STEPS: usize = u8::MAX as usize;

// Numeric Kernel.
pub const OP_ABS: u8 = 0x01;
pub const OP_SQRT: u8 = 0x02;
pub const OP_EXP: u8 = 0x03;
pub const OP_LOG: u8 = 0x04;
pub const OP_CLAMP: u8 = 0x05;
pub const OP_ADD: u8 = 0x11;
pub const OP_SUB: u8 = 0x12;
pub const OP_MUL: u8 = 0x13;
pub const OP_DIV: u8 = 0x14;

// Tensor Transform.
pub const OP_TRANSPOSE: u8 = 0x20;
pub const OP_RESHAPE: u8 = 0x21;
pub const OP_PERMUTE: u8 = 0x22;
pub const OP_SLICE: u8 = 0x23;

// Linear Algebra.
pub const OP_L2_NORM: u8 = 0x30;
pub const OP_DOT: u8 = 0x31;
pub const OP_L2_DISTANCE: u8 = 0x32;
pub const OP_MATMUL: u8 = 0x33;
pub const OP_COSINE_SIMILARITY: u8 = 0x34;

// Statistics.
pub const OP_SUM: u8 = 0x40;
pub const OP_MEAN: u8 = 0x41;
pub const OP_VARIANCE_POPULATION: u8 = 0x42;
pub const OP_STD_POPULATION: u8 = 0x43;
pub const OP_MIN: u8 = 0x44;
pub const OP_MAX: u8 = 0x45;

// Probability.
pub const OP_NORMALIZE: u8 = 0x50;
pub const OP_ENTROPY: u8 = 0x51;
pub const OP_CROSS_ENTROPY: u8 = 0x52;
pub const OP_KL_DIVERGENCE: u8 = 0x53;

const ARITY_UNARY: u8 = 1;
const ARITY_BINARY: u8 = 2;
const PARAM_NONE: u8 = 0;
const PARAM_CLAMP: u8 = 1;
const PARAM_EPSILON: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MathStep {
    op: u8,
    arity: u8,
    in_a: u8,
    in_b: u8,
    out: u8,
    param_kind: u8,
    param_a_bits: u32,
    param_b_bits: u32,
    shape_params: Option<FixedShapeParams>,
}

impl MathStep {
    fn plain(op: u8, arity: u8, in_a: u8, in_b: u8, out: u8) -> Self {
        Self {
            op,
            arity,
            in_a,
            in_b,
            out,
            param_kind: PARAM_NONE,
            param_a_bits: 0,
            param_b_bits: 0,
            shape_params: None,
        }
    }

    fn scalar1(
        op: u8,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
        param_kind: u8,
        value: f32,
    ) -> Self {
        Self {
            op,
            arity,
            in_a,
            in_b,
            out,
            param_kind,
            param_a_bits: canonical_f32_bits(value),
            param_b_bits: 0,
            shape_params: None,
        }
    }

    fn scalar2(
        op: u8,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
        param_kind: u8,
        a: f32,
        b: f32,
    ) -> Self {
        Self {
            op,
            arity,
            in_a,
            in_b,
            out,
            param_kind,
            param_a_bits: canonical_f32_bits(a),
            param_b_bits: canonical_f32_bits(b),
            shape_params: None,
        }
    }

    fn shape(
        op: u8,
        input: u8,
        output: u8,
        shape_params: FixedShapeParams,
    ) -> Self {
        Self {
            op,
            arity: ARITY_UNARY,
            in_a: input,
            in_b: 0,
            out: output,
            param_kind: shape_params.kind(),
            param_a_bits: 0,
            param_b_bits: 0,
            shape_params: Some(shape_params),
        }
    }

    fn param_a(self) -> f32 {
        f32::from_bits(self.param_a_bits)
    }

    fn param_b(self) -> f32 {
        f32::from_bits(self.param_b_bits)
    }

    fn has_scalar_parameters(self) -> bool {
        self.shape_params.is_none() && self.param_kind != PARAM_NONE
    }

    fn has_shape_parameters(self) -> bool {
        self.shape_params.is_some()
    }

    fn v3_payload(self) -> [u8; SHAPE_PARAM_BYTES] {
        if let Some(params) = self.shape_params {
            return params.encode();
        }
        let mut payload = [0u8; SHAPE_PARAM_BYTES];
        payload[0..4].copy_from_slice(&self.param_a_bits.to_le_bytes());
        payload[4..8].copy_from_slice(&self.param_b_bits.to_le_bytes());
        payload
    }
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn expected_arity(op: u8) -> Option<u8> {
    match op {
        OP_ABS
        | OP_SQRT
        | OP_EXP
        | OP_LOG
        | OP_CLAMP
        | OP_TRANSPOSE
        | OP_RESHAPE
        | OP_PERMUTE
        | OP_SLICE
        | OP_L2_NORM
        | OP_SUM
        | OP_MEAN
        | OP_VARIANCE_POPULATION
        | OP_STD_POPULATION
        | OP_MIN
        | OP_MAX
        | OP_NORMALIZE
        | OP_ENTROPY => Some(ARITY_UNARY),
        OP_ADD
        | OP_SUB
        | OP_MUL
        | OP_DIV
        | OP_DOT
        | OP_L2_DISTANCE
        | OP_MATMUL
        | OP_COSINE_SIMILARITY
        | OP_CROSS_ENTROPY
        | OP_KL_DIVERGENCE => Some(ARITY_BINARY),
        _ => None,
    }
}

fn op_name(op: u8) -> &'static str {
    match op {
        OP_ABS => "abs",
        OP_SQRT => "sqrt",
        OP_EXP => "exp",
        OP_LOG => "log",
        OP_CLAMP => "clamp",
        OP_ADD => "add",
        OP_SUB => "sub",
        OP_MUL => "mul",
        OP_DIV => "div",
        OP_TRANSPOSE => "transpose",
        OP_RESHAPE => "reshape",
        OP_PERMUTE => "permute",
        OP_SLICE => "slice",
        OP_L2_NORM => "l2Norm",
        OP_DOT => "dot",
        OP_L2_DISTANCE => "l2Distance",
        OP_MATMUL => "matmul",
        OP_COSINE_SIMILARITY => "cosineSimilarity",
        OP_SUM => "sum",
        OP_MEAN => "mean",
        OP_VARIANCE_POPULATION => "variancePopulation",
        OP_STD_POPULATION => "stdPopulation",
        OP_MIN => "min",
        OP_MAX => "max",
        OP_NORMALIZE => "normalize",
        OP_ENTROPY => "entropy",
        OP_CROSS_ENTROPY => "crossEntropy",
        OP_KL_DIVERGENCE => "klDivergence",
        _ => "unknown",
    }
}

fn bit(slot: u8) -> u64 {
    1u64 << slot
}

fn initial_filled(num_inputs: u8) -> u64 {
    (1u64 << num_inputs) - 1
}

fn validate_program_shape(num_inputs: u8, num_slots: u8) -> Result<(), String> {
    if !(1..=2).contains(&num_inputs) {
        return Err(format!(
            "MathProgram: num_inputs must be 1 or 2, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgram: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
            num_inputs + 1
        ));
    }
    Ok(())
}

fn validate_shape_parameters(
    step: MathStep,
    expected_kind: u8,
    context: &str,
) -> Result<(), String> {
    if step.param_kind != expected_kind || step.param_a_bits != 0 || step.param_b_bits != 0 {
        return Err(format!(
            "{context}: opcode {} requires canonical fixed rank-4 metadata",
            op_name(step.op)
        ));
    }
    let params = step.shape_params.ok_or_else(|| {
        format!(
            "{context}: opcode {} is missing fixed rank-4 metadata",
            op_name(step.op)
        )
    })?;
    if params.kind() != expected_kind {
        return Err(format!(
            "{context}: opcode {} parameter kind mismatch",
            op_name(step.op)
        ));
    }
    Ok(())
}

fn validate_step_parameters(step: MathStep, context: &str) -> Result<(), String> {
    match step.op {
        OP_CLAMP => {
            if step.shape_params.is_some() || step.param_kind != PARAM_CLAMP {
                return Err(format!(
                    "{context}: opcode clamp requires scalar min/max parameters"
                ));
            }
            let min = step.param_a();
            let max = step.param_b();
            if !min.is_finite() || !max.is_finite() {
                return Err(format!(
                    "{context}: clamp bounds must be finite, got min={min}, max={max}"
                ));
            }
            if min > max {
                return Err(format!(
                    "{context}: clamp requires min <= max, got min={min}, max={max}"
                ));
            }
            if step.param_a_bits != canonical_f32_bits(min)
                || step.param_b_bits != canonical_f32_bits(max)
            {
                return Err(format!(
                    "{context}: clamp parameters are not canonically encoded"
                ));
            }
        }
        OP_COSINE_SIMILARITY => {
            if step.shape_params.is_some()
                || step.param_kind != PARAM_EPSILON
                || step.param_b_bits != 0
            {
                return Err(format!(
                    "{context}: opcode cosineSimilarity requires exactly one epsilon parameter"
                ));
            }
            let epsilon = step.param_a();
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(format!(
                    "{context}: cosineSimilarity epsilon must be finite and > 0, got {epsilon}"
                ));
            }
            if step.param_a_bits != canonical_f32_bits(epsilon) {
                return Err(format!(
                    "{context}: cosineSimilarity epsilon is not canonically encoded"
                ));
            }
        }
        OP_RESHAPE => validate_shape_parameters(step, PARAM_RESHAPE_RANK4, context)?,
        OP_PERMUTE => validate_shape_parameters(step, PARAM_PERMUTE_RANK4, context)?,
        OP_SLICE => validate_shape_parameters(step, PARAM_SLICE_RANK4, context)?,
        _ => {
            if step.param_kind != PARAM_NONE
                || step.param_a_bits != 0
                || step.param_b_bits != 0
                || step.shape_params.is_some()
            {
                return Err(format!(
                    "{context}: opcode {} does not accept parameters",
                    op_name(step.op)
                ));
            }
        }
    }
    Ok(())
}

fn validate_step(
    step: MathStep,
    num_slots: u8,
    filled: u64,
    written: u64,
    context: &str,
) -> Result<(u64, u64), String> {
    let expected = expected_arity(step.op).ok_or_else(|| {
        format!(
            "{context}: unknown opcode 0x{:02X}; Math Program is fail-closed",
            step.op
        )
    })?;
    if step.arity != expected {
        return Err(format!(
            "{context}: opcode {} (0x{:02X}) requires arity {expected}, got {}",
            op_name(step.op),
            step.op,
            step.arity
        ));
    }
    validate_step_parameters(step, context)?;
    if step.in_a >= num_slots || step.in_b >= num_slots || step.out >= num_slots {
        return Err(format!(
            "{context}: slot out of range for num_slots={num_slots}: in_a={}, in_b={}, out={}",
            step.in_a, step.in_b, step.out
        ));
    }
    if filled & bit(step.in_a) == 0 {
        return Err(format!(
            "{context}: input slot {} is read before write",
            step.in_a
        ));
    }
    if step.arity == ARITY_BINARY && filled & bit(step.in_b) == 0 {
        return Err(format!(
            "{context}: second input slot {} is read before write",
            step.in_b
        ));
    }
    if step.arity == ARITY_UNARY && step.in_b != 0 {
        return Err(format!(
            "{context}: unary opcode {} must encode in_b=0 canonically, got {}",
            op_name(step.op),
            step.in_b
        ));
    }
    if filled & bit(step.out) != 0 {
        return Err(format!(
            "{context}: output slot {} is already filled; Math Program slots are write-once",
            step.out
        ));
    }

    Ok((filled | bit(step.out), written | bit(step.out)))
}

fn plan_version(steps: &[MathStep]) -> u8 {
    if steps.iter().copied().any(MathStep::has_shape_parameters) {
        PLAN_VERSION_V3
    } else if steps.iter().copied().any(MathStep::has_scalar_parameters) {
        PLAN_VERSION_V2
    } else {
        PLAN_VERSION_V1
    }
}

fn step_bytes(version: u8) -> Result<usize, String> {
    match version {
        PLAN_VERSION_V1 => Ok(PLAN_STEP_BYTES_V1),
        PLAN_VERSION_V2 => Ok(PLAN_STEP_BYTES_V2),
        PLAN_VERSION_V3 => Ok(PLAN_STEP_BYTES_V3),
        _ => Err(format!(
            "MathProgram: unsupported plan version {version}; supported versions are 1, 2, and 3"
        )),
    }
}

fn encode_plan(
    num_inputs: u8,
    num_slots: u8,
    steps: &[MathStep],
    out_slot: u8,
) -> Result<Vec<u8>, String> {
    validate_program_shape(num_inputs, num_slots)?;
    if steps.is_empty() {
        return Err("MathProgram: program must contain at least one step".into());
    }
    if steps.len() > MAX_STEPS {
        return Err(format!(
            "MathProgram: step count {} exceeds maximum {MAX_STEPS}",
            steps.len()
        ));
    }

    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    for (index, step) in steps.iter().copied().enumerate() {
        (filled, written) = validate_step(
            step,
            num_slots,
            filled,
            written,
            &format!("MathProgram step {index}"),
        )?;
    }
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgram: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgram: output slot {out_slot} must be produced by a program step"
        ));
    }

    let version = plan_version(steps);
    let step_bytes = step_bytes(version)?;
    let mut plan = Vec::with_capacity(
        PLAN_HEADER_BYTES + steps.len() * step_bytes + PLAN_OUTPUT_BYTES,
    );
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(version);
    plan.push(num_inputs);
    plan.push(num_slots);
    plan.push(steps.len() as u8);
    for step in steps {
        plan.push(step.op);
        plan.push(step.arity);
        plan.push(step.in_a);
        plan.push(step.in_b);
        plan.push(step.out);
        if version == PLAN_VERSION_V2 {
            plan.push(step.param_kind);
            plan.extend_from_slice(&step.param_a_bits.to_le_bytes());
            plan.extend_from_slice(&step.param_b_bits.to_le_bytes());
        } else if version == PLAN_VERSION_V3 {
            plan.push(step.param_kind);
            plan.extend_from_slice(&step.v3_payload());
        }
    }
    plan.push(out_slot);
    Ok(plan)
}

fn decode_v3_parameters(
    param_kind: u8,
    payload: &[u8],
) -> Result<(u32, u32, Option<FixedShapeParams>), String> {
    if payload.len() != SHAPE_PARAM_BYTES {
        return Err(format!(
            "MathProgram: malformed v3 parameter payload length {}, expected {SHAPE_PARAM_BYTES}",
            payload.len()
        ));
    }
    if matches!(
        param_kind,
        PARAM_RESHAPE_RANK4 | PARAM_PERMUTE_RANK4 | PARAM_SLICE_RANK4
    ) {
        return Ok((
            0,
            0,
            Some(FixedShapeParams::decode(param_kind, payload)?),
        ));
    }
    if payload[8..].iter().any(|byte| *byte != 0) {
        return Err("MathProgram: unused v3 scalar parameter bytes must be zero".into());
    }
    let a = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let b = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    Ok((a, b, None))
}

fn decode_plan(plan: &[u8]) -> Result<(u8, u8, Vec<MathStep>, u8), String> {
    if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
        return Err("MathProgram: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgram: invalid plan magic".into());
    }
    let version = plan[4];
    let step_bytes = step_bytes(version)?;

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgram: plan must contain at least one step".into());
    }
    let expected_len = PLAN_HEADER_BYTES
        .checked_add(num_steps.checked_mul(step_bytes).ok_or_else(|| {
            "MathProgram: plan step length overflow".to_string()
        })?)
        .and_then(|len| len.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgram: plan length overflow".to_string())?;
    if plan.len() != expected_len {
        return Err(format!(
            "MathProgram: malformed plan length: expected {expected_len}, got {}",
            plan.len()
        ));
    }

    let mut steps = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;
    for index in 0..num_steps {
        let (param_kind, param_a_bits, param_b_bits, shape_params) = match version {
            PLAN_VERSION_V1 => (PARAM_NONE, 0, 0, None),
            PLAN_VERSION_V2 => (
                plan[offset + 5],
                u32::from_le_bytes([
                    plan[offset + 6],
                    plan[offset + 7],
                    plan[offset + 8],
                    plan[offset + 9],
                ]),
                u32::from_le_bytes([
                    plan[offset + 10],
                    plan[offset + 11],
                    plan[offset + 12],
                    plan[offset + 13],
                ]),
                None,
            ),
            PLAN_VERSION_V3 => {
                let param_kind = plan[offset + 5];
                let payload = &plan[offset + 6..offset + 6 + SHAPE_PARAM_BYTES];
                let (a, b, shape) = decode_v3_parameters(param_kind, payload)?;
                (param_kind, a, b, shape)
            }
            _ => unreachable!(),
        };
        let step = MathStep {
            op: plan[offset],
            arity: plan[offset + 1],
            in_a: plan[offset + 2],
            in_b: plan[offset + 3],
            out: plan[offset + 4],
            param_kind,
            param_a_bits,
            param_b_bits,
            shape_params,
        };
        (filled, written) = validate_step(
            step,
            num_slots,
            filled,
            written,
            &format!("MathProgram replay step {index}"),
        )?;
        steps.push(step);
        offset += step_bytes;
    }

    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgram: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgram: output slot {out_slot} must be produced by a program step"
        ));
    }

    Ok((num_inputs, num_slots, steps, out_slot))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn math_program_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v3\",",
        "\"plan_schemas\":[\"burn-research.math-program-plan.v1\",\"burn-research.math-program-plan.v2\",\"burn-research.math-program-plan.v3\"],",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"inputs\":\"one_or_two\",",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"replay\":true,",
        "\"v1_identity_compatibility\":true,",
        "\"v2_identity_compatibility\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false,",
        "\"parameterized_ops\":{\"scalar_v2\":[\"clamp\",\"cosineSimilarity\"],\"fixed_rank4_v3\":[\"reshape\",\"permute\",\"slice\"],\"variable_length\":\"deferred_selectAxis\"},",
        "\"opcodes\":{",
        "\"abs\":1,\"sqrt\":2,\"exp\":3,\"log\":4,\"clamp\":5,",
        "\"add\":17,\"sub\":18,\"mul\":19,\"div\":20,",
        "\"transpose\":32,\"reshape\":33,\"permute\":34,\"slice\":35,",
        "\"l2Norm\":48,\"dot\":49,\"l2Distance\":50,\"matmul\":51,\"cosineSimilarity\":52,",
        "\"sum\":64,\"mean\":65,\"variancePopulation\":66,\"stdPopulation\":67,\"min\":68,\"max\":69,",
        "\"normalize\":80,\"entropy\":81,\"crossEntropy\":82,\"klDivergence\":83",
        "}",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramBuilder {
    num_inputs: u8,
    num_slots: u8,
    steps: Vec<MathStep>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramBuilder {
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<Self, String> {
        validate_program_shape(num_inputs, num_slots)?;
        Ok(Self {
            num_inputs,
            num_slots,
            steps: Vec::new(),
            filled: initial_filled(num_inputs),
            written: 0,
            out_slot: None,
        })
    }

    fn push_step(&mut self, step: MathStep, context: &str) -> Result<(), String> {
        if self.steps.len() >= MAX_STEPS {
            return Err(format!(
                "{context}: step count exceeds maximum {MAX_STEPS}"
            ));
        }
        let (filled, written) = validate_step(
            step,
            self.num_slots,
            self.filled,
            self.written,
            context,
        )?;
        self.steps.push(step);
        self.filled = filled;
        self.written = written;
        Ok(())
    }

    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.push_step(
            MathStep::plain(op, ARITY_UNARY, input, 0, output),
            "MathProgramBuilder.addUnary",
        )
    }

    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        self.push_step(
            MathStep::plain(op, ARITY_BINARY, lhs, rhs, output),
            "MathProgramBuilder.addBinary",
        )
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        self.push_step(
            MathStep::scalar2(
                OP_CLAMP,
                ARITY_UNARY,
                input,
                0,
                output,
                PARAM_CLAMP,
                min,
                max,
            ),
            "MathProgramBuilder.addClamp",
        )
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.push_step(
            MathStep::scalar1(
                OP_COSINE_SIMILARITY,
                ARITY_BINARY,
                lhs,
                rhs,
                output,
                PARAM_EPSILON,
                epsilon,
            ),
            "MathProgramBuilder.addCosineSimilarity",
        )
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::reshape(shape)?;
        self.push_step(
            MathStep::shape(OP_RESHAPE, input, output, params),
            "MathProgramBuilder.addReshape",
        )
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::permute(axes)?;
        self.push_step(
            MathStep::shape(OP_PERMUTE, input, output, params),
            "MathProgramBuilder.addPermute",
        )
    }

    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::slice(starts, ends)?;
        self.push_step(
            MathStep::shape(OP_SLICE, input, output, params),
            "MathProgramBuilder.addSlice",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramBuilder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramBuilder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgram, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramBuilder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.steps, out_slot)?;
        Ok(MathProgram {
            num_inputs: self.num_inputs,
            num_slots: self.num_slots,
            steps: self.steps.clone(),
            out_slot,
            canonical_plan,
        })
    }

    pub fn num_inputs(&self) -> u8 {
        self.num_inputs
    }

    pub fn num_slots(&self) -> u8 {
        self.num_slots
    }

    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

#[derive(Clone, Debug)]
pub struct MathProgram {
    num_inputs: u8,
    num_slots: u8,
    steps: Vec<MathStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgram {
    pub fn from_plan(plan: &[u8]) -> Result<Self, String> {
        let (num_inputs, num_slots, steps, out_slot) = decode_plan(plan)?;
        let canonical_plan = encode_plan(num_inputs, num_slots, &steps, out_slot)?;
        if canonical_plan != plan {
            return Err("MathProgram: replay plan is not canonical".into());
        }
        Ok(Self {
            num_inputs,
            num_slots,
            steps,
            out_slot,
            canonical_plan,
        })
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        if self.num_inputs != 1 {
            return Err(format!(
                "MathProgram.run1: program requires {} inputs",
                self.num_inputs
            ));
        }
        self.execute(&[input.clone()])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        if self.num_inputs != 2 {
            return Err(format!(
                "MathProgram.run2: program requires {} inputs",
                self.num_inputs
            ));
        }
        self.execute(&[a.clone(), b.clone()])
    }

    fn execute(&self, inputs: &[WasmTensor]) -> Result<WasmTensor, String> {
        if inputs.len() != self.num_inputs as usize {
            return Err(format!(
                "MathProgram: expected {} inputs, got {}",
                self.num_inputs,
                inputs.len()
            ));
        }

        let numeric = WasmNumericKernel::new();
        let tensor = WasmTensorTransform::new();
        let linalg = WasmLinearAlgebra::new();
        let statistics = WasmStatistics::new();
        let probability = WasmProbability::new();

        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (index, input) in inputs.iter().enumerate() {
            slots[index] = Some(input.clone());
        }

        for (index, step) in self.steps.iter().copied().enumerate() {
            let a = slots[step.in_a as usize].as_ref().ok_or_else(|| {
                format!("MathProgram.run: step {index} input slot {} is empty", step.in_a)
            })?;

            let output = if step.arity == ARITY_UNARY {
                match step.op {
                    OP_ABS => numeric.abs(a),
                    OP_SQRT => numeric.sqrt(a),
                    OP_EXP => numeric.exp(a),
                    OP_LOG => numeric.log(a),
                    OP_CLAMP => numeric.clamp(a, step.param_a(), step.param_b()),
                    OP_TRANSPOSE => Ok(tensor.transpose(a)),
                    OP_RESHAPE => {
                        let shape = step
                            .shape_params
                            .and_then(FixedShapeParams::reshape_usize)
                            .ok_or_else(|| {
                                format!("MathProgram.run: step {index} reshape metadata missing")
                            })?;
                        tensor.reshape(a, &shape)
                    }
                    OP_PERMUTE => {
                        let axes = step
                            .shape_params
                            .and_then(FixedShapeParams::permute_usize)
                            .ok_or_else(|| {
                                format!("MathProgram.run: step {index} permute metadata missing")
                            })?;
                        tensor.permute(a, &axes)
                    }
                    OP_SLICE => {
                        let (starts, ends) = step
                            .shape_params
                            .and_then(FixedShapeParams::slice_usize)
                            .ok_or_else(|| {
                                format!("MathProgram.run: step {index} slice metadata missing")
                            })?;
                        tensor.slice(a, &starts, &ends)
                    }
                    OP_L2_NORM => linalg.l2_norm(a),
                    OP_SUM => statistics.sum(a),
                    OP_MEAN => statistics.mean(a),
                    OP_VARIANCE_POPULATION => statistics.variance_population(a),
                    OP_STD_POPULATION => statistics.std_population(a),
                    OP_MIN => statistics.min(a),
                    OP_MAX => statistics.max(a),
                    OP_NORMALIZE => probability.normalize(a),
                    OP_ENTROPY => probability.entropy(a),
                    _ => Err(format!(
                        "MathProgram.run: step {index} unsupported unary opcode 0x{:02X}",
                        step.op
                    )),
                }
            } else {
                let b = slots[step.in_b as usize].as_ref().ok_or_else(|| {
                    format!(
                        "MathProgram.run: step {index} second input slot {} is empty",
                        step.in_b
                    )
                })?;
                match step.op {
                    OP_ADD => numeric.add(a, b),
                    OP_SUB => numeric.sub(a, b),
                    OP_MUL => numeric.mul(a, b),
                    OP_DIV => numeric.div(a, b),
                    OP_DOT => linalg.dot(a, b),
                    OP_L2_DISTANCE => linalg.l2_distance(a, b),
                    OP_MATMUL => linalg.matmul(a, b),
                    OP_COSINE_SIMILARITY => {
                        linalg.cosine_similarity(a, b, Some(step.param_a() as f64))
                    }
                    OP_CROSS_ENTROPY => probability.cross_entropy(a, b),
                    OP_KL_DIVERGENCE => probability.kl_divergence(a, b),
                    _ => Err(format!(
                        "MathProgram.run: step {index} unsupported binary opcode 0x{:02X}",
                        step.op
                    )),
                }
            }
            .map_err(|error| format!("MathProgram.run step {index} {}: {error}", op_name(step.op)))?;

            slots[step.out as usize] = Some(output);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgram.run: output slot {} is empty", self.out_slot))
    }

    pub fn program_plan(&self) -> Vec<u8> {
        self.canonical_plan.clone()
    }

    pub fn program_identity(&self) -> String {
        format!(
            "{{\"schema\":\"burn-research.math-program-identity.v1\",\"plan_hex\":\"{}\"}}",
            hex(&self.canonical_plan)
        )
    }

    pub fn num_inputs(&self) -> u8 {
        self.num_inputs
    }

    pub fn num_slots(&self) -> u8 {
        self.num_slots
    }

    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    pub fn out_slot(&self) -> u8 {
        self.out_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected} within {tolerance}"
            );
        }
    }

    #[test]
    fn numeric_to_statistics_composition_executes_deterministically() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        builder.add_unary(OP_SQRT, 0, 1).unwrap();
        builder.add_unary(OP_MEAN, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();

        let input = WasmTensor::new(&[1.0, 4.0, 9.0], &[1, 3, 1, 1]);
        let output = program.run1(&input).unwrap();
        assert_eq!(output.shape(), vec![1, 1, 1, 1]);
        assert_close(&output.to_array(), &[2.0], 1e-6);
    }

    #[test]
    fn probability_composition_normalize_then_entropy_is_valid() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        builder.add_unary(OP_NORMALIZE, 0, 1).unwrap();
        builder.add_unary(OP_ENTROPY, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();

        let weights = WasmTensor::new(&[1.0, 1.0], &[1, 2, 1, 1]);
        let output = program.run1(&weights).unwrap();
        assert_close(&output.to_array(), &[std::f32::consts::LN_2], 1e-6);
    }

    #[test]
    fn binary_numeric_to_linalg_composition_is_supported() {
        let mut builder = MathProgramBuilder::new(2, 4).unwrap();
        builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
        builder.add_unary(OP_L2_NORM, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let program = builder.compile().unwrap();

        let a = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let b = WasmTensor::new(&[2.0, 2.0], &[1, 2, 1, 1]);
        let output = program.run2(&a, &b).unwrap();
        assert_close(&output.to_array(), &[25.0_f32.sqrt()], 1e-6);
    }

    #[test]
    fn canonical_v1_plan_replays_same_identity_and_behavior() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        builder.add_unary(OP_SUM, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        assert_eq!(program.program_plan()[4], PLAN_VERSION_V1);
        let replay = MathProgram::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), program.program_plan());
        assert_eq!(replay.program_identity(), program.program_identity());

        let input = WasmTensor::new(&[-1.0, 2.0, -3.0], &[1, 3, 1, 1]);
        assert_eq!(
            replay.run1(&input).unwrap().to_array(),
            program.run1(&input).unwrap().to_array()
        );
    }

    #[test]
    fn clamp_uses_v2_plan_and_replays_identically() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        builder.add_clamp(0, 1, -1.0, 1.0).unwrap();
        builder.add_unary(OP_SUM, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        assert_eq!(program.program_plan()[4], PLAN_VERSION_V2);

        let replay = MathProgram::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), program.program_plan());
        assert_eq!(replay.program_identity(), program.program_identity());

        let input = WasmTensor::new(&[-2.0, 0.5, 3.0], &[1, 3, 1, 1]);
        assert_close(&program.run1(&input).unwrap().to_array(), &[0.5], 1e-6);
        assert_eq!(
            replay.run1(&input).unwrap().to_array(),
            program.run1(&input).unwrap().to_array()
        );
    }

    #[test]
    fn cosine_similarity_parameter_changes_identity_and_executes() {
        let mut tight = MathProgramBuilder::new(2, 3).unwrap();
        tight
            .add_cosine_similarity(0, 1, 2, 1e-6)
            .unwrap();
        tight.set_output(2).unwrap();
        let tight = tight.compile().unwrap();

        let mut loose = MathProgramBuilder::new(2, 3).unwrap();
        loose
            .add_cosine_similarity(0, 1, 2, 1e-3)
            .unwrap();
        loose.set_output(2).unwrap();
        let loose = loose.compile().unwrap();

        assert_eq!(tight.program_plan()[4], PLAN_VERSION_V2);
        assert_ne!(tight.program_identity(), loose.program_identity());

        let a = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);
        let b = WasmTensor::new(&[1.0, 0.0], &[1, 2, 1, 1]);
        assert_close(&tight.run2(&a, &b).unwrap().to_array(), &[1.0], 1e-6);
        let replay = MathProgram::from_plan(&tight.program_plan()).unwrap();
        assert_eq!(replay.program_identity(), tight.program_identity());
        assert_close(&replay.run2(&a, &b).unwrap().to_array(), &[1.0], 1e-6);
    }

    #[test]
    fn fixed_shape_transforms_use_v3_and_replay_identically() {
        let mut builder = MathProgramBuilder::new(1, 4).unwrap();
        builder.add_reshape(0, 1, &[1, 1, 3, 2]).unwrap();
        builder.add_permute(1, 2, &[0, 1, 3, 2]).unwrap();
        builder
            .add_slice(2, 3, &[0, 0, 0, 1], &[1, 1, 2, 3])
            .unwrap();
        builder.set_output(3).unwrap();
        let program = builder.compile().unwrap();
        assert_eq!(program.program_plan()[4], PLAN_VERSION_V3);

        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]);
        let output = program.run1(&input).unwrap();
        assert_eq!(output.shape(), vec![1, 1, 2, 2]);

        let replay = MathProgram::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), program.program_plan());
        assert_eq!(replay.program_identity(), program.program_identity());
        assert_eq!(replay.run1(&input).unwrap().to_array(), output.to_array());
    }

    #[test]
    fn fixed_shape_metadata_changes_identity() {
        let mut a = MathProgramBuilder::new(1, 2).unwrap();
        a.add_reshape(0, 1, &[1, 1, 2, 3]).unwrap();
        a.set_output(1).unwrap();
        let a = a.compile().unwrap();

        let mut b = MathProgramBuilder::new(1, 2).unwrap();
        b.add_reshape(0, 1, &[1, 1, 3, 2]).unwrap();
        b.set_output(1).unwrap();
        let b = b.compile().unwrap();

        assert_eq!(a.program_plan()[4], PLAN_VERSION_V3);
        assert_ne!(a.program_identity(), b.program_identity());
    }

    #[test]
    fn invalid_shape_metadata_is_rejected_without_builder_mutation() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        assert!(builder.add_unary(OP_RESHAPE, 0, 1).is_err());
        assert!(builder.add_reshape(0, 1, &[1, 0, 2, 3]).is_err());
        assert!(builder.add_permute(0, 1, &[0, 1, 1, 3]).is_err());
        assert!(builder
            .add_slice(0, 1, &[0, 0, 1, 0], &[1, 1, 1, 1])
            .is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_reshape(0, 1, &[1, 1, 2, 3]).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn runtime_shape_failure_preserves_identity_and_program_is_reusable() {
        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_reshape(0, 1, &[1, 1, 2, 2]).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let identity = program.program_identity();

        let invalid = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        assert!(program.run1(&invalid).is_err());
        assert_eq!(program.program_identity(), identity);

        let valid = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
        let output = program.run1(&valid).unwrap();
        assert_eq!(output.shape(), vec![1, 1, 2, 2]);
        assert_eq!(program.program_identity(), identity);
    }

    #[test]
    fn invalid_scalar_parameters_are_rejected_without_builder_mutation() {
        let mut unary = MathProgramBuilder::new(1, 3).unwrap();
        assert!(unary.add_unary(OP_CLAMP, 0, 1).is_err());
        assert!(unary.add_clamp(0, 1, f32::NAN, 1.0).is_err());
        assert!(unary.add_clamp(0, 1, 2.0, 1.0).is_err());
        assert_eq!(unary.num_steps(), 0);
        unary.add_unary(OP_ABS, 0, 1).unwrap();
        assert_eq!(unary.num_steps(), 1);

        let mut binary = MathProgramBuilder::new(2, 3).unwrap();
        assert!(binary
            .add_binary(OP_COSINE_SIMILARITY, 0, 1, 2)
            .is_err());
        assert!(binary
            .add_cosine_similarity(0, 1, 2, 0.0)
            .is_err());
        assert!(binary
            .add_cosine_similarity(0, 1, 2, f32::INFINITY)
            .is_err());
        assert_eq!(binary.num_steps(), 0);
        binary.add_binary(OP_DOT, 0, 1, 2).unwrap();
        assert_eq!(binary.num_steps(), 1);
    }

    #[test]
    fn v2_plan_without_parameters_is_noncanonical() {
        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let v1 = builder.compile().unwrap().program_plan();
        assert_eq!(v1[4], PLAN_VERSION_V1);

        let mut v2 = Vec::with_capacity(PLAN_HEADER_BYTES + PLAN_STEP_BYTES_V2 + 1);
        v2.extend_from_slice(&v1[..PLAN_HEADER_BYTES]);
        v2[4] = PLAN_VERSION_V2;
        v2.extend_from_slice(&v1[PLAN_HEADER_BYTES..PLAN_HEADER_BYTES + PLAN_STEP_BYTES_V1]);
        v2.extend_from_slice(&[0u8; PLAN_STEP_BYTES_V2 - PLAN_STEP_BYTES_V1]);
        v2.push(*v1.last().unwrap());
        assert!(MathProgram::from_plan(&v2).is_err());
    }

    #[test]
    fn v3_plan_without_shape_parameters_is_noncanonical() {
        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_clamp(0, 1, -1.0, 1.0).unwrap();
        builder.set_output(1).unwrap();
        let v2 = builder.compile().unwrap().program_plan();
        assert_eq!(v2[4], PLAN_VERSION_V2);

        let mut v3 = Vec::with_capacity(PLAN_HEADER_BYTES + PLAN_STEP_BYTES_V3 + 1);
        v3.extend_from_slice(&v2[..PLAN_HEADER_BYTES]);
        v3[4] = PLAN_VERSION_V3;
        let step = &v2[PLAN_HEADER_BYTES..PLAN_HEADER_BYTES + PLAN_STEP_BYTES_V2];
        v3.extend_from_slice(&step[..6]);
        v3.extend_from_slice(&step[6..14]);
        v3.extend_from_slice(&[0u8; SHAPE_PARAM_BYTES - 8]);
        v3.push(*v2.last().unwrap());
        assert!(MathProgram::from_plan(&v3).is_err());
    }

    #[test]
    fn builder_rejects_unknown_ops_read_before_write_and_duplicate_writes() {
        let mut builder = MathProgramBuilder::new(1, 4).unwrap();
        assert!(builder.add_unary(0xFF, 0, 1).is_err());
        assert!(builder.add_unary(OP_ABS, 2, 1).is_err());
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        assert!(builder.add_unary(OP_SQRT, 1, 1).is_err());
        assert!(builder.set_output(0).is_err());
    }

    #[test]
    fn malformed_or_noncanonical_replay_is_rejected() {
        assert!(MathProgram::from_plan(&[]).is_err());

        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let mut plan = program.program_plan();
        plan[8] = 0xFF;
        assert!(MathProgram::from_plan(&plan).is_err());

        let mut truncated = program.program_plan();
        truncated.pop();
        assert!(MathProgram::from_plan(&truncated).is_err());
    }

    #[test]
    fn failed_execution_preserves_identity_and_program_is_reusable() {
        let mut builder = MathProgramBuilder::new(1, 2).unwrap();
        builder.add_unary(OP_LOG, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let identity = program.program_identity();

        let invalid = WasmTensor::new(&[0.0], &[1, 1, 1, 1]);
        assert!(program.run1(&invalid).is_err());
        assert_eq!(program.program_identity(), identity);

        let valid = WasmTensor::new(&[std::f32::consts::E], &[1, 1, 1, 1]);
        assert_close(&program.run1(&valid).unwrap().to_array(), &[1.0], 1e-6);
        assert_eq!(program.program_identity(), identity);
    }

    #[test]
    fn run_entrypoint_must_match_declared_input_count() {
        let mut unary_builder = MathProgramBuilder::new(1, 2).unwrap();
        unary_builder.add_unary(OP_ABS, 0, 1).unwrap();
        unary_builder.set_output(1).unwrap();
        let unary = unary_builder.compile().unwrap();

        let a = WasmTensor::new(&[1.0], &[1, 1, 1, 1]);
        let b = WasmTensor::new(&[2.0], &[1, 1, 1, 1]);
        assert!(unary.run2(&a, &b).is_err());

        let mut binary_builder = MathProgramBuilder::new(2, 3).unwrap();
        binary_builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
        binary_builder.set_output(2).unwrap();
        let binary = binary_builder.compile().unwrap();
        assert!(binary.run1(&a).is_err());
    }

    #[test]
    fn capabilities_document_core_boundary() {
        let caps = math_program_capabilities();
        assert!(caps.contains("burn-research.math-program.v3"));
        assert!(caps.contains("burn-research.math-program-plan.v1"));
        assert!(caps.contains("burn-research.math-program-plan.v2"));
        assert!(caps.contains("burn-research.math-program-plan.v3"));
        assert!(caps.contains("\"slot_semantics\":\"write_once_read_after_write\""));
        assert!(caps.contains("\"v1_identity_compatibility\":true"));
        assert!(caps.contains("\"v2_identity_compatibility\":true"));
        assert!(caps.contains("\"registry_dependency\":false"));
        assert!(caps.contains("\"scalar_v2\":[\"clamp\",\"cosineSimilarity\"]"));
        assert!(caps.contains("\"fixed_rank4_v3\":[\"reshape\",\"permute\",\"slice\"]"));
        assert!(caps.contains("\"variable_length\":\"deferred_selectAxis\""));
    }
}
