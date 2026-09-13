use crate::math::{
    WasmLinearAlgebra, WasmNumericKernel, WasmProbability, WasmStatistics, WasmTensorTransform,
};
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION: u8 = 1;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_STEP_BYTES: usize = 5;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_SLOTS: u8 = 64;
const MAX_STEPS: usize = u8::MAX as usize;

// Numeric Kernel v1.
pub const OP_ABS: u8 = 0x01;
pub const OP_SQRT: u8 = 0x02;
pub const OP_EXP: u8 = 0x03;
pub const OP_LOG: u8 = 0x04;
pub const OP_ADD: u8 = 0x11;
pub const OP_SUB: u8 = 0x12;
pub const OP_MUL: u8 = 0x13;
pub const OP_DIV: u8 = 0x14;

// Tensor Transform v1. Parameterized transforms are intentionally deferred.
pub const OP_TRANSPOSE: u8 = 0x20;

// Linear Algebra v1. Cosine similarity is deferred because epsilon is parameterized.
pub const OP_L2_NORM: u8 = 0x30;
pub const OP_DOT: u8 = 0x31;
pub const OP_L2_DISTANCE: u8 = 0x32;
pub const OP_MATMUL: u8 = 0x33;

// Statistics v1.
pub const OP_SUM: u8 = 0x40;
pub const OP_MEAN: u8 = 0x41;
pub const OP_VARIANCE_POPULATION: u8 = 0x42;
pub const OP_STD_POPULATION: u8 = 0x43;
pub const OP_MIN: u8 = 0x44;
pub const OP_MAX: u8 = 0x45;

// Probability v1.
pub const OP_NORMALIZE: u8 = 0x50;
pub const OP_ENTROPY: u8 = 0x51;
pub const OP_CROSS_ENTROPY: u8 = 0x52;
pub const OP_KL_DIVERGENCE: u8 = 0x53;

const ARITY_UNARY: u8 = 1;
const ARITY_BINARY: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MathStep {
    op: u8,
    arity: u8,
    in_a: u8,
    in_b: u8,
    out: u8,
}

fn expected_arity(op: u8) -> Option<u8> {
    match op {
        OP_ABS
        | OP_SQRT
        | OP_EXP
        | OP_LOG
        | OP_TRANSPOSE
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
        OP_ADD => "add",
        OP_SUB => "sub",
        OP_MUL => "mul",
        OP_DIV => "div",
        OP_TRANSPOSE => "transpose",
        OP_L2_NORM => "l2Norm",
        OP_DOT => "dot",
        OP_L2_DISTANCE => "l2Distance",
        OP_MATMUL => "matmul",
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
            "MathProgram: num_inputs must be 1 or 2 in v1, got {num_inputs}"
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

fn validate_step(
    step: MathStep,
    num_slots: u8,
    filled: u64,
    written: u64,
    context: &str,
) -> Result<(u64, u64), String> {
    let expected = expected_arity(step.op).ok_or_else(|| {
        format!(
            "{context}: unknown opcode 0x{:02X}; Math Program v1 is fail-closed",
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
            "{context}: output slot {} is already filled; Math Program v1 slots are write-once",
            step.out
        ));
    }

    Ok((filled | bit(step.out), written | bit(step.out)))
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
            "MathProgram: step count {} exceeds v1 maximum {MAX_STEPS}",
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

    let mut plan = Vec::with_capacity(
        PLAN_HEADER_BYTES + steps.len() * PLAN_STEP_BYTES + PLAN_OUTPUT_BYTES,
    );
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION);
    plan.push(num_inputs);
    plan.push(num_slots);
    plan.push(steps.len() as u8);
    for step in steps {
        plan.push(step.op);
        plan.push(step.arity);
        plan.push(step.in_a);
        plan.push(step.in_b);
        plan.push(step.out);
    }
    plan.push(out_slot);
    Ok(plan)
}

fn decode_plan(plan: &[u8]) -> Result<(u8, u8, Vec<MathStep>, u8), String> {
    if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
        return Err("MathProgram: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgram: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION {
        return Err(format!(
            "MathProgram: unsupported plan version {}, expected {PLAN_VERSION}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgram: plan must contain at least one step".into());
    }
    let expected_len = PLAN_HEADER_BYTES
        .checked_add(num_steps.checked_mul(PLAN_STEP_BYTES).ok_or_else(|| {
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
        let step = MathStep {
            op: plan[offset],
            arity: plan[offset + 1],
            in_a: plan[offset + 2],
            in_b: plan[offset + 3],
            out: plan[offset + 4],
        };
        (filled, written) = validate_step(
            step,
            num_slots,
            filled,
            written,
            &format!("MathProgram replay step {index}"),
        )?;
        steps.push(step);
        offset += PLAN_STEP_BYTES;
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
        "\"schema\":\"burn-research.math-program.v1\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v1\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"inputs\":\"one_or_two\",",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"replay\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false,",
        "\"parameterized_ops\":\"deferred_v1\",",
        "\"opcodes\":{",
        "\"abs\":1,\"sqrt\":2,\"exp\":3,\"log\":4,",
        "\"add\":17,\"sub\":18,\"mul\":19,\"div\":20,",
        "\"transpose\":32,",
        "\"l2Norm\":48,\"dot\":49,\"l2Distance\":50,\"matmul\":51,",
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

    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        if self.steps.len() >= MAX_STEPS {
            return Err(format!(
                "MathProgramBuilder.addUnary: step count exceeds v1 maximum {MAX_STEPS}"
            ));
        }
        let step = MathStep {
            op,
            arity: ARITY_UNARY,
            in_a: input,
            in_b: 0,
            out: output,
        };
        let (filled, written) = validate_step(
            step,
            self.num_slots,
            self.filled,
            self.written,
            "MathProgramBuilder.addUnary",
        )?;
        self.steps.push(step);
        self.filled = filled;
        self.written = written;
        Ok(())
    }

    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        if self.steps.len() >= MAX_STEPS {
            return Err(format!(
                "MathProgramBuilder.addBinary: step count exceeds v1 maximum {MAX_STEPS}"
            ));
        }
        let step = MathStep {
            op,
            arity: ARITY_BINARY,
            in_a: lhs,
            in_b: rhs,
            out: output,
        };
        let (filled, written) = validate_step(
            step,
            self.num_slots,
            self.filled,
            self.written,
            "MathProgramBuilder.addBinary",
        )?;
        self.steps.push(step);
        self.filled = filled;
        self.written = written;
        Ok(())
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
                    OP_TRANSPOSE => Ok(tensor.transpose(a)),
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
    fn canonical_plan_replays_same_identity_and_behavior() {
        let mut builder = MathProgramBuilder::new(1, 3).unwrap();
        builder.add_unary(OP_ABS, 0, 1).unwrap();
        builder.add_unary(OP_SUM, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
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
        assert!(caps.contains("burn-research.math-program.v1"));
        assert!(caps.contains("\"slot_semantics\":\"write_once_read_after_write\""));
        assert!(caps.contains("\"registry_dependency\":false"));
        assert!(caps.contains("\"parameterized_ops\":\"deferred_v1\""));
    }
}
