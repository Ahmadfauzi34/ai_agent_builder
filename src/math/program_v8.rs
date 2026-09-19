//! Math Program v8: v7 semantics plus identity-bound generic rank-4 reduction.
//!
//! Historical v1-v7 decoders remain frozen. Every non-reduction step is validated and executed
//! through a canonical one-step Math Program v7 replay. V8 therefore owns only one new semantic
//! family: explicit keepdim reduction over an identity-bound axis.

use crate::math::program_reduction_params::{ReductionAxisParams, PARAM_REDUCTION_AXIS};
use crate::math::program_step_record::{ProgramStepRecord, STEP_HEADER_BYTES};
use crate::math::program_v7::{MathProgramV7, MathProgramV7Builder};
use crate::math::reduction::TensorReduction;
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V7: u8 = 7;
const PLAN_VERSION_V8: u8 = 8;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_PLAN_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLOTS: u8 = 64;

pub const MIN_V8_EXTERNAL_INPUTS: u8 = 1;
pub const MAX_V8_EXTERNAL_INPUTS: u8 = 8;
pub const OP_SUM_AXIS: u8 = 0x60;
pub const OP_MEAN_AXIS: u8 = 0x61;
pub const OP_MIN_AXIS: u8 = 0x62;
pub const OP_MAX_AXIS: u8 = 0x63;

type RawStepRecord = ProgramStepRecord;
const STEP_CONTEXT: &str = "MathProgramV8 step";

fn raw_step_record(
    op: u8,
    arity: u8,
    in_a: u8,
    in_b: u8,
    out: u8,
    param_kind: u8,
    payload: Vec<u8>,
) -> Result<RawStepRecord, String> {
    ProgramStepRecord::new(
        op,
        arity,
        in_a,
        in_b,
        out,
        param_kind,
        payload,
        STEP_CONTEXT,
    )
}

fn decode_raw_step_prefix(bytes: &[u8]) -> Result<(RawStepRecord, usize), String> {
    ProgramStepRecord::decode_prefix(bytes, STEP_CONTEXT)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReductionOp {
    Sum,
    Mean,
    Min,
    Max,
}

fn reduction_op(op: u8) -> Option<ReductionOp> {
    match op {
        OP_SUM_AXIS => Some(ReductionOp::Sum),
        OP_MEAN_AXIS => Some(ReductionOp::Mean),
        OP_MIN_AXIS => Some(ReductionOp::Min),
        OP_MAX_AXIS => Some(ReductionOp::Max),
        _ => None,
    }
}

#[derive(Clone, Debug)]
enum ExecutableStep {
    V7 {
        program: MathProgramV7,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
    },
    Reduction {
        op: ReductionOp,
        input: u8,
        out: u8,
        axis: u32,
    },
}

#[derive(Clone, Debug)]
struct DecodedPlan {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
}

fn bit(slot: u8) -> u64 {
    1u64 << slot
}

fn initial_filled(num_inputs: u8) -> u64 {
    (1u64 << num_inputs) - 1
}

fn validate_program_shape(num_inputs: u8, num_slots: u8) -> Result<(), String> {
    if !(MIN_V8_EXTERNAL_INPUTS..=MAX_V8_EXTERNAL_INPUTS).contains(&num_inputs) {
        return Err(format!(
            "MathProgramV8: num_inputs must be in {MIN_V8_EXTERNAL_INPUTS}..={MAX_V8_EXTERNAL_INPUTS}, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgramV8: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
            num_inputs + 1
        ));
    }
    Ok(())
}

fn extract_single_v7_record(program: MathProgramV7) -> Result<RawStepRecord, String> {
    let plan = program.program_plan();
    if plan.len() < PLAN_HEADER_BYTES + STEP_HEADER_BYTES + PLAN_OUTPUT_BYTES
        || &plan[..4] != PLAN_MAGIC
        || plan[4] != PLAN_VERSION_V7
        || plan[7] != 1
    {
        return Err("MathProgramV8: expected canonical one-step v7 plan".into());
    }
    let record_bytes = &plan[PLAN_HEADER_BYTES..plan.len() - PLAN_OUTPUT_BYTES];
    let (record, consumed) = decode_raw_step_prefix(record_bytes)?;
    if consumed != record_bytes.len() {
        return Err("MathProgramV8: one-step v7 plan contained trailing record bytes".into());
    }
    Ok(record)
}

fn v7_unary_record<F>(
    global_input: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV7Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV7Builder::new(1, 2)?;
    build(&mut builder)?;
    builder.set_output(1)?;
    let mut record = extract_single_v7_record(builder.compile()?)?;
    record.in_a = global_input;
    record.in_b = 0;
    record.out = global_output;
    Ok(record)
}

fn v7_binary_record<F>(
    global_lhs: u8,
    global_rhs: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV7Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV7Builder::new(2, 3)?;
    build(&mut builder)?;
    builder.set_output(2)?;
    let mut record = extract_single_v7_record(builder.compile()?)?;
    record.in_a = global_lhs;
    record.in_b = global_rhs;
    record.out = global_output;
    Ok(record)
}

fn canonical_one_step_v7_plan(record: &RawStepRecord) -> Result<Vec<u8>, String> {
    if record.arity != 1 && record.arity != 2 {
        return Err(format!(
            "MathProgramV8: v7 delegated step has unsupported arity {}; expected 1 or 2",
            record.arity
        ));
    }
    let out = record.arity;
    let local = raw_step_record(
        record.op,
        record.arity,
        0,
        if record.arity == 2 { 1 } else { 0 },
        out,
        record.param_kind,
        record.payload.clone(),
    )?;
    let mut plan = Vec::with_capacity(PLAN_HEADER_BYTES + local.encoded_len() + PLAN_OUTPUT_BYTES);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V7);
    plan.push(record.arity);
    plan.push(record.arity + 1);
    plan.push(1);
    plan.extend_from_slice(&local.encode());
    plan.push(out);
    Ok(plan)
}

fn compile_record(record: &RawStepRecord) -> Result<ExecutableStep, String> {
    if let Some(op) = reduction_op(record.op) {
        if record.arity != 1 || record.in_b != 0 || record.param_kind != PARAM_REDUCTION_AXIS {
            return Err(
                "MathProgramV8: generic reduction requires unary arity, in_b=0, and reduction-axis parameters"
                    .into(),
            );
        }
        let params = ReductionAxisParams::decode(&record.payload)?;
        return Ok(ExecutableStep::Reduction {
            op,
            input: record.in_a,
            out: record.out,
            axis: params.axis(),
        });
    }

    let program = MathProgramV7::from_plan(&canonical_one_step_v7_plan(record)?)?;
    Ok(ExecutableStep::V7 {
        program,
        arity: record.arity,
        in_a: record.in_a,
        in_b: record.in_b,
        out: record.out,
    })
}

fn validate_topology(
    record: &RawStepRecord,
    num_slots: u8,
    filled: u64,
    written: u64,
    context: &str,
) -> Result<(u64, u64), String> {
    if record.in_a >= num_slots || record.in_b >= num_slots || record.out >= num_slots {
        return Err(format!(
            "{context}: slot out of range for num_slots={num_slots}: in_a={}, in_b={}, out={}",
            record.in_a, record.in_b, record.out
        ));
    }
    if record.arity != 1 && record.arity != 2 {
        return Err(format!(
            "{context}: unsupported arity {}; expected 1 or 2",
            record.arity
        ));
    }
    if filled & bit(record.in_a) == 0 {
        return Err(format!("{context}: input slot {} is read before write", record.in_a));
    }
    if record.arity == 2 && filled & bit(record.in_b) == 0 {
        return Err(format!(
            "{context}: second input slot {} is read before write",
            record.in_b
        ));
    }
    if record.arity == 1 && record.in_b != 0 {
        return Err(format!(
            "{context}: unary step must encode in_b=0, got {}",
            record.in_b
        ));
    }
    if filled & bit(record.out) != 0 {
        return Err(format!(
            "{context}: output slot {} is already filled; slots are write-once",
            record.out
        ));
    }
    Ok((filled | bit(record.out), written | bit(record.out)))
}

fn validate_records(
    num_inputs: u8,
    num_slots: u8,
    records: &[RawStepRecord],
    out_slot: u8,
    context: &str,
) -> Result<(), String> {
    validate_program_shape(num_inputs, num_slots)?;
    if records.is_empty() {
        return Err(format!("{context}: program must contain at least one step"));
    }
    if records.len() > MAX_STEPS {
        return Err(format!(
            "{context}: step count {} exceeds maximum {MAX_STEPS}",
            records.len()
        ));
    }

    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    for (index, record) in records.iter().enumerate() {
        compile_record(record)?;
        (filled, written) = validate_topology(
            record,
            num_slots,
            filled,
            written,
            &format!("{context} step {index}"),
        )?;
    }
    if out_slot >= num_slots {
        return Err(format!(
            "{context}: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "{context}: output slot {out_slot} must be produced by a program step"
        ));
    }
    Ok(())
}

fn encode_plan(
    num_inputs: u8,
    num_slots: u8,
    records: &[RawStepRecord],
    out_slot: u8,
) -> Result<Vec<u8>, String> {
    validate_records(num_inputs, num_slots, records, out_slot, "MathProgramV8")?;
    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV8: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV8: plan length overflow".to_string())?;
    if plan_len > MAX_PLAN_BYTES {
        return Err(format!(
            "MathProgramV8: plan size {plan_len} exceeds maximum {MAX_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V8);
    plan.push(num_inputs);
    plan.push(num_slots);
    plan.push(records.len() as u8);
    for record in records {
        plan.extend_from_slice(&record.encode());
    }
    plan.push(out_slot);
    Ok(plan)
}

fn decode_plan(plan: &[u8]) -> Result<DecodedPlan, String> {
    if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
        return Err("MathProgramV8: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgramV8: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION_V8 {
        return Err(format!(
            "MathProgramV8: expected plan version {PLAN_VERSION_V8}, got {}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgramV8: plan must contain at least one step".into());
    }

    let mut records = Vec::with_capacity(num_steps);
    let mut steps = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;

    for index in 0..num_steps {
        if offset >= plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err(format!("MathProgramV8: plan is truncated before step {index}"));
        }
        let (record, consumed) = decode_raw_step_prefix(&plan[offset..])?;
        let step = compile_record(&record)?;
        (filled, written) = validate_topology(
            &record,
            num_slots,
            filled,
            written,
            &format!("MathProgramV8 replay step {index}"),
        )?;
        records.push(record);
        steps.push(step);
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| "MathProgramV8: plan offset overflow".to_string())?;
    }

    if offset + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "MathProgramV8: malformed plan length: expected output byte at offset {offset}, got {} total bytes",
            plan.len()
        ));
    }
    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgramV8: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgramV8: output slot {out_slot} must be produced by a program step"
        ));
    }

    Ok(DecodedPlan {
        num_inputs,
        num_slots,
        records,
        steps,
        out_slot,
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn math_program_v8_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v8\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v8\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"min_inputs\":1,",
        "\"max_inputs\":8,",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"legacy_step_validation\":\"canonical_one_step_v7_replay\",",
        "\"generic_reduction_ops\":[\"sumAxis\",\"meanAxis\",\"minAxis\",\"maxAxis\"],",
        "\"reduction_axis_binding\":\"identity_bound_plan_metadata\",",
        "\"reduction_keepdim_rank4\":true,",
        "\"zero_sized_dimensions\":false,",
        "\"implicit_broadcasting\":false,",
        "\"legacy_v1_v7_decoders_frozen\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV8Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV8Builder {
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<Self, String> {
        validate_program_shape(num_inputs, num_slots)?;
        Ok(Self {
            num_inputs,
            num_slots,
            records: Vec::new(),
            filled: initial_filled(num_inputs),
            written: 0,
            out_slot: None,
        })
    }

    fn push_record(&mut self, record: RawStepRecord, context: &str) -> Result<(), String> {
        if self.records.len() >= MAX_STEPS {
            return Err(format!("{context}: step count exceeds maximum {MAX_STEPS}"));
        }
        compile_record(&record)?;
        let (filled, written) = validate_topology(
            &record,
            self.num_slots,
            self.filled,
            self.written,
            context,
        )?;
        self.records.push(record);
        self.filled = filled;
        self.written = written;
        Ok(())
    }

    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| builder.add_unary(op, 0, 1))?;
        self.push_record(record, "MathProgramV8Builder.addUnary")
    }

    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        let record = v7_binary_record(lhs, rhs, output, |builder| {
            builder.add_binary(op, 0, 1, 2)
        })?;
        self.push_record(record, "MathProgramV8Builder.addBinary")
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| {
            builder.add_clamp(0, 1, min, max)
        })?;
        self.push_record(record, "MathProgramV8Builder.addClamp")
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        let record = v7_binary_record(lhs, rhs, output, |builder| {
            builder.add_cosine_similarity(0, 1, 2, epsilon)
        })?;
        self.push_record(record, "MathProgramV8Builder.addCosineSimilarity")
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| {
            builder.add_reshape(0, 1, shape)
        })?;
        self.push_record(record, "MathProgramV8Builder.addReshape")
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| {
            builder.add_permute(0, 1, axes)
        })?;
        self.push_record(record, "MathProgramV8Builder.addPermute")
    }

    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| {
            builder.add_slice(0, 1, starts, ends)
        })?;
        self.push_record(record, "MathProgramV8Builder.addSlice")
    }

    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        let record = v7_unary_record(input, output, |builder| {
            builder.add_select_axis(0, 1, axis, indices)
        })?;
        self.push_record(record, "MathProgramV8Builder.addSelectAxis")
    }

    pub fn add_fill_like(
        &mut self,
        reference: u8,
        output: u8,
        scalar: f32,
    ) -> Result<(), String> {
        let record = v7_unary_record(reference, output, |builder| {
            builder.add_fill_like(0, 1, scalar)
        })?;
        self.push_record(record, "MathProgramV8Builder.addFillLike")
    }

    pub fn add_expand_like(
        &mut self,
        source: u8,
        reference: u8,
        output: u8,
    ) -> Result<(), String> {
        let record = v7_binary_record(source, reference, output, |builder| {
            builder.add_expand_like(0, 1, 2)
        })?;
        self.push_record(record, "MathProgramV8Builder.addExpandLike")
    }

    fn add_reduction(
        &mut self,
        op: u8,
        input: u8,
        output: u8,
        axis: u32,
        context: &str,
    ) -> Result<(), String> {
        let params = ReductionAxisParams::new(axis)?;
        self.push_record(
            raw_step_record(
                op,
                1,
                input,
                0,
                output,
                PARAM_REDUCTION_AXIS,
                params.encode().to_vec(),
            )?,
            context,
        )
    }

    pub fn add_sum_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.add_reduction(OP_SUM_AXIS, input, output, axis, "MathProgramV8Builder.addSumAxis")
    }

    pub fn add_mean_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.add_reduction(OP_MEAN_AXIS, input, output, axis, "MathProgramV8Builder.addMeanAxis")
    }

    pub fn add_min_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.add_reduction(OP_MIN_AXIS, input, output, axis, "MathProgramV8Builder.addMinAxis")
    }

    pub fn add_max_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.add_reduction(OP_MAX_AXIS, input, output, axis, "MathProgramV8Builder.addMaxAxis")
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV8Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV8Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV8, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV8Builder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.records, out_slot)?;
        let steps = self
            .records
            .iter()
            .map(compile_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MathProgramV8 {
            num_inputs: self.num_inputs,
            num_slots: self.num_slots,
            records: self.records.clone(),
            steps,
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
        self.records.len()
    }
}

#[derive(Clone, Debug)]
pub struct MathProgramV8 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV8 {
    pub fn from_plan(plan: &[u8]) -> Result<Self, String> {
        let DecodedPlan {
            num_inputs,
            num_slots,
            records,
            steps,
            out_slot,
        } = decode_plan(plan)?;
        let canonical_plan = encode_plan(num_inputs, num_slots, &records, out_slot)?;
        if canonical_plan != plan {
            return Err("MathProgramV8: replay plan is not canonical".into());
        }
        Ok(Self {
            num_inputs,
            num_slots,
            records,
            steps,
            out_slot,
            canonical_plan,
        })
    }

    pub fn run_inputs(&self, inputs: &[WasmTensor]) -> Result<WasmTensor, String> {
        if inputs.len() != self.num_inputs as usize {
            return Err(format!(
                "MathProgramV8.runInputs: expected {} inputs, got {}",
                self.num_inputs,
                inputs.len()
            ));
        }

        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (index, input) in inputs.iter().enumerate() {
            slots[index] = Some(input.clone());
        }
        let reduction = TensorReduction::new();

        for (index, step) in self.steps.iter().enumerate() {
            let (out, value) = match step {
                ExecutableStep::V7 {
                    program,
                    arity,
                    in_a,
                    in_b,
                    out,
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV8.runInputs: step {index} input slot {in_a} is empty"
                        )
                    })?;
                    let value = if *arity == 1 {
                        program.run_inputs(&[a.clone()])?
                    } else {
                        let b = slots[*in_b as usize].as_ref().ok_or_else(|| {
                            format!(
                                "MathProgramV8.runInputs: step {index} second input slot {in_b} is empty"
                            )
                        })?;
                        program.run_inputs(&[a.clone(), b.clone()])?
                    };
                    (*out, value)
                }
                ExecutableStep::Reduction {
                    op,
                    input,
                    out,
                    axis,
                } => {
                    let input_value = slots[*input as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV8.runInputs: step {index} input slot {input} is empty"
                        )
                    })?;
                    let value = match op {
                        ReductionOp::Sum => reduction.sum_axis(input_value, *axis)?,
                        ReductionOp::Mean => reduction.mean_axis(input_value, *axis)?,
                        ReductionOp::Min => reduction.min_axis(input_value, *axis)?,
                        ReductionOp::Max => reduction.max_axis(input_value, *axis)?,
                    };
                    (*out, value)
                }
            };
            slots[out as usize] = Some(value);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV8.runInputs: output slot {} is empty", self.out_slot))
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
        self.records.len()
    }

    pub fn out_slot(&self) -> u8 {
        self.out_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::program::{OP_ADD, OP_DIV, OP_EXP, OP_MUL, OP_SUB};

    fn tensor(values: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor::new(values, shape)
    }

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
    fn stable_softmax_axis2_is_fully_expressible_inside_v8() {
        let mut builder = MathProgramV8Builder::new(1, 8).unwrap();
        builder.add_max_axis(0, 1, 2).unwrap();
        builder.add_expand_like(1, 0, 2).unwrap();
        builder.add_binary(OP_SUB, 0, 2, 3).unwrap();
        builder.add_unary(OP_EXP, 3, 4).unwrap();
        builder.add_sum_axis(4, 5, 2).unwrap();
        builder.add_expand_like(5, 4, 6).unwrap();
        builder.add_binary(OP_DIV, 4, 6, 7).unwrap();
        builder.set_output(7).unwrap();
        let program = builder.compile().unwrap();

        let input = tensor(&[1000.0, 1001.0, 1002.0, 1.0, 2.0, 3.0], &[1, 2, 3, 1]);
        let output = program.run_inputs(&[input]).unwrap();
        let values = output.to_array();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values[0..3].iter().sum::<f32>() - 1.0).abs() <= 3e-6);
        assert!((values[3..6].iter().sum::<f32>() - 1.0).abs() <= 3e-6);
        assert_close(&values[0..3], &values[3..6], 2e-6);
    }

    #[test]
    fn reduction_family_and_keepdim_shapes_are_available() {
        let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2, 1]);

        let mut mean_builder = MathProgramV8Builder::new(1, 2).unwrap();
        mean_builder.add_mean_axis(0, 1, 0).unwrap();
        mean_builder.set_output(1).unwrap();
        let mean = mean_builder.compile().unwrap().run_inputs(&[input.clone()]).unwrap();
        assert_eq!(mean.shape(), vec![1, 2, 2, 1]);
        assert_close(&mean.to_array(), &[3.0, 4.0, 5.0, 6.0], 1e-6);

        let mut min_builder = MathProgramV8Builder::new(1, 2).unwrap();
        min_builder.add_min_axis(0, 1, 1).unwrap();
        min_builder.set_output(1).unwrap();
        let min = min_builder.compile().unwrap().run_inputs(&[input.clone()]).unwrap();
        assert_eq!(min.shape(), vec![2, 1, 2, 1]);
        assert_eq!(min.to_array(), vec![1.0, 2.0, 5.0, 6.0]);

        let mut max_builder = MathProgramV8Builder::new(1, 2).unwrap();
        max_builder.add_max_axis(0, 1, 2).unwrap();
        max_builder.set_output(1).unwrap();
        let max = max_builder.compile().unwrap().run_inputs(&[input.clone()]).unwrap();
        assert_eq!(max.shape(), vec![2, 2, 1, 1]);
        assert_eq!(max.to_array(), vec![2.0, 4.0, 6.0, 8.0]);

        let mut sum_builder = MathProgramV8Builder::new(1, 2).unwrap();
        sum_builder.add_sum_axis(0, 1, 3).unwrap();
        sum_builder.set_output(1).unwrap();
        let sum = sum_builder.compile().unwrap().run_inputs(&[input]).unwrap();
        assert_eq!(sum.shape(), vec![2, 2, 2, 1]);
        assert_eq!(sum.to_array(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn reduction_axis_is_identity_bound_and_replay_stable() {
        let mut a = MathProgramV8Builder::new(1, 2).unwrap();
        a.add_sum_axis(0, 1, 0).unwrap();
        a.set_output(1).unwrap();
        let a = a.compile().unwrap();

        let mut b = MathProgramV8Builder::new(1, 2).unwrap();
        b.add_sum_axis(0, 1, 1).unwrap();
        b.set_output(1).unwrap();
        let b = b.compile().unwrap();
        assert_ne!(a.program_identity(), b.program_identity());

        let replay = MathProgramV8::from_plan(&a.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), a.program_plan());
        assert_eq!(replay.program_identity(), a.program_identity());
    }

    #[test]
    fn invalid_axis_construction_is_atomic() {
        let mut builder = MathProgramV8Builder::new(1, 3).unwrap();
        assert_eq!(builder.num_steps(), 0);
        assert!(builder.add_sum_axis(0, 1, 4).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_fill_like(0, 1, 2.0).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn runtime_failure_is_controlled_and_program_reusable() {
        let mut builder = MathProgramV8Builder::new(1, 2).unwrap();
        builder.add_max_axis(0, 1, 1).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let identity = program.program_identity();

        let bad = tensor(&[1.0, f32::INFINITY], &[1, 2, 1, 1]);
        assert!(program.run_inputs(&[bad]).is_err());
        assert_eq!(program.program_identity(), identity);

        let good = tensor(&[1.0, 3.0], &[1, 2, 1, 1]);
        assert_eq!(program.run_inputs(&[good]).unwrap().to_array(), vec![3.0]);
        assert_eq!(program.program_identity(), identity);
    }

    #[test]
    fn v7_fill_and_expand_semantics_are_reused_inside_v8() {
        let mut builder = MathProgramV8Builder::new(1, 6).unwrap();
        builder.add_fill_like(0, 1, 0.5).unwrap();
        builder.add_binary(OP_MUL, 0, 1, 2).unwrap();
        builder.add_fill_like(0, 3, 3.0).unwrap();
        builder.add_binary(OP_ADD, 2, 3, 4).unwrap();
        builder.add_sum_axis(4, 5, 1).unwrap();
        builder.set_output(5).unwrap();
        let program = builder.compile().unwrap();
        let input = tensor(&[2.0, 4.0, 6.0], &[1, 3, 1, 1]);
        assert_close(&program.run_inputs(&[input]).unwrap().to_array(), &[15.0], 1e-6);
    }

    #[test]
    fn versions_remain_isolated() {
        let mut builder = MathProgramV8Builder::new(1, 2).unwrap();
        builder.add_sum_axis(0, 1, 1).unwrap();
        builder.set_output(1).unwrap();
        let v8 = builder.compile().unwrap();
        assert!(MathProgramV7::from_plan(&v8.program_plan()).is_err());

        let mut v7_builder = MathProgramV7Builder::new(2, 3).unwrap();
        v7_builder.add_expand_like(0, 1, 2).unwrap();
        v7_builder.set_output(2).unwrap();
        let v7 = v7_builder.compile().unwrap();
        assert!(MathProgramV8::from_plan(&v7.program_plan()).is_err());
    }

    #[test]
    fn malformed_reduction_axis_record_fails_closed() {
        let mut builder = MathProgramV8Builder::new(1, 2).unwrap();
        builder.add_sum_axis(0, 1, 1).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let mut plan = program.program_plan();
        let payload_offset = PLAN_HEADER_BYTES + STEP_HEADER_BYTES;
        plan[payload_offset] = 4;
        assert!(MathProgramV8::from_plan(&plan).is_err());
    }

    #[test]
    fn implicit_broadcasting_remains_rejected() {
        let mut builder = MathProgramV8Builder::new(2, 3).unwrap();
        builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let vector = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let scalar = tensor(&[10.0], &[1, 1, 1, 1]);
        assert!(program.run_inputs(&[vector, scalar]).is_err());
    }

    #[test]
    fn capabilities_describe_only_the_new_reduction_semantics() {
        let caps = math_program_v8_capabilities();
        assert!(caps.contains("burn-research.math-program.v8"));
        assert!(caps.contains("canonical_one_step_v7_replay"));
        assert!(caps.contains("identity_bound_plan_metadata"));
        assert!(caps.contains("\"reduction_keepdim_rank4\":true"));
        assert!(caps.contains("\"legacy_v1_v7_decoders_frozen\":true"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
    }
}
