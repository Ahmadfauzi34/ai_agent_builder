//! Math Program v7: v6 semantics plus explicit runtime `expandLike`.
//!
//! Historical v1-v6 decoders remain frozen. Every non-`expandLike` step is validated and executed
//! through a canonical one-step Math Program v6 replay, so this version introduces exactly one new
//! execution semantic: explicit singleton expansion to a reference tensor shape.

use crate::math::program_v6::{MathProgramV6, MathProgramV6Builder};
use crate::math::program_runtime_shape::expand_like;
use crate::math::program_step_record::ProgramStepRecord;
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V6: u8 = 6;
const PLAN_VERSION_V7: u8 = 7;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_PLAN_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLOTS: u8 = 64;
const PARAM_NONE: u8 = 0;

pub const MIN_V7_EXTERNAL_INPUTS: u8 = 1;
pub const MAX_V7_EXTERNAL_INPUTS: u8 = 8;
pub const OP_EXPAND_LIKE: u8 = 0x26;

type RawStepRecord = ProgramStepRecord;
const STEP_CONTEXT: &str = "MathProgramV7 step";

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

#[derive(Clone, Debug)]
enum ExecutableStep {
    V6 {
        program: MathProgramV6,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
    },
    ExpandLike {
        source: u8,
        reference: u8,
        out: u8,
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
    if !(MIN_V7_EXTERNAL_INPUTS..=MAX_V7_EXTERNAL_INPUTS).contains(&num_inputs) {
        return Err(format!(
            "MathProgramV7: num_inputs must be in {MIN_V7_EXTERNAL_INPUTS}..={MAX_V7_EXTERNAL_INPUTS}, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgramV7: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
            num_inputs + 1
        ));
    }
    Ok(())
}

fn extract_single_v6_record(program: MathProgramV6) -> Result<RawStepRecord, String> {
    let plan = program.program_plan();
    if plan.len() < PLAN_HEADER_BYTES + STEP_HEADER_BYTES + PLAN_OUTPUT_BYTES
        || &plan[..4] != PLAN_MAGIC
        || plan[4] != PLAN_VERSION_V6
        || plan[7] != 1
    {
        return Err("MathProgramV7: expected canonical one-step v6 plan".into());
    }
    let record_bytes = &plan[PLAN_HEADER_BYTES..plan.len() - PLAN_OUTPUT_BYTES];
    let (record, consumed) = decode_raw_step_prefix(record_bytes)?;
    if consumed != record_bytes.len() {
        return Err("MathProgramV7: one-step v6 plan contained trailing record bytes".into());
    }
    Ok(record)
}

fn v6_unary_record<F>(
    global_input: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV6Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV6Builder::new(1, 2)?;
    build(&mut builder)?;
    builder.set_output(1)?;
    let mut record = extract_single_v6_record(builder.compile()?)?;
    record.in_a = global_input;
    record.in_b = 0;
    record.out = global_output;
    Ok(record)
}

fn v6_binary_record<F>(
    global_lhs: u8,
    global_rhs: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV6Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV6Builder::new(2, 3)?;
    build(&mut builder)?;
    builder.set_output(2)?;
    let mut record = extract_single_v6_record(builder.compile()?)?;
    record.in_a = global_lhs;
    record.in_b = global_rhs;
    record.out = global_output;
    Ok(record)
}

fn canonical_one_step_v6_plan(record: &RawStepRecord) -> Result<Vec<u8>, String> {
    if record.arity != 1 && record.arity != 2 {
        return Err(format!(
            "MathProgramV7: v6 delegated step has unsupported arity {}; expected 1 or 2",
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
    plan.push(PLAN_VERSION_V6);
    plan.push(record.arity);
    plan.push(record.arity + 1);
    plan.push(1);
    plan.extend_from_slice(&local.encode());
    plan.push(out);
    Ok(plan)
}

fn compile_record(record: &RawStepRecord) -> Result<ExecutableStep, String> {
    if record.op == OP_EXPAND_LIKE {
        if record.arity != 2
            || record.param_kind != PARAM_NONE
            || !record.payload.is_empty()
        {
            return Err(
                "MathProgramV7: expandLike requires binary arity and empty plain parameters"
                    .into(),
            );
        }
        return Ok(ExecutableStep::ExpandLike {
            source: record.in_a,
            reference: record.in_b,
            out: record.out,
        });
    }

    let program = MathProgramV6::from_plan(&canonical_one_step_v6_plan(record)?)?;
    Ok(ExecutableStep::V6 {
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
        return Err(format!(
            "{context}: input slot {} is read before write",
            record.in_a
        ));
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
    validate_records(num_inputs, num_slots, records, out_slot, "MathProgramV7")?;
    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV7: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV7: plan length overflow".to_string())?;
    if plan_len > MAX_PLAN_BYTES {
        return Err(format!(
            "MathProgramV7: plan size {plan_len} exceeds maximum {MAX_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V7);
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
        return Err("MathProgramV7: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgramV7: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION_V7 {
        return Err(format!(
            "MathProgramV7: expected plan version {PLAN_VERSION_V7}, got {}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgramV7: plan must contain at least one step".into());
    }

    let mut records = Vec::with_capacity(num_steps);
    let mut steps = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;

    for index in 0..num_steps {
        if offset >= plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err(format!(
                "MathProgramV7: plan is truncated before step {index}"
            ));
        }
        let (record, consumed) = decode_raw_step_prefix(&plan[offset..])?;
        let step = compile_record(&record)?;
        (filled, written) = validate_topology(
            &record,
            num_slots,
            filled,
            written,
            &format!("MathProgramV7 replay step {index}"),
        )?;
        records.push(record);
        steps.push(step);
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| "MathProgramV7: plan offset overflow".to_string())?;
    }

    if offset + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "MathProgramV7: malformed plan length: expected output byte at offset {offset}, got {} total bytes",
            plan.len()
        ));
    }
    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgramV7: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgramV7: output slot {out_slot} must be produced by a program step"
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

pub fn math_program_v7_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v7\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v7\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"min_inputs\":1,",
        "\"max_inputs\":8,",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"legacy_step_validation\":\"canonical_one_step_v6_replay\",",
        "\"runtime_shape_ops\":[\"expandLike\"],",
        "\"expand_like_rule\":\"per_axis_equal_or_singleton\",",
        "\"zero_sized_dimensions\":false,",
        "\"implicit_broadcasting\":false,",
        "\"legacy_v1_v6_decoders_frozen\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV7Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV7Builder {
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
        let record = v6_unary_record(input, output, |builder| builder.add_unary(op, 0, 1))?;
        self.push_record(record, "MathProgramV7Builder.addUnary")
    }

    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        let record = v6_binary_record(lhs, rhs, output, |builder| {
            builder.add_binary(op, 0, 1, 2)
        })?;
        self.push_record(record, "MathProgramV7Builder.addBinary")
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        let record = v6_unary_record(input, output, |builder| {
            builder.add_clamp(0, 1, min, max)
        })?;
        self.push_record(record, "MathProgramV7Builder.addClamp")
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        let record = v6_binary_record(lhs, rhs, output, |builder| {
            builder.add_cosine_similarity(0, 1, 2, epsilon)
        })?;
        self.push_record(record, "MathProgramV7Builder.addCosineSimilarity")
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let record = v6_unary_record(input, output, |builder| {
            builder.add_reshape(0, 1, shape)
        })?;
        self.push_record(record, "MathProgramV7Builder.addReshape")
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let record = v6_unary_record(input, output, |builder| {
            builder.add_permute(0, 1, axes)
        })?;
        self.push_record(record, "MathProgramV7Builder.addPermute")
    }

    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        let record = v6_unary_record(input, output, |builder| {
            builder.add_slice(0, 1, starts, ends)
        })?;
        self.push_record(record, "MathProgramV7Builder.addSlice")
    }

    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        let record = v6_unary_record(input, output, |builder| {
            builder.add_select_axis(0, 1, axis, indices)
        })?;
        self.push_record(record, "MathProgramV7Builder.addSelectAxis")
    }

    pub fn add_fill_like(
        &mut self,
        reference: u8,
        output: u8,
        scalar: f32,
    ) -> Result<(), String> {
        let record = v6_unary_record(reference, output, |builder| {
            builder.add_fill_like(0, 1, scalar)
        })?;
        self.push_record(record, "MathProgramV7Builder.addFillLike")
    }

    pub fn add_expand_like(
        &mut self,
        source: u8,
        reference: u8,
        output: u8,
    ) -> Result<(), String> {
        self.push_record(
            raw_step_record(
                OP_EXPAND_LIKE,
                2,
                source,
                reference,
                output,
                PARAM_NONE,
                vec![],
            )?,
            "MathProgramV7Builder.addExpandLike",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV7Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV7Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV7, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV7Builder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.records, out_slot)?;
        let steps = self
            .records
            .iter()
            .map(compile_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MathProgramV7 {
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
pub struct MathProgramV7 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV7 {
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
            return Err("MathProgramV7: replay plan is not canonical".into());
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
                "MathProgramV7.runInputs: expected {} inputs, got {}",
                self.num_inputs,
                inputs.len()
            ));
        }

        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (index, input) in inputs.iter().enumerate() {
            slots[index] = Some(input.clone());
        }

        for (index, step) in self.steps.iter().enumerate() {
            let (out, value) = match step {
                ExecutableStep::V6 {
                    program,
                    arity,
                    in_a,
                    in_b,
                    out,
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV7.runInputs: step {index} input slot {in_a} is empty"
                        )
                    })?;
                    let value = if *arity == 1 {
                        program.run_inputs(&[a.clone()])?
                    } else {
                        let b = slots[*in_b as usize].as_ref().ok_or_else(|| {
                            format!(
                                "MathProgramV7.runInputs: step {index} second input slot {in_b} is empty"
                            )
                        })?;
                        program.run_inputs(&[a.clone(), b.clone()])?
                    };
                    (*out, value)
                }
                ExecutableStep::ExpandLike {
                    source,
                    reference,
                    out,
                } => {
                    let source_value = slots[*source as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV7.runInputs: step {index} source slot {source} is empty"
                        )
                    })?;
                    let reference_value = slots[*reference as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV7.runInputs: step {index} reference slot {reference} is empty"
                        )
                    })?;
                    (*out, expand_like(source_value, reference_value)?)
                }
            };
            slots[out as usize] = Some(value);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV7.runInputs: output slot {} is empty", self.out_slot))
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
    use crate::math::program::{OP_ADD, OP_DIV, OP_EXP, OP_SUM};

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
    fn decomposed_feature_softmax_is_expressible_without_implicit_broadcasting() {
        let mut builder = MathProgramV7Builder::new(1, 5).unwrap();
        builder.add_unary(OP_EXP, 0, 1).unwrap();
        builder.add_unary(OP_SUM, 1, 2).unwrap();
        builder.add_expand_like(2, 1, 3).unwrap();
        builder.add_binary(OP_DIV, 1, 3, 4).unwrap();
        builder.set_output(4).unwrap();
        let program = builder.compile().unwrap();

        let input = tensor(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3, 1, 1]);
        let output = program.run_inputs(&[input]).unwrap();
        let e1 = 1.0f32.exp();
        let e2 = 2.0f32.exp();
        let e3 = 3.0f32.exp();
        let sum = e1 + e2 + e3;
        let expected = [e1 / sum, e2 / sum, e3 / sum, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        assert_close(&output.to_array(), &expected, 2e-6);
        let values = output.to_array();
        assert!((values[0..3].iter().sum::<f32>() - 1.0).abs() <= 2e-6);
        assert!((values[3..6].iter().sum::<f32>() - 1.0).abs() <= 2e-6);
    }

    #[test]
    fn expand_like_runtime_failure_is_controlled_and_program_reusable() {
        let mut builder = MathProgramV7Builder::new(2, 3).unwrap();
        builder.add_expand_like(0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let identity = program.program_identity();

        let bad_source = tensor(&[1.0, 2.0], &[1, 2, 1, 1]);
        let reference = tensor(&[0.0, 0.0, 0.0], &[1, 3, 1, 1]);
        assert!(program.run_inputs(&[bad_source, reference.clone()]).is_err());
        assert_eq!(program.program_identity(), identity);

        let good_source = tensor(&[4.0], &[1, 1, 1, 1]);
        let output = program.run_inputs(&[good_source, reference]).unwrap();
        assert_eq!(output.to_array(), vec![4.0, 4.0, 4.0]);
        assert_eq!(program.program_identity(), identity);
    }

    #[test]
    fn construction_failure_is_atomic() {
        let mut builder = MathProgramV7Builder::new(1, 3).unwrap();
        assert_eq!(builder.num_steps(), 0);
        assert!(builder.add_expand_like(0, 2, 1).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_fill_like(0, 1, 2.0).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn v6_fill_like_semantics_are_reused_inside_v7() {
        let mut builder = MathProgramV7Builder::new(1, 2).unwrap();
        builder.add_fill_like(0, 1, 2.5).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let input = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let output = program.run_inputs(&[input]).unwrap();
        assert_eq!(output.to_array(), vec![2.5, 2.5, 2.5]);
    }

    #[test]
    fn implicit_broadcasting_remains_rejected() {
        let mut builder = MathProgramV7Builder::new(2, 3).unwrap();
        builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let vector = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let scalar = tensor(&[10.0], &[1, 1, 1, 1]);
        assert!(program.run_inputs(&[vector, scalar]).is_err());
    }

    #[test]
    fn replay_identity_is_stable_and_versions_remain_isolated() {
        let mut builder = MathProgramV7Builder::new(2, 3).unwrap();
        builder.add_expand_like(0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let replay = MathProgramV7::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), program.program_plan());
        assert_eq!(replay.program_identity(), program.program_identity());
        assert!(MathProgramV6::from_plan(&program.program_plan()).is_err());

        let mut v6_builder = MathProgramV6Builder::new(1, 2).unwrap();
        v6_builder.add_fill_like(0, 1, 1.0).unwrap();
        v6_builder.set_output(1).unwrap();
        let v6 = v6_builder.compile().unwrap();
        assert!(MathProgramV7::from_plan(&v6.program_plan()).is_err());
    }

    #[test]
    fn malformed_expand_like_record_fails_closed() {
        let mut builder = MathProgramV7Builder::new(2, 3).unwrap();
        builder.add_expand_like(0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let mut plan = program.program_plan();
        plan[PLAN_HEADER_BYTES + 1] = 1;
        assert!(MathProgramV7::from_plan(&plan).is_err());
    }

    #[test]
    fn capabilities_describe_explicit_not_implicit_expansion() {
        let caps = math_program_v7_capabilities();
        assert!(caps.contains("burn-research.math-program.v7"));
        assert!(caps.contains("expandLike"));
        assert!(caps.contains("canonical_one_step_v6_replay"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
        assert!(caps.contains("\"legacy_v1_v6_decoders_frozen\":true"));
    }
}
