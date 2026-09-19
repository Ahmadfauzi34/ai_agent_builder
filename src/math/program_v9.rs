//! Math Program v9: frozen v8 semantics plus delegated positional-index and numeric-comparison steps.
//!
//! Historical v1-v8 decoders remain frozen. Every legacy step is validated and executed through a
//! canonical one-step Math Program v8 replay. V9 does not implement new index/comparison math: its
//! two new plan opcodes delegate to the independently proven `TensorIndexSource` and
//! `TensorComparison` primitive boundaries.

use crate::math::comparison::TensorComparison;
use crate::math::index_source::TensorIndexSource;
use crate::math::program_index_params::{IndexAxisParams, PARAM_INDEX_AXIS};
use crate::math::program_step_record::{ProgramStepRecord, STEP_HEADER_BYTES};
use crate::math::program_v8::{MathProgramV8, MathProgramV8Builder};
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V8: u8 = 8;
const PLAN_VERSION_V9: u8 = 9;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_PLAN_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLOTS: u8 = 64;
const PARAM_NONE: u8 = 0;

pub const MIN_V9_EXTERNAL_INPUTS: u8 = 1;
pub const MAX_V9_EXTERNAL_INPUTS: u8 = 8;
pub const OP_INDICES_LIKE: u8 = 0x70;
pub const OP_LESS_EQUAL_01: u8 = 0x71;

type RawStepRecord = ProgramStepRecord;
const STEP_CONTEXT: &str = "MathProgramV9 step";

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
    V8 {
        program: MathProgramV8,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
    },
    IndicesLike {
        reference: u8,
        out: u8,
        axis: u32,
    },
    LessEqual01 {
        lhs: u8,
        rhs: u8,
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
    if !(MIN_V9_EXTERNAL_INPUTS..=MAX_V9_EXTERNAL_INPUTS).contains(&num_inputs) {
        return Err(format!(
            "MathProgramV9: num_inputs must be in {MIN_V9_EXTERNAL_INPUTS}..={MAX_V9_EXTERNAL_INPUTS}, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgramV9: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
            num_inputs + 1
        ));
    }
    Ok(())
}

fn extract_single_v8_record(program: MathProgramV8) -> Result<RawStepRecord, String> {
    let plan = program.program_plan();
    if plan.len() < PLAN_HEADER_BYTES + STEP_HEADER_BYTES + PLAN_OUTPUT_BYTES
        || &plan[..4] != PLAN_MAGIC
        || plan[4] != PLAN_VERSION_V8
        || plan[7] != 1
    {
        return Err("MathProgramV9: expected canonical one-step v8 plan".into());
    }
    let record_bytes = &plan[PLAN_HEADER_BYTES..plan.len() - PLAN_OUTPUT_BYTES];
    let (record, consumed) = decode_raw_step_prefix(record_bytes)?;
    if consumed != record_bytes.len() {
        return Err("MathProgramV9: one-step v8 plan contained trailing record bytes".into());
    }
    Ok(record)
}

fn v8_unary_record<F>(
    global_input: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV8Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV8Builder::new(1, 2)?;
    build(&mut builder)?;
    builder.set_output(1)?;
    let mut record = extract_single_v8_record(builder.compile()?)?;
    record.in_a = global_input;
    record.in_b = 0;
    record.out = global_output;
    Ok(record)
}

fn v8_binary_record<F>(
    global_lhs: u8,
    global_rhs: u8,
    global_output: u8,
    build: F,
) -> Result<RawStepRecord, String>
where
    F: FnOnce(&mut MathProgramV8Builder) -> Result<(), String>,
{
    let mut builder = MathProgramV8Builder::new(2, 3)?;
    build(&mut builder)?;
    builder.set_output(2)?;
    let mut record = extract_single_v8_record(builder.compile()?)?;
    record.in_a = global_lhs;
    record.in_b = global_rhs;
    record.out = global_output;
    Ok(record)
}

fn canonical_one_step_v8_plan(record: &RawStepRecord) -> Result<Vec<u8>, String> {
    if record.arity != 1 && record.arity != 2 {
        return Err(format!(
            "MathProgramV9: v8 delegated step has unsupported arity {}; expected 1 or 2",
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
    plan.push(PLAN_VERSION_V8);
    plan.push(record.arity);
    plan.push(record.arity + 1);
    plan.push(1);
    plan.extend_from_slice(&local.encode());
    plan.push(out);
    Ok(plan)
}

fn compile_record(record: &RawStepRecord) -> Result<ExecutableStep, String> {
    match record.op {
        OP_INDICES_LIKE => {
            if record.arity != 1 || record.in_b != 0 || record.param_kind != PARAM_INDEX_AXIS {
                return Err(
                    "MathProgramV9: indicesLike requires unary arity, in_b=0, and index-axis parameters"
                        .into(),
                );
            }
            let params = IndexAxisParams::decode(&record.payload)?;
            Ok(ExecutableStep::IndicesLike {
                reference: record.in_a,
                out: record.out,
                axis: params.axis(),
            })
        }
        OP_LESS_EQUAL_01 => {
            if record.arity != 2 || record.param_kind != PARAM_NONE || !record.payload.is_empty() {
                return Err(
                    "MathProgramV9: lessEqual01 requires binary arity and no parameters".into(),
                );
            }
            Ok(ExecutableStep::LessEqual01 {
                lhs: record.in_a,
                rhs: record.in_b,
                out: record.out,
            })
        }
        _ => {
            let program = MathProgramV8::from_plan(&canonical_one_step_v8_plan(record)?)?;
            Ok(ExecutableStep::V8 {
                program,
                arity: record.arity,
                in_a: record.in_a,
                in_b: record.in_b,
                out: record.out,
            })
        }
    }
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
    validate_records(num_inputs, num_slots, records, out_slot, "MathProgramV9")?;
    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV9: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV9: plan length overflow".to_string())?;
    if plan_len > MAX_PLAN_BYTES {
        return Err(format!(
            "MathProgramV9: plan size {plan_len} exceeds maximum {MAX_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V9);
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
        return Err("MathProgramV9: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgramV9: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION_V9 {
        return Err(format!(
            "MathProgramV9: expected plan version {PLAN_VERSION_V9}, got {}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgramV9: plan must contain at least one step".into());
    }

    let mut records = Vec::with_capacity(num_steps);
    let mut steps = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;

    for index in 0..num_steps {
        if offset >= plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err(format!("MathProgramV9: plan is truncated before step {index}"));
        }
        let (record, consumed) = decode_raw_step_prefix(&plan[offset..])?;
        let step = compile_record(&record)?;
        (filled, written) = validate_topology(
            &record,
            num_slots,
            filled,
            written,
            &format!("MathProgramV9 replay step {index}"),
        )?;
        records.push(record);
        steps.push(step);
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| "MathProgramV9: plan offset overflow".to_string())?;
    }

    if offset + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "MathProgramV9: malformed plan length: expected output byte at offset {offset}, got {} total bytes",
            plan.len()
        ));
    }
    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgramV9: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgramV9: output slot {out_slot} must be produced by a program step"
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

pub fn math_program_v9_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v9\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v9\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"min_inputs\":1,",
        "\"max_inputs\":8,",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"legacy_step_validation\":\"canonical_one_step_v8_replay\",",
        "\"delegated_ops\":[\"indicesLike\",\"lessEqual01\"],",
        "\"index_axis_binding\":\"identity_bound_plan_metadata\",",
        "\"index_source_semantics\":\"delegate_index_source_v1\",",
        "\"comparison_semantics\":\"delegate_comparison_v1\",",
        "\"causal_mask_special_case\":false,",
        "\"select_where_required\":false,",
        "\"boolean_tensor_family\":false,",
        "\"implicit_broadcasting\":false,",
        "\"legacy_v1_v8_decoders_frozen\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false,",
        "\"grants_authority\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV9Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV9Builder {
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
        let record = v8_unary_record(input, output, |builder| builder.add_unary(op, 0, 1))?;
        self.push_record(record, "MathProgramV9Builder.addUnary")
    }

    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        let record = v8_binary_record(lhs, rhs, output, |builder| {
            builder.add_binary(op, 0, 1, 2)
        })?;
        self.push_record(record, "MathProgramV9Builder.addBinary")
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_clamp(0, 1, min, max)
        })?;
        self.push_record(record, "MathProgramV9Builder.addClamp")
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        let record = v8_binary_record(lhs, rhs, output, |builder| {
            builder.add_cosine_similarity(0, 1, 2, epsilon)
        })?;
        self.push_record(record, "MathProgramV9Builder.addCosineSimilarity")
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_reshape(0, 1, shape)
        })?;
        self.push_record(record, "MathProgramV9Builder.addReshape")
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_permute(0, 1, axes)
        })?;
        self.push_record(record, "MathProgramV9Builder.addPermute")
    }

    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_slice(0, 1, starts, ends)
        })?;
        self.push_record(record, "MathProgramV9Builder.addSlice")
    }

    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_select_axis(0, 1, axis, indices)
        })?;
        self.push_record(record, "MathProgramV9Builder.addSelectAxis")
    }

    pub fn add_fill_like(
        &mut self,
        reference: u8,
        output: u8,
        scalar: f32,
    ) -> Result<(), String> {
        let record = v8_unary_record(reference, output, |builder| {
            builder.add_fill_like(0, 1, scalar)
        })?;
        self.push_record(record, "MathProgramV9Builder.addFillLike")
    }

    pub fn add_expand_like(
        &mut self,
        source: u8,
        reference: u8,
        output: u8,
    ) -> Result<(), String> {
        let record = v8_binary_record(source, reference, output, |builder| {
            builder.add_expand_like(0, 1, 2)
        })?;
        self.push_record(record, "MathProgramV9Builder.addExpandLike")
    }

    pub fn add_sum_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_sum_axis(0, 1, axis)
        })?;
        self.push_record(record, "MathProgramV9Builder.addSumAxis")
    }

    pub fn add_mean_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_mean_axis(0, 1, axis)
        })?;
        self.push_record(record, "MathProgramV9Builder.addMeanAxis")
    }

    pub fn add_min_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_min_axis(0, 1, axis)
        })?;
        self.push_record(record, "MathProgramV9Builder.addMinAxis")
    }

    pub fn add_max_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        let record = v8_unary_record(input, output, |builder| {
            builder.add_max_axis(0, 1, axis)
        })?;
        self.push_record(record, "MathProgramV9Builder.addMaxAxis")
    }

    pub fn add_indices_like(
        &mut self,
        reference: u8,
        output: u8,
        axis: u32,
    ) -> Result<(), String> {
        let params = IndexAxisParams::new(axis)?;
        self.push_record(
            raw_step_record(
                OP_INDICES_LIKE,
                1,
                reference,
                0,
                output,
                PARAM_INDEX_AXIS,
                params.encode().to_vec(),
            )?,
            "MathProgramV9Builder.addIndicesLike",
        )
    }

    pub fn add_less_equal_01(&mut self, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.push_record(
            raw_step_record(
                OP_LESS_EQUAL_01,
                2,
                lhs,
                rhs,
                output,
                PARAM_NONE,
                Vec::new(),
            )?,
            "MathProgramV9Builder.addLessEqual01",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV9Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV9Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV9, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV9Builder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.records, out_slot)?;
        let steps = self
            .records
            .iter()
            .map(compile_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MathProgramV9 {
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
pub struct MathProgramV9 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<RawStepRecord>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV9 {
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
            return Err("MathProgramV9: replay plan is not canonical".into());
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
                "MathProgramV9.runInputs: expected {} inputs, got {}",
                self.num_inputs,
                inputs.len()
            ));
        }

        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (index, input) in inputs.iter().enumerate() {
            slots[index] = Some(input.clone());
        }
        let index_source = TensorIndexSource::new();
        let comparison = TensorComparison::new();

        for (index, step) in self.steps.iter().enumerate() {
            let (out, value) = match step {
                ExecutableStep::V8 {
                    program,
                    arity,
                    in_a,
                    in_b,
                    out,
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV9.runInputs: step {index} input slot {in_a} is empty"
                        )
                    })?;
                    let value = if *arity == 1 {
                        program.run_inputs(&[a.clone()])?
                    } else {
                        let b = slots[*in_b as usize].as_ref().ok_or_else(|| {
                            format!(
                                "MathProgramV9.runInputs: step {index} second input slot {in_b} is empty"
                            )
                        })?;
                        program.run_inputs(&[a.clone(), b.clone()])?
                    };
                    (*out, value)
                }
                ExecutableStep::IndicesLike {
                    reference,
                    out,
                    axis,
                } => {
                    let reference_value = slots[*reference as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV9.runInputs: step {index} reference slot {reference} is empty"
                        )
                    })?;
                    (*out, index_source.indices_like(reference_value, *axis)?)
                }
                ExecutableStep::LessEqual01 { lhs, rhs, out } => {
                    let lhs_value = slots[*lhs as usize].as_ref().ok_or_else(|| {
                        format!("MathProgramV9.runInputs: step {index} lhs slot {lhs} is empty")
                    })?;
                    let rhs_value = slots[*rhs as usize].as_ref().ok_or_else(|| {
                        format!("MathProgramV9.runInputs: step {index} rhs slot {rhs} is empty")
                    })?;
                    (*out, comparison.less_equal_01(lhs_value, rhs_value)?)
                }
            };
            slots[out as usize] = Some(value);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV9.runInputs: output slot {} is empty", self.out_slot))
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
    use crate::math::program::{OP_ADD, OP_MUL, OP_SUB};

    fn tensor(values: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor::new(values, shape)
    }

    #[test]
    fn indices_like_all_axes_reuse_proven_source_semantics() {
        let reference = tensor(&[9.0; 24], &[2, 3, 2, 2]);
        let expected = [
            [vec![0.0; 12], vec![1.0; 12]].concat(),
            [
                vec![0.0; 4],
                vec![1.0; 4],
                vec![2.0; 4],
                vec![0.0; 4],
                vec![1.0; 4],
                vec![2.0; 4],
            ]
            .concat(),
            vec![0.0, 0.0, 1.0, 1.0].repeat(6),
            vec![0.0, 1.0].repeat(12),
        ];

        for axis in 0..4 {
            let mut builder = MathProgramV9Builder::new(1, 2).unwrap();
            builder.add_indices_like(0, 1, axis).unwrap();
            builder.set_output(1).unwrap();
            let output = builder.compile().unwrap().run_inputs(&[reference.clone()]).unwrap();
            assert_eq!(output.shape(), vec![2, 3, 2, 2]);
            assert_eq!(output.to_array(), expected[axis as usize]);
        }
    }

    #[test]
    fn index_axis_is_identity_bound_and_replay_stable() {
        let mut a = MathProgramV9Builder::new(1, 2).unwrap();
        a.add_indices_like(0, 1, 1).unwrap();
        a.set_output(1).unwrap();
        let a = a.compile().unwrap();

        let mut b = MathProgramV9Builder::new(1, 2).unwrap();
        b.add_indices_like(0, 1, 2).unwrap();
        b.set_output(1).unwrap();
        let b = b.compile().unwrap();
        assert_ne!(a.program_identity(), b.program_identity());

        let replay = MathProgramV9::from_plan(&a.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), a.program_plan());
        assert_eq!(replay.program_identity(), a.program_identity());

        let input = tensor(&[3.0; 6], &[1, 2, 3, 1]);
        assert_eq!(
            replay.run_inputs(&[input]).unwrap().to_array(),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn less_equal_01_reuses_comparison_contract_and_failures() {
        let mut builder = MathProgramV9Builder::new(2, 3).unwrap();
        builder.add_less_equal_01(0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();

        let lhs = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[1, 6, 1, 1]);
        let rhs = tensor(&[-1.0, -1.0, 0.0, 0.0, 3.0, 2.0], &[1, 6, 1, 1]);
        assert_eq!(
            program.run_inputs(&[lhs, rhs]).unwrap().to_array(),
            vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        );

        let vector = tensor(&[1.0, 2.0], &[1, 2, 1, 1]);
        let mismatch = tensor(&[1.0, 2.0], &[1, 1, 2, 1]);
        assert!(program.run_inputs(&[vector.clone(), mismatch]).is_err());
        let nan = tensor(&[1.0, f32::NAN], &[1, 2, 1, 1]);
        assert!(program.run_inputs(&[nan, vector]).is_err());
    }

    #[test]
    fn causal_additive_mask_is_fully_expressible_inside_v9() {
        let mut builder = MathProgramV9Builder::new(1, 9).unwrap();
        builder.add_indices_like(0, 1, 1).unwrap();
        builder.add_indices_like(0, 2, 2).unwrap();
        builder.add_less_equal_01(2, 1, 3).unwrap();
        builder.add_fill_like(0, 4, 1.0).unwrap();
        builder.add_binary(OP_SUB, 4, 3, 5).unwrap();
        builder.add_fill_like(0, 6, -100.0).unwrap();
        builder.add_binary(OP_MUL, 5, 6, 7).unwrap();
        builder.add_binary(OP_ADD, 0, 7, 8).unwrap();
        builder.set_output(8).unwrap();
        let program = builder.compile().unwrap();

        let scores = tensor(
            &[
                0.0, 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 11.0,
                12.0, 13.0, 14.0, 15.0,
            ],
            &[1, 4, 4, 1],
        );
        let masked = program.run_inputs(&[scores]).unwrap();
        assert_eq!(masked.shape(), vec![1, 4, 4, 1]);
        assert_eq!(
            masked.to_array(),
            vec![
                0.0, -99.0, -98.0, -97.0,
                4.0, 5.0, -94.0, -93.0,
                8.0, 9.0, 10.0, -89.0,
                12.0, 13.0, 14.0, 15.0,
            ]
        );

        let replay = MathProgramV9::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_identity(), program.program_identity());
        assert_eq!(replay.num_steps(), 8);
    }

    #[test]
    fn invalid_index_axis_construction_is_atomic() {
        let mut builder = MathProgramV9Builder::new(1, 3).unwrap();
        assert_eq!(builder.num_steps(), 0);
        assert!(builder.add_indices_like(0, 1, 4).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_indices_like(0, 1, 1).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn malformed_new_records_fail_closed() {
        let mut index_builder = MathProgramV9Builder::new(1, 2).unwrap();
        index_builder.add_indices_like(0, 1, 1).unwrap();
        index_builder.set_output(1).unwrap();
        let index_plan = index_builder.compile().unwrap().program_plan();

        let mut bad_axis = index_plan.clone();
        bad_axis[PLAN_HEADER_BYTES + STEP_HEADER_BYTES] = 4;
        assert!(MathProgramV9::from_plan(&bad_axis).is_err());

        let mut bad_arity = index_plan.clone();
        bad_arity[PLAN_HEADER_BYTES + 1] = 2;
        assert!(MathProgramV9::from_plan(&bad_arity).is_err());

        let mut bad_param_kind = index_plan;
        bad_param_kind[PLAN_HEADER_BYTES + 5] = PARAM_NONE;
        assert!(MathProgramV9::from_plan(&bad_param_kind).is_err());

        let mut cmp_builder = MathProgramV9Builder::new(2, 3).unwrap();
        cmp_builder.add_less_equal_01(0, 1, 2).unwrap();
        cmp_builder.set_output(2).unwrap();
        let mut cmp_plan = cmp_builder.compile().unwrap().program_plan();
        cmp_plan[PLAN_HEADER_BYTES + 1] = 1;
        assert!(MathProgramV9::from_plan(&cmp_plan).is_err());
    }

    #[test]
    fn legacy_v8_steps_are_replayed_without_semantic_fork() {
        let mut builder = MathProgramV9Builder::new(1, 5).unwrap();
        builder.add_fill_like(0, 1, 2.0).unwrap();
        builder.add_binary(OP_MUL, 0, 1, 2).unwrap();
        builder.add_sum_axis(2, 3, 1).unwrap();
        builder.add_fill_like(3, 4, 1.0).unwrap();
        builder.set_output(4).unwrap();
        let program = builder.compile().unwrap();
        let input = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        assert_eq!(program.run_inputs(&[input]).unwrap().to_array(), vec![1.0]);
    }

    #[test]
    fn versions_remain_isolated() {
        let mut v9_builder = MathProgramV9Builder::new(1, 2).unwrap();
        v9_builder.add_indices_like(0, 1, 1).unwrap();
        v9_builder.set_output(1).unwrap();
        let v9 = v9_builder.compile().unwrap();
        assert!(MathProgramV8::from_plan(&v9.program_plan()).is_err());

        let mut v8_builder = MathProgramV8Builder::new(1, 2).unwrap();
        v8_builder.add_sum_axis(0, 1, 1).unwrap();
        v8_builder.set_output(1).unwrap();
        let v8 = v8_builder.compile().unwrap();
        assert!(MathProgramV9::from_plan(&v8.program_plan()).is_err());
    }

    #[test]
    fn topology_still_rejects_read_before_write_and_double_write() {
        let mut builder = MathProgramV9Builder::new(1, 4).unwrap();
        assert!(builder.add_less_equal_01(0, 2, 1).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_indices_like(0, 1, 1).unwrap();
        assert!(builder.add_indices_like(0, 1, 2).is_err());
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn capabilities_describe_delegation_not_new_math_engines() {
        let caps = math_program_v9_capabilities();
        assert!(caps.contains("burn-research.math-program.v9"));
        assert!(caps.contains("canonical_one_step_v8_replay"));
        assert!(caps.contains("delegate_index_source_v1"));
        assert!(caps.contains("delegate_comparison_v1"));
        assert!(caps.contains("\"causal_mask_special_case\":false"));
        assert!(caps.contains("\"select_where_required\":false"));
        assert!(caps.contains("\"boolean_tensor_family\":false"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
        assert!(caps.contains("\"grants_authority\":false"));
        assert!(caps.contains("\"legacy_v1_v8_decoders_frozen\":true"));
    }
}
