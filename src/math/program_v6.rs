//! Math Program v6: v5-compatible DAG orchestration plus identity-bound `fillLike` values.
//!
//! Historical v1-v5 decoders remain frozen. Legacy steps inside a v6 plan are validated and
//! executed through a canonical one-step v5 program, so v6 does not fork existing math semantics.
//! The only new execution semantic in this version is `fillLike(reference, scalar)`.

use crate::math::program::{
    OP_CLAMP, OP_COSINE_SIMILARITY, OP_PERMUTE, OP_RESHAPE, OP_SLICE,
};
use crate::math::program_select_params::{SelectAxisParams, PARAM_SELECT_AXIS};
use crate::math::program_shape_params::{
    FixedShapeParams, PARAM_PERMUTE_RANK4, PARAM_RESHAPE_RANK4, PARAM_SLICE_RANK4,
};
use crate::math::program_v4::OP_SELECT_AXIS;
use crate::math::program_v4_step::{
    V4StepRecord, PARAM_CLAMP as RECORD_PARAM_CLAMP, PARAM_EPSILON as RECORD_PARAM_EPSILON,
    PARAM_NONE,
};
use crate::math::program_v5::MathProgramV5;
use crate::math::program_value_source::{
    fill_like, FillLikeParams, FILL_LIKE_PARAM_BYTES, PARAM_FILL_LIKE,
};
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V5: u8 = 5;
const PLAN_VERSION_V6: u8 = 6;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const V6_STEP_HEADER_BYTES: usize = 10;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_V6_PLAN_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLOTS: u8 = 64;

pub const MIN_V6_EXTERNAL_INPUTS: u8 = 1;
pub const MAX_V6_EXTERNAL_INPUTS: u8 = 8;
pub const OP_FILL_LIKE: u8 = 0x25;

#[derive(Clone, Debug, PartialEq, Eq)]
enum V6Record {
    Legacy(V4StepRecord),
    FillLike {
        reference: u8,
        out: u8,
        params: FillLikeParams,
    },
}

impl V6Record {
    fn encoded_len(&self) -> usize {
        match self {
            Self::Legacy(record) => record.encoded_len(),
            Self::FillLike { .. } => V6_STEP_HEADER_BYTES + FILL_LIKE_PARAM_BYTES,
        }
    }

    fn encode(&self) -> Vec<u8> {
        match self {
            Self::Legacy(record) => record.encode(),
            Self::FillLike {
                reference,
                out,
                params,
            } => {
                let payload = params.encode();
                let mut bytes = Vec::with_capacity(V6_STEP_HEADER_BYTES + payload.len());
                bytes.push(OP_FILL_LIKE);
                bytes.push(1);
                bytes.push(*reference);
                bytes.push(0);
                bytes.push(*out);
                bytes.push(PARAM_FILL_LIKE);
                bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&payload);
                bytes
            }
        }
    }

    fn decode_prefix(bytes: &[u8]) -> Result<(Self, usize), String> {
        if bytes.len() < V6_STEP_HEADER_BYTES {
            return Err(format!(
                "MathProgramV6 step: truncated header: expected at least {V6_STEP_HEADER_BYTES} bytes, got {}",
                bytes.len()
            ));
        }

        if bytes[5] != PARAM_FILL_LIKE {
            let (record, consumed) = V4StepRecord::decode_prefix(bytes)?;
            return Ok((Self::Legacy(record), consumed));
        }

        let payload_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if payload_len != FILL_LIKE_PARAM_BYTES {
            return Err(format!(
                "MathProgramV6 fillLike: payload length must be {FILL_LIKE_PARAM_BYTES}, got {payload_len}"
            ));
        }
        let record_len = V6_STEP_HEADER_BYTES
            .checked_add(payload_len)
            .ok_or_else(|| "MathProgramV6 fillLike: record length overflow".to_string())?;
        if bytes.len() < record_len {
            return Err(format!(
                "MathProgramV6 fillLike: truncated payload: record needs {record_len} bytes, got {}",
                bytes.len()
            ));
        }
        if bytes[0] != OP_FILL_LIKE || bytes[1] != 1 || bytes[3] != 0 {
            return Err(
                "MathProgramV6 fillLike: requires opcode 0x25, unary arity, and canonical in_b=0"
                    .into(),
            );
        }
        let params = FillLikeParams::decode(&bytes[V6_STEP_HEADER_BYTES..record_len])?;
        let record = Self::FillLike {
            reference: bytes[2],
            out: bytes[4],
            params,
        };
        if record.encode().as_slice() != &bytes[..record_len] {
            return Err("MathProgramV6 fillLike: noncanonical record encoding".into());
        }
        Ok((record, record_len))
    }

    fn arity(&self) -> u8 {
        match self {
            Self::Legacy(record) => record.arity,
            Self::FillLike { .. } => 1,
        }
    }

    fn in_a(&self) -> u8 {
        match self {
            Self::Legacy(record) => record.in_a,
            Self::FillLike { reference, .. } => *reference,
        }
    }

    fn in_b(&self) -> u8 {
        match self {
            Self::Legacy(record) => record.in_b,
            Self::FillLike { .. } => 0,
        }
    }

    fn out(&self) -> u8 {
        match self {
            Self::Legacy(record) => record.out,
            Self::FillLike { out, .. } => *out,
        }
    }
}

#[derive(Clone, Debug)]
enum ExecutableStep {
    Legacy {
        program: MathProgramV5,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
    },
    FillLike {
        reference: u8,
        out: u8,
        params: FillLikeParams,
    },
}

#[derive(Clone, Debug)]
struct DecodedPlan {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V6Record>,
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
    if !(MIN_V6_EXTERNAL_INPUTS..=MAX_V6_EXTERNAL_INPUTS).contains(&num_inputs) {
        return Err(format!(
            "MathProgramV6: num_inputs must be in {MIN_V6_EXTERNAL_INPUTS}..={MAX_V6_EXTERNAL_INPUTS}, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgramV6: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
            num_inputs + 1
        ));
    }
    Ok(())
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn scalar_payload(a: f32, b: f32) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&canonical_f32_bits(a).to_le_bytes());
    payload.extend_from_slice(&canonical_f32_bits(b).to_le_bytes());
    payload
}

fn canonical_one_step_v5_plan(record: &V4StepRecord) -> Result<Vec<u8>, String> {
    let local = V4StepRecord::new(
        record.op,
        record.arity,
        0,
        if record.arity == 2 { 1 } else { 0 },
        3,
        record.param_kind,
        record.payload.clone(),
    )?;
    let mut plan = Vec::with_capacity(PLAN_HEADER_BYTES + local.encoded_len() + PLAN_OUTPUT_BYTES);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V5);
    plan.push(3);
    plan.push(4);
    plan.push(1);
    plan.extend_from_slice(&local.encode());
    plan.push(3);
    Ok(plan)
}

fn compile_record(record: &V6Record) -> Result<ExecutableStep, String> {
    match record {
        V6Record::Legacy(record) => {
            let program = MathProgramV5::from_plan(&canonical_one_step_v5_plan(record)?)?;
            Ok(ExecutableStep::Legacy {
                program,
                arity: record.arity,
                in_a: record.in_a,
                in_b: record.in_b,
                out: record.out,
            })
        }
        V6Record::FillLike {
            reference,
            out,
            params,
        } => Ok(ExecutableStep::FillLike {
            reference: *reference,
            out: *out,
            params: *params,
        }),
    }
}

fn validate_topology(
    record: &V6Record,
    num_slots: u8,
    filled: u64,
    written: u64,
    context: &str,
) -> Result<(u64, u64), String> {
    let in_a = record.in_a();
    let in_b = record.in_b();
    let out = record.out();
    let arity = record.arity();

    if in_a >= num_slots || in_b >= num_slots || out >= num_slots {
        return Err(format!(
            "{context}: slot out of range for num_slots={num_slots}: in_a={in_a}, in_b={in_b}, out={out}"
        ));
    }
    if filled & bit(in_a) == 0 {
        return Err(format!("{context}: input slot {in_a} is read before write"));
    }
    if arity == 2 && filled & bit(in_b) == 0 {
        return Err(format!(
            "{context}: second input slot {in_b} is read before write"
        ));
    }
    if arity == 1 && in_b != 0 {
        return Err(format!(
            "{context}: unary step must encode in_b=0, got {in_b}"
        ));
    }
    if arity != 1 && arity != 2 {
        return Err(format!(
            "{context}: unsupported arity {arity}; expected 1 or 2"
        ));
    }
    if filled & bit(out) != 0 {
        return Err(format!(
            "{context}: output slot {out} is already filled; slots are write-once"
        ));
    }
    Ok((filled | bit(out), written | bit(out)))
}

fn validate_records(
    num_inputs: u8,
    num_slots: u8,
    records: &[V6Record],
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
    records: &[V6Record],
    out_slot: u8,
) -> Result<Vec<u8>, String> {
    validate_records(num_inputs, num_slots, records, out_slot, "MathProgramV6")?;

    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV6: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV6: plan length overflow".to_string())?;
    if plan_len > MAX_V6_PLAN_BYTES {
        return Err(format!(
            "MathProgramV6: plan size {plan_len} exceeds maximum {MAX_V6_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V6);
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
        return Err("MathProgramV6: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgramV6: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION_V6 {
        return Err(format!(
            "MathProgramV6: expected plan version {PLAN_VERSION_V6}, got {}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgramV6: plan must contain at least one step".into());
    }

    let mut records = Vec::with_capacity(num_steps);
    let mut executable = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;

    for index in 0..num_steps {
        if offset >= plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err(format!(
                "MathProgramV6: plan is truncated before step {index}"
            ));
        }
        let (record, consumed) = V6Record::decode_prefix(&plan[offset..])?;
        let step = compile_record(&record)?;
        (filled, written) = validate_topology(
            &record,
            num_slots,
            filled,
            written,
            &format!("MathProgramV6 replay step {index}"),
        )?;
        records.push(record);
        executable.push(step);
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| "MathProgramV6: plan offset overflow".to_string())?;
    }

    if offset + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "MathProgramV6: malformed plan length: expected output byte at offset {offset}, got {} total bytes",
            plan.len()
        ));
    }
    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgramV6: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgramV6: output slot {out_slot} must be produced by a program step"
        ));
    }

    Ok(DecodedPlan {
        num_inputs,
        num_slots,
        records,
        steps: executable,
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

pub fn math_program_v6_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v6\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v6\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"min_inputs\":1,",
        "\"max_inputs\":8,",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"legacy_step_validation\":\"canonical_one_step_v5_replay\",",
        "\"value_sources\":[\"fillLike\"],",
        "\"fill_like_scalar\":\"finite_canonical_f32_identity_bound\",",
        "\"implicit_broadcasting\":false,",
        "\"legacy_v1_v5_decoders_frozen\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV6Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V6Record>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV6Builder {
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

    fn push_record(&mut self, record: V6Record, context: &str) -> Result<(), String> {
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

    fn push_legacy(&mut self, record: V4StepRecord, context: &str) -> Result<(), String> {
        self.push_record(V6Record::Legacy(record), context)
    }

    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.push_legacy(
            V4StepRecord::new(op, 1, input, 0, output, PARAM_NONE, vec![])?,
            "MathProgramV6Builder.addUnary",
        )
    }

    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        self.push_legacy(
            V4StepRecord::new(op, 2, lhs, rhs, output, PARAM_NONE, vec![])?,
            "MathProgramV6Builder.addBinary",
        )
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        self.push_legacy(
            V4StepRecord::new(
                OP_CLAMP,
                1,
                input,
                0,
                output,
                RECORD_PARAM_CLAMP,
                scalar_payload(min, max),
            )?,
            "MathProgramV6Builder.addClamp",
        )
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.push_legacy(
            V4StepRecord::new(
                OP_COSINE_SIMILARITY,
                2,
                lhs,
                rhs,
                output,
                RECORD_PARAM_EPSILON,
                scalar_payload(epsilon, 0.0),
            )?,
            "MathProgramV6Builder.addCosineSimilarity",
        )
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::reshape(shape)?;
        self.push_legacy(
            V4StepRecord::new(
                OP_RESHAPE,
                1,
                input,
                0,
                output,
                PARAM_RESHAPE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV6Builder.addReshape",
        )
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::permute(axes)?;
        self.push_legacy(
            V4StepRecord::new(
                OP_PERMUTE,
                1,
                input,
                0,
                output,
                PARAM_PERMUTE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV6Builder.addPermute",
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
        self.push_legacy(
            V4StepRecord::new(
                OP_SLICE,
                1,
                input,
                0,
                output,
                PARAM_SLICE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV6Builder.addSlice",
        )
    }

    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        let params = SelectAxisParams::new(axis, indices)?;
        self.push_legacy(
            V4StepRecord::new(
                OP_SELECT_AXIS,
                1,
                input,
                0,
                output,
                PARAM_SELECT_AXIS,
                params.encode(),
            )?,
            "MathProgramV6Builder.addSelectAxis",
        )
    }

    pub fn add_fill_like(
        &mut self,
        reference: u8,
        output: u8,
        scalar: f32,
    ) -> Result<(), String> {
        let params = FillLikeParams::new(scalar)?;
        self.push_record(
            V6Record::FillLike {
                reference,
                out: output,
                params,
            },
            "MathProgramV6Builder.addFillLike",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV6Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV6Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV6, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV6Builder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.records, out_slot)?;
        let steps = self
            .records
            .iter()
            .map(compile_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MathProgramV6 {
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
pub struct MathProgramV6 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V6Record>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV6 {
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
            return Err("MathProgramV6: replay plan is not canonical".into());
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
                "MathProgramV6.runInputs: expected {} inputs, got {}",
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
                ExecutableStep::Legacy {
                    program,
                    arity,
                    in_a,
                    in_b,
                    out,
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV6.runInputs: step {index} input slot {in_a} is empty"
                        )
                    })?;
                    let b = if *arity == 2 {
                        slots[*in_b as usize].as_ref().ok_or_else(|| {
                            format!(
                                "MathProgramV6.runInputs: step {index} second input slot {in_b} is empty"
                            )
                        })?
                    } else {
                        a
                    };
                    let local_inputs = [a.clone(), b.clone(), a.clone()];
                    (*out, program.run_inputs(&local_inputs)?)
                }
                ExecutableStep::FillLike {
                    reference,
                    out,
                    params,
                } => {
                    let reference = slots[*reference as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV6.runInputs: step {index} reference slot {reference} is empty"
                        )
                    })?;
                    (*out, fill_like(reference, *params)?)
                }
            };
            slots[out as usize] = Some(value);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV6.runInputs: output slot {} is empty", self.out_slot))
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
    use crate::math::program::{OP_ADD, OP_MUL};

    fn tensor(values: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor::new(values, shape)
    }

    #[test]
    fn fill_like_composes_affine_formula_without_host_constant_tensors() {
        let mut builder = MathProgramV6Builder::new(1, 5).unwrap();
        builder.add_fill_like(0, 1, 0.5).unwrap();
        builder.add_binary(OP_MUL, 0, 1, 2).unwrap();
        builder.add_fill_like(0, 3, 3.0).unwrap();
        builder.add_binary(OP_ADD, 2, 3, 4).unwrap();
        builder.set_output(4).unwrap();
        let program = builder.compile().unwrap();

        let input = tensor(&[2.0, 4.0, 6.0], &[1, 3, 1, 1]);
        let output = program.run_inputs(&[input]).unwrap();
        assert_eq!(output.shape(), vec![1, 3, 1, 1]);
        assert_eq!(output.to_array(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn fill_like_scalar_is_identity_bound_and_replay_is_exact() {
        let mut a = MathProgramV6Builder::new(1, 2).unwrap();
        a.add_fill_like(0, 1, 2.0).unwrap();
        a.set_output(1).unwrap();
        let a = a.compile().unwrap();

        let mut b = MathProgramV6Builder::new(1, 2).unwrap();
        b.add_fill_like(0, 1, 3.0).unwrap();
        b.set_output(1).unwrap();
        let b = b.compile().unwrap();

        assert_ne!(a.program_identity(), b.program_identity());
        let replay = MathProgramV6::from_plan(&a.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), a.program_plan());
        assert_eq!(replay.program_identity(), a.program_identity());
    }

    #[test]
    fn negative_zero_is_canonical_and_nonfinite_add_is_atomic() {
        let mut positive = MathProgramV6Builder::new(1, 2).unwrap();
        positive.add_fill_like(0, 1, 0.0).unwrap();
        positive.set_output(1).unwrap();
        let positive = positive.compile().unwrap();

        let mut negative = MathProgramV6Builder::new(1, 2).unwrap();
        negative.add_fill_like(0, 1, -0.0).unwrap();
        negative.set_output(1).unwrap();
        let negative = negative.compile().unwrap();
        assert_eq!(positive.program_plan(), negative.program_plan());

        let mut builder = MathProgramV6Builder::new(1, 3).unwrap();
        assert_eq!(builder.num_steps(), 0);
        assert!(builder.add_fill_like(0, 1, f32::NAN).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_fill_like(0, 1, 1.0).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn fill_like_preserves_nontrivial_reference_shape() {
        let mut builder = MathProgramV6Builder::new(1, 2).unwrap();
        builder.add_fill_like(0, 1, -2.0).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let output = program.run_inputs(&[input]).unwrap();
        assert_eq!(output.shape(), vec![1, 2, 2, 1]);
        assert_eq!(output.to_array(), vec![-2.0; 4]);
    }

    #[test]
    fn implicit_broadcasting_remains_rejected() {
        let mut builder = MathProgramV6Builder::new(2, 3).unwrap();
        builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        let vector = tensor(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let scalar = tensor(&[10.0], &[1, 1, 1, 1]);
        assert!(program.run_inputs(&[vector, scalar]).is_err());
    }

    #[test]
    fn old_v4_codec_and_v5_plan_decoder_do_not_learn_fill_like() {
        let params = FillLikeParams::new(1.25).unwrap();
        let record = V6Record::FillLike {
            reference: 0,
            out: 1,
            params,
        };
        assert!(V4StepRecord::decode_exact(&record.encode()).is_err());

        let mut builder = MathProgramV6Builder::new(1, 2).unwrap();
        builder.add_fill_like(0, 1, 1.25).unwrap();
        builder.set_output(1).unwrap();
        let v6 = builder.compile().unwrap();
        assert!(MathProgramV5::from_plan(&v6.program_plan()).is_err());
    }

    #[test]
    fn malformed_fill_like_payload_fails_closed() {
        let mut builder = MathProgramV6Builder::new(1, 2).unwrap();
        builder.add_fill_like(0, 1, 1.0).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let mut plan = program.program_plan();
        plan[PLAN_HEADER_BYTES + 6..PLAN_HEADER_BYTES + 10]
            .copy_from_slice(&8u32.to_le_bytes());
        assert!(MathProgramV6::from_plan(&plan).is_err());
    }

    #[test]
    fn capabilities_describe_the_new_boundary() {
        let caps = math_program_v6_capabilities();
        assert!(caps.contains("burn-research.math-program.v6"));
        assert!(caps.contains("fillLike"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
        assert!(caps.contains("\"legacy_v1_v5_decoders_frozen\":true"));
    }
}
