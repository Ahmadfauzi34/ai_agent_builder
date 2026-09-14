use crate::math::program::{
    MathProgram, MathProgramBuilder, OP_CLAMP, OP_COSINE_SIMILARITY, OP_PERMUTE, OP_RESHAPE,
    OP_SLICE,
};
use crate::math::program_select_params::{SelectAxisParams, PARAM_SELECT_AXIS};
use crate::math::program_shape_params::{
    FixedShapeParams, PARAM_PERMUTE_RANK4, PARAM_RESHAPE_RANK4, PARAM_SLICE_RANK4,
};
use crate::math::program_v4::{MathProgramV4, MathProgramV4Builder, OP_SELECT_AXIS};
use crate::math::program_v4_step::{
    V4StepRecord, PARAM_CLAMP as RECORD_PARAM_CLAMP, PARAM_EPSILON as RECORD_PARAM_EPSILON,
    PARAM_NONE,
};
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V5: u8 = 5;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_V5_PLAN_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLOTS: u8 = 64;

pub const MIN_V5_EXTERNAL_INPUTS: u8 = 3;
pub const MAX_V5_EXTERNAL_INPUTS: u8 = 8;

#[derive(Clone, Debug)]
enum ExecutableStep {
    Core {
        program: MathProgram,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
    },
    SelectAxis {
        program: MathProgramV4,
        input: u8,
        out: u8,
    },
}

fn bit(slot: u8) -> u64 {
    1u64 << slot
}

fn initial_filled(num_inputs: u8) -> u64 {
    (1u64 << num_inputs) - 1
}

fn validate_program_shape(num_inputs: u8, num_slots: u8) -> Result<(), String> {
    if !(MIN_V5_EXTERNAL_INPUTS..=MAX_V5_EXTERNAL_INPUTS).contains(&num_inputs) {
        return Err(format!(
            "MathProgramV5: num_inputs must be in {MIN_V5_EXTERNAL_INPUTS}..={MAX_V5_EXTERNAL_INPUTS}, got {num_inputs}"
        ));
    }
    if num_slots <= num_inputs || num_slots > MAX_SLOTS {
        return Err(format!(
            "MathProgramV5: num_slots must be in {}..={MAX_SLOTS}, got {num_slots}",
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

fn parse_scalar_payload(payload: &[u8]) -> Result<(f32, f32), String> {
    if payload.len() != 8 {
        return Err(format!(
            "MathProgramV5: scalar payload must contain 8 bytes, got {}",
            payload.len()
        ));
    }
    Ok((
        f32::from_bits(u32::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3],
        ])),
        f32::from_bits(u32::from_le_bytes([
            payload[4], payload[5], payload[6], payload[7],
        ])),
    ))
}

fn compile_core_record(record: &V4StepRecord) -> Result<MathProgram, String> {
    match record.param_kind {
        PARAM_NONE => match record.arity {
            1 => {
                let mut builder = MathProgramBuilder::new(1, 2)?;
                builder.add_unary(record.op, 0, 1)?;
                builder.set_output(1)?;
                builder.compile()
            }
            2 => {
                let mut builder = MathProgramBuilder::new(2, 3)?;
                builder.add_binary(record.op, 0, 1, 2)?;
                builder.set_output(2)?;
                builder.compile()
            }
            arity => Err(format!(
                "MathProgramV5: unsupported core arity {arity}; expected 1 or 2"
            )),
        },
        RECORD_PARAM_CLAMP => {
            if record.op != OP_CLAMP || record.arity != 1 {
                return Err("MathProgramV5: clamp payload must use unary clamp opcode".into());
            }
            let (min, max) = parse_scalar_payload(&record.payload)?;
            if scalar_payload(min, max) != record.payload {
                return Err("MathProgramV5: clamp payload is not canonically encoded".into());
            }
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_clamp(0, 1, min, max)?;
            builder.set_output(1)?;
            builder.compile()
        }
        RECORD_PARAM_EPSILON => {
            if record.op != OP_COSINE_SIMILARITY || record.arity != 2 {
                return Err(
                    "MathProgramV5: epsilon payload must use binary cosineSimilarity opcode"
                        .into(),
                );
            }
            let (epsilon, reserved) = parse_scalar_payload(&record.payload)?;
            if reserved.to_bits() != 0 || scalar_payload(epsilon, 0.0) != record.payload {
                return Err("MathProgramV5: cosine epsilon payload is not canonical".into());
            }
            let mut builder = MathProgramBuilder::new(2, 3)?;
            builder.add_cosine_similarity(0, 1, 2, epsilon)?;
            builder.set_output(2)?;
            builder.compile()
        }
        PARAM_RESHAPE_RANK4 => {
            if record.op != OP_RESHAPE || record.arity != 1 {
                return Err("MathProgramV5: reshape payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let shape = params
                .reshape_usize()
                .ok_or_else(|| "MathProgramV5: reshape metadata mismatch".to_string())?
                .map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_reshape(0, 1, &shape)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_PERMUTE_RANK4 => {
            if record.op != OP_PERMUTE || record.arity != 1 {
                return Err("MathProgramV5: permute payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let axes = params
                .permute_usize()
                .ok_or_else(|| "MathProgramV5: permute metadata mismatch".to_string())?
                .map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_permute(0, 1, &axes)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_SLICE_RANK4 => {
            if record.op != OP_SLICE || record.arity != 1 {
                return Err("MathProgramV5: slice payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let (starts, ends) = params
                .slice_usize()
                .ok_or_else(|| "MathProgramV5: slice metadata mismatch".to_string())?;
            let starts = starts.map(|value| value as u32);
            let ends = ends.map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_slice(0, 1, &starts, &ends)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_SELECT_AXIS => Err("MathProgramV5: selectAxis is not a v1-v3 core step".into()),
        kind => Err(format!(
            "MathProgramV5: unsupported parameter kind {kind}; fail-closed"
        )),
    }
}

fn compile_record(record: &V4StepRecord) -> Result<ExecutableStep, String> {
    if record.param_kind == PARAM_SELECT_AXIS {
        if record.op != OP_SELECT_AXIS || record.arity != 1 || record.in_b != 0 {
            return Err(
                "MathProgramV5: selectAxis requires unary opcode 0x24 with canonical in_b=0"
                    .into(),
            );
        }
        let params = SelectAxisParams::decode(&record.payload)?;
        let mut builder = MathProgramV4Builder::new(1, 2)?;
        builder.add_select_axis(0, 1, params.axis() as u32, params.indices())?;
        builder.set_output(1)?;
        return Ok(ExecutableStep::SelectAxis {
            program: builder.compile()?,
            input: record.in_a,
            out: record.out,
        });
    }

    Ok(ExecutableStep::Core {
        program: compile_core_record(record)?,
        arity: record.arity,
        in_a: record.in_a,
        in_b: record.in_b,
        out: record.out,
    })
}

fn validate_topology(
    record: &V4StepRecord,
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
    records: &[V4StepRecord],
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
    records: &[V4StepRecord],
    out_slot: u8,
) -> Result<Vec<u8>, String> {
    validate_records(
        num_inputs,
        num_slots,
        records,
        out_slot,
        "MathProgramV5",
    )?;

    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV5: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV5: plan length overflow".to_string())?;
    if plan_len > MAX_V5_PLAN_BYTES {
        return Err(format!(
            "MathProgramV5: plan size {plan_len} exceeds maximum {MAX_V5_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V5);
    plan.push(num_inputs);
    plan.push(num_slots);
    plan.push(records.len() as u8);
    for record in records {
        plan.extend_from_slice(&record.encode());
    }
    plan.push(out_slot);
    Ok(plan)
}

fn decode_plan(
    plan: &[u8],
) -> Result<(u8, u8, Vec<V4StepRecord>, Vec<ExecutableStep>, u8), String> {
    if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
        return Err("MathProgramV5: plan is truncated".into());
    }
    if &plan[..4] != PLAN_MAGIC {
        return Err("MathProgramV5: invalid plan magic".into());
    }
    if plan[4] != PLAN_VERSION_V5 {
        return Err(format!(
            "MathProgramV5: expected plan version {PLAN_VERSION_V5}, got {}",
            plan[4]
        ));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    validate_program_shape(num_inputs, num_slots)?;
    if num_steps == 0 {
        return Err("MathProgramV5: plan must contain at least one step".into());
    }

    let mut records = Vec::with_capacity(num_steps);
    let mut executable = Vec::with_capacity(num_steps);
    let mut filled = initial_filled(num_inputs);
    let mut written = 0u64;
    let mut offset = PLAN_HEADER_BYTES;

    for index in 0..num_steps {
        if offset >= plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err(format!(
                "MathProgramV5: plan is truncated before step {index}"
            ));
        }
        let (record, consumed) = V4StepRecord::decode_prefix(&plan[offset..])?;
        let step = compile_record(&record)?;
        (filled, written) = validate_topology(
            &record,
            num_slots,
            filled,
            written,
            &format!("MathProgramV5 replay step {index}"),
        )?;
        records.push(record);
        executable.push(step);
        offset = offset
            .checked_add(consumed)
            .ok_or_else(|| "MathProgramV5: plan offset overflow".to_string())?;
    }

    if offset + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "MathProgramV5: malformed plan length: expected output byte at offset {offset}, got {} total bytes",
            plan.len()
        ));
    }
    let out_slot = plan[offset];
    if out_slot >= num_slots {
        return Err(format!(
            "MathProgramV5: output slot {out_slot} is out of range for num_slots={num_slots}"
        ));
    }
    if written & bit(out_slot) == 0 {
        return Err(format!(
            "MathProgramV5: output slot {out_slot} must be produced by a program step"
        ));
    }

    Ok((num_inputs, num_slots, records, executable, out_slot))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn math_program_v5_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v5\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v5\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"min_inputs\":3,",
        "\"max_inputs\":8,",
        "\"max_slots\":64,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"step_record_codec\":\"v4_self_delimiting\",",
        "\"core_execution_reuse\":\"MathProgram.v1-v3.one-step\",",
        "\"select_axis_execution_reuse\":\"MathProgram.v4.one-step\",",
        "\"legacy_v1_v4_input_contract_frozen\":true,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV5Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V4StepRecord>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV5Builder {
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

    fn push_record(&mut self, record: V4StepRecord, context: &str) -> Result<(), String> {
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
        self.push_record(
            V4StepRecord::new(op, 1, input, 0, output, PARAM_NONE, vec![])?,
            "MathProgramV5Builder.addUnary",
        )
    }

    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        self.push_record(
            V4StepRecord::new(op, 2, lhs, rhs, output, PARAM_NONE, vec![])?,
            "MathProgramV5Builder.addBinary",
        )
    }

    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
        self.push_record(
            V4StepRecord::new(
                OP_CLAMP,
                1,
                input,
                0,
                output,
                RECORD_PARAM_CLAMP,
                scalar_payload(min, max),
            )?,
            "MathProgramV5Builder.addClamp",
        )
    }

    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.push_record(
            V4StepRecord::new(
                OP_COSINE_SIMILARITY,
                2,
                lhs,
                rhs,
                output,
                RECORD_PARAM_EPSILON,
                scalar_payload(epsilon, 0.0),
            )?,
            "MathProgramV5Builder.addCosineSimilarity",
        )
    }

    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::reshape(shape)?;
        self.push_record(
            V4StepRecord::new(
                OP_RESHAPE,
                1,
                input,
                0,
                output,
                PARAM_RESHAPE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV5Builder.addReshape",
        )
    }

    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
        let params = FixedShapeParams::permute(axes)?;
        self.push_record(
            V4StepRecord::new(
                OP_PERMUTE,
                1,
                input,
                0,
                output,
                PARAM_PERMUTE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV5Builder.addPermute",
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
        self.push_record(
            V4StepRecord::new(
                OP_SLICE,
                1,
                input,
                0,
                output,
                PARAM_SLICE_RANK4,
                params.encode().to_vec(),
            )?,
            "MathProgramV5Builder.addSlice",
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
        self.push_record(
            V4StepRecord::new(
                OP_SELECT_AXIS,
                1,
                input,
                0,
                output,
                PARAM_SELECT_AXIS,
                params.encode(),
            )?,
            "MathProgramV5Builder.addSelectAxis",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV5Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV5Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV5, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV5Builder.compile: output slot is not set".to_string())?;
        let canonical_plan = encode_plan(self.num_inputs, self.num_slots, &self.records, out_slot)?;
        let steps = self
            .records
            .iter()
            .map(compile_record)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MathProgramV5 {
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
pub struct MathProgramV5 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V4StepRecord>,
    steps: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV5 {
    pub fn from_plan(plan: &[u8]) -> Result<Self, String> {
        let (num_inputs, num_slots, records, steps, out_slot) = decode_plan(plan)?;
        let canonical_plan = encode_plan(num_inputs, num_slots, &records, out_slot)?;
        if canonical_plan != plan {
            return Err("MathProgramV5: replay plan is not canonical".into());
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
                "MathProgramV5.runInputs: expected {} inputs, got {}",
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
                ExecutableStep::Core {
                    program,
                    arity,
                    in_a,
                    in_b,
                    out,
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV5.runInputs: step {index} input slot {in_a} is empty"
                        )
                    })?;
                    let value = match *arity {
                        1 => program.run1(a),
                        2 => {
                            let b = slots[*in_b as usize].as_ref().ok_or_else(|| {
                                format!(
                                    "MathProgramV5.runInputs: step {index} second input slot {in_b} is empty"
                                )
                            })?;
                            program.run2(a, b)
                        }
                        other => Err(format!(
                            "MathProgramV5.runInputs: step {index} unsupported arity {other}"
                        )),
                    }?;
                    (*out, value)
                }
                ExecutableStep::SelectAxis {
                    program,
                    input,
                    out,
                } => {
                    let value = slots[*input as usize].as_ref().ok_or_else(|| {
                        format!(
                            "MathProgramV5.runInputs: step {index} input slot {input} is empty"
                        )
                    })?;
                    (*out, program.run1(value)?)
                }
            };
            slots[out as usize] = Some(value);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV5.runInputs: output slot {} is empty", self.out_slot))
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
