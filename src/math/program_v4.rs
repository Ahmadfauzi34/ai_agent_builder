use crate::math::program::{
    MathProgram, MathProgramBuilder, OP_CLAMP, OP_COSINE_SIMILARITY, OP_MEAN, OP_PERMUTE,
    OP_RESHAPE, OP_SLICE,
};
use crate::math::program_select_params::{SelectAxisParams, PARAM_SELECT_AXIS};
use crate::math::program_shape_params::{
    FixedShapeParams, PARAM_PERMUTE_RANK4, PARAM_RESHAPE_RANK4, PARAM_SLICE_RANK4,
};
use crate::math::program_v4_step::{
    V4StepRecord, PARAM_CLAMP as V4_PARAM_CLAMP, PARAM_EPSILON as V4_PARAM_EPSILON,
    PARAM_NONE,
};
use crate::math::WasmTensorTransform;
use crate::WasmTensor;

const PLAN_MAGIC: &[u8; 4] = b"BRMP";
const PLAN_VERSION_V4: u8 = 4;
const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;
const MAX_STEPS: usize = u8::MAX as usize;
const MAX_V4_PLAN_BYTES: usize = 4 * 1024 * 1024;

pub const OP_SELECT_AXIS: u8 = 0x24;

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
        params: SelectAxisParams,
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
    MathProgramBuilder::new(num_inputs, num_slots).map(|_| ())
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
            "MathProgramV4: scalar payload must contain 8 bytes, got {}",
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
                "MathProgramV4: unsupported core arity {arity}; expected 1 or 2"
            )),
        },
        V4_PARAM_CLAMP => {
            if record.op != OP_CLAMP || record.arity != 1 {
                return Err("MathProgramV4: clamp payload must use unary clamp opcode".into());
            }
            let (min, max) = parse_scalar_payload(&record.payload)?;
            if scalar_payload(min, max) != record.payload {
                return Err("MathProgramV4: clamp payload is not canonically encoded".into());
            }
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_clamp(0, 1, min, max)?;
            builder.set_output(1)?;
            builder.compile()
        }
        V4_PARAM_EPSILON => {
            if record.op != OP_COSINE_SIMILARITY || record.arity != 2 {
                return Err(
                    "MathProgramV4: epsilon payload must use binary cosineSimilarity opcode"
                        .into(),
                );
            }
            let (epsilon, reserved) = parse_scalar_payload(&record.payload)?;
            if reserved.to_bits() != 0 || scalar_payload(epsilon, 0.0) != record.payload {
                return Err("MathProgramV4: cosine epsilon payload is not canonical".into());
            }
            let mut builder = MathProgramBuilder::new(2, 3)?;
            builder.add_cosine_similarity(0, 1, 2, epsilon)?;
            builder.set_output(2)?;
            builder.compile()
        }
        PARAM_RESHAPE_RANK4 => {
            if record.op != OP_RESHAPE || record.arity != 1 {
                return Err("MathProgramV4: reshape payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let shape = params
                .reshape_usize()
                .ok_or_else(|| "MathProgramV4: reshape metadata mismatch".to_string())?;
            let shape = shape.map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_reshape(0, 1, &shape)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_PERMUTE_RANK4 => {
            if record.op != OP_PERMUTE || record.arity != 1 {
                return Err("MathProgramV4: permute payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let axes = params
                .permute_usize()
                .ok_or_else(|| "MathProgramV4: permute metadata mismatch".to_string())?;
            let axes = axes.map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_permute(0, 1, &axes)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_SLICE_RANK4 => {
            if record.op != OP_SLICE || record.arity != 1 {
                return Err("MathProgramV4: slice payload/opcode mismatch".into());
            }
            let params = FixedShapeParams::decode(record.param_kind, &record.payload)?;
            let (starts, ends) = params
                .slice_usize()
                .ok_or_else(|| "MathProgramV4: slice metadata mismatch".to_string())?;
            let starts = starts.map(|value| value as u32);
            let ends = ends.map(|value| value as u32);
            let mut builder = MathProgramBuilder::new(1, 2)?;
            builder.add_slice(0, 1, &starts, &ends)?;
            builder.set_output(1)?;
            builder.compile()
        }
        PARAM_SELECT_AXIS => Err("MathProgramV4: selectAxis is not a core v1-v3 step".into()),
        kind => Err(format!(
            "MathProgramV4: unsupported parameter kind {kind}; fail-closed"
        )),
    }
}

fn compile_record(record: &V4StepRecord) -> Result<ExecutableStep, String> {
    if record.param_kind == PARAM_SELECT_AXIS {
        if record.op != OP_SELECT_AXIS || record.arity != 1 || record.in_b != 0 {
            return Err(
                "MathProgramV4: selectAxis requires unary opcode 0x24 with canonical in_b=0"
                    .into(),
            );
        }
        let params = SelectAxisParams::decode(&record.payload)?;
        return Ok(ExecutableStep::SelectAxis {
            params,
            input: record.in_a,
            out: record.out,
        });
    }

    let program = compile_core_record(record)?;
    Ok(ExecutableStep::Core {
        program,
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

fn encode_plan(
    num_inputs: u8,
    num_slots: u8,
    records: &[V4StepRecord],
    out_slot: u8,
) -> Result<Vec<u8>, String> {
    validate_program_shape(num_inputs, num_slots)?;
    if records.is_empty() {
        return Err("MathProgramV4: program must contain at least one step".into());
    }
    if records.len() > MAX_STEPS {
        return Err(format!(
            "MathProgramV4: step count {} exceeds maximum {MAX_STEPS}",
            records.len()
        ));
    }
    if !records.iter().any(|record| record.param_kind == PARAM_SELECT_AXIS) {
        return Err(
            "MathProgramV4: v4 is noncanonical without selectAxis; use MathProgram v1-v3"
                .into(),
        );
    }

    let records_bytes = records.iter().try_fold(0usize, |total, record| {
        total
            .checked_add(record.encoded_len())
            .ok_or_else(|| "MathProgramV4: plan length overflow".to_string())
    })?;
    let plan_len = PLAN_HEADER_BYTES
        .checked_add(records_bytes)
        .and_then(|value| value.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "MathProgramV4: plan length overflow".to_string())?;
    if plan_len > MAX_V4_PLAN_BYTES {
        return Err(format!(
            "MathProgramV4: plan size {plan_len} exceeds maximum {MAX_V4_PLAN_BYTES} bytes"
        ));
    }

    let mut plan = Vec::with_capacity(plan_len);
    plan.extend_from_slice(PLAN_MAGIC);
    plan.push(PLAN_VERSION_V4);
    plan.push(num_inputs);
    plan.push(num_slots);
    plan.push(records.len() as u8);
    for record in records {
        plan.extend_from_slice(&record.encode());
    }
    plan.push(out_slot);
    Ok(plan)
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

pub fn math_program_v4_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.math-program.v4\",",
        "\"plan_schema\":\"burn-research.math-program-plan.v4\",",
        "\"identity_schema\":\"burn-research.math-program-identity.v1\",",
        "\"requires_select_axis\":true,",
        "\"max_plan_bytes\":4194304,",
        "\"slot_semantics\":\"write_once_read_after_write\",",
        "\"core_execution_reuse\":\"MathProgram.v1-v3.one-step\",",
        "\"select_axis_opcode\":36,",
        "\"registry_dependency\":false,",
        "\"mutable_state\":false",
        "}"
    )
    .to_string()
}

#[derive(Clone, Debug)]
pub struct MathProgramV4Builder {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V4StepRecord>,
    filled: u64,
    written: u64,
    out_slot: Option<u8>,
}

impl MathProgramV4Builder {
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
            "MathProgramV4Builder.addUnary",
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
            "MathProgramV4Builder.addBinary",
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
                V4_PARAM_CLAMP,
                scalar_payload(min, max),
            )?,
            "MathProgramV4Builder.addClamp",
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
                V4_PARAM_EPSILON,
                scalar_payload(epsilon, 0.0),
            )?,
            "MathProgramV4Builder.addCosineSimilarity",
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
                params.kind(),
                params.encode().to_vec(),
            )?,
            "MathProgramV4Builder.addReshape",
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
                params.kind(),
                params.encode().to_vec(),
            )?,
            "MathProgramV4Builder.addPermute",
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
                params.kind(),
                params.encode().to_vec(),
            )?,
            "MathProgramV4Builder.addSlice",
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
            "MathProgramV4Builder.addSelectAxis",
        )
    }

    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        if slot >= self.num_slots {
            return Err(format!(
                "MathProgramV4Builder.setOutput: slot {slot} is out of range for num_slots={}",
                self.num_slots
            ));
        }
        if self.written & bit(slot) == 0 {
            return Err(format!(
                "MathProgramV4Builder.setOutput: slot {slot} must be produced by a program step"
            ));
        }
        self.out_slot = Some(slot);
        Ok(())
    }

    pub fn compile(&self) -> Result<MathProgramV4, String> {
        let out_slot = self
            .out_slot
            .ok_or_else(|| "MathProgramV4Builder.compile: output slot is not set".to_string())?;
        MathProgramV4::from_records(
            self.num_inputs,
            self.num_slots,
            self.records.clone(),
            out_slot,
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
}

#[derive(Clone, Debug)]
pub struct MathProgramV4 {
    num_inputs: u8,
    num_slots: u8,
    records: Vec<V4StepRecord>,
    executable: Vec<ExecutableStep>,
    out_slot: u8,
    canonical_plan: Vec<u8>,
}

impl MathProgramV4 {
    fn from_records(
        num_inputs: u8,
        num_slots: u8,
        records: Vec<V4StepRecord>,
        out_slot: u8,
    ) -> Result<Self, String> {
        validate_program_shape(num_inputs, num_slots)?;
        let mut filled = initial_filled(num_inputs);
        let mut written = 0u64;
        let mut executable = Vec::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            let compiled = compile_record(record)?;
            (filled, written) = validate_topology(
                record,
                num_slots,
                filled,
                written,
                &format!("MathProgramV4 step {index}"),
            )?;
            executable.push(compiled);
        }
        if out_slot >= num_slots || written & bit(out_slot) == 0 {
            return Err(format!(
                "MathProgramV4: output slot {out_slot} must be in range and produced by a step"
            ));
        }
        let canonical_plan = encode_plan(num_inputs, num_slots, &records, out_slot)?;
        Ok(Self {
            num_inputs,
            num_slots,
            records,
            executable,
            out_slot,
            canonical_plan,
        })
    }

    pub fn from_plan(plan: &[u8]) -> Result<Self, String> {
        if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
            return Err("MathProgramV4: plan is truncated".into());
        }
        if plan.len() > MAX_V4_PLAN_BYTES {
            return Err(format!(
                "MathProgramV4: plan size {} exceeds maximum {MAX_V4_PLAN_BYTES} bytes",
                plan.len()
            ));
        }
        if &plan[..4] != PLAN_MAGIC {
            return Err("MathProgramV4: invalid plan magic".into());
        }
        if plan[4] != PLAN_VERSION_V4 {
            return Err(format!(
                "MathProgramV4: unsupported plan version {}; expected 4",
                plan[4]
            ));
        }
        let num_inputs = plan[5];
        let num_slots = plan[6];
        let num_steps = plan[7] as usize;
        validate_program_shape(num_inputs, num_slots)?;
        if num_steps == 0 {
            return Err("MathProgramV4: plan must contain at least one step".into());
        }

        let mut offset = PLAN_HEADER_BYTES;
        let mut records = Vec::with_capacity(num_steps);
        for _ in 0..num_steps {
            if offset >= plan.len() {
                return Err("MathProgramV4: plan is truncated before output slot".into());
            }
            let (record, consumed) = V4StepRecord::decode_prefix(&plan[offset..])?;
            offset = offset
                .checked_add(consumed)
                .ok_or_else(|| "MathProgramV4: plan offset overflow".to_string())?;
            records.push(record);
        }
        if offset + PLAN_OUTPUT_BYTES != plan.len() {
            return Err(format!(
                "MathProgramV4: malformed plan tail: expected one output byte after steps, got {} bytes",
                plan.len().saturating_sub(offset)
            ));
        }
        let out_slot = plan[offset];
        let program = Self::from_records(num_inputs, num_slots, records, out_slot)?;
        if program.canonical_plan != plan {
            return Err("MathProgramV4: replay plan is not canonical".into());
        }
        Ok(program)
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        if self.num_inputs != 1 {
            return Err(format!(
                "MathProgramV4.run1: program requires {} inputs",
                self.num_inputs
            ));
        }
        self.execute(&[input.clone()])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        if self.num_inputs != 2 {
            return Err(format!(
                "MathProgramV4.run2: program requires {} inputs",
                self.num_inputs
            ));
        }
        self.execute(&[a.clone(), b.clone()])
    }

    fn execute(&self, inputs: &[WasmTensor]) -> Result<WasmTensor, String> {
        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (index, input) in inputs.iter().enumerate() {
            slots[index] = Some(input.clone());
        }
        let tensor = WasmTensorTransform::new();

        for (index, step) in self.executable.iter().enumerate() {
            let output = match step {
                ExecutableStep::Core {
                    program,
                    arity,
                    in_a,
                    in_b,
                    ..
                } => {
                    let a = slots[*in_a as usize].as_ref().ok_or_else(|| {
                        format!("MathProgramV4.run step {index}: input slot {in_a} is empty")
                    })?;
                    if *arity == 1 {
                        program.run1(a)
                    } else {
                        let b = slots[*in_b as usize].as_ref().ok_or_else(|| {
                            format!(
                                "MathProgramV4.run step {index}: second input slot {in_b} is empty"
                            )
                        })?;
                        program.run2(a, b)
                    }
                }
                ExecutableStep::SelectAxis { params, input, .. } => {
                    let a = slots[*input as usize].as_ref().ok_or_else(|| {
                        format!("MathProgramV4.run step {index}: input slot {input} is empty")
                    })?;
                    let indices = params.indices_usize();
                    tensor.select_axis(a, params.axis_usize(), &indices)
                }
            }
            .map_err(|error| format!("MathProgramV4.run step {index}: {error}"))?;

            let out = match step {
                ExecutableStep::Core { out, .. } | ExecutableStep::SelectAxis { out, .. } => *out,
            };
            slots[out as usize] = Some(output);
        }

        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("MathProgramV4.run: output slot {} is empty", self.out_slot))
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

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn select_axis_then_existing_statistics_op_executes_through_core_reuse() {
        let mut builder = MathProgramV4Builder::new(1, 3).unwrap();
        builder.add_select_axis(0, 1, 1, &[2, 0]).unwrap();
        builder.add_unary(OP_MEAN, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();

        let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let output = program.run1(&input).unwrap();
        assert_close(&output.to_array(), &[2.0], 1e-6);
    }

    #[test]
    fn v4_plan_replays_byte_identically_with_same_identity() {
        let mut builder = MathProgramV4Builder::new(1, 3).unwrap();
        builder.add_select_axis(0, 1, 1, &[1, 0]).unwrap();
        builder.add_clamp(1, 2, -1.0, 1.0).unwrap();
        builder.set_output(2).unwrap();
        let program = builder.compile().unwrap();
        assert_eq!(program.program_plan()[4], PLAN_VERSION_V4);

        let replay = MathProgramV4::from_plan(&program.program_plan()).unwrap();
        assert_eq!(replay.program_plan(), program.program_plan());
        assert_eq!(replay.program_identity(), program.program_identity());
    }

    #[test]
    fn v4_without_select_axis_is_noncanonical() {
        let mut builder = MathProgramV4Builder::new(1, 2).unwrap();
        builder.add_unary(OP_MEAN, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        assert!(builder.compile().is_err());
    }

    #[test]
    fn invalid_select_parameters_are_atomic() {
        let mut builder = MathProgramV4Builder::new(1, 2).unwrap();
        assert!(builder.add_select_axis(0, 1, 4, &[0]).is_err());
        assert!(builder.add_select_axis(0, 1, 1, &[]).is_err());
        assert_eq!(builder.num_steps(), 0);
        builder.add_select_axis(0, 1, 1, &[0]).unwrap();
        assert_eq!(builder.num_steps(), 1);
    }

    #[test]
    fn input_dependent_select_failure_preserves_identity_and_reusability() {
        let mut builder = MathProgramV4Builder::new(1, 2).unwrap();
        builder.add_select_axis(0, 1, 1, &[3]).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let identity = program.program_identity();

        let too_small = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        assert!(program.run1(&too_small).is_err());
        assert_eq!(program.program_identity(), identity);

        let valid = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
        assert_eq!(program.run1(&valid).unwrap().to_array(), vec![4.0]);
        assert_eq!(program.program_identity(), identity);
    }

    #[test]
    fn malformed_or_lower_version_replay_fails_closed() {
        let mut builder = MathProgramV4Builder::new(1, 2).unwrap();
        builder.add_select_axis(0, 1, 1, &[0]).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();

        let mut wrong_version = program.program_plan();
        wrong_version[4] = 3;
        assert!(MathProgramV4::from_plan(&wrong_version).is_err());

        let mut trailing = program.program_plan();
        trailing.push(0);
        assert!(MathProgramV4::from_plan(&trailing).is_err());

        let mut truncated = program.program_plan();
        truncated.pop();
        assert!(MathProgramV4::from_plan(&truncated).is_err());
    }

    #[test]
    fn capabilities_document_v4_boundary() {
        let caps = math_program_v4_capabilities();
        assert!(caps.contains("burn-research.math-program.v4"));
        assert!(caps.contains("\"requires_select_axis\":true"));
        assert!(caps.contains("MathProgram.v1-v3.one-step"));
        assert!(caps.contains("\"registry_dependency\":false"));
    }
}
