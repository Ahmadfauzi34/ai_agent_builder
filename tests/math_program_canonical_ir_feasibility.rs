use burn_research::math::program::{
    MathProgram, MathProgramBuilder, OP_ABS, OP_ADD,
};
use burn_research::math::program_v4::{MathProgramV4, MathProgramV4Builder};
use burn_research::math::program_v5::{MathProgramV5, MathProgramV5Builder};
use burn_research::math::program_v6::{MathProgramV6, MathProgramV6Builder};
use burn_research::math::program_v7::{MathProgramV7, MathProgramV7Builder};
use burn_research::math::program_v8::{MathProgramV8, MathProgramV8Builder};
use burn_research::math::program_v9::{MathProgramV9, MathProgramV9Builder};

const PLAN_HEADER_BYTES: usize = 8;
const PLAN_OUTPUT_BYTES: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
struct AuditStep {
    op: u8,
    arity: u8,
    in_a: u8,
    in_b: u8,
    out: u8,
    param_kind: u8,
    raw_payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AuditPlan {
    source_version: u8,
    num_inputs: u8,
    num_slots: u8,
    steps: Vec<AuditStep>,
    output_slot: u8,
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn normalize_for_audit(plan: &[u8]) -> Result<AuditPlan, String> {
    if plan.len() < PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES {
        return Err(format!(
            "audit: truncated plan: expected at least {} bytes, got {}",
            PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES,
            plan.len()
        ));
    }
    if &plan[..4] != b"BRMP" {
        return Err("audit: wrong Math Program magic".into());
    }

    let version = plan[4];
    if !(1..=9).contains(&version) {
        return Err(format!("audit: unsupported version {version}"));
    }

    let num_inputs = plan[5];
    let num_slots = plan[6];
    let num_steps = plan[7] as usize;
    let mut cursor = PLAN_HEADER_BYTES;
    let mut steps = Vec::with_capacity(num_steps);

    for _ in 0..num_steps {
        let fixed_payload_len = match version {
            1 => Some(0usize),
            2 => Some(8usize),
            3 => Some(32usize),
            _ => None,
        };

        if let Some(payload_len) = fixed_payload_len {
            let record_len = 5 + if version == 1 { 0 } else { 1 + payload_len };
            let end = cursor
                .checked_add(record_len)
                .ok_or_else(|| "audit: fixed record length overflow".to_string())?;
            if end > plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
                return Err("audit: truncated fixed-width record".into());
            }

            let record = &plan[cursor..end];
            let (param_kind, raw_payload) = if version == 1 {
                (0, Vec::new())
            } else {
                (record[5], record[6..].to_vec())
            };

            steps.push(AuditStep {
                op: record[0],
                arity: record[1],
                in_a: record[2],
                in_b: record[3],
                out: record[4],
                param_kind,
                raw_payload,
            });
            cursor = end;
            continue;
        }

        let header_end = cursor
            .checked_add(10)
            .ok_or_else(|| "audit: variable record header overflow".to_string())?;
        if header_end > plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err("audit: truncated variable-width record header".into());
        }

        let record = &plan[cursor..];
        let payload_len = read_u32(&record[6..10]) as usize;
        let record_len = 10usize
            .checked_add(payload_len)
            .ok_or_else(|| "audit: variable record length overflow".to_string())?;
        let end = cursor
            .checked_add(record_len)
            .ok_or_else(|| "audit: variable record end overflow".to_string())?;
        if end > plan.len().saturating_sub(PLAN_OUTPUT_BYTES) {
            return Err("audit: truncated variable-width record payload".into());
        }

        steps.push(AuditStep {
            op: record[0],
            arity: record[1],
            in_a: record[2],
            in_b: record[3],
            out: record[4],
            param_kind: record[5],
            raw_payload: plan[cursor + 10..end].to_vec(),
        });
        cursor = end;
    }

    if cursor + PLAN_OUTPUT_BYTES != plan.len() {
        return Err(format!(
            "audit: trailing or missing bytes after {} decoded steps: cursor={cursor}, len={}",
            steps.len(),
            plan.len()
        ));
    }

    Ok(AuditPlan {
        source_version: version,
        num_inputs,
        num_slots,
        steps,
        output_slot: plan[cursor],
    })
}

fn assert_common_shape(plan: &AuditPlan, version: u8, expected_inputs: u8) {
    assert_eq!(plan.source_version, version);
    assert_eq!(plan.num_inputs, expected_inputs);
    assert!(!plan.steps.is_empty());
    assert!(plan.num_slots > plan.num_inputs);
    assert!(plan.output_slot < plan.num_slots);
}

#[test]
fn math_program_v1_to_v9_normalize_into_one_structural_ir_and_replay_exactly() {
    // v1: plain opcode / no parameter payload.
    let mut v1_builder = MathProgramBuilder::new(1, 2).unwrap();
    v1_builder.add_unary(OP_ABS, 0, 1).unwrap();
    v1_builder.set_output(1).unwrap();
    let v1 = v1_builder.compile().unwrap();
    let v1_plan = v1.program_plan();
    let v1_ir = normalize_for_audit(&v1_plan).unwrap();
    assert_common_shape(&v1_ir, 1, 1);
    assert_eq!(v1_ir.steps[0].raw_payload, Vec::<u8>::new());
    let v1_replay = MathProgram::from_plan(&v1_plan).unwrap();
    assert_eq!(v1_replay.program_plan(), v1_plan);
    assert_eq!(v1_replay.program_identity(), v1.program_identity());

    // v2: scalar parameter area.
    let mut v2_builder = MathProgramBuilder::new(1, 2).unwrap();
    v2_builder.add_clamp(0, 1, -1.0, 1.0).unwrap();
    v2_builder.set_output(1).unwrap();
    let v2 = v2_builder.compile().unwrap();
    let v2_plan = v2.program_plan();
    let v2_ir = normalize_for_audit(&v2_plan).unwrap();
    assert_common_shape(&v2_ir, 2, 1);
    assert_eq!(v2_ir.steps[0].raw_payload.len(), 8);
    let v2_replay = MathProgram::from_plan(&v2_plan).unwrap();
    assert_eq!(v2_replay.program_plan(), v2_plan);
    assert_eq!(v2_replay.program_identity(), v2.program_identity());

    // v3: fixed rank-4 shape metadata area.
    let mut v3_builder = MathProgramBuilder::new(1, 2).unwrap();
    v3_builder.add_reshape(0, 1, &[1, 1, 1, 1]).unwrap();
    v3_builder.set_output(1).unwrap();
    let v3 = v3_builder.compile().unwrap();
    let v3_plan = v3.program_plan();
    let v3_ir = normalize_for_audit(&v3_plan).unwrap();
    assert_common_shape(&v3_ir, 3, 1);
    assert_eq!(v3_ir.steps[0].raw_payload.len(), 32);
    let v3_replay = MathProgram::from_plan(&v3_plan).unwrap();
    assert_eq!(v3_replay.program_plan(), v3_plan);
    assert_eq!(v3_replay.program_identity(), v3.program_identity());

    // v4: first self-delimiting record family via selectAxis.
    let mut v4_builder = MathProgramV4Builder::new(1, 2).unwrap();
    v4_builder.add_select_axis(0, 1, 0, &[0]).unwrap();
    v4_builder.set_output(1).unwrap();
    let v4 = v4_builder.compile().unwrap();
    let v4_plan = v4.program_plan();
    let v4_ir = normalize_for_audit(&v4_plan).unwrap();
    assert_common_shape(&v4_ir, 4, 1);
    assert!(!v4_ir.steps[0].raw_payload.is_empty());
    let v4_replay = MathProgramV4::from_plan(&v4_plan).unwrap();
    assert_eq!(v4_replay.program_plan(), v4_plan);
    assert_eq!(v4_replay.program_identity(), v4.program_identity());

    // v5: wider external-input envelope, same self-delimiting record structure.
    let mut v5_builder = MathProgramV5Builder::new(3, 4).unwrap();
    v5_builder.add_binary(OP_ADD, 0, 1, 3).unwrap();
    v5_builder.set_output(3).unwrap();
    let v5 = v5_builder.compile().unwrap();
    let v5_plan = v5.program_plan();
    let v5_ir = normalize_for_audit(&v5_plan).unwrap();
    assert_common_shape(&v5_ir, 5, 3);
    let v5_replay = MathProgramV5::from_plan(&v5_plan).unwrap();
    assert_eq!(v5_replay.program_plan(), v5_plan);
    assert_eq!(v5_replay.program_identity(), v5.program_identity());

    // v6: identity-bound value source.
    let mut v6_builder = MathProgramV6Builder::new(1, 2).unwrap();
    v6_builder.add_fill_like(0, 1, 0.5).unwrap();
    v6_builder.set_output(1).unwrap();
    let v6 = v6_builder.compile().unwrap();
    let v6_plan = v6.program_plan();
    let v6_ir = normalize_for_audit(&v6_plan).unwrap();
    assert_common_shape(&v6_ir, 6, 1);
    assert_eq!(v6_ir.steps[0].arity, 1);
    assert!(!v6_ir.steps[0].raw_payload.is_empty());
    let v6_replay = MathProgramV6::from_plan(&v6_plan).unwrap();
    assert_eq!(v6_replay.program_plan(), v6_plan);
    assert_eq!(v6_replay.program_identity(), v6.program_identity());

    // v7: explicit broadcast/expand relation.
    let mut v7_builder = MathProgramV7Builder::new(2, 3).unwrap();
    v7_builder.add_expand_like(0, 1, 2).unwrap();
    v7_builder.set_output(2).unwrap();
    let v7 = v7_builder.compile().unwrap();
    let v7_plan = v7.program_plan();
    let v7_ir = normalize_for_audit(&v7_plan).unwrap();
    assert_common_shape(&v7_ir, 7, 2);
    assert_eq!(v7_ir.steps[0].arity, 2);
    let v7_replay = MathProgramV7::from_plan(&v7_plan).unwrap();
    assert_eq!(v7_replay.program_plan(), v7_plan);
    assert_eq!(v7_replay.program_identity(), v7.program_identity());

    // v8: identity-bound generic reduction axis.
    let mut v8_builder = MathProgramV8Builder::new(1, 2).unwrap();
    v8_builder.add_sum_axis(0, 1, 2).unwrap();
    v8_builder.set_output(1).unwrap();
    let v8 = v8_builder.compile().unwrap();
    let v8_plan = v8.program_plan();
    let v8_ir = normalize_for_audit(&v8_plan).unwrap();
    assert_common_shape(&v8_ir, 8, 1);
    assert!(!v8_ir.steps[0].raw_payload.is_empty());
    let v8_replay = MathProgramV8::from_plan(&v8_plan).unwrap();
    assert_eq!(v8_replay.program_plan(), v8_plan);
    assert_eq!(v8_replay.program_identity(), v8.program_identity());

    // v9: positional index source, still the same self-delimiting structural record.
    let mut v9_builder = MathProgramV9Builder::new(1, 2).unwrap();
    v9_builder.add_indices_like(0, 1, 2).unwrap();
    v9_builder.set_output(1).unwrap();
    let v9 = v9_builder.compile().unwrap();
    let v9_plan = v9.program_plan();
    let v9_ir = normalize_for_audit(&v9_plan).unwrap();
    assert_common_shape(&v9_ir, 9, 1);
    assert!(!v9_ir.steps[0].raw_payload.is_empty());
    let v9_replay = MathProgramV9::from_plan(&v9_plan).unwrap();
    assert_eq!(v9_replay.program_plan(), v9_plan);
    assert_eq!(v9_replay.program_identity(), v9.program_identity());

    // The structural IR is intentionally common, but source-version identity remains explicit.
    let all = [
        v1_ir, v2_ir, v3_ir, v4_ir, v5_ir, v6_ir, v7_ir, v8_ir, v9_ir,
    ];
    assert_eq!(
        all.iter().map(|plan| plan.source_version).collect::<Vec<_>>(),
        (1u8..=9).collect::<Vec<_>>()
    );
}

#[test]
fn audit_normalizer_rejects_trailing_or_truncated_envelopes() {
    let mut builder = MathProgramV9Builder::new(1, 2).unwrap();
    builder.add_indices_like(0, 1, 0).unwrap();
    builder.set_output(1).unwrap();
    let plan = builder.compile().unwrap().program_plan();

    let mut trailing = plan.clone();
    trailing.push(0xFF);
    assert!(normalize_for_audit(&trailing).is_err());

    let mut truncated = plan;
    truncated.pop();
    assert!(normalize_for_audit(&truncated).is_err());
}
