use burn_research::math::{
    math_program_capabilities, MathProgram, MathProgramBuilder, MathProgramV4Builder,
    MathProgramV5, MathProgramV5Builder, MAX_V5_EXTERNAL_INPUTS, MIN_V5_EXTERNAL_INPUTS, OP_ADD,
    OP_MUL,
};
use burn_research::WasmTensor;

fn tensor(values: &[f32]) -> WasmTensor {
    WasmTensor::new(values, &[1, values.len(), 1, 1])
}

fn scalar(value: f32) -> WasmTensor {
    WasmTensor::new(&[value], &[1, 1, 1, 1])
}

#[test]
fn three_external_inputs_fan_in_and_replay_deterministically() {
    let mut builder = MathProgramV5Builder::new(3, 5).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 3).unwrap();
    builder.add_binary(OP_ADD, 3, 2, 4).unwrap();
    builder.set_output(4).unwrap();
    let program = builder.compile().unwrap();

    assert_eq!(program.num_inputs(), 3);
    assert_eq!(program.program_plan()[4], 5);
    assert_eq!(program.program_plan()[5], 3);

    let inputs = [tensor(&[1.0, 2.0]), tensor(&[3.0, 4.0]), tensor(&[5.0, 6.0])];
    assert_eq!(program.run_inputs(&inputs).unwrap().to_array(), vec![9.0, 12.0]);

    let replay = MathProgramV5::from_plan(&program.program_plan()).unwrap();
    assert_eq!(replay.program_plan(), program.program_plan());
    assert_eq!(replay.program_identity(), program.program_identity());
    assert_eq!(replay.run_inputs(&inputs).unwrap().to_array(), vec![9.0, 12.0]);
}

#[test]
fn four_inputs_can_branch_and_merge_without_packing() {
    let mut builder = MathProgramV5Builder::new(4, 7).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 4).unwrap();
    builder.add_binary(OP_MUL, 2, 3, 5).unwrap();
    builder.add_binary(OP_ADD, 4, 5, 6).unwrap();
    builder.set_output(6).unwrap();
    let program = builder.compile().unwrap();

    let inputs = [
        tensor(&[1.0, 2.0]),
        tensor(&[3.0, 4.0]),
        tensor(&[2.0, 3.0]),
        tensor(&[4.0, 5.0]),
    ];
    assert_eq!(program.run_inputs(&inputs).unwrap().to_array(), vec![12.0, 21.0]);
}

#[test]
fn eight_inputs_is_the_explicit_bounded_maximum() {
    assert_eq!(MIN_V5_EXTERNAL_INPUTS, 3);
    assert_eq!(MAX_V5_EXTERNAL_INPUTS, 8);

    let mut builder = MathProgramV5Builder::new(8, 15).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 8).unwrap();
    builder.add_binary(OP_ADD, 8, 2, 9).unwrap();
    builder.add_binary(OP_ADD, 9, 3, 10).unwrap();
    builder.add_binary(OP_ADD, 10, 4, 11).unwrap();
    builder.add_binary(OP_ADD, 11, 5, 12).unwrap();
    builder.add_binary(OP_ADD, 12, 6, 13).unwrap();
    builder.add_binary(OP_ADD, 13, 7, 14).unwrap();
    builder.set_output(14).unwrap();
    let program = builder.compile().unwrap();

    let inputs = [
        scalar(1.0),
        scalar(2.0),
        scalar(3.0),
        scalar(4.0),
        scalar(5.0),
        scalar(6.0),
        scalar(7.0),
        scalar(8.0),
    ];
    assert_eq!(program.run_inputs(&inputs).unwrap().to_array(), vec![36.0]);

    assert!(MathProgramV5Builder::new(9, 10).is_err());
    assert!(MathProgramV5Builder::new(2, 3).is_err());
}

#[test]
fn select_axis_reuses_v4_inside_three_input_program() {
    let mut builder = MathProgramV5Builder::new(3, 6).unwrap();
    builder.add_select_axis(0, 3, 1, &[2, 0]).unwrap();
    builder.add_binary(OP_ADD, 1, 2, 4).unwrap();
    builder.add_binary(OP_ADD, 3, 4, 5).unwrap();
    builder.set_output(5).unwrap();
    let program = builder.compile().unwrap();

    let inputs = [
        tensor(&[10.0, 20.0, 30.0]),
        tensor(&[1.0, 1.0]),
        tensor(&[2.0, 2.0]),
    ];
    assert_eq!(program.run_inputs(&inputs).unwrap().to_array(), vec![33.0, 13.0]);
}

#[test]
fn wrong_runtime_input_count_is_controlled_and_program_is_reusable() {
    let mut builder = MathProgramV5Builder::new(3, 5).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 3).unwrap();
    builder.add_binary(OP_ADD, 3, 2, 4).unwrap();
    builder.set_output(4).unwrap();
    let program = builder.compile().unwrap();
    let identity = program.program_identity();

    let too_few = [scalar(1.0), scalar(2.0)];
    assert!(program.run_inputs(&too_few).is_err());
    assert_eq!(program.program_identity(), identity);

    let valid = [scalar(1.0), scalar(2.0), scalar(3.0)];
    assert_eq!(program.run_inputs(&valid).unwrap().to_array(), vec![6.0]);
    assert_eq!(program.program_identity(), identity);
}

#[test]
fn legacy_v1_to_v4_input_contract_remains_frozen() {
    assert!(math_program_capabilities().contains("\"inputs\":\"one_or_two\""));

    let mut legacy = MathProgramBuilder::new(2, 3).unwrap();
    legacy.add_binary(OP_ADD, 0, 1, 2).unwrap();
    legacy.set_output(2).unwrap();
    let legacy = legacy.compile().unwrap();
    assert_eq!(legacy.program_plan()[4], 1);
    assert_eq!(MathProgram::from_plan(&legacy.program_plan()).unwrap().program_plan(), legacy.program_plan());
    assert!(MathProgramBuilder::new(3, 4).is_err());

    let mut v4 = MathProgramV4Builder::new(2, 3).unwrap();
    v4.add_select_axis(0, 2, 1, &[0]).unwrap();
    v4.set_output(2).unwrap();
    let v4 = v4.compile().unwrap();
    assert_eq!(v4.program_plan()[4], 4);
    assert!(MathProgramV4Builder::new(3, 4).is_err());
}
