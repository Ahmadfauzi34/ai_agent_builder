//! Fase 2 structural audit: entry-validation + unchecked-step invariant.
//!
//! Fase 2 removed per-step re-validation of intermediate tensors. Program
//! entry points validate inputs once; step dispatch uses input-unchecked
//! kernel twins (`*_unchecked`) that retain output validation and all
//! stronger domain checks (nonzero, positive, normalized, ...).
//!
//! These tests are the structural enforcement for that invariant. They lock:
//!  1. every program version rejects non-finite inputs at entry;
//!  2. every `_unchecked` twin agrees bit-for-bit with its checked twin on
//!     valid inputs;
//!  3. every `_unchecked` twin retains domain checks and output validation;
//!  4. every program version's full step set executes with exact expected
//!     values through the unchecked dispatch path.
//!
//! MAINTENANCE CONTRACT: when adding a new step variant or kernel op that is
//! reachable from a program, extend the tables below. A new producer that
//! forgets output validation will silently poison downstream steps under
//! Fase 2 — these tests are the alarm for that.

use crate::math::program::{
    MathProgramBuilder, OP_ABS, OP_ADD, OP_CROSS_ENTROPY, OP_DIV, OP_DOT, OP_ENTROPY, OP_EXP,
    OP_KL_DIVERGENCE, OP_L2_DISTANCE, OP_L2_NORM, OP_LOG, OP_MATMUL, OP_MAX, OP_MEAN, OP_MIN,
    OP_MUL, OP_NORMALIZE, OP_SQRT, OP_STD_POPULATION, OP_SUB, OP_SUM, OP_VARIANCE_POPULATION,
};
use crate::math::{
    WasmLinearAlgebra, WasmNumericKernel, WasmProbability, WasmStatistics,
};
use crate::WasmTensor;

fn t(values: &[f32], shape: &[usize]) -> WasmTensor {
    WasmTensor::new(values, shape)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {actual:?} vs {expected:?}"
    );
    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "index {index}: {a} != {e} within {tolerance}"
        );
    }
}

fn assert_all_finite(values: &[f32]) {
    for (index, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "non-finite value at index {index}: {v}");
    }
}

fn nan_tensor() -> WasmTensor {
    t(&[1.0, f32::NAN], &[1, 2, 1, 1])
}

fn inf_tensor() -> WasmTensor {
    t(&[1.0, f32::INFINITY], &[1, 2, 1, 1])
}

// ---------------------------------------------------------------------------
// 1. Entry validation: every version rejects non-finite program inputs.
// ---------------------------------------------------------------------------

#[test]
fn fase2_v1_entry_rejects_nonfinite_inputs() {
    let mut builder = MathProgramBuilder::new(1, 2).unwrap();
    builder.add_unary(OP_ABS, 0, 1).unwrap();
    builder.set_output(1).unwrap();
    let program = builder.compile().unwrap();

    for bad in [nan_tensor(), inf_tensor()] {
        let err = program
            .run1(&bad)
            .err()
            .expect("NaN/Inf input must be rejected");
        assert!(
            err.contains("run1 input"),
            "error must come from entry validation, got: {err}"
        );
    }

    let mut builder = MathProgramBuilder::new(2, 3).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let good = t(&[1.0, 2.0], &[1, 2, 1, 1]);
    assert!(program.run2(&good, &nan_tensor()).is_err());
    assert!(program.run2(&nan_tensor(), &good).is_err());
    // Valid inputs still work.
    assert_close(
        &program.run2(&good, &good).unwrap().to_array(),
        &[2.0, 4.0],
        1e-6,
    );
}

#[test]
fn fase2_v4_entry_rejects_nonfinite_inputs() {
    use crate::math::program_v4::MathProgramV4Builder;
    // v4 canonicality requires a selectAxis step; 1 input, 3 slots.
    let mut builder = MathProgramV4Builder::new(1, 3).unwrap();
    builder.add_unary(OP_ABS, 0, 1).unwrap();
    builder.add_select_axis(1, 2, 1, &[0, 1]).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    for bad in [nan_tensor(), inf_tensor()] {
        let err = program
            .run1(&bad)
            .err()
            .expect("NaN/Inf input must be rejected");
        assert!(err.contains("run1 input"), "entry validation, got: {err}");
    }
    // Valid input flows through both steps.
    let out = program.run1(&t(&[-1.0, 2.0], &[1, 2, 1, 1])).unwrap();
    assert_close(&out.to_array(), &[1.0, 2.0], 1e-6);
}

#[test]
fn fase2_v5_thru_v9_entry_rejects_nonfinite_inputs() {
    use crate::math::program_v5::MathProgramV5Builder;
    use crate::math::program_v6::MathProgramV6Builder;
    use crate::math::program_v7::MathProgramV7Builder;
    use crate::math::program_v8::MathProgramV8Builder;
    use crate::math::program_v9::MathProgramV9Builder;

    let mut b5 = MathProgramV5Builder::new(3, 4).unwrap();
    b5.add_unary(OP_ABS, 0, 3).unwrap();
    b5.set_output(3).unwrap();
    let p5 = b5.compile().unwrap();

    let mut b6 = MathProgramV6Builder::new(1, 2).unwrap();
    b6.add_unary(OP_ABS, 0, 1).unwrap();
    b6.set_output(1).unwrap();
    let p6 = b6.compile().unwrap();

    let mut b7 = MathProgramV7Builder::new(1, 2).unwrap();
    b7.add_unary(OP_ABS, 0, 1).unwrap();
    b7.set_output(1).unwrap();
    let p7 = b7.compile().unwrap();

    let mut b8 = MathProgramV8Builder::new(1, 2).unwrap();
    b8.add_unary(OP_ABS, 0, 1).unwrap();
    b8.set_output(1).unwrap();
    let p8 = b8.compile().unwrap();

    let mut b9 = MathProgramV9Builder::new(1, 2).unwrap();
    b9.add_unary(OP_ABS, 0, 1).unwrap();
    b9.set_output(1).unwrap();
    let p9 = b9.compile().unwrap();

    let good3 = || {
        [
            t(&[1.0, 2.0], &[1, 2, 1, 1]),
            t(&[1.0, 2.0], &[1, 2, 1, 1]),
            t(&[1.0, 2.0], &[1, 2, 1, 1]),
        ]
    };
    for bad in [nan_tensor(), inf_tensor()] {
        let mut v5_inputs = good3();
        v5_inputs[1] = bad.clone();
        let single_input = [bad];
        for (version, result) in [
            ("v5", p5.run_inputs(&v5_inputs)),
            ("v6", p6.run_inputs(&single_input)),
            ("v7", p7.run_inputs(&single_input)),
            ("v8", p8.run_inputs(&single_input)),
            ("v9", p9.run_inputs(&single_input)),
        ] {
            let err = result.err().unwrap_or_else(|| {
                panic!("{version} must reject NaN/Inf at entry")
            });
            assert!(
                err.contains("runInputs input"),
                "{version} error must come from entry validation, got: {err}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Twin agreement: checked(op) == unchecked(op) bit-for-bit on valid inputs.
// ---------------------------------------------------------------------------

#[test]
fn fase2_numeric_twins_agree_on_valid_inputs() {
    let k = WasmNumericKernel::new();
    let a = t(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let b = t(&[2.0, 4.0], &[1, 2, 1, 1]);

    assert_eq!(k.abs(&a).unwrap().to_array(), k.abs_unchecked(&a).unwrap().to_array());
    assert_eq!(k.sqrt(&b).unwrap().to_array(), k.sqrt_unchecked(&b).unwrap().to_array());
    assert_eq!(k.exp(&a).unwrap().to_array(), k.exp_unchecked(&a).unwrap().to_array());
    assert_eq!(k.log(&b).unwrap().to_array(), k.log_unchecked(&b).unwrap().to_array());
    assert_eq!(
        k.clamp(&a, 0.0, 2.0).unwrap().to_array(),
        k.clamp_unchecked(&a, 0.0, 2.0).unwrap().to_array()
    );
    assert_eq!(k.add(&a, &b).unwrap().to_array(), k.add_unchecked(&a, &b).unwrap().to_array());
    assert_eq!(k.sub(&a, &b).unwrap().to_array(), k.sub_unchecked(&a, &b).unwrap().to_array());
    assert_eq!(k.mul(&a, &b).unwrap().to_array(), k.mul_unchecked(&a, &b).unwrap().to_array());
    assert_eq!(k.div(&a, &b).unwrap().to_array(), k.div_unchecked(&a, &b).unwrap().to_array());
}

#[test]
fn fase2_linalg_twins_agree_on_valid_inputs() {
    let l = WasmLinearAlgebra::new();
    let a = t(&[1.0, 2.0], &[1, 2, 1, 1]);
    let b = t(&[3.0, 4.0], &[1, 2, 1, 1]);
    assert_eq!(l.l2_norm(&a).unwrap().to_array(), l.l2_norm_unchecked(&a).unwrap().to_array());
    assert_eq!(l.dot(&a, &b).unwrap().to_array(), l.dot_unchecked(&a, &b).unwrap().to_array());
    assert_eq!(
        l.l2_distance(&a, &b).unwrap().to_array(),
        l.l2_distance_unchecked(&a, &b).unwrap().to_array()
    );
    let m1 = t(&[1.0, 2.0], &[1, 1, 1, 2]);
    let m2 = t(&[3.0, 4.0], &[1, 1, 2, 1]);
    assert_eq!(
        l.matmul(&m1, &m2).unwrap().to_array(),
        l.matmul_unchecked(&m1, &m2).unwrap().to_array()
    );
    assert_eq!(
        l.cosine_similarity(&a, &b, None).unwrap().to_array(),
        l.cosine_similarity_unchecked(&a, &b, None).unwrap().to_array()
    );
}

#[test]
fn fase2_statistics_twins_agree_on_valid_inputs() {
    let s = WasmStatistics::new();
    let a = t(&[1.0, 3.0], &[1, 2, 1, 1]);
    assert_eq!(s.sum(&a).unwrap().to_array(), s.sum_unchecked(&a).unwrap().to_array());
    assert_eq!(s.mean(&a).unwrap().to_array(), s.mean_unchecked(&a).unwrap().to_array());
    assert_eq!(
        s.variance_population(&a).unwrap().to_array(),
        s.variance_population_unchecked(&a).unwrap().to_array()
    );
    assert_eq!(
        s.std_population(&a).unwrap().to_array(),
        s.std_population_unchecked(&a).unwrap().to_array()
    );
    assert_eq!(s.min(&a).unwrap().to_array(), s.min_unchecked(&a).unwrap().to_array());
    assert_eq!(s.max(&a).unwrap().to_array(), s.max_unchecked(&a).unwrap().to_array());
}

#[test]
fn fase2_probability_twins_agree_on_valid_inputs() {
    let p = WasmProbability::new();
    let d = t(&[0.25, 0.75], &[1, 2, 1, 1]);
    assert_eq!(p.normalize(&d).unwrap().to_array(), p.normalize_unchecked(&d).unwrap().to_array());
    assert_eq!(p.entropy(&d).unwrap().to_array(), p.entropy_unchecked(&d).unwrap().to_array());
    assert_eq!(
        p.cross_entropy(&d, &d).unwrap().to_array(),
        p.cross_entropy_unchecked(&d, &d).unwrap().to_array()
    );
    assert_eq!(
        p.kl_divergence(&d, &d).unwrap().to_array(),
        p.kl_divergence_unchecked(&d, &d).unwrap().to_array()
    );
}

#[test]
fn fase2_reduction_and_comparison_twins_agree_on_valid_inputs() {
    use crate::math::comparison::TensorComparison;
    use crate::math::reduction::TensorReduction;
    let r = TensorReduction::new();
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
    for axis in [0u32, 1] {
        assert_eq!(
            r.sum_axis(&a, axis).unwrap().to_array(),
            r.sum_axis_unchecked(&a, axis).unwrap().to_array()
        );
        assert_eq!(
            r.mean_axis(&a, axis).unwrap().to_array(),
            r.mean_axis_unchecked(&a, axis).unwrap().to_array()
        );
        assert_eq!(
            r.min_axis(&a, axis).unwrap().to_array(),
            r.min_axis_unchecked(&a, axis).unwrap().to_array()
        );
        assert_eq!(
            r.max_axis(&a, axis).unwrap().to_array(),
            r.max_axis_unchecked(&a, axis).unwrap().to_array()
        );
    }
    let c = TensorComparison::new();
    let lhs = t(&[1.0, 2.0], &[1, 2, 1, 1]);
    let rhs = t(&[2.0, 1.0], &[1, 2, 1, 1]);
    assert_eq!(
        c.less_equal_01(&lhs, &rhs).unwrap().to_array(),
        c.less_equal_01_unchecked(&lhs, &rhs).unwrap().to_array()
    );
}

// ---------------------------------------------------------------------------
// 3. Unchecked twins retain domain checks and output validation.
// ---------------------------------------------------------------------------

#[test]
fn fase2_unchecked_twins_retain_domain_checks() {
    let k = WasmNumericKernel::new();
    let p = WasmProbability::new();
    let l = WasmLinearAlgebra::new();

    // div by zero still rejected (domain check retained, finite scan skipped).
    let num = t(&[1.0, 2.0], &[1, 2, 1, 1]);
    let zero = t(&[1.0, 0.0], &[1, 2, 1, 1]);
    assert!(k.div_unchecked(&num, &zero).is_err());

    // sqrt of negative / log of non-positive still rejected.
    let neg = t(&[-1.0, 2.0], &[1, 2, 1, 1]);
    assert!(k.sqrt_unchecked(&neg).is_err());
    assert!(k.log_unchecked(&neg).is_err());
    assert!(k.log_unchecked(&zero).is_err());

    // clamp with non-finite bounds still rejected.
    assert!(k.clamp_unchecked(&num, f32::NAN, 1.0).is_err());
    assert!(k.clamp_unchecked(&num, 2.0, 1.0).is_err());

    // matmul shape mismatch still rejected.
    let m1 = t(&[1.0, 2.0], &[1, 1, 1, 2]);
    let m2 = t(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
    assert!(l.matmul_unchecked(&m1, &m2).is_err());

    // probability domain checks retained: negative / zero-mass / bad support.
    let neg_mass = t(&[-0.2, 0.2], &[1, 2, 1, 1]);
    let zero_mass = t(&[0.0, 0.0], &[1, 2, 1, 1]);
    assert!(p.normalize_unchecked(&neg_mass).is_err());
    assert!(p.normalize_unchecked(&zero_mass).is_err());
    // normalize accepts positive-mass input and still validates its output.
    let unnorm = t(&[0.2, 0.2], &[1, 2, 1, 1]);
    let out = p.normalize_unchecked(&unnorm).unwrap();
    assert_close(&out.to_array(), &[0.5, 0.5], 1e-6);
    // entropy requires an actual distribution (sums to 1).
    assert!(p.entropy_unchecked(&unnorm).is_err());
    let d = t(&[0.5, 0.5], &[1, 2, 1, 1]);
    let bad_q = t(&[1.0, 0.0], &[1, 2, 1, 1]);
    assert!(p.cross_entropy_unchecked(&d, &bad_q).is_err());
    assert!(p.kl_divergence_unchecked(&d, &bad_q).is_err());
}

#[test]
fn fase2_unchecked_twins_retain_output_validation() {
    let k = WasmNumericKernel::new();
    // exp(100) overflows f32 -> output validation must still reject.
    let big = t(&[100.0, 1.0], &[1, 2, 1, 1]);
    assert!(k.exp(&big).is_err());
    assert!(k.exp_unchecked(&big).is_err());
}

// ---------------------------------------------------------------------------
// 4. v1 dispatch wiring: every dispatched op through the unchecked path.
// ---------------------------------------------------------------------------

fn v1_unary_program(op: u8, input: WasmTensor) -> WasmTensor {
    let mut builder = MathProgramBuilder::new(1, 2).unwrap();
    builder.add_unary(op, 0, 1).unwrap();
    builder.set_output(1).unwrap();
    builder.compile().unwrap().run1(&input).unwrap()
}

fn v1_binary_program(op: u8, a: WasmTensor, b: WasmTensor) -> WasmTensor {
    let mut builder = MathProgramBuilder::new(2, 3).unwrap();
    builder.add_binary(op, 0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    builder.compile().unwrap().run2(&a, &b).unwrap()
}

#[test]
fn fase2_v1_dispatch_wiring_all_unary_ops() {
    let f = |v: &[f32]| t(v, &[1, 2, 1, 1]);
    assert_close(&v1_unary_program(OP_ABS, f(&[-2.0, 3.0])).to_array(), &[2.0, 3.0], 1e-6);
    assert_close(&v1_unary_program(OP_SQRT, f(&[4.0, 9.0])).to_array(), &[2.0, 3.0], 1e-6);
    assert_close(&v1_unary_program(OP_EXP, f(&[0.0, 1.0])).to_array(), &[1.0, std::f32::consts::E], 1e-5);
    assert_close(&v1_unary_program(OP_LOG, f(&[1.0, std::f32::consts::E])).to_array(), &[0.0, 1.0], 1e-6);
    assert_close(&v1_unary_program(OP_L2_NORM, f(&[3.0, 4.0])).to_array(), &[5.0], 1e-5);
    assert_close(&v1_unary_program(OP_SUM, f(&[1.0, 2.0])).to_array(), &[3.0], 1e-6);
    assert_close(&v1_unary_program(OP_MEAN, f(&[1.0, 3.0])).to_array(), &[2.0], 1e-6);
    assert_close(&v1_unary_program(OP_VARIANCE_POPULATION, f(&[1.0, 3.0])).to_array(), &[1.0], 1e-6);
    assert_close(&v1_unary_program(OP_STD_POPULATION, f(&[1.0, 3.0])).to_array(), &[1.0], 1e-6);
    assert_close(&v1_unary_program(OP_MIN, f(&[1.0, 2.0])).to_array(), &[1.0], 1e-6);
    assert_close(&v1_unary_program(OP_MAX, f(&[1.0, 2.0])).to_array(), &[2.0], 1e-6);
    assert_close(&v1_unary_program(OP_NORMALIZE, f(&[1.0, 3.0])).to_array(), &[0.25, 0.75], 1e-6);
    assert_close(
        &v1_unary_program(OP_ENTROPY, f(&[0.5, 0.5])).to_array(),
        &[std::f32::consts::LN_2],
        1e-5,
    );

    // clamp via dedicated builder method.
    let mut builder = MathProgramBuilder::new(1, 2).unwrap();
    builder.add_clamp(0, 1, 0.0, 1.0).unwrap();
    builder.set_output(1).unwrap();
    let out = builder.compile().unwrap().run1(&f(&[-5.0, 0.5])).unwrap();
    assert_close(&out.to_array(), &[0.0, 0.5], 1e-6);
}

#[test]
fn fase2_v1_dispatch_wiring_all_binary_ops() {
    let f = |v: &[f32]| t(v, &[1, 2, 1, 1]);
    assert_close(&v1_binary_program(OP_ADD, f(&[1.0, 2.0]), f(&[3.0, 4.0])).to_array(), &[4.0, 6.0], 1e-6);
    assert_close(&v1_binary_program(OP_SUB, f(&[5.0, 6.0]), f(&[1.0, 2.0])).to_array(), &[4.0, 4.0], 1e-6);
    assert_close(&v1_binary_program(OP_MUL, f(&[2.0, 3.0]), f(&[4.0, 5.0])).to_array(), &[8.0, 15.0], 1e-6);
    assert_close(&v1_binary_program(OP_DIV, f(&[6.0, 8.0]), f(&[2.0, 4.0])).to_array(), &[3.0, 2.0], 1e-6);
    assert_close(&v1_binary_program(OP_DOT, f(&[1.0, 2.0]), f(&[3.0, 4.0])).to_array(), &[11.0], 1e-5);
    assert_close(&v1_binary_program(OP_L2_DISTANCE, f(&[1.0, 2.0]), f(&[4.0, 6.0])).to_array(), &[5.0], 1e-5);
    assert_close(
        &v1_binary_program(OP_CROSS_ENTROPY, f(&[0.5, 0.5]), f(&[0.5, 0.5])).to_array(),
        &[std::f32::consts::LN_2],
        1e-5,
    );
    assert_close(
        &v1_binary_program(OP_KL_DIVERGENCE, f(&[0.5, 0.5]), f(&[0.5, 0.5])).to_array(),
        &[0.0],
        1e-6,
    );

    // matmul needs [B,G,M,K] @ [B,G,K,N] layout.
    let mut builder = MathProgramBuilder::new(2, 3).unwrap();
    builder.add_binary(OP_MATMUL, 0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let out = program
        .run2(&t(&[1.0, 2.0], &[1, 1, 1, 2]), &t(&[3.0, 4.0], &[1, 1, 2, 1]))
        .unwrap();
    assert_close(&out.to_array(), &[11.0], 1e-5);

    // cosine similarity via dedicated builder method.
    let mut builder = MathProgramBuilder::new(2, 3).unwrap();
    builder.add_cosine_similarity(0, 1, 2, 1e-6).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let out = program.run2(&f(&[1.0, 0.0]), &f(&[1.0, 0.0])).unwrap();
    assert_close(&out.to_array(), &[1.0], 1e-5);
}

#[test]
fn fase2_v1_multistep_intermediate_skips_revalidation() {
    // add(0,1->2) then mul(2,0->3): slot 2 is an intermediate consumed unchecked.
    let mut builder = MathProgramBuilder::new(2, 4).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 2).unwrap();
    builder.add_binary(OP_MUL, 2, 0, 3).unwrap();
    builder.set_output(3).unwrap();
    let program = builder.compile().unwrap();
    let a = t(&[1.0, 2.0], &[1, 2, 1, 1]);
    let b = t(&[3.0, 4.0], &[1, 2, 1, 1]);
    let out = program.run2(&a, &b).unwrap();
    assert_close(&out.to_array(), &[4.0, 12.0], 1e-6);

    // Overflow in an intermediate is still caught by output validation.
    let mut builder = MathProgramBuilder::new(1, 3).unwrap();
    builder.add_unary(OP_EXP, 0, 1).unwrap();
    builder.add_unary(OP_ABS, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    assert!(program.run1(&t(&[100.0], &[1, 1, 1, 1])).is_err());
}

// ---------------------------------------------------------------------------
// 5. Per-version new step variants through the unchecked delegation chain.
// ---------------------------------------------------------------------------

#[test]
fn fase2_v4_select_axis_exact_values() {
    use crate::math::program_v4::MathProgramV4Builder;
    let mut builder = MathProgramV4Builder::new(1, 2).unwrap();
    builder.add_select_axis(0, 1, 1, &[2, 0]).unwrap();
    builder.set_output(1).unwrap();
    let program = builder.compile().unwrap();
    let out = program.run1(&t(&[10.0, 20.0, 30.0], &[1, 3, 1, 1])).unwrap();
    assert_close(&out.to_array(), &[30.0, 10.0], 1e-6);
}

#[test]
fn fase2_v5_core_and_select_axis_delegation() {
    use crate::math::program_v5::MathProgramV5Builder;
    let mut builder = MathProgramV5Builder::new(3, 5).unwrap();
    builder.add_binary(OP_ADD, 0, 1, 3).unwrap();
    builder.add_select_axis(3, 4, 1, &[1, 0]).unwrap();
    builder.set_output(4).unwrap();
    let program = builder.compile().unwrap();
    let out = program
        .run_inputs(&[
            t(&[1.0, 2.0], &[1, 2, 1, 1]),
            t(&[3.0, 4.0], &[1, 2, 1, 1]),
            t(&[0.0, 0.0], &[1, 2, 1, 1]),
        ])
        .unwrap();
    // add -> [4,6]; select_axis [1,0] -> [6,4].
    assert_close(&out.to_array(), &[6.0, 4.0], 1e-6);
}

#[test]
fn fase2_v6_fill_like_exact_values() {
    use crate::math::program_v6::MathProgramV6Builder;
    let mut builder = MathProgramV6Builder::new(1, 3).unwrap();
    builder.add_unary(OP_ABS, 0, 1).unwrap();
    builder.add_fill_like(1, 2, 7.0).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let out = program.run_inputs(&[t(&[-1.0, -2.0], &[1, 2, 1, 1])]).unwrap();
    assert_close(&out.to_array(), &[7.0, 7.0], 1e-6);
    assert_all_finite(&out.to_array());
}

#[test]
fn fase2_v7_expand_like_exact_values() {
    use crate::math::program_v7::MathProgramV7Builder;
    let mut builder = MathProgramV7Builder::new(2, 3).unwrap();
    builder.add_expand_like(0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let out = program
        .run_inputs(&[
            t(&[5.0], &[1, 1, 1, 1]),
            t(&[0.0, 0.0, 0.0], &[1, 3, 1, 1]),
        ])
        .unwrap();
    assert_close(&out.to_array(), &[5.0, 5.0, 5.0], 1e-6);
}

#[test]
fn fase2_v8_reduction_axes_exact_values() {
    use crate::math::program_v8::MathProgramV8Builder;
    for (add, expected) in [
        (
            MathProgramV8Builder::add_sum_axis as fn(&mut MathProgramV8Builder, u8, u8, u32) -> Result<(), String>,
            [10.0],
        ),
        (
            MathProgramV8Builder::add_mean_axis as fn(&mut MathProgramV8Builder, u8, u8, u32) -> Result<(), String>,
            [2.5],
        ),
        (
            MathProgramV8Builder::add_min_axis as fn(&mut MathProgramV8Builder, u8, u8, u32) -> Result<(), String>,
            [1.0],
        ),
        (
            MathProgramV8Builder::add_max_axis as fn(&mut MathProgramV8Builder, u8, u8, u32) -> Result<(), String>,
            [4.0],
        ),
    ] {
        let mut builder = MathProgramV8Builder::new(1, 2).unwrap();
        add(&mut builder, 0, 1, 1).unwrap();
        builder.set_output(1).unwrap();
        let program = builder.compile().unwrap();
        let out = program
            .run_inputs(&[t(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1])])
            .unwrap();
        assert_eq!(out.shape(), vec![1, 1, 1, 1]);
        assert_close(&out.to_array(), &expected, 1e-6);
    }
}

#[test]
fn fase2_v9_indices_like_and_less_equal_exact_values() {
    use crate::math::program_v9::MathProgramV9Builder;
    let mut builder = MathProgramV9Builder::new(1, 3).unwrap();
    builder.add_indices_like(0, 1, 1).unwrap();
    builder.add_less_equal_01(0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    // input [5,5,5] -> indices [0,1,2]; less_equal([5,5,5],[0,1,2]) -> [0,0,0].
    let out = program.run_inputs(&[t(&[5.0, 5.0, 5.0], &[1, 3, 1, 1])]).unwrap();
    assert_close(&out.to_array(), &[0.0, 0.0, 0.0], 1e-6);

    let mut builder = MathProgramV9Builder::new(2, 3).unwrap();
    builder.add_less_equal_01(0, 1, 2).unwrap();
    builder.set_output(2).unwrap();
    let program = builder.compile().unwrap();
    let out = program
        .run_inputs(&[t(&[1.0, 2.0], &[1, 2, 1, 1]), t(&[2.0, 1.0], &[1, 2, 1, 1])])
        .unwrap();
    assert_close(&out.to_array(), &[1.0, 0.0], 1e-6);
}
