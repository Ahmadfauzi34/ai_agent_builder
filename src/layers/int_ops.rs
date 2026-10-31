use burn::prelude::*;
use burn::tensor::Int;
use burn::tensor::TensorData;

use crate::WasmBackend;

// --- INTEGER OPS LAYER FAMILY (Boundary #5) ---
//
// Native integer path: Burn Int tensors are i64 on NdArray, so integer
// arithmetic is exact up to 2^63 - 1 (vs f32, which loses integer
// exactness above 2^24). Two layer types:
//
// - IntBinary (add/sub/mul/matmul/div/rem): elementwise integer ops +
//   integer matmul, mirroring the f32 Binary family.
// - IntCast (to_float/from_float): the bridge between the int path and
//   the f32 path.
//
// Overflow policy (checked, unconditional): every integer op runs
// through Rust `checked_*` semantics on plain i64 data — overflow,
// division by zero, and MIN / -1 are all `Err`, in debug AND release
// builds alike. There is no wrapping mode in the engine and no flag to
// enable one: Burn's NdArray int ops wrap in release builds, so this
// layer never calls them for arithmetic. Two's-complement wraparound
// exists only as an explicit caller-side emulation (the `--wrap`
// short-circuit in `helper/bin/int.mjs`), never as engine behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBinaryOp {
    Add,
    Sub,
    Mul,
    Matmul,
    Div,
    Rem,
}

#[derive(Debug, Clone, Copy)]
pub struct IntBinary {
    op: IntBinaryOp,
}

impl IntBinary {
    pub fn new(op: IntBinaryOp) -> Self {
        Self { op }
    }

    pub fn op(&self) -> IntBinaryOp {
        self.op
    }

    pub fn forward<B: Backend>(
        &self,
        a: Tensor<B, 4, Int>,
        b: Tensor<B, 4, Int>,
    ) -> Result<Tensor<B, 4, Int>, String> {
        let da = a.dims();
        let db = b.dims();
        match self.op {
            IntBinaryOp::Add
            | IntBinaryOp::Sub
            | IntBinaryOp::Mul
            | IntBinaryOp::Div
            | IntBinaryOp::Rem => {
                let name = match self.op {
                    IntBinaryOp::Add => "add",
                    IntBinaryOp::Sub => "sub",
                    IntBinaryOp::Mul => "mul",
                    IntBinaryOp::Div => "div",
                    _ => "rem",
                };
                if da != db {
                    return Err(format!(
                        "int binary {name}: shape mismatch {:?} vs {:?}",
                        da, db
                    ));
                }
                // Checked arithmetic on plain i64 data: Burn's NdArray
                // int ops wrap in release builds, so they are never
                // used for the arithmetic itself.
                let device = a.device();
                let va = a
                    .to_data()
                    .into_vec::<i64>()
                    .map_err(|e| format!("int binary {name}: cannot read i64 data: {e}"))?;
                let vb = b
                    .to_data()
                    .into_vec::<i64>()
                    .map_err(|e| format!("int binary {name}: cannot read i64 data: {e}"))?;
                let mut out = Vec::with_capacity(va.len());
                for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                    let v = match self.op {
                        IntBinaryOp::Add => x.checked_add(*y).ok_or_else(|| {
                            format!(
                                "int binary add: i64 overflow at index {i} ({x} + {y} = {})",
                                *x as i128 + *y as i128
                            )
                        }),
                        IntBinaryOp::Sub => x.checked_sub(*y).ok_or_else(|| {
                            format!(
                                "int binary sub: i64 overflow at index {i} ({x} - {y} = {})",
                                *x as i128 - *y as i128
                            )
                        }),
                        IntBinaryOp::Mul => x.checked_mul(*y).ok_or_else(|| {
                            format!(
                                "int binary mul: i64 overflow at index {i} ({x} * {y} = {})",
                                *x as i128 * *y as i128
                            )
                        }),
                        IntBinaryOp::Div => {
                            if *y == 0 {
                                Err(format!("int binary div: division by zero at index {i}"))
                            } else if *x == i64::MIN && *y == -1 {
                                Err(format!(
                                    "int binary div: i64 overflow at index {i} (i64::MIN / -1)"
                                ))
                            } else {
                                Ok(x / y)
                            }
                        }
                        _ => {
                            if *y == 0 {
                                Err(format!("int binary rem: division by zero at index {i}"))
                            } else if *x == i64::MIN && *y == -1 {
                                Err(format!(
                                    "int binary rem: i64 overflow at index {i} (i64::MIN % -1)"
                                ))
                            } else {
                                Ok(x % y)
                            }
                        }
                    }?;
                    out.push(v);
                }
                Ok(Tensor::from_data(TensorData::new(out, da), &device))
            }
            IntBinaryOp::Matmul => {
                // Batched matmul over the last two dims, same convention
                // as the f32 Binary::Matmul. Manual loop: Burn's matmul
                // kernel cannot be used here because its accumulation
                // wraps in release builds; each product and each
                // accumulation step is checked individually (same
                // structure as verifyMatmulNoOverflow in int.mjs).
                if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
                    return Err(format!(
                        "int binary matmul: incompatible shapes {:?} @ {:?}",
                        da, db
                    ));
                }
                crate::protocol::check_numel(
                    &[da[0], da[1], da[2], db[3]],
                    "int binary matmul output",
                )?;
                let device = a.device();
                let va = a
                    .to_data()
                    .into_vec::<i64>()
                    .map_err(|e| format!("int binary matmul: cannot read i64 data: {e}"))?;
                let vb = b
                    .to_data()
                    .into_vec::<i64>()
                    .map_err(|e| format!("int binary matmul: cannot read i64 data: {e}"))?;
                let (b0n, b1n, m, k, n) = (da[0], da[1], da[2], da[3], db[3]);
                let mut out = Vec::with_capacity(b0n * b1n * m * n);
                for b0 in 0..b0n {
                    for b1 in 0..b1n {
                        let batch = b0 * b1n + b1;
                        for mi in 0..m {
                            for ni in 0..n {
                                let mut acc: i64 = 0;
                                for ki in 0..k {
                                    let x = va[(batch * m + mi) * k + ki];
                                    let y = vb[(batch * k + ki) * n + ni];
                                    let prod = x.checked_mul(y).ok_or_else(|| {
                                        format!(
                                            "int binary matmul: i64 overflow in product at [{b0},{b1},{mi},{ni}] (k={ki}): {x} * {y} = {}",
                                            x as i128 * y as i128
                                        )
                                    })?;
                                    acc = acc.checked_add(prod).ok_or_else(|| {
                                        format!(
                                            "int binary matmul: i64 overflow in accumulator at [{b0},{b1},{mi},{ni}]: {acc} + {prod} = {}",
                                            acc as i128 + prod as i128
                                        )
                                    })?;
                                }
                                out.push(acc);
                            }
                        }
                    }
                }
                Ok(Tensor::from_data(
                    TensorData::new(out, [b0n, b1n, m, n]),
                    &device,
                ))
            }
        }
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntCastKind {
    /// i64 -> f32. Exact only for |x| <= 2^24; larger magnitudes round,
    /// exactly like any f32 conversion. The int path exists so callers
    /// can avoid this direction for counting.
    ToFloat,
    /// f32 -> i64, truncating toward zero (Rust `as i64` semantics).
    /// NaN converts to 0; out-of-range values saturate (Rust `as`
    /// semantics, NOT an error) — documented, not hidden.
    FromFloat,
}

#[derive(Debug, Clone, Copy)]
pub struct IntCast {
    kind: IntCastKind,
}

impl IntCast {
    pub fn new(kind: IntCastKind) -> Self {
        Self { kind }
    }

    pub fn kind(&self) -> IntCastKind {
        self.kind
    }

    pub fn to_float<B: Backend>(&self, input: Tensor<B, 4, Int>) -> Tensor<B, 4> {
        input.float()
    }

    pub fn from_float<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4, Int> {
        input.int()
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_tensor(data: Vec<i64>, dims: [usize; 4]) -> Tensor<WasmBackend, 4, Int> {
        let device: <WasmBackend as Backend>::Device = Default::default();
        Tensor::from_data(TensorData::new(data, dims), &device)
    }

    fn to_vec(t: Tensor<WasmBackend, 4, Int>) -> Vec<i64> {
        t.to_data().into_vec::<i64>().expect("i64 vec")
    }

    #[test]
    fn int_add_is_exact_beyond_f32_range() {
        // The core promise of boundary #5: 2^24 + 1 is exact here,
        // while f32 would round it.
        let op = IntBinary::new(IntBinaryOp::Add);
        let a = int_tensor(vec![16_777_217], [1, 1, 1, 1]);
        let b = int_tensor(vec![1], [1, 1, 1, 1]);
        let y = op.forward(a, b).unwrap();
        assert_eq!(to_vec(y), vec![16_777_218]);
    }

    #[test]
    fn int_sub_mul_elementwise() {
        let a = int_tensor(vec![10, 20, 30, 40], [1, 4, 1, 1]);
        let b = int_tensor(vec![1, 2, 3, 4], [1, 4, 1, 1]);
        let sub = IntBinary::new(IntBinaryOp::Sub);
        assert_eq!(
            to_vec(sub.forward(a.clone(), b.clone()).unwrap()),
            vec![9, 18, 27, 36]
        );
        let mul = IntBinary::new(IntBinaryOp::Mul);
        assert_eq!(
            to_vec(mul.forward(a, b).unwrap()),
            vec![10, 40, 90, 160]
        );
    }

    #[test]
    fn int_matmul_matches_hand_computation() {
        // [1,1,2,2] @ [1,1,2,2] = [[19,22],[43,50]]
        let op = IntBinary::new(IntBinaryOp::Matmul);
        let a = int_tensor(vec![1, 2, 3, 4], [1, 1, 2, 2]);
        let b = int_tensor(vec![5, 6, 7, 8], [1, 1, 2, 2]);
        let y = op.forward(a, b).unwrap();
        assert_eq!(y.dims(), [1, 1, 2, 2]);
        assert_eq!(to_vec(y), vec![19, 22, 43, 50]);
    }

    #[test]
    fn int_ops_reject_shape_mismatch() {
        let op = IntBinary::new(IntBinaryOp::Add);
        let a = int_tensor(vec![1, 2], [1, 2, 1, 1]);
        let b = int_tensor(vec![1, 2, 3], [1, 3, 1, 1]);
        assert!(op.forward(a, b).is_err());
        let mm = IntBinary::new(IntBinaryOp::Matmul);
        let a = int_tensor(vec![1, 2, 3, 4], [1, 1, 2, 2]);
        let b = int_tensor(vec![1, 2, 3, 4, 5, 6], [1, 1, 3, 2]);
        assert!(mm.forward(a, b).is_err());
    }

    #[test]
    fn int_add_sub_mul_overflow_is_err_not_wrap() {
        // MAX + 1, MIN - 1, and 3037000500^2 must Err in release
        // builds too (previously wrapped silently).
        let add = IntBinary::new(IntBinaryOp::Add);
        let err = add
            .forward(
                int_tensor(vec![i64::MAX], [1, 1, 1, 1]),
                int_tensor(vec![1], [1, 1, 1, 1]),
            )
            .err()
            .expect("add overflow must err");
        assert!(err.contains("i64 overflow"), "add err: {err}");
        let sub = IntBinary::new(IntBinaryOp::Sub);
        let err = sub
            .forward(
                int_tensor(vec![i64::MIN], [1, 1, 1, 1]),
                int_tensor(vec![1], [1, 1, 1, 1]),
            )
            .err()
            .expect("sub overflow must err");
        assert!(err.contains("i64 overflow"), "sub err: {err}");
        let mul = IntBinary::new(IntBinaryOp::Mul);
        let err = mul
            .forward(
                int_tensor(vec![3_037_000_500], [1, 1, 1, 1]),
                int_tensor(vec![3_037_000_500], [1, 1, 1, 1]),
            )
            .err()
            .expect("mul overflow must err");
        assert!(err.contains("i64 overflow"), "mul err: {err}");
    }

    #[test]
    fn int_matmul_overflow_product_and_accumulator_are_err() {
        let mm = IntBinary::new(IntBinaryOp::Matmul);
        // Product overflow: 3037000500^2 in a 1x1 matmul.
        let err = mm
            .forward(
                int_tensor(vec![3_037_000_500], [1, 1, 1, 1]),
                int_tensor(vec![3_037_000_500], [1, 1, 1, 1]),
            )
            .err()
            .expect("matmul product overflow must err");
        assert!(err.contains("product"), "matmul product err: {err}");
        // Accumulator overflow: two in-range products whose sum
        // overflows: [MAX, 1] @ [[1], [1]] = MAX + 1.
        let err = mm
            .forward(
                int_tensor(vec![i64::MAX, 1], [1, 1, 1, 2]),
                int_tensor(vec![1, 1], [1, 1, 2, 1]),
            )
            .err()
            .expect("matmul accumulator overflow must err");
        assert!(err.contains("accumulator"), "matmul acc err: {err}");
    }

    #[test]
    fn int_div_rem_truncate_and_reject() {
        let div = IntBinary::new(IntBinaryOp::Div);
        let y = div
            .forward(
                int_tensor(vec![10, -10, 7, -7], [1, 4, 1, 1]),
                int_tensor(vec![3, 3, 2, 2], [1, 4, 1, 1]),
            )
            .unwrap();
        assert_eq!(to_vec(y), vec![3, -3, 3, -3]);
        let rem = IntBinary::new(IntBinaryOp::Rem);
        let y = rem
            .forward(
                int_tensor(vec![10, -10, 7, -7], [1, 4, 1, 1]),
                int_tensor(vec![3, 3, 2, 2], [1, 4, 1, 1]),
            )
            .unwrap();
        assert_eq!(to_vec(y), vec![1, -1, 1, -1]);
        // Division by zero.
        let err = div
            .forward(
                int_tensor(vec![5], [1, 1, 1, 1]),
                int_tensor(vec![0], [1, 1, 1, 1]),
            )
            .err()
            .expect("div by zero must err");
        assert!(err.contains("division by zero"), "div zero err: {err}");
        let err = rem
            .forward(
                int_tensor(vec![5], [1, 1, 1, 1]),
                int_tensor(vec![0], [1, 1, 1, 1]),
            )
            .err()
            .expect("rem by zero must err");
        assert!(err.contains("division by zero"), "rem zero err: {err}");
        // MIN / -1 and MIN % -1.
        let err = div
            .forward(
                int_tensor(vec![i64::MIN], [1, 1, 1, 1]),
                int_tensor(vec![-1], [1, 1, 1, 1]),
            )
            .err()
            .expect("MIN/-1 must err");
        assert!(err.contains("i64 overflow"), "div MIN/-1 err: {err}");
        let err = rem
            .forward(
                int_tensor(vec![i64::MIN], [1, 1, 1, 1]),
                int_tensor(vec![-1], [1, 1, 1, 1]),
            )
            .err()
            .expect("MIN%-1 must err");
        assert!(err.contains("i64 overflow"), "rem MIN/-1 err: {err}");
    }

    #[test]
    fn int_cast_roundtrip_small_values() {
        let cast = IntCast::new(IntCastKind::ToFloat);
        let a = int_tensor(vec![7, -3], [1, 2, 1, 1]);
        let f: Tensor<WasmBackend, 4> = cast.to_float(a);
        let v: Vec<f32> = f.to_data().into_vec::<f32>().expect("f32 vec");
        assert_eq!(v, vec![7.0, -3.0]);

        let back = IntCast::new(IntCastKind::FromFloat);
        let f2: Tensor<WasmBackend, 4> = {
            let device: <WasmBackend as Backend>::Device = Default::default();
            Tensor::from_data(TensorData::new(vec![7.9f32, -3.9], [1, 2, 1, 1]), &device)
        };
        let i = back.from_float(f2);
        // Truncation toward zero.
        assert_eq!(to_vec(i), vec![7, -3]);
    }

    #[test]
    fn int_layers_are_stateless() {
        assert_eq!(IntBinary::new(IntBinaryOp::Add).num_params(), 0);
        assert_eq!(IntCast::new(IntCastKind::ToFloat).num_params(), 0);
    }
}
