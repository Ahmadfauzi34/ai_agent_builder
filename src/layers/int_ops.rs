use burn::prelude::*;
use burn::tensor::Int;

#[cfg(test)]
use burn::tensor::TensorData;

use crate::WasmBackend;

// --- INTEGER OPS LAYER FAMILY (Boundary #5) ---
//
// Native integer path: Burn Int tensors are i64 on NdArray, so integer
// arithmetic is exact up to 2^63 - 1 (vs f32, which loses integer
// exactness above 2^24). Two layer types:
//
// - IntBinary (add/sub/mul/matmul): elementwise integer ops + integer
//   matmul, mirroring the f32 Binary family.
// - IntCast (to_float/from_float): the bridge between the int path and
//   the f32 path.
//
// Overflow policy (honest): Burn's NdArray int ops use the primitive
// integer operators — they panic on overflow in debug builds and wrap
// in release builds. This matches Rust's native `+`/`*` behavior; it is
// NOT a silent-precision-loss like f32, but callers counting on
// "never wraps" must range-check inputs themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBinaryOp {
    Add,
    Sub,
    Mul,
    Matmul,
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
            IntBinaryOp::Add | IntBinaryOp::Sub | IntBinaryOp::Mul => {
                if da != db {
                    let name = match self.op {
                        IntBinaryOp::Add => "add",
                        IntBinaryOp::Sub => "sub",
                        _ => "mul",
                    };
                    return Err(format!(
                        "int binary {name}: shape mismatch {:?} vs {:?}",
                        da, db
                    ));
                }
                Ok(match self.op {
                    IntBinaryOp::Add => a.add(b),
                    IntBinaryOp::Sub => a.sub(b),
                    _ => a.mul(b),
                })
            }
            IntBinaryOp::Matmul => {
                // Batched matmul over the last two dims, same convention
                // as the f32 Binary::Matmul.
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
                Ok(a.matmul(b))
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
