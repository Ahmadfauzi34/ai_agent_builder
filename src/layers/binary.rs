pub use crate::facade::wasm_types::WasmBinary;
use crate::WasmTensor;
use burn::prelude::*;
use wasm_bindgen::prelude::*;

// Parameter-free binary op. `dim` hanya bermakna untuk Concat.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Matmul,
    Div,
    Concat,
}

#[derive(Debug)]
pub struct Binary {
    op: BinaryOp,
    dim: usize,
}

impl Binary {
    pub fn new(op: BinaryOp, dim: usize) -> Self {
        Self { op, dim }
    }

    // Validasi shape manual -> Err rapi (bukan panic/trap).
    pub fn forward<B: Backend>(
        &self,
        a: Tensor<B, 4>,
        b: Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>, String> {
        let da = a.dims();
        let db = b.dims();
        match self.op {
            BinaryOp::Add => {
                if da != db {
                    return Err(format!("binary add: shape mismatch {:?} vs {:?}", da, db));
                }
                Ok(a.add(b))
            }
            BinaryOp::Sub => {
                if da != db {
                    return Err(format!("binary sub: shape mismatch {:?} vs {:?}", da, db));
                }
                Ok(a.sub(b))
            }
            BinaryOp::Mul => {
                if da != db {
                    return Err(format!("binary mul: shape mismatch {:?} vs {:?}", da, db));
                }
                Ok(a.mul(b))
            }
            BinaryOp::Matmul => {
                // batched matmul atas 2 dim terakhir: a[*,*,m,k] @ b[*,*,k,n] = [*,*,m,n]
                if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
                    return Err(format!(
                        "binary matmul: incompatible shapes {:?} @ {:?}",
                        da, db
                    ));
                }
                // Complaint #18: output [*,*,m,n] can dwarf both inputs
                // (e.g. 16384^2); validate the budget before the matmul so
                // the run fails structured instead of trapping unreachable.
                crate::protocol::check_numel(
                    &[da[0], da[1], da[2], db[3]],
                    "binary matmul output",
                )?;
                Ok(a.matmul(b))
            }
            BinaryOp::Div => {
                if da != db {
                    return Err(format!("binary div: shape mismatch {:?} vs {:?}", da, db));
                }
                // Checked div, kernel parity (WasmNumericKernel::div):
                // finiteness first, then nonzero divisor, then checked
                // finite output. The validators in src/math/numeric.rs
                // are WasmTensor-typed and cannot run on these generic
                // Burn tensors, so the same scans run here inline with
                // the same messages and precedence. One materialization
                // per tensor; -0.0 counts as zero (== 0.0 in Rust).
                let a_data = a.to_data();
                let a_vals = a_data
                    .as_slice::<f32>()
                    .map_err(|_| "binary div lhs: expected f32 tensor".to_string())?;
                for (index, &value) in a_vals.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(format!(
                            "binary div lhs: non-finite value at index {index}: {value}"
                        ));
                    }
                }
                let b_data = b.to_data();
                let b_vals = b_data
                    .as_slice::<f32>()
                    .map_err(|_| "binary div rhs: expected f32 tensor".to_string())?;
                for (index, &value) in b_vals.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(format!(
                            "binary div rhs: non-finite value at index {index}: {value}"
                        ));
                    }
                }
                for (index, &value) in b_vals.iter().enumerate() {
                    if value == 0.0 {
                        return Err(format!(
                            "binary div rhs: zero denominator at index {index}"
                        ));
                    }
                }
                let out = a.div(b);
                let out_data = out.to_data();
                let out_vals = out_data
                    .as_slice::<f32>()
                    .map_err(|_| "binary div output: expected f32 tensor".to_string())?;
                for (index, &value) in out_vals.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(format!(
                            "binary div output: non-finite value at index {index}: {value}"
                        ));
                    }
                }
                Ok(out)
            }
            BinaryOp::Concat => {
                let d = self.dim;
                if d >= 4 {
                    return Err(format!("binary concat: dim {} out of range (rank 4)", d));
                }
                for i in 0..4 {
                    if i == d {
                        continue;
                    }
                    if da[i] != db[i] {
                        return Err(format!(
                            "binary concat: non-concat dim {} differs ({:?} vs {:?})",
                            i, da, db
                        ));
                    }
                }
                Ok(Tensor::cat(vec![a, b], d))
            }
        }
    }
}

// --- WASM WRAPPER (stateless; named constructors infallible, seperti pool/shift) ---
