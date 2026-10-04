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
