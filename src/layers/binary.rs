use burn::prelude::*;
use wasm_bindgen::prelude::*;
use crate::WasmTensor;

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
#[wasm_bindgen]
pub struct WasmBinary {
    inner: Binary,
}

#[wasm_bindgen]
impl WasmBinary {
    #[wasm_bindgen(js_name = newAdd)]
    pub fn new_add() -> WasmBinary {
        WasmBinary { inner: Binary::new(BinaryOp::Add, 0) }
    }
    #[wasm_bindgen(js_name = newSub)]
    pub fn new_sub() -> WasmBinary {
        WasmBinary { inner: Binary::new(BinaryOp::Sub, 0) }
    }
    #[wasm_bindgen(js_name = newMul)]
    pub fn new_mul() -> WasmBinary {
        WasmBinary { inner: Binary::new(BinaryOp::Mul, 0) }
    }
    #[wasm_bindgen(js_name = newMatmul)]
    pub fn new_matmul() -> WasmBinary {
        WasmBinary { inner: Binary::new(BinaryOp::Matmul, 0) }
    }
    #[wasm_bindgen(js_name = newConcat)]
    pub fn new_concat(dim: usize) -> WasmBinary {
        WasmBinary { inner: Binary::new(BinaryOp::Concat, dim) }
    }

    /// Dua input. Shape-mismatch -> thrown string (bukan trap).
    #[wasm_bindgen(js_name = forwardBinary)]
    pub fn forward_binary(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        let out = self.inner.forward(a.inner.clone(), b.inner.clone())?;
        Ok(WasmTensor { inner: out })
    }

    // Parameter-free
    pub fn num_params(&self) -> usize {
        0
    }
}
