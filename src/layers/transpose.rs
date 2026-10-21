use burn::prelude::*;

#[cfg(test)]
use burn::tensor::TensorData;

// --- TRANSPOSE LAYER (Parameter-Free) ---
//
// Swaps two axes of a rank-4 tensor: out[d0] <-> out[d1].
// This is the missing primitive that unlocks QK^T attention (K transposed
// from [B,S,F,1] to [B,F,S,1] via dims (1,2)), bilinear forms (x^T A y),
// outer products, and axis-agnostic linear projections
// (transpose -> linear -> transpose back).
#[derive(Debug, Clone, Copy)]
pub struct Transpose {
    dim0: usize,
    dim1: usize,
}

impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Result<Self, String> {
        for (name, d) in [("dim0", dim0), ("dim1", dim1)] {
            if d >= 4 {
                return Err(format!(
                    "transpose: {name} must be in 0..4, got {d}"
                ));
            }
        }
        Ok(Self { dim0, dim1 })
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.dim0, self.dim1)
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // Self-swap is a well-defined no-op.
        if self.dim0 == self.dim1 {
            return input;
        }
        Tensor::swap_dims(input, self.dim0, self.dim1)
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WasmBackend;

    fn tensor_of(data: Vec<f32>, dims: [usize; 4]) -> Tensor<WasmBackend, 4> {
        let device: <WasmBackend as Backend>::Device = Default::default();
        Tensor::from_data(TensorData::new(data, dims), &device)
    }

    #[test]
    fn transpose_swaps_last_two_axes() {
        // [1,1,2,3] -> [1,1,3,2]
        let t = Transpose::new(2, 3).unwrap();
        let x = tensor_of(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 2, 3]);
        let y = t.forward(x);
        assert_eq!(y.dims(), [1, 1, 3, 2]);
        let v: Vec<f32> = y.to_data().into_vec::<f32>().expect("f32 vec");
        assert_eq!(v, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_sequence_layout_for_attention() {
        // K: [B,S,F,1] -> [B,F,S,1] via dims (1,2): the K^T of QK^T attention.
        let t = Transpose::new(1, 2).unwrap();
        let x = tensor_of(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2, 1]);
        let y = t.forward(x);
        assert_eq!(y.dims(), [1, 2, 2, 1]);
        let v: Vec<f32> = y.to_data().into_vec::<f32>().expect("f32 vec");
        assert_eq!(v, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_self_swap_is_noop() {
        let t = Transpose::new(1, 1).unwrap();
        let x = tensor_of(vec![1.0, 2.0], [1, 2, 1, 1]);
        let y = t.forward(x);
        assert_eq!(y.dims(), [1, 2, 1, 1]);
    }

    #[test]
    fn transpose_rejects_out_of_range_axis() {
        assert!(Transpose::new(0, 4).is_err());
        assert!(Transpose::new(4, 0).is_err());
    }

    #[test]
    fn transpose_is_stateless() {
        assert_eq!(Transpose::new(1, 2).unwrap().num_params(), 0);
    }
}
