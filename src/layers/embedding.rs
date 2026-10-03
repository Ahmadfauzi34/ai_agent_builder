use super::state_record::deterministic_record_bytes;
pub use crate::facade::wasm_types::WasmEmbedding;
use crate::layers::shape_contract::require_singleton_spatial;
use crate::{WasmBackend, WasmTensor};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

pub(crate) fn validate_embedding_indices(values: &[f32], vocab_size: usize) -> Result<(), String> {
    for (position, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "Embedding forward: index at position {position} must be finite, got {value}"
            ));
        }
        if value.fract() != 0.0 {
            return Err(format!(
                "Embedding forward: index at position {position} must be an integer-valued float, got {value}"
            ));
        }
        if value < 0.0 {
            return Err(format!(
                "Embedding forward: index at position {position} must be >= 0, got {value}"
            ));
        }
        let index = value as usize;
        if index >= vocab_size {
            return Err(format!(
                "Embedding forward: index at position {position} is out of range: {index} >= vocab_size {vocab_size}"
            ));
        }
    }
    Ok(())
}

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum EmbeddingConfigEnum {
    Basic(EmbeddingConfig),
}

impl EmbeddingConfigEnum {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingLayer<B> {
        match self {
            EmbeddingConfigEnum::Basic(c) => EmbeddingLayer::Basic(c.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum EmbeddingLayer<B: Backend> {
    Basic(Embedding<B>),
}

impl<B: Backend> EmbeddingLayer<B> {
    // Input: Tensor 4D Float (dari WasmTensor)
    // Output: Tensor 4D Float
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            EmbeddingLayer::Basic(layer) => {
                // The public boundary validates that these f32 values are exact, in-range indices.
                let x_int = input.int();

                // 2. Reshape: 4D -> 2D
                // Asumsi input [Batch, Seq_Len, 1, 1] -> jadi [Batch, Seq_Len]
                let [b, s, _, _] = x_int.dims();
                let x_2d = x_int.reshape([b, s]);

                // 3. Proses Embedding
                // Outputnya adalah [Batch, Seq_Len, D_Model]
                let out = layer.forward(x_2d);

                // 4. Reshape Balik: 3D -> 4D
                // Menjadi [Batch, Seq_Len, D_Model, 1] agar muat di WasmTensor
                let [b_out, s_out, d_out] = out.dims();
                out.reshape([b_out, s_out, d_out, 1])
            }
        }
    }
}

pub(crate) fn validate_embedding_state_structure(
    current: &EmbeddingLayerRecord<WasmBackend>,
    incoming: &EmbeddingLayerRecord<WasmBackend>,
) -> Result<(), String> {
    match (current, incoming) {
        (EmbeddingLayerRecord::Basic(expected), EmbeddingLayerRecord::Basic(actual)) => {
            if expected.weight.dims() != actual.weight.dims() {
                return Err(format!(
                    "Embedding loadState: weight shape mismatch: expected {:?}, got {:?}",
                    expected.weight.dims(),
                    actual.weight.dims()
                ));
            }
            Ok(())
        }
    }
}

// ============================================================
// FLOAT-BRIDGE (M1) — embedding. Record = { weight: Param<T2> } (tanpa bias).
// ============================================================

// ============================================================
// WEIGHT LAYOUT (M2) — embedding. Hanya weight (tanpa bias).
// ============================================================

#[cfg(test)]
mod tests {
    use super::validate_embedding_indices;

    #[test]
    fn embedding_indices_accept_integer_values_inside_vocabulary() {
        assert!(validate_embedding_indices(&[0.0, 1.0, 3.0], 4).is_ok());
    }

    #[test]
    fn embedding_indices_reject_negative_and_upper_bound() {
        assert!(validate_embedding_indices(&[-1.0], 4).is_err());
        assert!(validate_embedding_indices(&[4.0], 4).is_err());
    }

    #[test]
    fn embedding_indices_reject_fractional_and_non_finite_values() {
        assert!(validate_embedding_indices(&[1.5], 4).is_err());
        assert!(validate_embedding_indices(&[f32::NAN], 4).is_err());
        assert!(validate_embedding_indices(&[f32::INFINITY], 4).is_err());
    }
}
