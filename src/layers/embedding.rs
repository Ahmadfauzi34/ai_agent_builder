use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};
use crate::layers::shape_contract::require_singleton_spatial;

fn validate_embedding_indices(values: &[f32], vocab_size: usize) -> Result<(), String> {
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

fn validate_embedding_state_structure(
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

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmEmbedding {
    inner: EmbeddingLayer<WasmBackend>,
}

#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(constructor)]
    pub fn new(vocab_size: usize, d_model: usize) -> WasmEmbedding {
        let device = Default::default();
        let config = EmbeddingConfig::new(vocab_size, d_model);
        WasmEmbedding {
            inner: EmbeddingConfigEnum::Basic(config).init(&device),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        self.try_forward(input)
            .unwrap_or_else(crate::layers::shape_contract::forward_fail)
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record: EmbeddingLayerRecord<WasmBackend> = crate::layers::state_record::decode_bin_record(
            data,
            &device,
            "Embedding loadState",
        )?;
        let current = self.inner.clone().into_record();
        validate_embedding_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        let record = self.inner.clone().into_record();
        let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
            .record(record, ())
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
}

impl WasmEmbedding {
    fn vocab_size(&self) -> usize {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => r.weight.dims()[0],
        }
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        require_singleton_spatial(input.inner.dims(), "Embedding forward")?;
        let input_data = input.inner.to_data();
        let values = input_data
            .as_slice::<f32>()
            .map_err(|_| "Embedding forward: input tensor is not f32".to_string())?;
        validate_embedding_indices(values, self.vocab_size())?;
        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

// ============================================================
// FLOAT-BRIDGE (M1) — embedding. Record = { weight: Param<T2> } (tanpa bias).
// ============================================================
#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(js_name = weightDims)]
    pub fn weight_dims(&self) -> Vec<usize> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => r.weight.dims().to_vec(),
        }
    }

    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => {
                let w = <Tensor<WasmBackend, 2> as Clone>::clone(&r.weight).into_data();
                w.as_slice::<f32>()
                    .map_err(|_| "getWeightsFlat: embedding weight not f32".to_string())
                    .map(|s| s.to_vec())
            }
        }
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.clone().into_record();
        match &mut rec {
            EmbeddingLayerRecord::Basic(r) => {
                let wd = r.weight.dims(); // [vocab, d_model]
                let need = wd[0] * wd[1];
                if data.len() != need {
                    return Err(format!("setWeightsFlat: expected {} floats, got {}", need, data.len()));
                }
                let device: <WasmBackend as Backend>::Device = Default::default();
                r.weight = burn::module::Param::from_data(
                    burn::tensor::TensorData::new(data[..need].to_vec(), wd),
                    &device,
                );
            }
        }
        self.inner = self.inner.clone().load_record(rec);
        Ok(())
    }
}

// ============================================================
// WEIGHT LAYOUT (M2) — embedding. Hanya weight (tanpa bias).
// ============================================================
impl WasmEmbedding {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => {
                vec![("weight", r.weight.dims().iter().product::<usize>())]
            }
        }
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}

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