use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

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
                // 1. Konversi Tipe Data: Float -> Int
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
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &device)
            .map_err(|e| e.to_string())?;
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
