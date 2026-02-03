use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};

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
                // (Karena WasmTensor menyimpan f32, tapi Embedding butuh index integer)
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
