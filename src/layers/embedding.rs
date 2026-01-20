use burn::prelude::*;
use burn::nn::{Embedding, EmbeddingConfig};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor}; // Import dari lib.rs

// --- CONFIGURATION ---
#[derive(Config, Debug)]
pub struct EmbeddingLayerConfig {
    pub n_vocab: usize,
    pub d_model: usize,
}

impl EmbeddingLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EmbeddingLayer<B> {
        // Panggil standar pabrik Burn
        let embed = EmbeddingConfig::new(self.n_vocab, self.d_model).init(device);
        EmbeddingLayer { inner: embed }
    }
}

// --- MODULE ---
#[derive(Module, Debug)]
pub struct EmbeddingLayer<B: Backend> {
    inner: Embedding<B>,
}

impl<B: Backend> EmbeddingLayer<B> {
    // Input: [Batch, Seq_Len] (Int)
    // Output: [Batch, Seq_Len, D_Model] (Float)
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.inner.forward(input)
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
    pub fn new(n_vocab: usize, d_model: usize) -> WasmEmbedding {
        let device = Default::default();
        let config = EmbeddingLayerConfig { n_vocab, d_model };
        WasmEmbedding { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        // 1. Ambil input (biasanya dari JS dikirim sebagai Float)
        let x_float = input.inner.clone();
        
        // 2. Cast ke Int (Karena Embedding butuh Index Integer, bukan Float)
        let x_int = x_float.int(); 
        
        // 3. Reshape ke 2D [Batch, Seq_Len]
        // WasmTensor kita 4D [B, S, 1, 1], kita ambil 2 dimensi pertama
        let [b, s, _, _] = x_int.dims();
        let x_2d = x_int.reshape([b, s]);

        // 4. Forward Pass
        let out = self.inner.forward(x_2d);
        
        // 5. Kembalikan ke format 4D [Batch, Seq, Dim, 1] agar masuk WasmTensor
        let [b_out, s_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, s_out, d_out, 1]);

        WasmTensor { inner: out_4d }
    }
}
