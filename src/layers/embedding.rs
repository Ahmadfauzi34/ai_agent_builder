use burn::prelude::*;

#[derive(Config, Debug)]
pub struct StrictEmbeddingConfig {
    pub n_vocab: usize,  // Jumlah kata di kamus
    pub d_model: usize,  // Ukuran vector per kata
}

impl StrictEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictEmbedding<B> {
        let embed = nn::EmbeddingConfig::new(self.n_vocab, self.d_model).init(device);
        StrictEmbedding { inner: embed }
    }
}

#[derive(Module, Debug)]
pub struct StrictEmbedding<B: Backend> {
    inner: nn::Embedding<B>,
}

impl<B: Backend> StrictEmbedding<B> {
    // Input: [Batch, Seq_Len] (Isinya ID kata, misal 1024, 50, 1)
    // Output: [Batch, Seq_Len, D_Model] (Isinya Vector Float)
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.inner.forward(input)
    }
}
