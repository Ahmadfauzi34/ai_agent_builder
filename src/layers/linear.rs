use burn::prelude::*;
use burn::module::{Module, Param}; // PENTING: Untuk load_record
use burn::tensor::TensorData;

// 1. CONFIG (Tetap sama)
#[derive(Config, Debug)]
pub struct StrictLinearConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl StrictLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictLinear<B> {
        let linear = nn::LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .init(device);

        StrictLinear {
            inner: linear,
            expected_input_dim: self.d_input,
        }
    }
}

// 2. MODULE
#[derive(Module, Debug)]
pub struct StrictLinear<B: Backend> {
    inner: nn::Linear<B>,
    expected_input_dim: usize,
}

impl<B: Backend> StrictLinear<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_batch, dim] = input.dims();
        
        if dim != self.expected_input_dim {
            panic!("STRICT ERROR: Input dim {} != Layer dim {}", dim, self.expected_input_dim);
        }

        self.inner.forward(input)
    }

    // --- FITUR BARU: LOAD WEIGHTS ---
    // Fungsi ini memakan layer lama (self), menyuntikkan weights, dan mengembalikan layer baru.
    pub fn load_weights(self, w_flat: Vec<f32>, b_flat: Option<Vec<f32>>) -> Self {
        let device = self.inner.weight.device();
        let [d_out, d_in] = self.inner.weight.dims(); // Shape asli layer

        // 1. Validasi Ukuran Weights
        if w_flat.len() != d_out * d_in {
            panic!("STRICT LOAD: Weight length mismatch. Expected {}, got {}", d_out * d_in, w_flat.len());
        }

        // 2. Buat Tensor Weight
        let w_data = TensorData::new(w_flat, [d_out, d_in]);
        let w_tensor = Tensor::from_data(w_data, &device);

        // 3. Buat Tensor Bias (Jika ada)
        let b_param = if let Some(b_data) = b_flat {
            if b_data.len() != d_out {
                panic!("STRICT LOAD: Bias length mismatch. Expected {}, got {}", d_out, b_data.len());
            }
            let b_tensor = Tensor::from_data(TensorData::new(b_data, [d_out]), &device);
            Some(Param::from_tensor(b_tensor))
        } else {
            None
        };

        // 4. Bungkus jadi Record (Format penyimpanan Burn)
        let record = nn::LinearRecord {
            weight: Param::from_tensor(w_tensor),
            bias: b_param,
        };

        // 5. Load Record ke dalam Module
        let new_inner = self.inner.load_record(record);

        // 6. Kembalikan Layer Baru yang sudah punya "Otak"
        StrictLinear {
            inner: new_inner,
            expected_input_dim: self.expected_input_dim,
        }
    }
}
