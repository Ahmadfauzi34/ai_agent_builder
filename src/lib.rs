use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData; // Kita hanya pakai TensorData sekarang

// Import modul layer
pub mod layers;
use layers::linear::{StrictLinear, StrictLinearConfig};
use layers::conv::{StrictConv2d, StrictConv2dConfig};
use layers::embedding::{StrictEmbedding, StrictEmbeddingConfig};
use layers::activation::{StrictRelu, StrictGelu};

// Kunci Backend ke NdArray (CPU) untuk WASM
// (Nanti bisa diganti Wgpu jika mau WebGPU)
type WasmBackend = burn_ndarray::NdArray<f32>;

#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<WasmBackend, 4>, 
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let device = Default::default();
        
        // Logic Shape 4D (Tetap sama)
        let mut dims = [1, 1, 1, 1];
        for (i, &d) in shape.iter().enumerate().take(4) {
            dims[i] = d;
        }

        // PERUBAHAN V0.20:
        // TensorData sekarang lebih strict soal bytes/dtype.
        // new(data, shape) biasanya masih didukung, tapi kita pastikan alurnya.
        let tensor_data = TensorData::new(
            data.to_vec(), 
            dims
        );

        let tensor = Tensor::from_data(tensor_data, &device);
        
        WasmTensor { inner: tensor }
    }

    pub fn to_array(&self) -> Vec<f32> {
        // PERUBAHAN V0.20:
        // Mengambil data balik ke Vec<f32>
        let data = self.inner.to_data();
        
        // Kita pastikan konversi ke slice aman
        data.as_slice::<f32>().unwrap().to_vec()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
}

// --- WRAPPER LAYER (Tidak banyak berubah karena API Layer Burn stabil) ---

#[wasm_bindgen]
pub struct WasmStrictLinear {
    inner: StrictLinear<WasmBackend>,
}

#[wasm_bindgen]
impl WasmStrictLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize) -> WasmStrictLinear {
        let device = Default::default();
        let config = StrictLinearConfig::new(in_dim, out_dim);
        let layer = config.init(&device);
        WasmStrictLinear { inner: layer }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let [b, d, _, _] = x.dims(); 
        let x_2d = x.reshape([b, d]); 
        let out = self.inner.forward(x_2d);
        let [b_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, d_out, 1, 1]);
        WasmTensor { inner: out_4d }
    }

    // --- TAMBAHAN BARU ---
    // JS memanggil: layer.load_weights(floatArrayWeights, floatArrayBias)
    pub fn load_weights(&mut self, weights: &[f32], bias: &[f32]) {
        // 1. Clone layer saat ini (karena kita akan menggantinya)
        // (Di Rust, memodifikasi struct in-place yang punya generic agak tricky, 
        //  jadi kita clone -> update -> replace).
        let current_layer = self.inner.clone();

        // 2. Cek apakah bias kosong (panjang 0)
        let bias_opt = if bias.is_empty() {
            None
        } else {
            Some(bias.to_vec())
        };

        // 3. Panggil fungsi load_weights yang kita buat di linear.rs
        let updated_layer = current_layer.load_weights(
            weights.to_vec(), 
            bias_opt
        );

        // 4. Ganti layer lama dengan layer baru yang sudah pintar
        self.inner = updated_layer;
    }
}

#[wasm_bindgen]
pub struct WasmConv2d {
    inner: StrictConv2d<WasmBackend>,
}

#[wasm_bindgen]
impl WasmConv2d {
    #[wasm_bindgen(constructor)]
    pub fn new(c_in: usize, c_out: usize, kernel: usize) -> WasmConv2d {
        let device = Default::default();
        let config = StrictConv2dConfig::new(c_in, c_out, kernel);
        WasmConv2d { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

#[wasm_bindgen]
pub struct WasmEmbedding {
    inner: StrictEmbedding<WasmBackend>,
}

#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(constructor)]
    pub fn new(n_vocab: usize, d_model: usize) -> WasmEmbedding {
        let device = Default::default();
        let config = StrictEmbeddingConfig::new(n_vocab, d_model);
        WasmEmbedding { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x_float = input.inner.clone();
        let x_int = x_float.int(); 
        let [b, s, _, _] = x_int.dims();
        let x_2d = x_int.reshape([b, s]);
        let out = self.inner.forward(x_2d);
        let [b_out, s_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, s_out, d_out, 1]);
        WasmTensor { inner: out_4d }
    }
}

#[wasm_bindgen]
pub struct WasmRelu {
    inner: StrictRelu,
}

#[wasm_bindgen]
impl WasmRelu {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmRelu {
        WasmRelu { inner: StrictRelu::new() }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

#[wasm_bindgen]
pub struct WasmGelu {
    inner: StrictGelu,
}

#[wasm_bindgen]
impl WasmGelu {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGelu {
        WasmGelu { inner: StrictGelu::new() }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}
