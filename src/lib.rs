use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;

// 1. IMPORT MODUL LAYER
pub mod layers;
use layers::linear::{StrictLinear, StrictLinearConfig};
use layers::conv::{StrictConv2d, StrictConv2dConfig};
use layers::embedding::{StrictEmbedding, StrictEmbeddingConfig};
use layers::activation::{StrictRelu, StrictGelu}; // Pastikan file activation.rs sudah ada

// 2. KUNCI BACKEND (CPU / NdArray)
type WasmBackend = burn_ndarray::NdArray<f32>;

// 3. JEMBATAN DATA (WasmTensor - Universal 4D)
// Kita ubah jadi 4D agar bisa menampung Gambar (B,C,H,W) maupun Teks/Linear
#[wasm_bindgen]
pub struct WasmTensor {
    inner: Tensor<WasmBackend, 4>, 
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let device = Default::default();
        
        // Logic: Paksa shape menjadi 4 Dimensi [d1, d2, d3, d4]
        // Jika user kirim [Batch, Dim], kita anggap [Batch, Dim, 1, 1]
        let mut dims = [1, 1, 1, 1];
        for (i, &d) in shape.iter().enumerate().take(4) {
            dims[i] = d;
        }

        let tensor_data = TensorData::new(data.to_vec(), dims);
        let tensor = Tensor::from_data(tensor_data, &device);
        
        WasmTensor { inner: tensor }
    }

    pub fn to_array(&self) -> Vec<f32> {
        let data = self.inner.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
}

// 4. WRAPPER LINEAR (Diupdate agar support WasmTensor 4D)
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
        WasmStrictLinear { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        // Linear butuh 2D [Batch, Dim]. Kita reshape dari 4D.
        let x = input.inner.clone();
        let [b, d, _, _] = x.dims(); 
        
        // Flatten ke 2D: [Batch, Dim * H * W] -> Asumsi input linear
        let x_2d = x.reshape([b, d]); 

        let out = self.inner.forward(x_2d);
        
        // Kembalikan ke 4D [Batch, OutDim, 1, 1] agar konsisten
        let [b_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, d_out, 1, 1]);

        WasmTensor { inner: out_4d }
    }
}

// 5. WRAPPER CONV2D (Baru)
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
        // Conv2d butuh 4D, WasmTensor sudah 4D. Langsung gas.
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

// 6. WRAPPER EMBEDDING (Baru)
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
        
        // Embedding butuh Integer (ID kata). Kita cast float -> int.
        let x_int = x_float.int(); 
        
        // Embedding butuh input 2D [Batch, SeqLen].
        // Kita ambil dimensi 0 dan 1 dari tensor 4D.
        let [b, s, _, _] = x_int.dims();
        let x_2d = x_int.reshape([b, s]);

        let out = self.inner.forward(x_2d);
        
        // Output Embedding adalah 3D [Batch, Seq, Dim].
        // Kita padding jadi 4D [Batch, Seq, Dim, 1]
        let [b_out, s_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, s_out, d_out, 1]);

        WasmTensor { inner: out_4d }
    }
}

// 7. WRAPPER AKTIVASI (Relu & Gelu)
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
