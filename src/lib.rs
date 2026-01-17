use wasm_bindgen::prelude::*;
use burn::prelude::*;

// 1. IMPORT MODUL LAYER KITA
// Pastikan folder src/layers/ ada dan berisi file linear.rs yang kita buat sebelumnya
pub mod layers;
use layers::linear::{StrictLinear, StrictLinearConfig};

// 2. KUNCI BACKEND (PENTING!)
// JavaScript tidak mengerti "Generic". Kita harus tentukan:
// "Versi WASM ini berjalan di CPU menggunakan NdArray dengan presisi Float 32-bit"
// (Nanti kalau mau pakai GPU, ganti ini jadi burn_wgpu::Wgpu)
type WasmBackend = burn_ndarray::NdArray<f32>;

// 3. JEMBATAN DATA (WasmTensor)
// Ini adalah amplop pembungkus Tensor Burn agar bisa dipegang oleh JS
#[wasm_bindgen]
pub struct WasmTensor {
    // Kita kunci tensor ini menjadi 2 Dimensi (Baris, Kolom) agar simpel
    inner: Tensor<WasmBackend, 2>,
}

#[wasm_bindgen]
impl WasmTensor {
    // Constructor: JS kirim Array Datar -> Rust ubah jadi Tensor 2D
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], rows: usize, cols: usize) -> WasmTensor {
        // Setup device (CPU)
        let device = Default::default();
        
        // Buat Data Burn dari array flat
        let tensor_data = burn::tensor::Data::new(
            data.to_vec(), 
            [rows, cols].into() // Shape
        );

        // Buat Tensor
        let tensor = Tensor::from_data(tensor_data, &device);
        
        WasmTensor { inner: tensor }
    }

    // Helper: Rust kirim balik data ke JS (sebagai Float32Array)
    pub fn to_array(&self) -> Vec<f32> {
        self.inner.to_data().value
    }
    
    // Helper: Cek Shape (untuk debug di JS console)
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
}

// 4. JEMBATAN LAYER (WasmStrictLinear)
// Kita bungkus layer generic kita menjadi layer konkret
#[wasm_bindgen]
pub struct WasmStrictLinear {
    inner: StrictLinear<WasmBackend>,
}

#[wasm_bindgen]
impl WasmStrictLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize) -> WasmStrictLinear {
        let device = Default::default();
        
        // Panggil Config dari kode Rust murni kita
        let config = StrictLinearConfig::new(in_dim, out_dim);
        
        // Init menjadi layer hidup
        let layer = config.init(&device);
        
        WasmStrictLinear { inner: layer }
    }

    // Forward Pass: Menerima WasmTensor, Mengembalikan WasmTensor
    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        // 1. Ambil tensor asli (Clone karena Burn tensor itu handle/pointer, jadi murah)
        let x = input.inner.clone();
        
        // 2. Proses pakai logika StrictLinear kita
        let output = self.inner.forward(x);
        
        // 3. Bungkus hasilnya
        WasmTensor { inner: output }
    }
}
