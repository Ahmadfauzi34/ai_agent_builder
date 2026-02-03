use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIG & MODULE ---
#[derive(Config, Debug)]
pub struct LinearLayerConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl LinearLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearLayer<B> {
        let linear = LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .init(device);
        LinearLayer { inner: linear }
    }
}

#[derive(Module, Debug, Clone)] // Tambahkan Clone di sini
pub struct LinearLayer<B: Backend> {
    inner: Linear<B>,
}

impl<B: Backend> LinearLayer<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmLinear {
    inner: LinearLayer<WasmBackend>,
}

#[wasm_bindgen]
impl WasmLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize, bias: bool) -> WasmLinear {
        let device = Default::default();
        let config = LinearLayerConfig { 
            d_input: in_dim, 
            d_output: out_dim, 
            bias 
        };
        WasmLinear { inner: config.init(&device) }
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

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &device)
            .map_err(|e| e.to_string())?;
            
        // PERBAIKAN: Clone dulu sebelum load_record
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
