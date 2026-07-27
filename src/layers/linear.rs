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

#[derive(Module, Debug)] 
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
// ============================================================
// FLOAT-BRIDGE (B) — baca/tulis bobot sebagai Vec<f32> flat.
// Ini kabel yang membuat ES (gradient-free) bisa menyentuh bobot,
// sehingga ES bisa melatih layer nyata, bukan cuma vektor abstrak.
//
// Urutan flat: weight row-major [in_dim * out_dim], lalu bias [out_dim] (kalau ada).
// Implementasi via Module Record (bobot jadi tensor plain) -> menghindari Parameter API.
// ============================================================
#[wasm_bindgen]
impl WasmLinear {
    /// [in_dim, out_dim] — supaya JS tahu cara memotong vektor flat.
    #[wasm_bindgen(js_name = weightDims)]
    pub fn weight_dims(&self) -> Vec<usize> {
        let rec = self.inner.inner.clone().into_record();          // TITIK API #1
        rec.weight.dims().to_vec()                                 // TITIK API #3 (field `weight`)
    }

    /// Baca seluruh bobot sebagai flat f32: [weight..., bias...].
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.inner.clone().into_record();          // TITIK API #1
        let w = rec.weight.into_data();                            // TITIK API #2
        let mut out = w
            .as_slice::<f32>()
            .map_err(|_| "getWeightsFlat: weight not f32".to_string())?
            .to_vec();
        if let Some(b) = rec.bias {                                // TITIK API #3 (field `bias`)
            let bv = b
                .into_data()
                .as_slice::<f32>()
                .map_err(|_| "getWeightsFlat: bias not f32".to_string())?
                .to_vec();
            out.extend(bv);
        }
        Ok(out)
    }

    /// Tulis bobot dari flat f32. Panjang HARUS = in*out (+ out kalau bias).
    /// Shape in/out & ada-tidaknya bias di-infer dari bobot yang sudah ada.
    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.inner.clone().into_record();      // TITIK API #1
        let wd = rec.weight.dims();                                // [in, out]  (TITIK API #3)
        let in_d = wd[0];
        let out_d = wd[1];
        let has_bias = rec.bias.is_some();
        let need = in_d * out_d + if has_bias { out_d } else { 0 };
        if data.len() != need {
            return Err(format!(
                "setWeightsFlat: expected {} floats (in*out{}), got {}",
                need,
                if has_bias { "+out" } else { "" },
                data.len()
            ));
        }
        let device: <WasmBackend as Backend>::Device = Default::default();
        // TITIK API #4: assign field record dengan tensor baru (bobot jadi tensor plain di record).
        rec.weight = Tensor::from_data(
            burn::tensor::TensorData::new(data[..in_d * out_d].to_vec(), [in_d, out_d]),
            &device,
        );
        if has_bias {
            rec.bias = Some(Tensor::from_data(
                burn::tensor::TensorData::new(data[in_d * out_d..].to_vec(), [out_d]),
                &device,
            ));
        }
        self.inner.inner = self.inner.inner.clone().load_record(rec); // TITIK API #5
        Ok(())
    }
}
