use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};
use crate::layers::shape_contract::{require_axis_size, require_singleton_spatial};

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
        self.try_forward(input)
            .unwrap_or_else(crate::layers::shape_contract::forward_fail)
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

impl WasmLinear {
    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let x = input.inner.clone();
        let shape = x.dims();
        require_singleton_spatial(shape, "Linear forward")?;
        let expected_features = self.weight_dims()[0];
        require_axis_size(shape, 1, expected_features, "Linear forward")?;

        let [b, d, _, _] = shape;
        let x_2d = x.reshape([b, d]);
        let out = self.inner.forward(x_2d);
        let [b_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, d_out, 1, 1]);
        Ok(WasmTensor { inner: out_4d })
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
        let rec = self.inner.inner.clone().into_record();
        rec.weight.dims().to_vec()
    }

    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.inner.clone().into_record();
        let w = <Tensor<WasmBackend, 2> as Clone>::clone(&rec.weight).into_data();
        let mut out = w
            .as_slice::<f32>()
            .map_err(|_| "getWeightsFlat: weight not f32".to_string())?
            .to_vec();
        if let Some(b) = &rec.bias {
            let bv = <Tensor<WasmBackend, 1> as Clone>::clone(b)
                .into_data()
                .as_slice::<f32>()
                .map_err(|_| "getWeightsFlat: bias not f32".to_string())?
                .to_vec();
            out.extend(bv);
        }
        Ok(out)
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.inner.clone().into_record();
        let wd = rec.weight.dims(); // [in, out]
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
        rec.weight = burn::module::Param::from_data(
            burn::tensor::TensorData::new(data[..in_d * out_d].to_vec(), [in_d, out_d]),
            &device,
        );
        if has_bias {
            rec.bias = Some(burn::module::Param::from_data(
                burn::tensor::TensorData::new(data[in_d * out_d..].to_vec(), [out_d]),
                &device,
            ));
        }
        self.inner.inner = self.inner.inner.clone().load_record(rec);
        Ok(())
    }
}

// ============================================================
// WEIGHT LAYOUT (M2) — linear. Mirror urutan getWeightsFlat.
// ============================================================
impl WasmLinear {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.inner.clone().into_record();
        let wlen = rec.weight.dims().iter().product::<usize>();
        let mut segs = vec![("weight", wlen)];
        if let Some(b) = &rec.bias {
            segs.push(("bias", b.dims().iter().product::<usize>()));
        }
        segs
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}
