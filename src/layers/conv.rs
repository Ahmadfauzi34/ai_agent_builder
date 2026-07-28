use burn::prelude::*;
use burn::nn::conv::{
    Conv1d, Conv1dConfig,
    Conv2d, Conv2dConfig,
    ConvTranspose2d, ConvTranspose2dConfig
};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum ConvolutionConfig {
    Conv1d(Conv1dConfig),
    Conv2d(Conv2dConfig),
    ConvTranspose2d(ConvTranspose2dConfig),
}

impl ConvolutionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Convolution<B> {
        match self {
            ConvolutionConfig::Conv1d(c) => Convolution::Conv1d(c.init(device)),
            ConvolutionConfig::Conv2d(c) => Convolution::Conv2d(c.init(device)),
            ConvolutionConfig::ConvTranspose2d(c) => Convolution::ConvTranspose2d(c.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Convolution<B: Backend> {
    Conv1d(Conv1d<B>),
    Conv2d(Conv2d<B>),
    ConvTranspose2d(ConvTranspose2d<B>),
}

impl<B: Backend> Convolution<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Convolution::Conv2d(layer) => layer.forward(input),
            Convolution::ConvTranspose2d(layer) => layer.forward(input),
            Convolution::Conv1d(layer) => {
                let [b, c, h, _w] = input.dims();
                let x_3d = input.reshape([b, c, h]);
                let out = layer.forward(x_3d);
                let [b_out, c_out, l_out] = out.dims();
                out.reshape([b_out, c_out, l_out, 1])
            }
        }
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmConv {
    inner: Convolution<WasmBackend>,
}

#[wasm_bindgen]
impl WasmConv {
    #[wasm_bindgen(js_name = newConv1d)]
    pub fn new_conv1d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = Conv1dConfig::new(in_channels, out_channels, kernel_size);
        if let Some(s) = stride {
            config.stride = s;
        }
        if let Some(p) = padding {
            config.padding = burn::nn::PaddingConfig1d::Explicit(p);
        }
        WasmConv {
            inner: ConvolutionConfig::Conv1d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newConv2d)]
    pub fn new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = Conv2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = burn::nn::PaddingConfig2d::Explicit(ph, pw);
        }
        WasmConv {
            inner: ConvolutionConfig::Conv2d(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newConvTranspose2d)]
    pub fn new_conv_transpose2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmConv {
        let device = Default::default();
        let mut config = ConvTranspose2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];
        }
        WasmConv {
            inner: ConvolutionConfig::ConvTranspose2d(config).init(&device),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(data.to_vec(), &device)
            .map_err(|e| e.to_string())?;
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
// FLOAT-BRIDGE (M1) — conv. Record per variant = { weight: Param<TD>, bias: Option<Param<T1>> }.
// D = 3 (conv1d) atau 4 (conv2d / transpose2d). Helper generic supaya rank statis & aman.
// ============================================================
fn push_param<B: Backend, const D: usize>(
    p: &burn::module::Param<Tensor<B, D>>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let t = <Tensor<B, D> as Clone>::clone(p).into_data();
    out.extend(
        t.as_slice::<f32>()
            .map_err(|_| "getWeightsFlat: conv param not f32".to_string())?,
    );
    Ok(())
}

fn set_conv_param<B: Backend, const D: usize>(
    weight: &mut burn::module::Param<Tensor<B, D>>,
    bias: &mut Option<burn::module::Param<Tensor<B, 1>>>,
    data: &[f32],
) -> Result<(), String> {
    let wd = weight.dims(); // [usize; D]
    let wlen = wd.iter().product::<usize>();
    let has_bias = bias.is_some();
    let blen = if has_bias {
        bias.as_ref().unwrap().dims().iter().product::<usize>()
    } else {
        0
    };
    if data.len() != wlen + blen {
        return Err(format!(
            "setWeightsFlat: conv expected {} floats (weight{}), got {}",
            wlen + blen,
            if has_bias { "+bias" } else { "" },
            data.len()
        ));
    }
    let device: <B as Backend>::Device = Default::default();
    *weight = burn::module::Param::from_data(
        burn::tensor::TensorData::new(data[..wlen].to_vec(), wd),
        &device,
    );
    if has_bias {
        *bias = Some(burn::module::Param::from_data(
            burn::tensor::TensorData::new(data[wlen..].to_vec(), [blen]),
            &device,
        ));
    }
    Ok(())
}

#[wasm_bindgen]
impl WasmConv {
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        let mut out = Vec::new();
        match rec {
            ConvolutionRecord::Conv1d(r) => { // TITIK API: nama record
                push_param::<WasmBackend, 3>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias { push_param::<WasmBackend, 1>(b, &mut out)?; }
            }
            ConvolutionRecord::Conv2d(r) => {
                push_param::<WasmBackend, 4>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias { push_param::<WasmBackend, 1>(b, &mut out)?; }
            }
            ConvolutionRecord::ConvTranspose2d(r) => {
                push_param::<WasmBackend, 4>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias { push_param::<WasmBackend, 1>(b, &mut out)?; }
            }
        }
        Ok(out)
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.clone().into_record();
        match &mut rec {
            ConvolutionRecord::Conv1d(r) => {
                set_conv_param::<WasmBackend, 3>(&mut r.weight, &mut r.bias, data)?;
            }
            ConvolutionRecord::Conv2d(r) => {
                set_conv_param::<WasmBackend, 4>(&mut r.weight, &mut r.bias, data)?;
            }
            ConvolutionRecord::ConvTranspose2d(r) => {
                set_conv_param::<WasmBackend, 4>(&mut r.weight, &mut r.bias, data)?;
            }
        }
        self.inner = self.inner.clone().load_record(rec);
        Ok(())
    }
}

// ============================================================
// WEIGHT LAYOUT (M2) — conv. Mirror urutan getWeightsFlat per variant.
// ============================================================
fn push_conv_segs<B: Backend, const D: usize>(
    weight: &burn::module::Param<Tensor<B, D>>,
    bias: &Option<burn::module::Param<Tensor<B, 1>>>,
    segs: &mut Vec<(&'static str, usize)>,
) {
    segs.push(("weight", weight.dims().iter().product::<usize>()));
    if let Some(b) = bias {
        segs.push(("bias", b.dims().iter().product::<usize>()));
    }
}

impl WasmConv {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        let mut segs = Vec::new();
        match rec {
            ConvolutionRecord::Conv1d(r) => { // TITIK API (terbukti di M1)
                push_conv_segs::<WasmBackend, 3>(&r.weight, &r.bias, &mut segs);
            }
            ConvolutionRecord::Conv2d(r) => {
                push_conv_segs::<WasmBackend, 4>(&r.weight, &r.bias, &mut segs);
            }
            ConvolutionRecord::ConvTranspose2d(r) => {
                push_conv_segs::<WasmBackend, 4>(&r.weight, &r.bias, &mut segs);
            }
        }
        segs
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}
