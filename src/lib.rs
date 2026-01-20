use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;

pub mod layers;
use layers::linear::{StrictLinear, StrictLinearConfig};
use layers::conv::{StrictConv2d, StrictConv2dConfig};
use layers::embedding::{StrictEmbedding, StrictEmbeddingConfig};
use layers::activation::{Activation, ActivationConfig};
use layers::norm::{Normalization, NormalizationConfig};

// PERBAIKAN: Import PReluConfig (R besar)
use burn::nn::{
    LeakyReluConfig, PReluConfig, HardSigmoidConfig, SoftplusConfig, SwiGluConfig,
    BatchNormConfig, LayerNormConfig, RmsNormConfig, GroupNormConfig, InstanceNormConfig
};

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
        let mut dims = [1, 1, 1, 1];
        for (i, &d) in shape.iter().enumerate().take(4) { dims[i] = d; }
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

#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    #[wasm_bindgen]
    pub fn new_relu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Relu.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_gelu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Gelu.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_sigmoid() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Sigmoid.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_tanh() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Tanh.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_hard_swish() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::HardSwish.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_leaky_relu(slope: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::LeakyRelu(
            LeakyReluConfig::new().with_negative_slope(slope)
        );
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_prelu() -> WasmActivation {
        let device = Default::default();
        // PERBAIKAN: Gunakan PReluConfig
        let config = ActivationConfig::PRelu(PReluConfig::new());
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_swiglu(d_model: usize) -> WasmActivation {
        let device = Default::default();
        // PERBAIKAN: SwiGluConfig butuh (d_input, d_output).
        // Kita asumsikan output sama dengan input (d_model)
        let config = ActivationConfig::SwiGlu(SwiGluConfig::new(d_model, d_model));
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_hard_sigmoid(alpha: f64, beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::HardSigmoid(
            HardSigmoidConfig::new().with_alpha(alpha).with_beta(beta)
        );
        WasmActivation { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_softplus(beta: f64) -> WasmActivation {
        let device = Default::default();
        let config = ActivationConfig::Softplus(
            SoftplusConfig::new().with_beta(beta)
        );
        WasmActivation { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

// --- WRAPPER NORMALISASI ---
#[wasm_bindgen]
pub struct WasmNorm {
    inner: Normalization<WasmBackend>,
}

#[wasm_bindgen]
impl WasmNorm {
    #[wasm_bindgen]
    pub fn new_rms_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Rms(RmsNormConfig::new(size).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_layer_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Layer(LayerNormConfig::new(size).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_batch_norm(num_features: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Batch(BatchNormConfig::new(num_features).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_group_norm(num_groups: usize, num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Group(GroupNormConfig::new(num_groups, num_channels).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    #[wasm_bindgen]
    pub fn new_instance_norm(num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Instance(InstanceNormConfig::new(num_channels).with_epsilon(eps));
        WasmNorm { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

// --- WRAPPER LINEAR ---
#[wasm_bindgen]
pub struct WasmLinear {
    inner: StrictLinear<WasmBackend>,
}

#[wasm_bindgen]
impl WasmLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize, bias: bool) -> WasmLinear {
        let device = Default::default();
        let config = StrictLinearConfig { d_input: in_dim, d_output: out_dim, bias };
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
}

// --- WRAPPER CONV2D ---
#[wasm_bindgen]
pub struct WasmConv2d {
    inner: StrictConv2d<WasmBackend>,
}

#[wasm_bindgen]
impl WasmConv2d {
    #[wasm_bindgen(constructor)]
    pub fn new(c_in: usize, c_out: usize, k_h: usize, k_w: usize, pad_h: usize, pad_w: usize, stride: usize) -> WasmConv2d {
        let device = Default::default();
        let config = StrictConv2dConfig {
            channels_in: c_in,
            channels_out: c_out,
            kernel_size: [k_h, k_w],
            stride: [stride, stride],
            padding: [pad_h, pad_w],
        };
        WasmConv2d { inner: config.init(&device) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }
}

// --- WRAPPER EMBEDDING ---
#[wasm_bindgen]
pub struct WasmEmbedding {
    inner: StrictEmbedding<WasmBackend>,
}

#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(constructor)]
    pub fn new(n_vocab: usize, d_model: usize) -> WasmEmbedding {
        let device = Default::default();
        let config = StrictEmbeddingConfig { n_vocab, d_model };
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
