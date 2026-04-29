use burn::prelude::*;
use burn::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu,
    PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
};
use burn::nn::activation::HardSwish;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use wasm_bindgen::prelude::*;
use crate::{WasmBackend, WasmTensor};

// --- HELPER STRUCTS (SOLUSI ERROR DERIVE) ---
#[derive(Module, Debug, Clone)]
pub struct StrictSoftmax {
    pub dim: usize,
}
impl StrictSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictLogSoftmax {
    pub dim: usize,
}
impl StrictLogSoftmax {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::log_softmax(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictGlu {
    pub dim: usize,
}
impl StrictGlu {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::glu(input, self.dim)
    }
}

#[derive(Module, Debug, Clone)]
pub struct StrictMish;
impl StrictMish {
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        burn::tensor::activation::mish(input)
    }
}

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum ActivationConfig {
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    HardSwish,
    LeakyRelu(LeakyReluConfig),
    PRelu(PReluConfig),
    SwiGlu(SwiGluConfig),
    HardSigmoid(HardSigmoidConfig),
    Softplus(SoftplusConfig),
    Mish,
    Softmax { dim: usize },
    LogSoftmax { dim: usize },
    Glu { dim: usize },
}

impl ActivationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Gelu => Activation::Gelu(Gelu::new()),
            ActivationConfig::Relu => Activation::Relu(Relu::new()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            ActivationConfig::Tanh => Activation::Tanh(Tanh::new()),
            ActivationConfig::HardSwish => Activation::HardSwish(HardSwish::new()),
            ActivationConfig::LeakyRelu(c) => Activation::LeakyRelu(c.init()),
            ActivationConfig::PRelu(c) => Activation::PRelu(c.init(device)),
            ActivationConfig::SwiGlu(c) => Activation::SwiGlu(c.init(device)),
            ActivationConfig::HardSigmoid(c) => Activation::HardSigmoid(c.init()),
            ActivationConfig::Softplus(c) => Activation::Softplus(c.init()),
            ActivationConfig::Mish => Activation::Mish(StrictMish),
            ActivationConfig::Softmax { dim } => Activation::Softmax(StrictSoftmax { dim: *dim }),
            ActivationConfig::LogSoftmax { dim } => Activation::LogSoftmax(StrictLogSoftmax { dim: *dim }),
            ActivationConfig::Glu { dim } => Activation::Glu(StrictGlu { dim: *dim }),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Activation<B: Backend> {
    Gelu(Gelu),
    Relu(Relu),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    HardSwish(HardSwish),
    LeakyRelu(LeakyRelu),
    PRelu(PRelu<B>),
    SwiGlu(SwiGlu<B>),
    HardSigmoid(HardSigmoid),
    Softplus(Softplus),
    Mish(StrictMish),
    Softmax(StrictSoftmax),
    LogSoftmax(StrictLogSoftmax),
    Glu(StrictGlu),
}

impl<B: Backend> Activation<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Activation::Gelu(m) => m.forward(input),
            Activation::Relu(m) => m.forward(input),
            Activation::Sigmoid(m) => m.forward(input),
            Activation::Tanh(m) => m.forward(input),
            Activation::HardSwish(m) => m.forward(input),
            Activation::LeakyRelu(m) => m.forward(input),
            Activation::PRelu(m) => m.forward(input),
            Activation::SwiGlu(m) => m.forward(input),
            Activation::HardSigmoid(m) => m.forward(input),
            Activation::Softplus(m) => m.forward(input),
            Activation::Mish(m) => m.forward(input),
            Activation::Softmax(m) => m.forward(input),
            Activation::LogSoftmax(m) => m.forward(input),
            Activation::Glu(m) => m.forward(input),
        }
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    #[wasm_bindgen(js_name = newGelu)]
    pub fn new_gelu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Gelu.init(&device) }
    }

    #[wasm_bindgen(js_name = newRelu)]
    pub fn new_relu() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Relu.init(&device) }
    }

    #[wasm_bindgen(js_name = newSigmoid)]
    pub fn new_sigmoid() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Sigmoid.init(&device) }
    }

    #[wasm_bindgen(js_name = newTanh)]
    pub fn new_tanh() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Tanh.init(&device) }
    }

    #[wasm_bindgen(js_name = newHardSwish)]
    pub fn new_hard_swish() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::HardSwish.init(&device) }
    }

    #[wasm_bindgen(js_name = newLeakyRelu)]
    pub fn new_leaky_relu(negative_slope: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = LeakyReluConfig::new();
        if let Some(s) = negative_slope {
            config = config.with_negative_slope(s);
        }
        WasmActivation { inner: ActivationConfig::LeakyRelu(config).init(&device) }
    }

    #[wasm_bindgen(js_name = newPRelu)]
    pub fn new_prelu(num_parameters: Option<usize>, alpha: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = PReluConfig::new();
        if let Some(n) = num_parameters {
            config = config.with_num_parameters(n);
        }
        if let Some(a) = alpha {
            config = config.with_alpha(a);
        }
        WasmActivation { inner: ActivationConfig::PRelu(config).init(&device) }
    }

    #[wasm_bindgen(js_name = newSwiGlu)]
    pub fn new_swiglu(d_input: usize, d_output: usize, bias: Option<bool>) -> WasmActivation {
        let device = Default::default();
        let mut config = SwiGluConfig::new(d_input, d_output);
        if let Some(b) = bias {
            config = config.with_bias(b);
        }
        WasmActivation { inner: ActivationConfig::SwiGlu(config).init(&device) }
    }

    #[wasm_bindgen(js_name = newHardSigmoid)]
    pub fn new_hard_sigmoid(alpha: Option<f64>, beta: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = HardSigmoidConfig::new();
        if let Some(a) = alpha {
            config = config.with_alpha(a);
        }
        if let Some(b) = beta {
            config = config.with_beta(b);
        }
        WasmActivation { inner: ActivationConfig::HardSigmoid(config).init(&device) }
    }

    #[wasm_bindgen(js_name = newSoftplus)]
    pub fn new_softplus(beta: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = SoftplusConfig::new();
        if let Some(b) = beta {
            config = config.with_beta(b);
        }
        WasmActivation { inner: ActivationConfig::Softplus(config).init(&device) }
    }

    #[wasm_bindgen(js_name = newMish)]
    pub fn new_mish() -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Mish.init(&device) }
    }

    #[wasm_bindgen(js_name = newSoftmax)]
    pub fn new_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Softmax { dim }.init(&device) }
    }

    #[wasm_bindgen(js_name = newLogSoftmax)]
    pub fn new_log_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::LogSoftmax { dim }.init(&device) }
    }

    #[wasm_bindgen(js_name = newGlu)]
    pub fn new_glu(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation { inner: ActivationConfig::Glu { dim }.init(&device) }
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
