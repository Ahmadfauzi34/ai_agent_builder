//! Fasad WASM tunggal — tipe adapter `Wasm*` (Opsi C, Fase 1, pindahan murni).
//!
//! Seluruh `#[wasm_bindgen] pub struct Wasm*` beserta SEMUA `impl` block-nya
//! dipindahkan ke sini tanpa perubahan nama/signature; permukaan JS identik.
//!
//! Kompatibilitas path:
//! - `burn_research::facade::wasm_types::<T>` — lokasi baru,
//! - `burn_research::<T>` — via re-export di `src/lib.rs`,
//! - path domain lama (`burn_research::layers::linear::WasmLinear`,
//!   `burn_research::math::WasmComparison`, ...) — via `pub use` di file asal.
//!
//! Helper privat yang dipakai adapter digeser menjadi `pub(crate)` di file
//! asalnya (bukan API publik) agar dapat di-import ke sini. Tidak ada
//! perubahan perilaku; hanya lokasi kode yang berubah.

use crate::layers::activation::{
    activation_forward_fail, validate_activation_state_structure, validate_glu_shape,
    validate_rank4_axis, Activation, ActivationConfig, ActivationRecord,
};
use crate::layers::binary::{Binary, BinaryOp};
use crate::layers::conv::{
    conv_fail, push_conv_segs, push_param, set_conv_param, validate_conv1d_stride,
    validate_conv2d_stride, validate_conv_state_structure, Convolution, ConvolutionConfig,
    ConvolutionRecord,
};
use crate::layers::custom::feature_norm::FeatureNorm;
use crate::layers::custom::ghost::{
    reject_invalid_config as ghost_reject_invalid_config, validate_ghost_state_structure,
    GhostModule, GhostModuleConfig, GhostModuleRecord,
};
use crate::layers::custom::seblock::{
    reject_invalid_config as seblock_reject_invalid_config, validate_seblock_state_structure,
    SeBlock, SeBlockConfig, SeBlockRecord,
};
use crate::layers::custom::shift::{Shift, ShiftDirection};
use crate::layers::embedding::{
    validate_embedding_indices, validate_embedding_state_structure, EmbeddingConfigEnum,
    EmbeddingLayer, EmbeddingLayerRecord,
};
use crate::layers::linear::{LinearLayer, LinearLayerConfig, LinearLayerRecord};
use crate::layers::norm::{
    norm_fail, norm_param_len, norm_trainable_refs, push_norm_param, set_norm_param,
    validate_group_config, validate_norm_axis, validate_norm_state_structure, validate_rms_epsilon,
    Normalization, NormalizationConfig, NormalizationRecord,
};
use crate::layers::pool::{
    pool_fail, validate_pool1d_params, validate_pool2d_params, Pooling, PoolingConfig,
};
use crate::layers::shape_contract::{
    require_axis_size, require_singleton_axis, require_singleton_spatial,
};
use crate::layers::state_record::deterministic_record_bytes;
use crate::math::linalg::{
    checked_output as linalg_checked_output, validate_epsilon, validate_feature_pair,
    validate_feature_shape, validate_finite as linalg_validate_finite, DEFAULT_EPSILON,
};
use crate::math::numeric::{
    checked_output as numeric_checked_output, validate_finite as numeric_validate_finite,
    validate_nonnegative, validate_nonzero, validate_positive, validate_same_shape,
};
use crate::math::probability::{
    batch_masses, checked_scalar_output, safe_log_input, validate_nonnegative_finite,
    validate_normalized, validate_pair,
};
use crate::math::statistics::{
    checked_output as statistics_checked_output, validate_feature_tensor,
};
use crate::math::tensor::{
    checked_element_count, parse_permutation, parse_rank4_shape, parse_slice_ranges,
    validate_select_indices,
};
use crate::math::{
    MathProgram, MathProgramBuilder, MathProgramV4, MathProgramV4Builder, MathProgramV5,
    MathProgramV5Builder, MathProgramV6, MathProgramV6Builder, MathProgramV7, MathProgramV7Builder,
    MathProgramV8, MathProgramV8Builder, MathProgramV9, MathProgramV9Builder, TensorComparison,
    TensorIndexSource, TensorReduction,
};
use crate::{WasmBackend, WasmTensor};
use burn::nn::activation::HardSwish;
use burn::nn::conv::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
};
use burn::nn::pool::{
    AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool1d, AvgPool1dConfig, AvgPool2d,
    AvgPool2dConfig, MaxPool1d, MaxPool1dConfig, MaxPool2d, MaxPool2dConfig,
};
use burn::nn::{
    BatchNormConfig, EmbeddingConfig, Gelu, GroupNormConfig, HardSigmoid, HardSigmoidConfig,
    InstanceNormConfig, LayerNormConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig,
    PaddingConfig1d, PaddingConfig2d, Relu, RmsNormConfig, Sigmoid, Softplus, SoftplusConfig,
    SwiGlu, SwiGluConfig, Tanh,
};
use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use wasm_bindgen::prelude::*;

// ======================================================================
// src/layers/custom/feature_norm.rs — WasmFeatureNorm
// ======================================================================

#[wasm_bindgen]
pub struct WasmFeatureNorm {
    inner: FeatureNorm,
}

#[wasm_bindgen]
impl WasmFeatureNorm {
    #[wasm_bindgen(js_name = newFeatureNorm)]
    pub fn new_feature_norm(epsilon: Option<f64>) -> Result<WasmFeatureNorm, String> {
        Ok(Self {
            inner: FeatureNorm::try_new(epsilon)?,
        })
    }

    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let out = self.inner.forward(input.inner.clone())?;
        Ok(WasmTensor { inner: out })
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

// ======================================================================
// src/layers/custom/shift.rs — WasmShift
// ======================================================================

#[wasm_bindgen]
pub struct WasmShift {
    inner: Shift,
}

#[wasm_bindgen]
impl WasmShift {
    #[wasm_bindgen(js_name = newShiftUp)]
    pub fn new_shift_up(shift_size: usize) -> WasmShift {
        WasmShift {
            inner: Shift::new(shift_size, ShiftDirection::Up),
        }
    }

    #[wasm_bindgen(js_name = newShiftDown)]
    pub fn new_shift_down(shift_size: usize) -> WasmShift {
        WasmShift {
            inner: Shift::new(shift_size, ShiftDirection::Down),
        }
    }

    #[wasm_bindgen(js_name = newShiftLeft)]
    pub fn new_shift_left(shift_size: usize) -> WasmShift {
        WasmShift {
            inner: Shift::new(shift_size, ShiftDirection::Left),
        }
    }

    #[wasm_bindgen(js_name = newShiftRight)]
    pub fn new_shift_right(shift_size: usize) -> WasmShift {
        WasmShift {
            inner: Shift::new(shift_size, ShiftDirection::Right),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    // Parameter-free: selalu 0
    pub fn num_params(&self) -> usize {
        0
    }
}

// ======================================================================
// src/layers/custom/ghost.rs — WasmGhostModule
// ======================================================================

#[wasm_bindgen]
pub struct WasmGhostModule {
    inner: GhostModule<WasmBackend>,
}

#[wasm_bindgen]
impl WasmGhostModule {
    #[wasm_bindgen(constructor)]
    /// Fallible constructor (complaint #14): invalid configs are a per-call
    /// `Err`, never a panic/`throw_str`, so corrupt bundle bytes cannot wedge
    /// the in-process WASM runtime.
    pub fn try_new(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        ratio: Option<usize>,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<WasmGhostModule, String> {
        let device = Default::default();
        let mut config =
            GhostModuleConfig::new(in_channels, out_channels, [kernel_size_h, kernel_size_w]);
        if let Some(r) = ratio {
            config.ratio = r;
        }
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];
        }
        let inner = config
            .try_init(&device)
            .unwrap_or_else(ghost_reject_invalid_config);
        Ok(WasmGhostModule { inner })
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
        let record: GhostModuleRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "GhostModule loadState")?;
        let current = self.inner.clone().into_record();
        validate_ghost_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

// ======================================================================
// src/layers/custom/seblock.rs — WasmSeBlock
// ======================================================================

#[wasm_bindgen]
pub struct WasmSeBlock {
    inner: SeBlock<WasmBackend>,
}

#[wasm_bindgen]
impl WasmSeBlock {
    #[wasm_bindgen(constructor)]
    /// Fallible constructor (complaint #14): invalid configs are a per-call
    /// `Err`, never a panic/`throw_str`, so corrupt bundle bytes cannot wedge
    /// the in-process WASM runtime.
    pub fn try_new(channels: usize, reduction: Option<usize>) -> Result<WasmSeBlock, String> {
        let device = Default::default();
        let mut config = SeBlockConfig::new(channels);
        if let Some(r) = reduction {
            config.reduction = r;
        }
        let inner = config
            .try_init(&device)
            .unwrap_or_else(seblock_reject_invalid_config);
        Ok(WasmSeBlock { inner })
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
        let record: SeBlockRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "SEBlock loadState")?;
        let current = self.inner.clone().into_record();
        validate_seblock_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

// ======================================================================
// src/layers/pool.rs — WasmPool
// ======================================================================

#[wasm_bindgen]
pub struct WasmPool {
    inner: Pooling,
}

#[wasm_bindgen]
impl WasmPool {
    #[wasm_bindgen(js_name = newMaxPool1d)]
    pub fn new_max_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        Self::try_new_max_pool1d(kernel_size, stride, padding).unwrap_or_else(pool_fail)
    }

    #[wasm_bindgen(js_name = newMaxPool2d)]
    pub fn new_max_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmPool {
        Self::try_new_max_pool2d(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(pool_fail)
    }

    #[wasm_bindgen(js_name = newAvgPool1d)]
    pub fn new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> WasmPool {
        Self::try_new_avg_pool1d(kernel_size, stride, padding).unwrap_or_else(pool_fail)
    }

    #[wasm_bindgen(js_name = newAvgPool2d)]
    pub fn new_avg_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> WasmPool {
        Self::try_new_avg_pool2d(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(pool_fail)
    }

    #[wasm_bindgen(js_name = newAdaptiveAvgPool2d)]
    pub fn new_adaptive_avg_pool2d(output_size_h: usize, output_size_w: usize) -> WasmPool {
        let config = AdaptiveAvgPool2dConfig::new([output_size_h, output_size_w]);
        WasmPool {
            inner: PoolingConfig::AdaptiveAvgPool2d(config).init(),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        self.try_forward(input)
            .unwrap_or_else(crate::layers::shape_contract::forward_fail)
    }

    pub fn num_params(&self) -> usize {
        0
    }
}

impl WasmPool {
    pub(crate) fn try_new_max_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool1d_params(kernel_size, stride, "MaxPool1d")?;
        let mut config = MaxPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(PaddingConfig1d::Explicit(p));
        }
        Ok(WasmPool {
            inner: PoolingConfig::MaxPool1d(config).init(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_max_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool2d_params(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            "MaxPool2d",
        )?;
        let mut config = MaxPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_strides([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding(PaddingConfig2d::Explicit(ph, pw));
        }
        Ok(WasmPool {
            inner: PoolingConfig::MaxPool2d(config).init(),
        })
    }

    pub(crate) fn try_new_avg_pool1d(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool1d_params(kernel_size, stride, "AvgPool1d")?;
        let mut config = AvgPool1dConfig::new(kernel_size);
        if let Some(s) = stride {
            config = config.with_stride(s);
        }
        if let Some(p) = padding {
            config = config.with_padding(PaddingConfig1d::Explicit(p));
        }
        Ok(WasmPool {
            inner: PoolingConfig::AvgPool1d(config).init(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_avg_pool2d(
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_pool2d_params(
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            "AvgPool2d",
        )?;
        let mut config = AvgPool2dConfig::new([kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config = config.with_strides([sh, sw]);
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config = config.with_padding(PaddingConfig2d::Explicit(ph, pw));
        }
        Ok(WasmPool {
            inner: PoolingConfig::AvgPool2d(config).init(),
        })
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        if matches!(&self.inner, Pooling::MaxPool1d(_) | Pooling::AvgPool1d(_)) {
            require_singleton_axis(input.inner.dims(), 3, "Pool1d forward")?;
        }
        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

// ======================================================================
// src/layers/activation.rs — WasmActivation
// ======================================================================

#[wasm_bindgen]
pub struct WasmActivation {
    inner: Activation<WasmBackend>,
}

#[wasm_bindgen]
impl WasmActivation {
    #[wasm_bindgen(js_name = newGelu)]
    pub fn new_gelu() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Gelu.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newRelu)]
    pub fn new_relu() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Relu.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newSigmoid)]
    pub fn new_sigmoid() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Sigmoid.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newTanh)]
    pub fn new_tanh() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Tanh.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newHardSwish)]
    pub fn new_hard_swish() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::HardSwish.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newLeakyRelu)]
    pub fn new_leaky_relu(negative_slope: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = LeakyReluConfig::new();
        if let Some(s) = negative_slope {
            config = config.with_negative_slope(s);
        }
        WasmActivation {
            inner: ActivationConfig::LeakyRelu(config).init(&device),
        }
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
        WasmActivation {
            inner: ActivationConfig::PRelu(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newSwiGlu)]
    pub fn new_swiglu(d_input: usize, d_output: usize, bias: Option<bool>) -> WasmActivation {
        let device = Default::default();
        let mut config = SwiGluConfig::new(d_input, d_output);
        if let Some(b) = bias {
            config = config.with_bias(b);
        }
        // Complaint #15: deterministic zero initial weights (no implicit RNG).
        let config = config.with_initializer(burn::module::Initializer::Zeros);
        WasmActivation {
            inner: ActivationConfig::SwiGlu(config).init(&device),
        }
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
        WasmActivation {
            inner: ActivationConfig::HardSigmoid(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newSoftplus)]
    pub fn new_softplus(beta: Option<f64>) -> WasmActivation {
        let device = Default::default();
        let mut config = SoftplusConfig::new();
        if let Some(b) = beta {
            config = config.with_beta(b);
        }
        WasmActivation {
            inner: ActivationConfig::Softplus(config).init(&device),
        }
    }

    #[wasm_bindgen(js_name = newMish)]
    pub fn new_mish() -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Mish.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newSoftmax)]
    pub fn new_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Softmax { dim }.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newLogSoftmax)]
    pub fn new_log_softmax(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::LogSoftmax { dim }.init(&device),
        }
    }

    #[wasm_bindgen(js_name = newGlu)]
    pub fn new_glu(dim: usize) -> WasmActivation {
        let device = Default::default();
        WasmActivation {
            inner: ActivationConfig::Glu { dim }.init(&device),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        self.try_forward(input)
            .unwrap_or_else(activation_forward_fail)
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record: ActivationRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "Activation loadState")?;
        let current = self.inner.clone().into_record();
        validate_activation_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

impl WasmActivation {
    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let shape = input.inner.dims();
        match &self.inner {
            Activation::Softmax(m) => validate_rank4_axis(m.dim, "Softmax forward")?,
            Activation::LogSoftmax(m) => validate_rank4_axis(m.dim, "LogSoftmax forward")?,
            Activation::Glu(m) => validate_glu_shape(shape, m.dim)?,
            _ => {}
        }

        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

// ======================================================================
// src/layers/binary.rs — WasmBinary
// ======================================================================

#[wasm_bindgen]
pub struct WasmBinary {
    inner: Binary,
}

#[wasm_bindgen]
impl WasmBinary {
    #[wasm_bindgen(js_name = newAdd)]
    pub fn new_add() -> WasmBinary {
        WasmBinary {
            inner: Binary::new(BinaryOp::Add, 0),
        }
    }
    #[wasm_bindgen(js_name = newSub)]
    pub fn new_sub() -> WasmBinary {
        WasmBinary {
            inner: Binary::new(BinaryOp::Sub, 0),
        }
    }
    #[wasm_bindgen(js_name = newMul)]
    pub fn new_mul() -> WasmBinary {
        WasmBinary {
            inner: Binary::new(BinaryOp::Mul, 0),
        }
    }
    #[wasm_bindgen(js_name = newMatmul)]
    pub fn new_matmul() -> WasmBinary {
        WasmBinary {
            inner: Binary::new(BinaryOp::Matmul, 0),
        }
    }
    #[wasm_bindgen(js_name = newConcat)]
    pub fn new_concat(dim: usize) -> WasmBinary {
        WasmBinary {
            inner: Binary::new(BinaryOp::Concat, dim),
        }
    }

    /// Dua input. Shape-mismatch -> thrown string (bukan trap).
    #[wasm_bindgen(js_name = forwardBinary)]
    pub fn forward_binary(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        let out = self.inner.forward(a.inner.clone(), b.inner.clone())?;
        Ok(WasmTensor { inner: out })
    }

    // Parameter-free
    pub fn num_params(&self) -> usize {
        0
    }
}

// ======================================================================
// src/layers/conv.rs — WasmConv
// ======================================================================

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
        Self::try_new_conv1d(in_channels, out_channels, kernel_size, stride, padding)
            .unwrap_or_else(conv_fail)
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
        Self::try_new_conv2d(
            in_channels,
            out_channels,
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(conv_fail)
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
        Self::try_new_conv_transpose2d(
            in_channels,
            out_channels,
            kernel_size_h,
            kernel_size_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        )
        .unwrap_or_else(conv_fail)
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
        let record: ConvolutionRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "Conv loadState")?;
        let current = self.inner.clone().into_record();
        validate_conv_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

impl WasmConv {
    pub(crate) fn try_new_conv1d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Self, String> {
        validate_conv1d_stride(stride, "Conv1d")?;
        let device = Default::default();
        let mut config = Conv1dConfig::new(in_channels, out_channels, kernel_size);
        if let Some(s) = stride {
            config.stride = s;
        }
        if let Some(p) = padding {
            config.padding = burn::nn::PaddingConfig1d::Explicit(p);
        }
        Ok(WasmConv {
            inner: ConvolutionConfig::Conv1d(
                config.with_initializer(burn::module::Initializer::Zeros),
            )
            .init(&device),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_conv2d_stride(stride_h, stride_w, "Conv2d")?;
        let device = Default::default();
        let mut config =
            Conv2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = burn::nn::PaddingConfig2d::Explicit(ph, pw);
        }
        Ok(WasmConv {
            inner: ConvolutionConfig::Conv2d(
                config.with_initializer(burn::module::Initializer::Zeros),
            )
            .init(&device),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_conv_transpose2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size_h: usize,
        kernel_size_w: usize,
        stride_h: Option<usize>,
        stride_w: Option<usize>,
        padding_h: Option<usize>,
        padding_w: Option<usize>,
    ) -> Result<Self, String> {
        validate_conv2d_stride(stride_h, stride_w, "ConvTranspose2d")?;
        let device = Default::default();
        let mut config =
            ConvTranspose2dConfig::new([in_channels, out_channels], [kernel_size_h, kernel_size_w]);
        if let (Some(sh), Some(sw)) = (stride_h, stride_w) {
            config.stride = [sh, sw];
        }
        if let (Some(ph), Some(pw)) = (padding_h, padding_w) {
            config.padding = [ph, pw];
        }
        Ok(WasmConv {
            inner: ConvolutionConfig::ConvTranspose2d(
                config.with_initializer(burn::module::Initializer::Zeros),
            )
            .init(&device),
        })
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        if matches!(&self.inner, Convolution::Conv1d(_)) {
            require_singleton_axis(input.inner.dims(), 3, "Conv1d forward")?;
        }
        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

#[wasm_bindgen]
impl WasmConv {
    /// WARNING — load-bearing invariant: the segment list and order here MUST
    /// stay in sync with `weight_segs()` / `weights_len()` below.
    /// `GraphParameterBinding::build()` derives binding offsets from
    /// `weights_len()` alone, so drift here silently corrupts parameter
    /// slicing in get/setGraphParametersFlat. If you change this function,
    /// update `weight_segs()` and extend the
    /// `weights_flat_len_is_length_aware_alias_of_flat_len` test.
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        let mut out = Vec::new();
        match rec {
            ConvolutionRecord::Conv1d(r) => {
                push_param::<WasmBackend, 3>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias {
                    push_param::<WasmBackend, 1>(b, &mut out)?;
                }
            }
            ConvolutionRecord::Conv2d(r) => {
                push_param::<WasmBackend, 4>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias {
                    push_param::<WasmBackend, 1>(b, &mut out)?;
                }
            }
            ConvolutionRecord::ConvTranspose2d(r) => {
                push_param::<WasmBackend, 4>(&r.weight, &mut out)?;
                if let Some(b) = &r.bias {
                    push_param::<WasmBackend, 1>(b, &mut out)?;
                }
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

impl WasmConv {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        let mut segs = Vec::new();
        match rec {
            ConvolutionRecord::Conv1d(r) => {
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

    /// Length-aware accessor: exactly `get_weights_flat().len()` without
    /// materializing the floats. The sum of `weight_segs()` lengths equals the
    /// flat length because both walk the same segments in the same order
    /// (guarded by `weight_layout_consistent_with_flat` and
    /// `weights_flat_len_is_length_aware_alias_of_flat_len`).
    /// Only dims metadata is read (via a cheap `into_record()` handle clone);
    /// no per-param `into_data()` full-buffer materialization happens here.
    pub(crate) fn weights_len(&self) -> usize {
        self.weight_segs().iter().map(|seg| seg.1).sum()
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}

// ======================================================================
// src/layers/embedding.rs — WasmEmbedding
// ======================================================================

#[wasm_bindgen]
pub struct WasmEmbedding {
    inner: EmbeddingLayer<WasmBackend>,
}

#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(constructor)]
    pub fn new(vocab_size: usize, d_model: usize) -> WasmEmbedding {
        let device = Default::default();
        // Complaint #15: deterministic zero initial weights (no implicit RNG).
        let config = EmbeddingConfig::new(vocab_size, d_model)
            .with_initializer(burn::module::Initializer::Zeros);
        WasmEmbedding {
            inner: EmbeddingConfigEnum::Basic(config).init(&device),
        }
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
        let record: EmbeddingLayerRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "Embedding loadState")?;
        let current = self.inner.clone().into_record();
        validate_embedding_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

impl WasmEmbedding {
    fn vocab_size(&self) -> usize {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => r.weight.dims()[0],
        }
    }

    fn d_model(&self) -> usize {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => r.weight.dims()[1],
        }
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        require_singleton_spatial(input.inner.dims(), "Embedding forward")?;
        // Complaint #18: output is [b, s, d_model], i.e. input_numel * d_model
        // elements — it can dwarf the input. Validate the budget before the
        // lookup materializes it, so the run fails structured.
        let input_numel: usize = input.inner.dims().iter().product();
        crate::protocol::check_numel(&[input_numel, self.d_model()], "Embedding forward output")?;
        let input_data = input.inner.to_data();
        let values = input_data
            .as_slice::<f32>()
            .map_err(|_| "Embedding forward: input tensor is not f32".to_string())?;
        validate_embedding_indices(values, self.vocab_size())?;
        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

#[wasm_bindgen]
impl WasmEmbedding {
    #[wasm_bindgen(js_name = weightDims)]
    pub fn weight_dims(&self) -> Vec<usize> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => r.weight.dims().to_vec(),
        }
    }

    /// WARNING — load-bearing invariant: the segment list and order here MUST
    /// stay in sync with `weight_segs()` / `weights_len()` below.
    /// `GraphParameterBinding::build()` derives binding offsets from
    /// `weights_len()` alone, so drift here silently corrupts parameter
    /// slicing in get/setGraphParametersFlat. If you change this function,
    /// update `weight_segs()` and extend the
    /// `weights_flat_len_is_length_aware_alias_of_flat_len` test.
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => {
                let w = <Tensor<WasmBackend, 2> as Clone>::clone(&r.weight).into_data();
                w.as_slice::<f32>()
                    .map_err(|_| "getWeightsFlat: embedding weight not f32".to_string())
                    .map(|s| s.to_vec())
            }
        }
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.clone().into_record();
        match &mut rec {
            EmbeddingLayerRecord::Basic(r) => {
                let wd = r.weight.dims(); // [vocab, d_model]
                let need = wd[0] * wd[1];
                if data.len() != need {
                    return Err(format!(
                        "setWeightsFlat: expected {} floats, got {}",
                        need,
                        data.len()
                    ));
                }
                let device: <WasmBackend as Backend>::Device = Default::default();
                r.weight = burn::module::Param::from_data(
                    burn::tensor::TensorData::new(data[..need].to_vec(), wd),
                    &device,
                );
            }
        }
        self.inner = self.inner.clone().load_record(rec);
        Ok(())
    }
}

impl WasmEmbedding {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        match rec {
            EmbeddingLayerRecord::Basic(r) => {
                vec![("weight", r.weight.dims().iter().product::<usize>())]
            }
        }
    }

    /// Length-aware accessor: exactly `get_weights_flat().len()` without
    /// materializing the floats (see WasmConv::weights_len for the invariant).
    pub(crate) fn weights_len(&self) -> usize {
        self.weight_segs().iter().map(|seg| seg.1).sum()
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}

// ======================================================================
// src/layers/linear.rs — WasmLinear
// ======================================================================

#[wasm_bindgen]
pub struct WasmLinear {
    inner: LinearLayer<WasmBackend>,
    in_dim: usize,
    out_dim: usize,
}

#[wasm_bindgen]
impl WasmLinear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_dim: usize, out_dim: usize, bias: bool) -> WasmLinear {
        let device = Default::default();
        let config = LinearLayerConfig {
            d_input: in_dim,
            d_output: out_dim,
            bias,
        };
        WasmLinear {
            inner: config.init(&device),
            in_dim,
            out_dim,
        }
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
        let record: LinearLayerRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "Linear loadState")?;
        let current = self.inner.clone().into_record();
        if record.inner.weight.dims() != current.inner.weight.dims() {
            return Err(format!(
                "Linear loadState: weight shape mismatch: expected {:?}, got {:?}",
                current.inner.weight.dims(),
                record.inner.weight.dims()
            ));
        }
        match (&current.inner.bias, &record.inner.bias) {
            (None, None) => {}
            (Some(expected), Some(actual)) if expected.dims() == actual.dims() => {}
            (Some(expected), Some(actual)) => {
                return Err(format!(
                    "Linear loadState: bias shape mismatch: expected {:?}, got {:?}",
                    expected.dims(),
                    actual.dims()
                ));
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err("Linear loadState: bias presence mismatch".to_string());
            }
        }
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

impl WasmLinear {
    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let x = input.inner.clone();
        let shape = x.dims();
        require_singleton_spatial(shape, "Linear forward")?;
        require_axis_size(shape, 1, self.in_dim, "Linear forward")?;

        let [b, d, _, _] = shape;
        // Complaint #18: the matmul below materializes [b, out_dim]; validate
        // the output against the allocation budget first so an oversized run
        // fails as a structured error instead of a raw `unreachable` trap.
        crate::protocol::check_numel(&[b, self.out_dim], "Linear forward output")?;
        let x_2d = x.reshape([b, d]);
        let out = self.inner.forward(x_2d);
        let [b_out, d_out] = out.dims();
        let out_4d = out.reshape([b_out, d_out, 1, 1]);
        Ok(WasmTensor { inner: out_4d })
    }
}

#[wasm_bindgen]
impl WasmLinear {
    /// [in_dim, out_dim] — supaya JS tahu cara memotong vektor flat.
    #[wasm_bindgen(js_name = weightDims)]
    pub fn weight_dims(&self) -> Vec<usize> {
        vec![self.in_dim, self.out_dim]
    }

    /// WARNING — load-bearing invariant: the segment list and order here MUST
    /// stay in sync with `weight_segs()` / `weights_len()` below.
    /// `GraphParameterBinding::build()` derives binding offsets from
    /// `weights_len()` alone, so drift here silently corrupts parameter
    /// slicing in get/setGraphParametersFlat. If you change this function,
    /// update `weight_segs()` and extend the
    /// `weights_flat_len_is_length_aware_alias_of_flat_len` test.
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
        debug_assert_eq!(wd, [self.in_dim, self.out_dim]);
        let in_d = self.in_dim;
        let out_d = self.out_dim;
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

    /// Length-aware accessor: exactly `get_weights_flat().len()` without
    /// materializing the floats (see WasmConv::weights_len for the invariant).
    pub(crate) fn weights_len(&self) -> usize {
        self.weight_segs().iter().map(|seg| seg.1).sum()
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}

// ======================================================================
// src/layers/norm.rs — WasmNorm
// ======================================================================

#[wasm_bindgen]
pub struct WasmNorm {
    inner: Normalization<WasmBackend>,
}

#[wasm_bindgen]
impl WasmNorm {
    #[wasm_bindgen]
    pub fn new_rms_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        Self::try_new_rms_norm(size, epsilon).unwrap_or_else(norm_fail)
    }

    #[wasm_bindgen]
    pub fn new_layer_norm(size: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Layer(LayerNormConfig::new(size).with_epsilon(eps));
        WasmNorm {
            inner: config.init(&device),
        }
    }

    #[wasm_bindgen]
    pub fn new_batch_norm(num_features: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config =
            NormalizationConfig::Batch(BatchNormConfig::new(num_features).with_epsilon(eps));
        WasmNorm {
            inner: config.init(&device),
        }
    }

    #[wasm_bindgen]
    pub fn new_group_norm(
        num_groups: usize,
        num_channels: usize,
        epsilon: Option<f64>,
    ) -> WasmNorm {
        Self::try_new_group_norm(num_groups, num_channels, epsilon).unwrap_or_else(norm_fail)
    }

    #[wasm_bindgen]
    pub fn new_instance_norm(num_channels: usize, epsilon: Option<f64>) -> WasmNorm {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config =
            NormalizationConfig::Instance(InstanceNormConfig::new(num_channels).with_epsilon(eps));
        WasmNorm {
            inner: config.init(&device),
        }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        self.try_forward(input).unwrap_or_else(norm_fail)
    }

    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let device = Default::default();
        let record: NormalizationRecord<WasmBackend> =
            crate::layers::state_record::decode_bin_record(data, &device, "Norm loadState")?;
        let current = self.inner.clone().into_record();
        validate_norm_state_structure(&current, &record)?;
        self.inner = self.inner.clone().load_record(record);
        Ok(())
    }

    pub fn get_state(&self) -> Result<Vec<u8>, String> {
        deterministic_record_bytes(&self.inner)
    }
}

impl WasmNorm {
    pub(crate) fn try_new_rms_norm(size: usize, epsilon: Option<f64>) -> Result<Self, String> {
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        validate_rms_epsilon(eps)?;
        let config = NormalizationConfig::Rms(RmsNormConfig::new(size).with_epsilon(eps));
        Ok(WasmNorm {
            inner: config.init(&device),
        })
    }

    pub(crate) fn try_new_group_norm(
        num_groups: usize,
        num_channels: usize,
        epsilon: Option<f64>,
    ) -> Result<Self, String> {
        validate_group_config(num_groups, num_channels)?;
        let device = Default::default();
        let eps = epsilon.unwrap_or(1e-5);
        let config = NormalizationConfig::Group(
            GroupNormConfig::new(num_groups, num_channels).with_epsilon(eps),
        );
        Ok(WasmNorm {
            inner: config.init(&device),
        })
    }

    pub(crate) fn try_forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let shape = input.inner.dims();
        match &self.inner {
            Normalization::Batch(norm) => {
                validate_norm_axis(shape, 1, norm.gamma.dims()[0], "BatchNorm forward")?;
            }
            Normalization::Group(norm) => {
                validate_norm_axis(shape, 1, norm.num_channels, "GroupNorm forward")?;
            }
            Normalization::Instance(norm) => {
                validate_norm_axis(shape, 1, norm.num_channels, "InstanceNorm forward")?;
            }
            Normalization::Layer(norm) => {
                validate_norm_axis(shape, 3, norm.gamma.dims()[0], "LayerNorm forward")?;
            }
            Normalization::Rms(norm) => {
                validate_norm_axis(shape, 3, norm.gamma.dims()[0], "RMSNorm forward")?;
            }
        }

        let out = self.inner.forward(input.inner.clone());
        Ok(WasmTensor { inner: out })
    }
}

#[wasm_bindgen]
impl WasmNorm {
    /// WARNING — load-bearing invariant: the segment list and order here MUST
    /// stay in sync with `weight_segs()` / `weights_len()` below.
    /// `GraphParameterBinding::build()` derives binding offsets from
    /// `weights_len()` alone, so drift here silently corrupts parameter
    /// slicing in get/setGraphParametersFlat. If you change this function,
    /// update `weight_segs()` and extend the
    /// `weights_flat_len_is_length_aware_alias_of_flat_len` test.
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self) -> Result<Vec<f32>, String> {
        let rec = self.inner.clone().into_record();
        let (gamma, beta) = norm_trainable_refs(&rec);
        let mut out = Vec::new();
        if let Some(g) = gamma {
            push_norm_param(g, &mut out)?;
        }
        if let Some(b) = beta {
            push_norm_param(b, &mut out)?;
        }
        Ok(out)
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, data: &[f32]) -> Result<(), String> {
        let mut rec = self.inner.clone().into_record();
        // panjang trainable: pinjam immut, lalu lepas (blok tersendiri)
        let (gl, bl) = {
            let (g, b) = norm_trainable_refs(&rec);
            (
                g.map(norm_param_len).unwrap_or(0),
                b.map(norm_param_len).unwrap_or(0),
            )
        };
        let total = gl + bl;
        if data.len() != total {
            return Err(format!(
                "setWeightsFlat: norm expected {} floats ({}{}), got {}",
                total,
                if gl > 0 { "gamma" } else { "" },
                if bl > 0 { "+beta" } else { "" },
                data.len()
            ));
        }
        // tulis per-field sekuensial (tanpa pinjam-mut bersamaan)
        match &mut rec {
            NormalizationRecord::Batch(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
                set_norm_param(&mut r.beta, &data[gl..])?;
            }
            NormalizationRecord::Group(r) => {
                if let Some(ref mut g) = r.gamma {
                    set_norm_param(g, &data[..gl])?;
                }
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Instance(r) => {
                if let Some(ref mut g) = r.gamma {
                    set_norm_param(g, &data[..gl])?;
                }
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Layer(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
                if let Some(ref mut b) = r.beta {
                    set_norm_param(b, &data[gl..])?;
                }
            }
            NormalizationRecord::Rms(r) => {
                set_norm_param(&mut r.gamma, &data[..gl])?;
            }
        }
        self.inner = self.inner.clone().load_record(rec);
        Ok(())
    }
}

impl WasmNorm {
    pub fn weight_segs(&self) -> Vec<(&'static str, usize)> {
        let rec = self.inner.clone().into_record();
        let (gamma, beta) = norm_trainable_refs(&rec);
        let mut segs = Vec::new();
        if let Some(g) = gamma {
            segs.push(("gamma", norm_param_len(g)));
        }
        if let Some(b) = beta {
            segs.push(("beta", norm_param_len(b)));
        }
        segs
    }

    /// Length-aware accessor: exactly `get_weights_flat().len()` without
    /// materializing the floats (see WasmConv::weights_len for the invariant).
    pub(crate) fn weights_len(&self) -> usize {
        self.weight_segs().iter().map(|seg| seg.1).sum()
    }

    pub fn weight_layout(&self) -> String {
        crate::layers::layout::segs_json(&self.weight_segs())
    }
}

// ======================================================================
// src/math/reduction_wasm.rs — WasmReduction
// ======================================================================

#[wasm_bindgen]
pub struct WasmReduction {
    inner: TensorReduction,
}

#[wasm_bindgen]
impl WasmReduction {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmReduction {
        WasmReduction {
            inner: TensorReduction::new(),
        }
    }

    #[wasm_bindgen(js_name = sumAxis)]
    pub fn sum_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.sum_axis(input, axis)
    }

    #[wasm_bindgen(js_name = meanAxis)]
    pub fn mean_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.mean_axis(input, axis)
    }

    #[wasm_bindgen(js_name = minAxis)]
    pub fn min_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.min_axis(input, axis)
    }

    #[wasm_bindgen(js_name = maxAxis)]
    pub fn max_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.max_axis(input, axis)
    }
}

// ======================================================================
// src/math/comparison_wasm.rs — WasmComparison
// ======================================================================

#[wasm_bindgen]
pub struct WasmComparison {
    inner: TensorComparison,
}

#[wasm_bindgen]
impl WasmComparison {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmComparison {
        WasmComparison {
            inner: TensorComparison::new(),
        }
    }

    #[wasm_bindgen(js_name = lessEqual01)]
    pub fn less_equal_01(&self, lhs: &WasmTensor, rhs: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.less_equal_01(lhs, rhs)
    }
}

// ======================================================================
// src/math/index_source_wasm.rs — WasmIndexSource
// ======================================================================

#[wasm_bindgen]
pub struct WasmIndexSource {
    inner: TensorIndexSource,
}

#[wasm_bindgen]
impl WasmIndexSource {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmIndexSource {
        WasmIndexSource {
            inner: TensorIndexSource::new(),
        }
    }

    #[wasm_bindgen(js_name = indicesLike)]
    pub fn indices_like(&self, reference: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        self.inner.indices_like(reference, axis)
    }
}

// ======================================================================
// src/math/linalg.rs — WasmLinearAlgebra
// ======================================================================

/// Stateless Burn-backed linear algebra surface for the canonical rank-4 bridge.
///
/// Vector operations use `[B,F,1,1]`. Matrix multiplication follows Burn's rank-4
/// batched semantics `[B,G,M,K] @ [B,G,K,N] -> [B,G,M,N]`.
#[wasm_bindgen]
pub struct WasmLinearAlgebra;

#[wasm_bindgen]
impl WasmLinearAlgebra {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLinearAlgebra {
        WasmLinearAlgebra
    }

    pub fn matmul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        let da = a.inner.dims();
        let db = b.inner.dims();
        if da.iter().any(|&d| d == 0) || db.iter().any(|&d| d == 0) {
            return Err(format!(
                "LinearAlgebra.matmul: zero-sized dimensions are not supported in v1: {da:?} @ {db:?}"
            ));
        }
        if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
            return Err(format!(
                "LinearAlgebra.matmul: incompatible shapes {da:?} @ {db:?}; expected [B,G,M,K] @ [B,G,K,N]"
            ));
        }
        linalg_validate_finite(a, "LinearAlgebra.matmul lhs")?;
        linalg_validate_finite(b, "LinearAlgebra.matmul rhs")?;
        linalg_checked_output(
            a.inner.clone().matmul(b.inner.clone()),
            "LinearAlgebra.matmul output",
        )
    }

    pub fn dot(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.dot")?;
        linalg_checked_output(
            a.inner.clone().mul(b.inner.clone()).sum_dim(1),
            "LinearAlgebra.dot output",
        )
    }

    #[wasm_bindgen(js_name = l2Norm)]
    pub fn l2_norm(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_shape(input.inner.dims(), "LinearAlgebra.l2Norm")?;
        linalg_validate_finite(input, "LinearAlgebra.l2Norm input")?;
        let squared = input.inner.clone().mul(input.inner.clone());
        linalg_checked_output(squared.sum_dim(1).sqrt(), "LinearAlgebra.l2Norm output")
    }

    #[wasm_bindgen(js_name = cosineSimilarity)]
    pub fn cosine_similarity(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        epsilon: Option<f64>,
    ) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.cosineSimilarity")?;
        let epsilon = validate_epsilon(
            epsilon.unwrap_or(DEFAULT_EPSILON),
            "LinearAlgebra.cosineSimilarity",
        )?;

        let dot = a.inner.clone().mul(b.inner.clone()).sum_dim(1);
        let norm_a = a.inner.clone().mul(a.inner.clone()).sum_dim(1).sqrt();
        let norm_b = b.inner.clone().mul(b.inner.clone()).sum_dim(1).sqrt();
        let denominator = norm_a.mul(norm_b).clamp_min(epsilon);
        linalg_checked_output(
            dot.div(denominator),
            "LinearAlgebra.cosineSimilarity output",
        )
    }

    #[wasm_bindgen(js_name = l2Distance)]
    pub fn l2_distance(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_pair(a, b, "LinearAlgebra.l2Distance")?;
        let delta = a.inner.clone().sub(b.inner.clone());
        let squared = delta.clone().mul(delta);
        linalg_checked_output(squared.sum_dim(1).sqrt(), "LinearAlgebra.l2Distance output")
    }
}

// ======================================================================
// src/math/numeric.rs — WasmNumericKernel
// ======================================================================

/// Stateless Burn-backed primitive math surface.
///
/// Numeric Kernel v1 intentionally forbids implicit broadcasting and rejects invalid numerical
/// domains as controlled errors instead of silently producing NaN/Inf. It is a lower-level math
/// primitive, not a neural layer or graph policy.
#[wasm_bindgen]
pub struct WasmNumericKernel;

#[wasm_bindgen]
impl WasmNumericKernel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmNumericKernel {
        WasmNumericKernel
    }

    pub fn add(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "add")?;
        numeric_validate_finite(a, "NumericKernel.add lhs")?;
        numeric_validate_finite(b, "NumericKernel.add rhs")?;
        numeric_checked_output(
            a.inner.clone().add(b.inner.clone()),
            "NumericKernel.add output",
        )
    }

    pub fn sub(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "sub")?;
        numeric_validate_finite(a, "NumericKernel.sub lhs")?;
        numeric_validate_finite(b, "NumericKernel.sub rhs")?;
        numeric_checked_output(
            a.inner.clone().sub(b.inner.clone()),
            "NumericKernel.sub output",
        )
    }

    pub fn mul(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "mul")?;
        numeric_validate_finite(a, "NumericKernel.mul lhs")?;
        numeric_validate_finite(b, "NumericKernel.mul rhs")?;
        numeric_checked_output(
            a.inner.clone().mul(b.inner.clone()),
            "NumericKernel.mul output",
        )
    }

    pub fn div(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        validate_same_shape(a, b, "div")?;
        numeric_validate_finite(a, "NumericKernel.div lhs")?;
        validate_nonzero(b, "NumericKernel.div rhs")?;
        numeric_checked_output(
            a.inner.clone().div(b.inner.clone()),
            "NumericKernel.div output",
        )
    }

    pub fn abs(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        numeric_validate_finite(input, "NumericKernel.abs input")?;
        numeric_checked_output(input.inner.clone().abs(), "NumericKernel.abs output")
    }

    pub fn sqrt(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_nonnegative(input, "NumericKernel.sqrt input")?;
        numeric_checked_output(input.inner.clone().sqrt(), "NumericKernel.sqrt output")
    }

    pub fn exp(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        numeric_validate_finite(input, "NumericKernel.exp input")?;
        numeric_checked_output(input.inner.clone().exp(), "NumericKernel.exp output")
    }

    pub fn log(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_positive(input, "NumericKernel.log input")?;
        numeric_checked_output(input.inner.clone().log(), "NumericKernel.log output")
    }

    pub fn clamp(&self, input: &WasmTensor, min: f32, max: f32) -> Result<WasmTensor, String> {
        if !min.is_finite() || !max.is_finite() {
            return Err(format!(
                "NumericKernel.clamp: bounds must be finite, got min={min}, max={max}"
            ));
        }
        if min > max {
            return Err(format!(
                "NumericKernel.clamp: min must be <= max, got min={min}, max={max}"
            ));
        }
        numeric_validate_finite(input, "NumericKernel.clamp input")?;
        numeric_checked_output(
            input.inner.clone().clamp(min, max),
            "NumericKernel.clamp output",
        )
    }

    #[wasm_bindgen(js_name = allFinite)]
    pub fn all_finite(&self, input: &WasmTensor) -> bool {
        // One materialization, no intermediate Vec: iterate the slice directly.
        let data = input.inner.to_data();
        let values = data
            .as_slice::<f32>()
            .expect("NumericKernel.allFinite: tensor is not f32");
        values.iter().all(|value| value.is_finite())
    }
}

// ======================================================================
// src/math/probability.rs — WasmProbability
// ======================================================================

/// Deterministic probability-distribution math over `[B,F,1,1]` tensors.
///
/// Entropy, cross-entropy and KL require normalized distributions. Zero-probability
/// P entries contribute exactly zero. Q support must cover every positive P entry.
#[wasm_bindgen]
pub struct WasmProbability;

#[wasm_bindgen]
impl WasmProbability {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmProbability {
        WasmProbability
    }

    pub fn normalize(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, values) = validate_nonnegative_finite(input, "Probability.normalize")?;
        let masses = batch_masses(&values, shape[0], shape[1]);
        for (batch_index, mass) in masses.into_iter().enumerate() {
            if !mass.is_finite() || mass <= 0.0 {
                return Err(format!(
                    "Probability.normalize: batch {batch_index} must have strictly positive finite mass; got {mass}"
                ));
            }
        }

        let mass = input.inner.clone().sum_dim(1);
        let denominator = mass.repeat_dim(1, shape[1]);
        let output = WasmTensor {
            inner: input.inner.clone().div(denominator),
        };
        validate_normalized(&output, "Probability.normalize output")?;
        Ok(output)
    }

    pub fn entropy(&self, p: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, _) = validate_normalized(p, "Probability.entropy")?;
        let safe_p = safe_log_input(p.inner.clone());
        let terms = p.inner.clone().mul(safe_p.log());
        checked_scalar_output(
            terms.sum_dim(1).neg(),
            shape[0],
            "Probability.entropy output",
        )
    }

    #[wasm_bindgen(js_name = crossEntropy)]
    pub fn cross_entropy(&self, p: &WasmTensor, q: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, _, _) = validate_pair(p, q, "Probability.crossEntropy")?;
        let safe_q = safe_log_input(q.inner.clone());
        let terms = p.inner.clone().mul(safe_q.log());
        checked_scalar_output(
            terms.sum_dim(1).neg(),
            shape[0],
            "Probability.crossEntropy output",
        )
    }

    #[wasm_bindgen(js_name = klDivergence)]
    pub fn kl_divergence(&self, p: &WasmTensor, q: &WasmTensor) -> Result<WasmTensor, String> {
        let (shape, _, _) = validate_pair(p, q, "Probability.klDivergence")?;
        let safe_p = safe_log_input(p.inner.clone());
        let safe_q = safe_log_input(q.inner.clone());
        let log_ratio = safe_p.log().sub(safe_q.log());
        let terms = p.inner.clone().mul(log_ratio);
        checked_scalar_output(
            terms.sum_dim(1),
            shape[0],
            "Probability.klDivergence output",
        )
    }
}

// ======================================================================
// src/math/program_v5_wasm.rs — WasmMathProgramV5, WasmMathProgramV5Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramV5Builder {
    inner: MathProgramV5Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV5Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV5Builder, String> {
        Ok(WasmMathProgramV5Builder {
            inner: MathProgramV5Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV5, String> {
        Ok(WasmMathProgramV5 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV5 {
    inner: MathProgramV5,
}

impl WasmMathProgramV5 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV5 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV5, String> {
        Ok(WasmMathProgramV5 {
            inner: MathProgramV5::from_plan(plan)?,
        })
    }

    pub fn run3(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
        h: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/program_v6_wasm.rs — WasmMathProgramV6, WasmMathProgramV6Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramV6Builder {
    inner: MathProgramV6Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV6Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV6Builder, String> {
        Ok(WasmMathProgramV6Builder {
            inner: MathProgramV6Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = addFillLike)]
    pub fn add_fill_like(&mut self, reference: u8, output: u8, scalar: f32) -> Result<(), String> {
        self.inner.add_fill_like(reference, output, scalar)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV6, String> {
        Ok(WasmMathProgramV6 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV6 {
    inner: MathProgramV6,
}

impl WasmMathProgramV6 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV6 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV6, String> {
        Ok(WasmMathProgramV6 {
            inner: MathProgramV6::from_plan(plan)?,
        })
    }

    pub fn run1(&self, a: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b])
    }

    pub fn run3(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
        h: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/program_v7_wasm.rs — WasmMathProgramV7, WasmMathProgramV7Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramV7Builder {
    inner: MathProgramV7Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV7Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV7Builder, String> {
        Ok(WasmMathProgramV7Builder {
            inner: MathProgramV7Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = addFillLike)]
    pub fn add_fill_like(&mut self, reference: u8, output: u8, scalar: f32) -> Result<(), String> {
        self.inner.add_fill_like(reference, output, scalar)
    }

    #[wasm_bindgen(js_name = addExpandLike)]
    pub fn add_expand_like(&mut self, source: u8, reference: u8, output: u8) -> Result<(), String> {
        self.inner.add_expand_like(source, reference, output)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV7, String> {
        Ok(WasmMathProgramV7 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV7 {
    inner: MathProgramV7,
}

impl WasmMathProgramV7 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV7 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV7, String> {
        Ok(WasmMathProgramV7 {
            inner: MathProgramV7::from_plan(plan)?,
        })
    }

    pub fn run1(&self, a: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b])
    }

    pub fn run3(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
        h: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/program_v8_wasm.rs — WasmMathProgramV8, WasmMathProgramV8Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramV8Builder {
    inner: MathProgramV8Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV8Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV8Builder, String> {
        Ok(WasmMathProgramV8Builder {
            inner: MathProgramV8Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = addFillLike)]
    pub fn add_fill_like(&mut self, reference: u8, output: u8, scalar: f32) -> Result<(), String> {
        self.inner.add_fill_like(reference, output, scalar)
    }

    #[wasm_bindgen(js_name = addExpandLike)]
    pub fn add_expand_like(&mut self, source: u8, reference: u8, output: u8) -> Result<(), String> {
        self.inner.add_expand_like(source, reference, output)
    }

    #[wasm_bindgen(js_name = addSumAxis)]
    pub fn add_sum_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_sum_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMeanAxis)]
    pub fn add_mean_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_mean_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMinAxis)]
    pub fn add_min_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_min_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMaxAxis)]
    pub fn add_max_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_max_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV8, String> {
        Ok(WasmMathProgramV8 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV8 {
    inner: MathProgramV8,
}

impl WasmMathProgramV8 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV8 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV8, String> {
        Ok(WasmMathProgramV8 {
            inner: MathProgramV8::from_plan(plan)?,
        })
    }

    pub fn run1(&self, a: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b])
    }

    pub fn run3(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
        h: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/program_v9_wasm.rs — WasmMathProgramV9, WasmMathProgramV9Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramV9Builder {
    inner: MathProgramV9Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV9Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV9Builder, String> {
        Ok(WasmMathProgramV9Builder {
            inner: MathProgramV9Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = addFillLike)]
    pub fn add_fill_like(&mut self, reference: u8, output: u8, scalar: f32) -> Result<(), String> {
        self.inner.add_fill_like(reference, output, scalar)
    }

    #[wasm_bindgen(js_name = addExpandLike)]
    pub fn add_expand_like(&mut self, source: u8, reference: u8, output: u8) -> Result<(), String> {
        self.inner.add_expand_like(source, reference, output)
    }

    #[wasm_bindgen(js_name = addSumAxis)]
    pub fn add_sum_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_sum_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMeanAxis)]
    pub fn add_mean_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_mean_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMinAxis)]
    pub fn add_min_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_min_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMaxAxis)]
    pub fn add_max_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_max_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addIndicesLike)]
    pub fn add_indices_like(&mut self, reference: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_indices_like(reference, output, axis)
    }

    #[wasm_bindgen(js_name = addLessEqual01)]
    pub fn add_less_equal_01(&mut self, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_less_equal_01(lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV9, String> {
        Ok(WasmMathProgramV9 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV9 {
    inner: MathProgramV9,
}

impl WasmMathProgramV9 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV9 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV9, String> {
        Ok(WasmMathProgramV9 {
            inner: MathProgramV9::from_plan(plan)?,
        })
    }

    pub fn run1(&self, a: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b])
    }

    pub fn run3(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(
        &self,
        a: &WasmTensor,
        b: &WasmTensor,
        c: &WasmTensor,
        d: &WasmTensor,
        e: &WasmTensor,
        f: &WasmTensor,
        g: &WasmTensor,
        h: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/program_wasm.rs — WasmMathProgram, WasmMathProgramBuilder, WasmMathProgramV4, WasmMathProgramV4Builder
// ======================================================================

#[wasm_bindgen]
pub struct WasmMathProgramBuilder {
    inner: MathProgramBuilder,
}

#[wasm_bindgen]
impl WasmMathProgramBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramBuilder, String> {
        Ok(WasmMathProgramBuilder {
            inner: MathProgramBuilder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgram, String> {
        Ok(WasmMathProgram {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgram {
    inner: MathProgram,
}

#[wasm_bindgen]
impl WasmMathProgram {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgram, String> {
        Ok(WasmMathProgram {
            inner: MathProgram::from_plan(plan)?,
        })
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run1(input)
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run2(a, b)
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV4Builder {
    inner: MathProgramV4Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV4Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV4Builder, String> {
        Ok(WasmMathProgramV4Builder {
            inner: MathProgramV4Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(&mut self, op: u8, lhs: u8, rhs: u8, output: u8) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV4, String> {
        Ok(WasmMathProgramV4 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV4 {
    inner: MathProgramV4,
}

#[wasm_bindgen]
impl WasmMathProgramV4 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV4, String> {
        Ok(WasmMathProgramV4 {
            inner: MathProgramV4::from_plan(plan)?,
        })
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run1(input)
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run2(a, b)
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}

// ======================================================================
// src/math/statistics.rs — WasmStatistics
// ======================================================================

/// Stateless descriptive-statistics surface for canonical feature tensors `[B,F,1,1]`.
///
/// All reducers operate across feature axis 1 and retain the canonical rank-4 bridge,
/// producing `[B,1,1,1]`. Variance and standard deviation use population semantics.
#[wasm_bindgen]
pub struct WasmStatistics;

#[wasm_bindgen]
impl WasmStatistics {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmStatistics {
        WasmStatistics
    }

    pub fn sum(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.sum")?;
        statistics_checked_output(input.inner.clone().sum_dim(1), "Statistics.sum output")
    }

    pub fn mean(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.mean")?;
        statistics_checked_output(input.inner.clone().mean_dim(1), "Statistics.mean output")
    }

    #[wasm_bindgen(js_name = variancePopulation)]
    pub fn variance_population(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.variancePopulation")?;
        statistics_checked_output(
            input.inner.clone().var_bias(1),
            "Statistics.variancePopulation output",
        )
    }

    #[wasm_bindgen(js_name = stdPopulation)]
    pub fn std_population(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.stdPopulation")?;
        statistics_checked_output(
            input.inner.clone().var_bias(1).sqrt(),
            "Statistics.stdPopulation output",
        )
    }

    pub fn min(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.min")?;
        statistics_checked_output(input.inner.clone().min_dim(1), "Statistics.min output")
    }

    pub fn max(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        validate_feature_tensor(input, "Statistics.max")?;
        statistics_checked_output(input.inner.clone().max_dim(1), "Statistics.max output")
    }
}

// ======================================================================
// src/math/tensor.rs — WasmTensorTransform
// ======================================================================

/// Stateless rank-4 tensor/layout transform surface.
///
/// Every operation validates shape/axis/range/index metadata before calling Burn so malformed
/// requests fail as controlled errors rather than backend panics or unchecked indexing behavior.
#[wasm_bindgen]
pub struct WasmTensorTransform;

#[wasm_bindgen]
impl WasmTensorTransform {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmTensorTransform {
        WasmTensorTransform
    }

    pub fn reshape(&self, input: &WasmTensor, shape: &[usize]) -> Result<WasmTensor, String> {
        let target = parse_rank4_shape(shape, "TensorTransform.reshape")?;
        let source = input.inner.dims();
        let source_count = checked_element_count(source, "TensorTransform.reshape source")?;
        let target_count = checked_element_count(target, "TensorTransform.reshape target")?;
        if source_count != target_count {
            return Err(format!(
                "TensorTransform.reshape: element-count mismatch: source {source:?} has {source_count}, target {target:?} has {target_count}"
            ));
        }
        Ok(WasmTensor {
            inner: input.inner.clone().reshape(target),
        })
    }

    pub fn transpose(&self, input: &WasmTensor) -> WasmTensor {
        WasmTensor {
            inner: input.inner.clone().swap_dims(2, 3),
        }
    }

    pub fn permute(&self, input: &WasmTensor, axes: &[usize]) -> Result<WasmTensor, String> {
        let axes = parse_permutation(axes)?;
        Ok(WasmTensor {
            inner: input.inner.clone().permute(axes),
        })
    }

    pub fn slice(
        &self,
        input: &WasmTensor,
        starts: &[usize],
        ends: &[usize],
    ) -> Result<WasmTensor, String> {
        let dims = input.inner.dims();
        let (starts, ends) = parse_slice_ranges(dims, starts, ends)?;
        Ok(WasmTensor {
            inner: input.inner.clone().slice([
                starts[0]..ends[0],
                starts[1]..ends[1],
                starts[2]..ends[2],
                starts[3]..ends[3],
            ]),
        })
    }

    #[wasm_bindgen(js_name = selectAxis)]
    pub fn select_axis(
        &self,
        input: &WasmTensor,
        axis: usize,
        indices: &[usize],
    ) -> Result<WasmTensor, String> {
        let dims = input.inner.dims();
        validate_select_indices(dims, axis, indices)?;

        let device = input.inner.device();
        let index_values: Vec<i64> = indices.iter().map(|&value| value as i64).collect();
        let index_tensor = Tensor::<WasmBackend, 1, Int>::from_data(
            TensorData::new(index_values, [indices.len()]),
            &device,
        );

        Ok(WasmTensor {
            inner: input.inner.clone().select(axis, index_tensor),
        })
    }
}
