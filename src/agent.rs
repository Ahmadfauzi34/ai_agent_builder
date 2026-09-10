use wasm_bindgen::prelude::*;

use crate::graph::CompiledGraph;
use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX,
    ACT_MISH, ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU,
    ACT_TANH, BINARY_ADD, BINARY_CONCAT, BINARY_MATMUL, BINARY_MUL, BINARY_SUB,
    CONV_CONV1D, CONV_CONV2D, CONV_CONVTRANSPOSE2D, FLAG_BIAS, LAYER_ACTIVATION,
    LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM,
    LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT, NORM_BATCH, NORM_GROUP, NORM_INSTANCE,
    NORM_LAYER, NORM_RMS, OP_INIT, POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D,
    POOL_AVGPOOL2D, POOL_MAXPOOL1D, POOL_MAXPOOL2D, PacketHeader, SHIFT_DOWN,
    SHIFT_LEFT, SHIFT_RIGHT, SHIFT_UP, VARIANT_NONE,
};
use crate::registry::LayerRegistry;

fn push_u32(payload: &mut Vec<u8>, value: u32) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn push_option_u32(payload: &mut Vec<u8>, value: Option<u32>) {
    payload.push(u8::from(value.is_some()));
    push_u32(payload, value.unwrap_or(0));
}

fn push_option_f64(payload: &mut Vec<u8>, value: Option<f64>) {
    payload.push(u8::from(value.is_some()));
    payload.extend_from_slice(&value.unwrap_or(0.0).to_le_bytes());
}

fn validate_positive(value: u32, context: &str) -> Result<(), String> {
    if value == 0 {
        return Err(format!("{context}: value must be > 0"));
    }
    Ok(())
}

fn validate_optional_positive(value: Option<u32>, context: &str) -> Result<(), String> {
    if value == Some(0) {
        return Err(format!("{context}: value must be > 0 when provided"));
    }
    Ok(())
}

fn validate_pair_presence(
    first: Option<u32>,
    second: Option<u32>,
    context: &str,
) -> Result<(), String> {
    if first.is_some() != second.is_some() {
        return Err(format!(
            "{context}: both components must be provided together or both omitted"
        ));
    }
    Ok(())
}

fn validate_finite(value: f64, context: &str) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{context}: value must be finite, got {value}"));
    }
    Ok(())
}

fn validate_positive_f64(value: f64, context: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "{context}: value must be finite and > 0, got {value}"
        ));
    }
    Ok(())
}

fn validate_optional_epsilon(value: Option<f64>, context: &str) -> Result<(), String> {
    if let Some(value) = value {
        validate_positive_f64(value, context)?;
    }
    Ok(())
}

#[wasm_bindgen]
pub struct AgentLayerSpec {
    layer_id: u32,
    layer_type: u8,
    variant: u8,
    flags: u8,
    payload: Vec<u8>,
}

impl AgentLayerSpec {
    fn from_payload(
        layer_id: u32,
        layer_type: u8,
        variant: u8,
        flags: u8,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            layer_id,
            layer_type,
            variant,
            flags,
            payload,
        }
    }

    fn id_only(layer_id: u32, layer_type: u8, variant: u8) -> Self {
        Self::from_payload(
            layer_id,
            layer_type,
            variant,
            0,
            layer_id.to_le_bytes().to_vec(),
        )
    }

    fn activation_with_dim(layer_id: u32, variant: u8, dim: u32) -> Self {
        let mut payload = Vec::with_capacity(8);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, dim);
        Self::from_payload(layer_id, LAYER_ACTIVATION, variant, 0, payload)
    }

    fn binary_with_dim(layer_id: u32, variant: u8, dim: u32) -> Self {
        let mut payload = Vec::with_capacity(8);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, dim);
        Self::from_payload(layer_id, LAYER_BINARY, variant, 0, payload)
    }

    fn norm_spec(
        layer_id: u32,
        variant: u8,
        size: u32,
        epsilon: Option<f64>,
        group: Option<(u32, u32)>,
    ) -> Self {
        let mut payload = Vec::with_capacity(if group.is_some() { 25 } else { 17 });
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, size);
        push_option_f64(&mut payload, epsilon);
        if let Some((groups, channels)) = group {
            push_u32(&mut payload, groups);
            push_u32(&mut payload, channels);
        }
        Self::from_payload(layer_id, LAYER_NORM, variant, 0, payload)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_spec(
        layer_id: u32,
        variant: u8,
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Self {
        let mut payload = Vec::with_capacity(40);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, in_channels);
        push_u32(&mut payload, out_channels);
        push_u32(&mut payload, kernel_h);
        push_u32(&mut payload, kernel_w);
        push_option_u32(&mut payload, stride_h);
        push_option_u32(&mut payload, stride_w);
        push_option_u32(&mut payload, padding_h);
        push_option_u32(&mut payload, padding_w);
        Self::from_payload(layer_id, LAYER_CONV, variant, 0, payload)
    }

    fn pool1d_spec(
        layer_id: u32,
        variant: u8,
        kernel: u32,
        stride: Option<u32>,
        padding: Option<u32>,
    ) -> Self {
        let mut payload = Vec::with_capacity(18);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, kernel);
        push_option_u32(&mut payload, stride);
        push_option_u32(&mut payload, padding);
        Self::from_payload(layer_id, LAYER_POOL, variant, 0, payload)
    }

    #[allow(clippy::too_many_arguments)]
    fn pool2d_spec(
        layer_id: u32,
        variant: u8,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Self {
        let mut payload = Vec::with_capacity(32);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, kernel_h);
        push_u32(&mut payload, kernel_w);
        push_option_u32(&mut payload, stride_h);
        push_option_u32(&mut payload, stride_w);
        push_option_u32(&mut payload, padding_h);
        push_option_u32(&mut payload, padding_w);
        Self::from_payload(layer_id, LAYER_POOL, variant, 0, payload)
    }

    fn shift_spec(layer_id: u32, variant: u8, shift_size: u32) -> Self {
        let mut payload = Vec::with_capacity(8);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, shift_size);
        Self::from_payload(layer_id, LAYER_SHIFT, variant, 0, payload)
    }

    pub(crate) fn header(&self) -> PacketHeader {
        PacketHeader {
            opcode: OP_INIT,
            layer_type: self.layer_type,
            variant: self.variant,
            flags: self.flags,
            payload_len: self.payload.len() as u32,
        }
    }
}

#[wasm_bindgen]
impl AgentLayerSpec {
    #[wasm_bindgen(js_name = relu)]
    pub fn relu(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_RELU)
    }

    #[wasm_bindgen(js_name = gelu)]
    pub fn gelu(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_GELU)
    }

    #[wasm_bindgen(js_name = sigmoid)]
    pub fn sigmoid(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_SIGMOID)
    }

    #[wasm_bindgen(js_name = tanh)]
    pub fn tanh(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_TANH)
    }

    #[wasm_bindgen(js_name = hardSwish)]
    pub fn hard_swish(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_HARDSWISH)
    }

    #[wasm_bindgen(js_name = leakyRelu)]
    pub fn leaky_relu(layer_id: u32, negative_slope: f64) -> Result<AgentLayerSpec, String> {
        validate_finite(negative_slope, "AgentLayerSpec.leakyRelu")?;
        if negative_slope < 0.0 {
            return Err(format!(
                "AgentLayerSpec.leakyRelu: negative_slope must be >= 0, got {negative_slope}"
            ));
        }
        let mut payload = Vec::with_capacity(13);
        push_u32(&mut payload, layer_id);
        push_option_f64(&mut payload, Some(negative_slope));
        Ok(Self::from_payload(
            layer_id,
            LAYER_ACTIVATION,
            ACT_LEAKYRELU,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = prelu)]
    pub fn prelu(
        layer_id: u32,
        num_parameters: u32,
        alpha: f64,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(num_parameters, "AgentLayerSpec.prelu.num_parameters")?;
        validate_finite(alpha, "AgentLayerSpec.prelu.alpha")?;
        let mut payload = Vec::with_capacity(18);
        push_u32(&mut payload, layer_id);
        push_option_u32(&mut payload, Some(num_parameters));
        push_option_f64(&mut payload, Some(alpha));
        Ok(Self::from_payload(
            layer_id,
            LAYER_ACTIVATION,
            ACT_PRELU,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = swiGlu)]
    pub fn swi_glu(
        layer_id: u32,
        d_input: u32,
        d_output: u32,
        bias: bool,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(d_input, "AgentLayerSpec.swiGlu.d_input")?;
        validate_positive(d_output, "AgentLayerSpec.swiGlu.d_output")?;
        let mut payload = Vec::with_capacity(17);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, d_input);
        push_u32(&mut payload, d_output);
        push_option_u32(&mut payload, Some(u32::from(bias)));
        Ok(Self::from_payload(
            layer_id,
            LAYER_ACTIVATION,
            ACT_SWIGLU,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = hardSigmoid)]
    pub fn hard_sigmoid(
        layer_id: u32,
        alpha: f64,
        beta: f64,
    ) -> Result<AgentLayerSpec, String> {
        validate_finite(alpha, "AgentLayerSpec.hardSigmoid.alpha")?;
        validate_finite(beta, "AgentLayerSpec.hardSigmoid.beta")?;
        let mut payload = Vec::with_capacity(22);
        push_u32(&mut payload, layer_id);
        push_option_f64(&mut payload, Some(alpha));
        push_option_f64(&mut payload, Some(beta));
        Ok(Self::from_payload(
            layer_id,
            LAYER_ACTIVATION,
            ACT_HARDSIGMOID,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = softplus)]
    pub fn softplus(layer_id: u32, beta: f64) -> Result<AgentLayerSpec, String> {
        validate_positive_f64(beta, "AgentLayerSpec.softplus.beta")?;
        let mut payload = Vec::with_capacity(13);
        push_u32(&mut payload, layer_id);
        push_option_f64(&mut payload, Some(beta));
        Ok(Self::from_payload(
            layer_id,
            LAYER_ACTIVATION,
            ACT_SOFTPLUS,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = mish)]
    pub fn mish(layer_id: u32) -> AgentLayerSpec {
        Self::id_only(layer_id, LAYER_ACTIVATION, ACT_MISH)
    }

    #[wasm_bindgen(js_name = softmax)]
    pub fn softmax(layer_id: u32, dim: u32) -> AgentLayerSpec {
        Self::activation_with_dim(layer_id, ACT_SOFTMAX, dim)
    }

    #[wasm_bindgen(js_name = logSoftmax)]
    pub fn log_softmax(layer_id: u32, dim: u32) -> AgentLayerSpec {
        Self::activation_with_dim(layer_id, ACT_LOGSOFTMAX, dim)
    }

    #[wasm_bindgen(js_name = glu)]
    pub fn glu(layer_id: u32, dim: u32) -> AgentLayerSpec {
        Self::activation_with_dim(layer_id, ACT_GLU, dim)
    }

    #[wasm_bindgen(js_name = linear)]
    pub fn linear(
        layer_id: u32,
        in_dim: u32,
        out_dim: u32,
        bias: bool,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(in_dim, "AgentLayerSpec.linear.in_dim")?;
        validate_positive(out_dim, "AgentLayerSpec.linear.out_dim")?;
        let mut payload = Vec::with_capacity(13);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, in_dim);
        push_u32(&mut payload, out_dim);
        payload.push(u8::from(bias));
        Ok(Self::from_payload(
            layer_id,
            LAYER_LINEAR,
            VARIANT_NONE,
            if bias { FLAG_BIAS } else { 0 },
            payload,
        ))
    }

    #[wasm_bindgen(js_name = batchNorm)]
    pub fn batch_norm(
        layer_id: u32,
        num_features: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(num_features, "AgentLayerSpec.batchNorm.num_features")?;
        validate_optional_epsilon(epsilon, "AgentLayerSpec.batchNorm.epsilon")?;
        Ok(Self::norm_spec(
            layer_id,
            NORM_BATCH,
            num_features,
            epsilon,
            None,
        ))
    }

    #[wasm_bindgen(js_name = groupNorm)]
    pub fn group_norm(
        layer_id: u32,
        num_groups: u32,
        num_channels: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(num_groups, "AgentLayerSpec.groupNorm.num_groups")?;
        validate_positive(num_channels, "AgentLayerSpec.groupNorm.num_channels")?;
        if !num_channels.is_multiple_of(num_groups) {
            return Err(format!(
                "AgentLayerSpec.groupNorm: num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            ));
        }
        validate_optional_epsilon(epsilon, "AgentLayerSpec.groupNorm.epsilon")?;
        Ok(Self::norm_spec(
            layer_id,
            NORM_GROUP,
            num_channels,
            epsilon,
            Some((num_groups, num_channels)),
        ))
    }

    #[wasm_bindgen(js_name = instanceNorm)]
    pub fn instance_norm(
        layer_id: u32,
        num_channels: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(num_channels, "AgentLayerSpec.instanceNorm.num_channels")?;
        validate_optional_epsilon(epsilon, "AgentLayerSpec.instanceNorm.epsilon")?;
        Ok(Self::norm_spec(
            layer_id,
            NORM_INSTANCE,
            num_channels,
            epsilon,
            None,
        ))
    }

    #[wasm_bindgen(js_name = layerNorm)]
    pub fn layer_norm(
        layer_id: u32,
        size: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(size, "AgentLayerSpec.layerNorm.size")?;
        validate_optional_epsilon(epsilon, "AgentLayerSpec.layerNorm.epsilon")?;
        Ok(Self::norm_spec(
            layer_id,
            NORM_LAYER,
            size,
            epsilon,
            None,
        ))
    }

    #[wasm_bindgen(js_name = rmsNorm)]
    pub fn rms_norm(
        layer_id: u32,
        size: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(size, "AgentLayerSpec.rmsNorm.size")?;
        validate_optional_epsilon(epsilon, "AgentLayerSpec.rmsNorm.epsilon")?;
        Ok(Self::norm_spec(
            layer_id,
            NORM_RMS,
            size,
            epsilon,
            None,
        ))
    }

    #[wasm_bindgen(js_name = conv1d)]
    pub fn conv1d(
        layer_id: u32,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: Option<u32>,
        padding: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(in_channels, "AgentLayerSpec.conv1d.in_channels")?;
        validate_positive(out_channels, "AgentLayerSpec.conv1d.out_channels")?;
        validate_positive(kernel_size, "AgentLayerSpec.conv1d.kernel_size")?;
        validate_optional_positive(stride, "AgentLayerSpec.conv1d.stride")?;
        Ok(Self::conv_spec(
            layer_id,
            CONV_CONV1D,
            in_channels,
            out_channels,
            kernel_size,
            1,
            stride,
            None,
            padding,
            None,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = conv2d)]
    pub fn conv2d(
        layer_id: u32,
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(in_channels, "AgentLayerSpec.conv2d.in_channels")?;
        validate_positive(out_channels, "AgentLayerSpec.conv2d.out_channels")?;
        validate_positive(kernel_h, "AgentLayerSpec.conv2d.kernel_h")?;
        validate_positive(kernel_w, "AgentLayerSpec.conv2d.kernel_w")?;
        validate_pair_presence(stride_h, stride_w, "AgentLayerSpec.conv2d.stride")?;
        validate_optional_positive(stride_h, "AgentLayerSpec.conv2d.stride_h")?;
        validate_optional_positive(stride_w, "AgentLayerSpec.conv2d.stride_w")?;
        validate_pair_presence(padding_h, padding_w, "AgentLayerSpec.conv2d.padding")?;
        Ok(Self::conv_spec(
            layer_id,
            CONV_CONV2D,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = convTranspose2d)]
    pub fn conv_transpose2d(
        layer_id: u32,
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(in_channels, "AgentLayerSpec.convTranspose2d.in_channels")?;
        validate_positive(out_channels, "AgentLayerSpec.convTranspose2d.out_channels")?;
        validate_positive(kernel_h, "AgentLayerSpec.convTranspose2d.kernel_h")?;
        validate_positive(kernel_w, "AgentLayerSpec.convTranspose2d.kernel_w")?;
        validate_pair_presence(
            stride_h,
            stride_w,
            "AgentLayerSpec.convTranspose2d.stride",
        )?;
        validate_optional_positive(stride_h, "AgentLayerSpec.convTranspose2d.stride_h")?;
        validate_optional_positive(stride_w, "AgentLayerSpec.convTranspose2d.stride_w")?;
        validate_pair_presence(
            padding_h,
            padding_w,
            "AgentLayerSpec.convTranspose2d.padding",
        )?;
        Ok(Self::conv_spec(
            layer_id,
            CONV_CONVTRANSPOSE2D,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        ))
    }

    #[wasm_bindgen(js_name = embedding)]
    pub fn embedding(
        layer_id: u32,
        vocab_size: u32,
        d_model: u32,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(vocab_size, "AgentLayerSpec.embedding.vocab_size")?;
        validate_positive(d_model, "AgentLayerSpec.embedding.d_model")?;
        let mut payload = Vec::with_capacity(12);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, vocab_size);
        push_u32(&mut payload, d_model);
        Ok(Self::from_payload(
            layer_id,
            LAYER_EMBEDDING,
            VARIANT_NONE,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = maxPool1d)]
    pub fn max_pool1d(
        layer_id: u32,
        kernel: u32,
        stride: Option<u32>,
        padding: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(kernel, "AgentLayerSpec.maxPool1d.kernel")?;
        validate_optional_positive(stride, "AgentLayerSpec.maxPool1d.stride")?;
        Ok(Self::pool1d_spec(
            layer_id,
            POOL_MAXPOOL1D,
            kernel,
            stride,
            padding,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = maxPool2d)]
    pub fn max_pool2d(
        layer_id: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(kernel_h, "AgentLayerSpec.maxPool2d.kernel_h")?;
        validate_positive(kernel_w, "AgentLayerSpec.maxPool2d.kernel_w")?;
        validate_pair_presence(stride_h, stride_w, "AgentLayerSpec.maxPool2d.stride")?;
        validate_optional_positive(stride_h, "AgentLayerSpec.maxPool2d.stride_h")?;
        validate_optional_positive(stride_w, "AgentLayerSpec.maxPool2d.stride_w")?;
        validate_pair_presence(padding_h, padding_w, "AgentLayerSpec.maxPool2d.padding")?;
        Ok(Self::pool2d_spec(
            layer_id,
            POOL_MAXPOOL2D,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        ))
    }

    #[wasm_bindgen(js_name = avgPool1d)]
    pub fn avg_pool1d(
        layer_id: u32,
        kernel: u32,
        stride: Option<u32>,
        padding: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(kernel, "AgentLayerSpec.avgPool1d.kernel")?;
        validate_optional_positive(stride, "AgentLayerSpec.avgPool1d.stride")?;
        Ok(Self::pool1d_spec(
            layer_id,
            POOL_AVGPOOL1D,
            kernel,
            stride,
            padding,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = avgPool2d)]
    pub fn avg_pool2d(
        layer_id: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(kernel_h, "AgentLayerSpec.avgPool2d.kernel_h")?;
        validate_positive(kernel_w, "AgentLayerSpec.avgPool2d.kernel_w")?;
        validate_pair_presence(stride_h, stride_w, "AgentLayerSpec.avgPool2d.stride")?;
        validate_optional_positive(stride_h, "AgentLayerSpec.avgPool2d.stride_h")?;
        validate_optional_positive(stride_w, "AgentLayerSpec.avgPool2d.stride_w")?;
        validate_pair_presence(padding_h, padding_w, "AgentLayerSpec.avgPool2d.padding")?;
        Ok(Self::pool2d_spec(
            layer_id,
            POOL_AVGPOOL2D,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        ))
    }

    #[wasm_bindgen(js_name = adaptiveAvgPool2d)]
    pub fn adaptive_avg_pool2d(
        layer_id: u32,
        output_h: u32,
        output_w: u32,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(output_h, "AgentLayerSpec.adaptiveAvgPool2d.output_h")?;
        validate_positive(output_w, "AgentLayerSpec.adaptiveAvgPool2d.output_w")?;
        let mut payload = Vec::with_capacity(12);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, output_h);
        push_u32(&mut payload, output_w);
        Ok(Self::from_payload(
            layer_id,
            LAYER_POOL,
            POOL_ADAPTIVEAVGPOOL2D,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = shiftUp)]
    pub fn shift_up(layer_id: u32, shift_size: u32) -> AgentLayerSpec {
        Self::shift_spec(layer_id, SHIFT_UP, shift_size)
    }

    #[wasm_bindgen(js_name = shiftDown)]
    pub fn shift_down(layer_id: u32, shift_size: u32) -> AgentLayerSpec {
        Self::shift_spec(layer_id, SHIFT_DOWN, shift_size)
    }

    #[wasm_bindgen(js_name = shiftLeft)]
    pub fn shift_left(layer_id: u32, shift_size: u32) -> AgentLayerSpec {
        Self::shift_spec(layer_id, SHIFT_LEFT, shift_size)
    }

    #[wasm_bindgen(js_name = shiftRight)]
    pub fn shift_right(layer_id: u32, shift_size: u32) -> AgentLayerSpec {
        Self::shift_spec(layer_id, SHIFT_RIGHT, shift_size)
    }

    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = ghost)]
    pub fn ghost(
        layer_id: u32,
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        ratio: u32,
        stride_h: Option<u32>,
        stride_w: Option<u32>,
        padding_h: Option<u32>,
        padding_w: Option<u32>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(in_channels, "AgentLayerSpec.ghost.in_channels")?;
        validate_positive(out_channels, "AgentLayerSpec.ghost.out_channels")?;
        validate_positive(kernel_h, "AgentLayerSpec.ghost.kernel_h")?;
        validate_positive(kernel_w, "AgentLayerSpec.ghost.kernel_w")?;
        validate_positive(ratio, "AgentLayerSpec.ghost.ratio")?;
        if !out_channels.is_multiple_of(ratio) {
            return Err(format!(
                "AgentLayerSpec.ghost: out_channels ({out_channels}) must be divisible by ratio ({ratio})"
            ));
        }
        validate_pair_presence(stride_h, stride_w, "AgentLayerSpec.ghost.stride")?;
        validate_optional_positive(stride_h, "AgentLayerSpec.ghost.stride_h")?;
        validate_optional_positive(stride_w, "AgentLayerSpec.ghost.stride_w")?;
        validate_pair_presence(padding_h, padding_w, "AgentLayerSpec.ghost.padding")?;

        let mut payload = Vec::with_capacity(45);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, in_channels);
        push_u32(&mut payload, out_channels);
        push_u32(&mut payload, kernel_h);
        push_u32(&mut payload, kernel_w);
        push_option_u32(&mut payload, Some(ratio));
        push_option_u32(&mut payload, stride_h);
        push_option_u32(&mut payload, stride_w);
        push_option_u32(&mut payload, padding_h);
        push_option_u32(&mut payload, padding_w);
        Ok(Self::from_payload(
            layer_id,
            LAYER_GHOST,
            VARIANT_NONE,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = seBlock)]
    pub fn se_block(
        layer_id: u32,
        channels: u32,
        reduction: u32,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(channels, "AgentLayerSpec.seBlock.channels")?;
        validate_positive(reduction, "AgentLayerSpec.seBlock.reduction")?;
        if channels < reduction {
            return Err(format!(
                "AgentLayerSpec.seBlock: channels ({channels}) must be >= reduction ({reduction})"
            ));
        }
        let mut payload = Vec::with_capacity(13);
        push_u32(&mut payload, layer_id);
        push_u32(&mut payload, channels);
        push_option_u32(&mut payload, Some(reduction));
        Ok(Self::from_payload(
            layer_id,
            LAYER_SEBLOCK,
            VARIANT_NONE,
            0,
            payload,
        ))
    }

    #[wasm_bindgen(js_name = add)]
    pub fn add(layer_id: u32) -> AgentLayerSpec {
        Self::binary_with_dim(layer_id, BINARY_ADD, 0)
    }

    #[wasm_bindgen(js_name = sub)]
    pub fn sub(layer_id: u32) -> AgentLayerSpec {
        Self::binary_with_dim(layer_id, BINARY_SUB, 0)
    }

    #[wasm_bindgen(js_name = mul)]
    pub fn mul(layer_id: u32) -> AgentLayerSpec {
        Self::binary_with_dim(layer_id, BINARY_MUL, 0)
    }

    #[wasm_bindgen(js_name = matmul)]
    pub fn matmul(layer_id: u32) -> AgentLayerSpec {
        Self::binary_with_dim(layer_id, BINARY_MATMUL, 0)
    }

    #[wasm_bindgen(js_name = concat)]
    pub fn concat(layer_id: u32, dim: u32) -> AgentLayerSpec {
        Self::binary_with_dim(layer_id, BINARY_CONCAT, dim)
    }

    #[wasm_bindgen(js_name = layerId)]
    pub fn layer_id(&self) -> u32 {
        self.layer_id
    }

    #[wasm_bindgen(js_name = layerType)]
    pub fn layer_type(&self) -> u8 {
        self.layer_type
    }

    pub fn variant(&self) -> u8 {
        self.variant
    }
}

#[derive(Clone, Copy)]
struct AgentGraphStep {
    arity: u8,
    layer_type: u8,
    layer_id: u32,
    in_slot: u8,
    in_slot2: u8,
    out_slot: u8,
}

#[wasm_bindgen]
pub struct AgentGraphBuilder {
    num_slots: u32,
    steps: Vec<AgentGraphStep>,
    output_slot: Option<u8>,
}

impl AgentGraphBuilder {
    fn validate_slot(&self, slot: u8, context: &str) -> Result<(), String> {
        if u32::from(slot) >= self.num_slots {
            return Err(format!(
                "{context}: slot {slot} is outside num_slots {}",
                self.num_slots
            ));
        }
        Ok(())
    }

    fn plan_bytes_with_output(&self, output_slot: u8) -> Result<Vec<u8>, String> {
        if self.steps.is_empty() {
            return Err("AgentGraphBuilder.compile: graph has no steps".into());
        }
        let num_steps = u32::try_from(self.steps.len())
            .map_err(|_| "AgentGraphBuilder.compile: too many steps".to_string())?;
        let capacity = self
            .steps
            .len()
            .checked_mul(9)
            .and_then(|n| n.checked_add(9))
            .ok_or_else(|| "AgentGraphBuilder.compile: plan size overflow".to_string())?;
        let mut plan = Vec::with_capacity(capacity);
        push_u32(&mut plan, num_steps);
        push_u32(&mut plan, self.num_slots);
        for step in &self.steps {
            plan.push(step.arity);
            plan.push(step.layer_type);
            push_u32(&mut plan, step.layer_id);
            plan.push(step.in_slot);
            plan.push(step.in_slot2);
            plan.push(step.out_slot);
        }
        plan.push(output_slot);
        Ok(plan)
    }

    fn plan_bytes(&self) -> Result<Vec<u8>, String> {
        let output_slot = self
            .output_slot
            .ok_or_else(|| "AgentGraphBuilder.compile: output slot is not set".to_string())?;
        self.plan_bytes_with_output(output_slot)
    }
}

#[wasm_bindgen]
impl AgentGraphBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_slots: u32) -> Result<AgentGraphBuilder, String> {
        if !(1..=64).contains(&num_slots) {
            return Err(format!(
                "AgentGraphBuilder: num_slots must be 1..=64, got {num_slots}"
            ));
        }
        Ok(Self {
            num_slots,
            steps: Vec::new(),
            output_slot: None,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(
        &mut self,
        spec: &AgentLayerSpec,
        input_slot: u8,
        output_slot: u8,
    ) -> Result<(), String> {
        if spec.layer_type == LAYER_BINARY {
            return Err("AgentGraphBuilder.addUnary: binary spec requires addBinary".into());
        }
        self.validate_slot(input_slot, "AgentGraphBuilder.addUnary")?;
        self.validate_slot(output_slot, "AgentGraphBuilder.addUnary")?;
        self.steps.push(AgentGraphStep {
            arity: 1,
            layer_type: spec.layer_type,
            layer_id: spec.layer_id,
            in_slot: input_slot,
            in_slot2: input_slot,
            out_slot: output_slot,
        });
        Ok(())
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(
        &mut self,
        spec: &AgentLayerSpec,
        left_slot: u8,
        right_slot: u8,
        output_slot: u8,
    ) -> Result<(), String> {
        if spec.layer_type != LAYER_BINARY {
            return Err("AgentGraphBuilder.addBinary: spec is not binary".into());
        }
        self.validate_slot(left_slot, "AgentGraphBuilder.addBinary")?;
        self.validate_slot(right_slot, "AgentGraphBuilder.addBinary")?;
        self.validate_slot(output_slot, "AgentGraphBuilder.addBinary")?;
        self.steps.push(AgentGraphStep {
            arity: 2,
            layer_type: LAYER_BINARY,
            layer_id: spec.layer_id,
            in_slot: left_slot,
            in_slot2: right_slot,
            out_slot: output_slot,
        });
        Ok(())
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, output_slot: u8) -> Result<(), String> {
        self.validate_slot(output_slot, "AgentGraphBuilder.setOutput")?;
        self.output_slot = Some(output_slot);
        Ok(())
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> u32 {
        self.steps.len() as u32
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    pub fn compile(&self, registry: &LayerRegistry) -> Result<CompiledGraph, String> {
        registry.compile_graph(&self.plan_bytes()?)
    }

    /// Compile using a temporary output selection without mutating the builder's configured output.
    /// This is the canonical path for stateless workspace orchestration.
    #[wasm_bindgen(js_name = compileWithOutput)]
    pub fn compile_with_output(
        &self,
        registry: &LayerRegistry,
        output_slot: u8,
    ) -> Result<CompiledGraph, String> {
        self.validate_slot(output_slot, "AgentGraphBuilder.compileWithOutput")?;
        registry.compile_graph(&self.plan_bytes_with_output(output_slot)?)
    }
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = initAgentLayer)]
    pub fn init_agent_layer(&mut self, spec: &AgentLayerSpec) -> Result<(), String> {
        let header = spec.header();
        self.init_layer(&header, &spec.payload)
    }
}

pub(crate) fn capability_manifest() -> String {
    format!(
        concat!(
            "{{\"schema_version\":1,",
            "\"engine\":\"burn-research\",",
            "\"purpose\":\"agent_math_coprocessor\",",
            "\"tensor\":{{\"dtype\":\"f32\",\"rank_max\":4,\"owned\":\"WasmTensor\",\"shared\":\"TensorView\"}},",
            "\"proof\":{{\"vector\":\"mathVerifyVectors\",\"graph_output\":\"CompiledGraph.verifyFlat\"}},",
            "\"graph\":{{\"registry\":\"LayerRegistry\",\"compile\":\"LayerRegistry.compileGraph\",\"run\":\"CompiledGraph.run\",\"max_slots\":64}},",
            "\"agent_facade\":{{\"layer_spec\":\"AgentLayerSpec\",\"registry_init\":\"LayerRegistry.initAgentLayer\",",
            "\"constructors\":[\"relu\",\"gelu\",\"sigmoid\",\"tanh\",\"hardSwish\",\"leakyRelu\",\"prelu\",\"swiGlu\",\"hardSigmoid\",\"softplus\",\"mish\",\"softmax\",\"logSoftmax\",\"glu\",\"linear\",\"batchNorm\",\"groupNorm\",\"instanceNorm\",\"layerNorm\",\"rmsNorm\",\"conv1d\",\"conv2d\",\"convTranspose2d\",\"embedding\",\"maxPool1d\",\"maxPool2d\",\"avgPool1d\",\"avgPool2d\",\"adaptiveAvgPool2d\",\"shiftUp\",\"shiftDown\",\"shiftLeft\",\"shiftRight\",\"ghost\",\"seBlock\",\"add\",\"sub\",\"mul\",\"matmul\",\"concat\"],",
            "\"constructor_signatures\":{{",
            "\"linear\":\"linear(id,in_dim,out_dim,bias)\",",
            "\"batchNorm\":\"batchNorm(id,num_features,epsilon?)\",",
            "\"groupNorm\":\"groupNorm(id,num_groups,num_channels,epsilon?)\",",
            "\"conv1d\":\"conv1d(id,in_ch,out_ch,kernel,stride?,padding?)\",",
            "\"conv2d\":\"conv2d(id,in_ch,out_ch,kh,kw,sh?,sw?,ph?,pw?)\",",
            "\"embedding\":\"embedding(id,vocab_size,d_model)\",",
            "\"maxPool2d\":\"maxPool2d(id,kh,kw,sh?,sw?,ph?,pw?)\",",
            "\"ghost\":\"ghost(id,in_ch,out_ch,kh,kw,ratio,sh?,sw?,ph?,pw?)\",",
            "\"seBlock\":\"seBlock(id,channels,reduction)\",",
            "\"swiGlu\":\"swiGlu(id,d_input,d_output,bias)\"}},",
            "\"graph_builder\":\"AgentGraphBuilder\",\"graph_methods\":[\"addUnary\",\"addBinary\",\"setOutput\",\"compile\",\"compileWithOutput\"]}},",
            "\"optimizer\":{{\"entry\":\"EsOptimizer\",\"strategies\":{{\"openes\":0,\"mu_lambda\":1}},\"lifecycle\":\"ask->tell\"}},",
            "\"recommended_flow\":[\"discover\",\"construct_reference\",\"run_external_candidate\",\"verify\",\"revise_or_accept\"],",
            "\"layers\":{{",
            "\"linear\":{{\"code\":{},\"arity\":1}},",
            "\"norm\":{{\"code\":{},\"arity\":1,\"variants\":{{\"batch\":{},\"group\":{},\"instance\":{},\"layer\":{},\"rms\":{}}}}},",
            "\"conv\":{{\"code\":{},\"arity\":1,\"variants\":{{\"conv1d\":{},\"conv2d\":{},\"conv_transpose2d\":{}}}}},",
            "\"activation\":{{\"code\":{},\"arity\":1,\"variants\":{{\"gelu\":{},\"relu\":{},\"sigmoid\":{},\"tanh\":{},\"hard_swish\":{},\"leaky_relu\":{},\"prelu\":{},\"swiglu\":{},\"hard_sigmoid\":{},\"softplus\":{},\"mish\":{},\"softmax\":{},\"log_softmax\":{},\"glu\":{}}}}},",
            "\"embedding\":{{\"code\":{},\"arity\":1}},",
            "\"pool\":{{\"code\":{},\"arity\":1,\"variants\":{{\"max_pool1d\":{},\"max_pool2d\":{},\"avg_pool1d\":{},\"avg_pool2d\":{},\"adaptive_avg_pool2d\":{}}}}},",
            "\"shift\":{{\"code\":{},\"arity\":1,\"variants\":{{\"up\":{},\"down\":{},\"left\":{},\"right\":{}}}}},",
            "\"ghost\":{{\"code\":{},\"arity\":1}},",
            "\"seblock\":{{\"code\":{},\"arity\":1}},",
            "\"binary\":{{\"code\":{},\"arity\":2,\"variants\":{{\"add\":{},\"sub\":{},\"mul\":{},\"matmul\":{},\"concat\":{}}}}}",
            "}}}}"
        ),
        LAYER_LINEAR,
        LAYER_NORM,
        NORM_BATCH,
        NORM_GROUP,
        NORM_INSTANCE,
        NORM_LAYER,
        NORM_RMS,
        LAYER_CONV,
        CONV_CONV1D,
        CONV_CONV2D,
        CONV_CONVTRANSPOSE2D,
        LAYER_ACTIVATION,
        ACT_GELU,
        ACT_RELU,
        ACT_SIGMOID,
        ACT_TANH,
        ACT_HARDSWISH,
        ACT_LEAKYRELU,
        ACT_PRELU,
        ACT_SWIGLU,
        ACT_HARDSIGMOID,
        ACT_SOFTPLUS,
        ACT_MISH,
        ACT_SOFTMAX,
        ACT_LOGSOFTMAX,
        ACT_GLU,
        LAYER_EMBEDDING,
        LAYER_POOL,
        POOL_MAXPOOL1D,
        POOL_MAXPOOL2D,
        POOL_AVGPOOL1D,
        POOL_AVGPOOL2D,
        POOL_ADAPTIVEAVGPOOL2D,
        LAYER_SHIFT,
        SHIFT_UP,
        SHIFT_DOWN,
        SHIFT_LEFT,
        SHIFT_RIGHT,
        LAYER_GHOST,
        LAYER_SEBLOCK,
        LAYER_BINARY,
        BINARY_ADD,
        BINARY_SUB,
        BINARY_MUL,
        BINARY_MATMUL,
        BINARY_CONCAT,
    )
}

/// Return a compact, machine-readable description of the stable WASM capabilities.
///
/// Agents should call this once before planning numerical work instead of inferring
/// features from generated JS glue or repeatedly probing exports.
#[wasm_bindgen(js_name = agentCapabilities)]
pub fn agent_capabilities() -> String {
    capability_manifest()
}

#[cfg(test)]
mod tests {
    use super::{AgentGraphBuilder, AgentLayerSpec, capability_manifest};
    use crate::protocol::{
        LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST,
        LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT,
    };
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    #[test]
    fn manifest_exposes_agent_workflow_and_core_entrypoints() {
        let manifest = capability_manifest();
        assert!(manifest.contains("\"schema_version\":1"));
        assert!(manifest.contains("\"purpose\":\"agent_math_coprocessor\""));
        assert!(manifest.contains("\"vector\":\"mathVerifyVectors\""));
        assert!(manifest.contains("\"graph_output\":\"CompiledGraph.verifyFlat\""));
        assert!(manifest.contains("\"compile\":\"LayerRegistry.compileGraph\""));
        assert!(manifest.contains("\"registry_init\":\"LayerRegistry.initAgentLayer\""));
        assert!(manifest.contains("\"graph_builder\":\"AgentGraphBuilder\""));
        assert!(manifest.contains("\"constructors\":[\"relu\""));
        assert!(manifest.contains("\"constructor_signatures\":{\"linear\""));
        assert!(manifest.contains("\"ghost\":\"ghost(id,in_ch,out_ch"));
        assert!(manifest.contains("\"lifecycle\":\"ask->tell\""));
        assert!(manifest.contains("compileWithOutput"));
    }

    #[test]
    fn manifest_layer_codes_follow_protocol_constants() {
        let manifest = capability_manifest();
        assert!(manifest.contains(&format!("\"norm\":{{\"code\":{}", LAYER_NORM)));
        assert!(manifest.contains(&format!("\"conv\":{{\"code\":{}", LAYER_CONV)));
        assert!(manifest.contains(&format!(
            "\"activation\":{{\"code\":{}",
            LAYER_ACTIVATION
        )));
        assert!(manifest.contains(&format!("\"binary\":{{\"code\":{}", LAYER_BINARY)));
    }

    #[test]
    fn typed_relu_spec_initializes_through_existing_registry_boundary() {
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(7);
        registry.init_agent_layer(&spec).unwrap();
        let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
        let output = registry
            .forward_layer(7, LAYER_ACTIVATION, &input)
            .unwrap();
        assert_eq!(output.to_array(), vec![0.0, 2.0]);
    }

    #[test]
    fn typed_linear_spec_rejects_zero_dimensions_before_backend_init() {
        assert!(AgentLayerSpec::linear(1, 0, 2, true).is_err());
        assert!(AgentLayerSpec::linear(1, 2, 0, true).is_err());

        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::linear(2, 3, 2, true).unwrap();
        registry.init_agent_layer(&spec).unwrap();
        assert_eq!(registry.total_params(), 8);
    }

    #[test]
    fn typed_binary_spec_initializes_without_raw_payload_bytes() {
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::add(9);
        registry.init_agent_layer(&spec).unwrap();
        let a = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let b = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        let output = registry.forward_binary_layer(9, &a, &b).unwrap();
        assert_eq!(output.to_array(), vec![4.0, 6.0]);
    }

    #[test]
    fn extended_typed_specs_initialize_all_layer_families() {
        let mut registry = LayerRegistry::new();
        let specs = vec![
            AgentLayerSpec::hard_swish(20),
            AgentLayerSpec::leaky_relu(21, 0.01).unwrap(),
            AgentLayerSpec::prelu(22, 1, 0.25).unwrap(),
            AgentLayerSpec::swi_glu(23, 2, 2, true).unwrap(),
            AgentLayerSpec::hard_sigmoid(24, 0.2, 0.5).unwrap(),
            AgentLayerSpec::softplus(25, 1.0).unwrap(),
            AgentLayerSpec::batch_norm(30, 2, Some(1e-5)).unwrap(),
            AgentLayerSpec::group_norm(31, 1, 2, Some(1e-5)).unwrap(),
            AgentLayerSpec::instance_norm(32, 2, Some(1e-5)).unwrap(),
            AgentLayerSpec::layer_norm(33, 2, Some(1e-5)).unwrap(),
            AgentLayerSpec::rms_norm(34, 2, Some(1e-5)).unwrap(),
            AgentLayerSpec::conv1d(40, 1, 1, 1, None, None).unwrap(),
            AgentLayerSpec::conv2d(41, 1, 1, 1, 1, None, None, None, None).unwrap(),
            AgentLayerSpec::conv_transpose2d(42, 1, 1, 1, 1, None, None, None, None)
                .unwrap(),
            AgentLayerSpec::embedding(50, 4, 2).unwrap(),
            AgentLayerSpec::max_pool1d(60, 1, None, None).unwrap(),
            AgentLayerSpec::max_pool2d(61, 1, 1, None, None, None, None).unwrap(),
            AgentLayerSpec::avg_pool1d(62, 1, None, None).unwrap(),
            AgentLayerSpec::avg_pool2d(63, 1, 1, None, None, None, None).unwrap(),
            AgentLayerSpec::adaptive_avg_pool2d(64, 1, 1).unwrap(),
            AgentLayerSpec::shift_up(70, 1),
            AgentLayerSpec::shift_down(71, 1),
            AgentLayerSpec::shift_left(72, 1),
            AgentLayerSpec::shift_right(73, 1),
            AgentLayerSpec::ghost(80, 1, 2, 1, 1, 2, None, None, None, None).unwrap(),
            AgentLayerSpec::se_block(81, 2, 1).unwrap(),
        ];

        for spec in &specs {
            registry.init_agent_layer(spec).unwrap();
            assert!(registry.layer_exists(spec.layer_type(), spec.layer_id()));
        }

        assert!(registry.layer_exists(LAYER_NORM, 30));
        assert!(registry.layer_exists(LAYER_CONV, 41));
        assert!(registry.layer_exists(LAYER_EMBEDDING, 50));
        assert!(registry.layer_exists(LAYER_POOL, 61));
        assert!(registry.layer_exists(LAYER_SHIFT, 70));
        assert!(registry.layer_exists(LAYER_GHOST, 80));
        assert!(registry.layer_exists(LAYER_SEBLOCK, 81));
    }

    #[test]
    fn extended_facade_rejects_invalid_configs_before_registry_init() {
        assert!(AgentLayerSpec::prelu(1, 0, 0.25).is_err());
        assert!(AgentLayerSpec::swi_glu(1, 0, 2, true).is_err());
        assert!(AgentLayerSpec::softplus(1, 0.0).is_err());
        assert!(AgentLayerSpec::batch_norm(1, 0, None).is_err());
        assert!(AgentLayerSpec::group_norm(1, 0, 4, None).is_err());
        assert!(AgentLayerSpec::group_norm(1, 3, 4, None).is_err());
        assert!(AgentLayerSpec::rms_norm(1, 4, Some(f64::NAN)).is_err());
        assert!(AgentLayerSpec::conv1d(1, 0, 1, 3, None, None).is_err());
        assert!(
            AgentLayerSpec::conv2d(1, 1, 1, 3, 3, Some(1), None, None, None).is_err()
        );
        assert!(AgentLayerSpec::embedding(1, 0, 4).is_err());
        assert!(AgentLayerSpec::max_pool1d(1, 0, None, None).is_err());
        assert!(AgentLayerSpec::adaptive_avg_pool2d(1, 0, 1).is_err());
        assert!(AgentLayerSpec::ghost(1, 1, 4, 3, 3, 0, None, None, None, None).is_err());
        assert!(AgentLayerSpec::ghost(1, 1, 5, 3, 3, 2, None, None, None, None).is_err());
        assert!(AgentLayerSpec::se_block(1, 4, 8).is_err());
    }

    #[test]
    fn graph_builder_compiles_and_runs_without_manual_plan_encoding() {
        let mut registry = LayerRegistry::new();
        let relu = AgentLayerSpec::relu(11);
        registry.init_agent_layer(&relu).unwrap();

        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&relu, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph = builder.compile(&registry).unwrap();

        let input = WasmTensor::new(&[-2.0, 5.0], &[1, 2, 1, 1]);
        let output = graph.run(&registry, &input).unwrap();
        assert_eq!(output.to_array(), vec![0.0, 5.0]);
    }

    #[test]
    fn compile_with_output_does_not_mutate_existing_output_selection() {
        let mut registry = LayerRegistry::new();
        let relu = AgentLayerSpec::relu(12);
        registry.init_agent_layer(&relu).unwrap();

        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_unary(&relu, 0, 1).unwrap();
        builder.add_unary(&relu, 1, 2).unwrap();
        builder.set_output(1).unwrap();

        let alternate = builder.compile_with_output(&registry, 2).unwrap();
        assert_eq!(alternate.output_slot(), 2);
        let configured = builder.compile(&registry).unwrap();
        assert_eq!(configured.output_slot(), 1);
    }

    #[test]
    fn graph_builder_rejects_arity_mismatch_before_compile() {
        let relu = AgentLayerSpec::relu(1);
        let add = AgentLayerSpec::add(2);
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        assert!(builder.add_binary(&relu, 0, 0, 1).is_err());
        assert!(builder.add_unary(&add, 0, 1).is_err());
    }
}
