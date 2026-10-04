//! Fasad WASM tunggal — domain `agent` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::agent::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::capability_manifest;

// Opsi C Fase 2: imports untuk #[wasm_bindgen] impl AgentGraphBuilder yang pindah ke sini.
use crate::agent::{
    push_option_f64, push_option_u32, push_u32, validate_finite, validate_optional_epsilon,
    validate_optional_positive, validate_pair_presence, validate_positive, validate_positive_f64,
    AgentGraphBuilder, AgentGraphStep, AgentLayerSpec,
};
use crate::graph::CompiledGraph;
use crate::layers::custom::feature_norm::DEFAULT_EPSILON as FEATURE_NORM_DEFAULT_EPSILON;
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX, ACT_MISH,
    ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU, ACT_TANH, BINARY_ADD,
    BINARY_CONCAT, BINARY_MATMUL, BINARY_MUL, BINARY_SUB, CONV_CONV1D, CONV_CONV2D,
    CONV_CONVTRANSPOSE2D, FLAG_BIAS, LAYER_ACTIVATION, LAYER_BINARY, LAYER_EMBEDDING,
    LAYER_FEATURE_NORM, LAYER_GHOST, LAYER_LINEAR, LAYER_POOL, LAYER_SEBLOCK, NORM_BATCH,
    NORM_GROUP, NORM_INSTANCE, NORM_LAYER, NORM_RMS, POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D,
    POOL_AVGPOOL2D, POOL_MAXPOOL1D, POOL_MAXPOOL2D, SHIFT_DOWN, SHIFT_LEFT, SHIFT_RIGHT, SHIFT_UP,
    VARIANT_NONE,
};
use crate::registry::LayerRegistry;

/// Return a compact, machine-readable description of the stable WASM capabilities.
///
/// Agents should call this once before planning numerical work instead of inferring
/// features from generated JS glue or repeatedly probing exports.
#[wasm_bindgen(js_name = agentCapabilities)]
pub fn agent_capabilities() -> String {
    capability_manifest()
}

// ============================================================
// Opsi C Fase 2 — pindahan murni dari `src/agent.rs`:
// #[wasm_bindgen] impl AgentGraphBuilder (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

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
            semantic_edge_bindings: Vec::new(),
            semantic_lifecycle_transitions: Vec::new(),
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

    #[wasm_bindgen(js_name = multiInputPlanV1)]
    pub fn multi_input_plan_v1(&self) -> Result<MultiInputGraphPlan, String> {
        MultiInputGraphPlan::new(self)
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

// ============================================================
// Opsi C Fase 2 repair — pindahan murni dari `src/agent.rs`:
// #[wasm_bindgen] impl AgentLayerSpec (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

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
    pub fn prelu(layer_id: u32, num_parameters: u32, alpha: f64) -> Result<AgentLayerSpec, String> {
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
    pub fn hard_sigmoid(layer_id: u32, alpha: f64, beta: f64) -> Result<AgentLayerSpec, String> {
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
        Ok(Self::norm_spec(layer_id, NORM_LAYER, size, epsilon, None))
    }

    #[wasm_bindgen(js_name = rmsNorm)]
    pub fn rms_norm(
        layer_id: u32,
        size: u32,
        epsilon: Option<f64>,
    ) -> Result<AgentLayerSpec, String> {
        validate_positive(size, "AgentLayerSpec.rmsNorm.size")?;
        validate_optional_epsilon(epsilon, "AgentLayerSpec.rmsNorm.epsilon")?;
        Ok(Self::norm_spec(layer_id, NORM_RMS, size, epsilon, None))
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
        validate_pair_presence(stride_h, stride_w, "AgentLayerSpec.convTranspose2d.stride")?;
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

    #[wasm_bindgen(js_name = featureNorm)]
    pub fn feature_norm(layer_id: u32, epsilon: Option<f64>) -> Result<AgentLayerSpec, String> {
        let epsilon = epsilon.unwrap_or(FEATURE_NORM_DEFAULT_EPSILON);
        validate_positive_f64(epsilon, "AgentLayerSpec.featureNorm.epsilon")?;
        let mut payload = Vec::with_capacity(13);
        push_u32(&mut payload, layer_id);
        push_option_f64(&mut payload, Some(epsilon));
        Ok(Self::from_payload(
            layer_id,
            LAYER_FEATURE_NORM,
            VARIANT_NONE,
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
