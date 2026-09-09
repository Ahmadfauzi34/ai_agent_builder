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
        if in_dim == 0 || out_dim == 0 {
            return Err(format!(
                "AgentLayerSpec.linear: in_dim and out_dim must be > 0, got {in_dim} and {out_dim}"
            ));
        }
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

    fn plan_bytes(&self) -> Result<Vec<u8>, String> {
        if self.steps.is_empty() {
            return Err("AgentGraphBuilder.compile: graph has no steps".into());
        }
        let output_slot = self
            .output_slot
            .ok_or_else(|| "AgentGraphBuilder.compile: output slot is not set".to_string())?;
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
            "\"agent_facade\":{{\"layer_spec\":\"AgentLayerSpec\",\"registry_init\":\"LayerRegistry.initAgentLayer\",\"constructors\":[\"relu\",\"gelu\",\"sigmoid\",\"tanh\",\"mish\",\"softmax\",\"logSoftmax\",\"glu\",\"linear\",\"add\",\"sub\",\"mul\",\"matmul\",\"concat\"],\"graph_builder\":\"AgentGraphBuilder\",\"graph_methods\":[\"addUnary\",\"addBinary\",\"setOutput\",\"compile\"]}},",
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
    use crate::protocol::{LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_NORM};
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
        assert!(manifest.contains("\"lifecycle\":\"ask->tell\""));
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
    fn graph_builder_rejects_arity_mismatch_before_compile() {
        let relu = AgentLayerSpec::relu(1);
        let add = AgentLayerSpec::add(2);
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        assert!(builder.add_binary(&relu, 0, 0, 1).is_err());
        assert!(builder.add_unary(&add, 0, 1).is_err());
    }
}
