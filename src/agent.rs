use wasm_bindgen::prelude::*;

use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX,
    ACT_MISH, ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU,
    ACT_TANH, BINARY_ADD, BINARY_CONCAT, BINARY_MATMUL, BINARY_MUL, BINARY_SUB,
    CONV_CONV1D, CONV_CONV2D, CONV_CONVTRANSPOSE2D, LAYER_ACTIVATION, LAYER_BINARY,
    LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL,
    LAYER_SEBLOCK, LAYER_SHIFT, NORM_BATCH, NORM_GROUP, NORM_INSTANCE, NORM_LAYER,
    NORM_RMS, POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D, POOL_AVGPOOL2D, POOL_MAXPOOL1D,
    POOL_MAXPOOL2D, SHIFT_DOWN, SHIFT_LEFT, SHIFT_RIGHT, SHIFT_UP,
};

pub(crate) fn capability_manifest() -> String {
    format!(
        concat!(
            "{{\"schema_version\":1,",
            "\"engine\":\"burn-research\",",
            "\"purpose\":\"agent_math_coprocessor\",",
            "\"tensor\":{{\"dtype\":\"f32\",\"rank_max\":4,\"owned\":\"WasmTensor\",\"shared\":\"TensorView\"}},",
            "\"proof\":{{\"vector\":\"mathVerifyVectors\",\"graph_output\":\"CompiledGraph.verifyFlat\"}},",
            "\"graph\":{{\"registry\":\"LayerRegistry\",\"compile\":\"LayerRegistry.compileGraph\",\"run\":\"CompiledGraph.run\",\"max_slots\":64}},",
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
    use super::capability_manifest;
    use crate::protocol::{LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_NORM};

    #[test]
    fn manifest_exposes_agent_workflow_and_core_entrypoints() {
        let manifest = capability_manifest();
        assert!(manifest.contains("\"schema_version\":1"));
        assert!(manifest.contains("\"purpose\":\"agent_math_coprocessor\""));
        assert!(manifest.contains("\"vector\":\"mathVerifyVectors\""));
        assert!(manifest.contains("\"graph_output\":\"CompiledGraph.verifyFlat\""));
        assert!(manifest.contains("\"compile\":\"LayerRegistry.compileGraph\""));
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
}
