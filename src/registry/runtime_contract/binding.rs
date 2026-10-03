pub use crate::facade::registry::layer_registry_operation_binding_capabilities;
use wasm_bindgen::prelude::*;

use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX, ACT_MISH,
    ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU, ACT_TANH, BINARY_ADD,
    BINARY_CONCAT, BINARY_MATMUL, BINARY_MUL, BINARY_SUB, CONV_CONV1D, CONV_CONV2D,
    CONV_CONVTRANSPOSE2D, LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING,
    LAYER_FEATURE_NORM, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK,
    LAYER_SHIFT, NORM_BATCH, NORM_GROUP, NORM_INSTANCE, NORM_LAYER, NORM_RMS,
    POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D, POOL_AVGPOOL2D, POOL_MAXPOOL1D, POOL_MAXPOOL2D,
    SHIFT_DOWN, SHIFT_LEFT, SHIFT_RIGHT, SHIFT_UP,
};

use super::super::LayerRegistry;
use super::inventory::{inventory_fingerprint_of, live_instance_records, sha256_text};

pub(crate) const BINDING_CAPABILITIES_V1: &str =
    include_str!("../../../docs/contracts/layer-registry-operation-binding.v1.json");
const BINDING_SCHEMA: &str = "burn-research.layer-registry-operation-binding-snapshot.v1";

/// Infer the host operation-ID for a canonical `(layer_type, variant)` protocol
/// identity.
///
/// Variant-discriminated layer types require an exact variant match and fail
/// closed otherwise. Variant-insensitive layer types (linear, embedding, ghost,
/// seblock, feature norm) collapse to a single host operation-ID, matching
/// [`LayerRegistry`] init acceptance, which does not discriminate on variant
/// for those types.
///
/// This is a pure projection: it performs no ranking, no selection, no
/// initialize-vs-reuse decision, no execution, and no mutation.
fn infer_operation_id(layer_type: u8, variant: u8) -> Result<&'static str, String> {
    let operation = match layer_type {
        LAYER_LINEAR => "linear",
        LAYER_EMBEDDING => "embedding",
        LAYER_GHOST => "ghost",
        LAYER_SEBLOCK => "seBlock",
        LAYER_FEATURE_NORM => "featureNorm",
        LAYER_NORM => match variant {
            NORM_BATCH => "batchNorm",
            NORM_GROUP => "groupNorm",
            NORM_INSTANCE => "instanceNorm",
            NORM_LAYER => "layerNorm",
            NORM_RMS => "rmsNorm",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown norm variant 0x{variant:02X}"
                ))
            }
        },
        LAYER_CONV => match variant {
            CONV_CONV1D => "conv1d",
            CONV_CONV2D => "conv2d",
            CONV_CONVTRANSPOSE2D => "convTranspose2d",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown conv variant 0x{variant:02X}"
                ))
            }
        },
        LAYER_ACTIVATION => match variant {
            ACT_GELU => "gelu",
            ACT_RELU => "relu",
            ACT_SIGMOID => "sigmoid",
            ACT_TANH => "tanh",
            ACT_HARDSWISH => "hardSwish",
            ACT_LEAKYRELU => "leakyRelu",
            ACT_PRELU => "prelu",
            ACT_SWIGLU => "swiGlu",
            ACT_HARDSIGMOID => "hardSigmoid",
            ACT_SOFTPLUS => "softplus",
            ACT_MISH => "mish",
            ACT_SOFTMAX => "softmax",
            ACT_LOGSOFTMAX => "logSoftmax",
            ACT_GLU => "glu",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown activation variant 0x{variant:02X}"
                ))
            }
        },
        LAYER_POOL => match variant {
            POOL_MAXPOOL1D => "maxPool1d",
            POOL_MAXPOOL2D => "maxPool2d",
            POOL_AVGPOOL1D => "avgPool1d",
            POOL_AVGPOOL2D => "avgPool2d",
            POOL_ADAPTIVEAVGPOOL2D => "adaptiveAvgPool2d",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown pool variant 0x{variant:02X}"
                ))
            }
        },
        LAYER_SHIFT => match variant {
            SHIFT_UP => "shiftUp",
            SHIFT_DOWN => "shiftDown",
            SHIFT_LEFT => "shiftLeft",
            SHIFT_RIGHT => "shiftRight",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown shift variant 0x{variant:02X}"
                ))
            }
        },
        LAYER_BINARY => match variant {
            BINARY_ADD => "add",
            BINARY_SUB => "sub",
            BINARY_MUL => "mul",
            BINARY_CONCAT => "concat",
            BINARY_MATMUL => "matmul",
            _ => {
                return Err(format!(
                    "operationBindingSnapshot: unknown binary variant 0x{variant:02X}"
                ))
            }
        },
        _ => {
            return Err(format!(
                "operationBindingSnapshot: unknown live layer type 0x{layer_type:02X}"
            ))
        }
    };
    Ok(operation)
}

fn operation_binding_snapshot(registry: &LayerRegistry) -> Result<String, String> {
    let (records, summed_params) = live_instance_records(registry)?;
    let inventory_fingerprint = inventory_fingerprint_of(&records, summed_params);

    let mut canonical_bindings = Vec::with_capacity(records.len());
    let mut json_bindings = Vec::with_capacity(records.len());

    for record in &records {
        let operation_id = infer_operation_id(record.layer_type, record.variant)?;
        canonical_bindings.push(format!(
            "type={:02x}|id={}|variant={:02x}|operation={operation_id}",
            record.layer_type, record.layer_id, record.variant
        ));
        json_bindings.push(format!(
            "{{\"layer_type\":{},\"layer_id\":{},\"variant\":{},\"operation_id\":\"{operation_id}\",\"init_fingerprint\":\"{}\",\"parameter_count\":{}}}",
            record.layer_type,
            record.layer_id,
            record.variant,
            record.init_fingerprint,
            record.parameter_count
        ));
    }

    let canonical = format!(
        "schema={BINDING_SCHEMA}|inventory={inventory_fingerprint}|bindings={}|{}",
        canonical_bindings.len(),
        canonical_bindings.join("||")
    );
    let binding_fingerprint = sha256_text(&canonical);

    Ok(format!(
        "{{\"schema\":\"{BINDING_SCHEMA}\",\"instance_count\":{},\"inventory_fingerprint\":\"{inventory_fingerprint}\",\"binding_fingerprint\":\"{binding_fingerprint}\",\"bindings\":[{}],\"execution_authorized\":false,\"mutation\":\"none\"}}",
        json_bindings.len(),
        json_bindings.join(",")
    ))
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = operationBindingSnapshot)]
    pub fn operation_binding_snapshot(&self) -> Result<String, String> {
        operation_binding_snapshot(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{infer_operation_id, operation_binding_snapshot, BINDING_SCHEMA};
    use crate::agent::AgentLayerSpec;
    use crate::protocol::{LAYER_ACTIVATION, LAYER_CONV, LAYER_LINEAR};
    use crate::registry::LayerRegistry;

    #[test]
    fn empty_binding_is_deterministic_and_nonexecuting() {
        let registry = LayerRegistry::new();
        let snapshot = operation_binding_snapshot(&registry).unwrap();
        assert!(snapshot.contains(&format!("\"schema\":\"{BINDING_SCHEMA}\"")));
        assert!(snapshot.contains("\"instance_count\":0"));
        assert!(snapshot.contains("\"bindings\":[]"));
        assert!(snapshot.contains("\"execution_authorized\":false"));
        assert!(snapshot.contains("\"mutation\":\"none\""));
        assert_eq!(
            snapshot,
            operation_binding_snapshot(&registry).unwrap(),
            "empty binding snapshot must be deterministic"
        );
    }

    #[test]
    fn live_instances_bind_to_exact_host_operation_ids() {
        let mut registry = LayerRegistry::new();
        registry
            .init_agent_layer(&AgentLayerSpec::linear(7, 4, 3, true).unwrap())
            .unwrap();
        registry.init_agent_layer(&AgentLayerSpec::relu(9)).unwrap();
        registry
            .init_agent_layer(
                &AgentLayerSpec::conv2d(11, 2, 3, 3, 2, None, None, None, None).unwrap(),
            )
            .unwrap();

        let snapshot = operation_binding_snapshot(&registry).unwrap();
        assert!(snapshot.contains("\"instance_count\":3"));
        // linear is variant-insensitive: AgentLayerSpec uses VARIANT_NONE (0xFF).
        assert!(snapshot.contains("\"layer_id\":7,\"variant\":255,\"operation_id\":\"linear\""));
        assert!(snapshot.contains("\"layer_id\":9,\"variant\":1,\"operation_id\":\"relu\""));
        assert!(snapshot.contains("\"layer_id\":11,\"variant\":1,\"operation_id\":\"conv2d\""));
    }

    #[test]
    fn binding_fingerprint_is_anchored_to_inventory_fingerprint() {
        let mut registry = LayerRegistry::new();
        registry.init_agent_layer(&AgentLayerSpec::gelu(5)).unwrap();

        let binding = operation_binding_snapshot(&registry).unwrap();
        let inventory = registry.inventory_snapshot().unwrap();

        let inventory_fingerprint = inventory
            .split("\"inventory_fingerprint\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("inventory snapshot carries its fingerprint");
        assert!(
            binding.contains(&format!(
                "\"inventory_fingerprint\":\"{inventory_fingerprint}\""
            )),
            "binding must be computed over the exact canonical inventory"
        );

        assert!(registry.destroy_layer(5, LAYER_ACTIVATION));
        let rebound = operation_binding_snapshot(&registry).unwrap();
        assert!(!rebound.contains(inventory_fingerprint));
        assert!(rebound.contains("\"instance_count\":0"));
    }

    #[test]
    fn unknown_protocol_identity_fails_closed_without_touching_registry() {
        assert!(infer_operation_id(0x99, 0x00).is_err());
        assert!(infer_operation_id(LAYER_ACTIVATION, 0xFF).is_err());
        assert!(infer_operation_id(LAYER_CONV, 0xFF).is_err());

        let mut registry = LayerRegistry::new();
        registry
            .init_agent_layer(&AgentLayerSpec::linear(3, 2, 2, false).unwrap())
            .unwrap();
        let live_before = registry.inventory_snapshot().unwrap();
        assert!(infer_operation_id(LAYER_LINEAR, 0x00).is_ok());
        assert_eq!(
            registry.inventory_snapshot().unwrap(),
            live_before,
            "failed inference must not mutate the registry"
        );
    }

    #[test]
    fn binding_covers_every_live_layer_type_family() {
        assert_eq!(infer_operation_id(LAYER_LINEAR, 0xAB).unwrap(), "linear");
        assert_eq!(infer_operation_id(0x02, 0x00).unwrap(), "batchNorm");
        assert_eq!(infer_operation_id(0x02, 0x04).unwrap(), "rmsNorm");
        assert_eq!(infer_operation_id(0x03, 0x02).unwrap(), "convTranspose2d");
        assert_eq!(infer_operation_id(0x04, 0x07).unwrap(), "swiGlu");
        assert_eq!(infer_operation_id(0x04, 0x0D).unwrap(), "glu");
        assert_eq!(infer_operation_id(0x05, 0x00).unwrap(), "embedding");
        assert_eq!(infer_operation_id(0x06, 0x04).unwrap(), "adaptiveAvgPool2d");
        assert_eq!(infer_operation_id(0x10, 0x03).unwrap(), "shiftRight");
        assert_eq!(infer_operation_id(0x11, 0x00).unwrap(), "ghost");
        assert_eq!(infer_operation_id(0x12, 0x00).unwrap(), "seBlock");
        assert_eq!(infer_operation_id(0x13, 0x04).unwrap(), "concat");
        assert_eq!(infer_operation_id(0x14, 0x00).unwrap(), "featureNorm");
    }
}
