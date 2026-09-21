use std::collections::BTreeMap;

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX,
    ACT_MISH, ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU,
    ACT_TANH, BINARY_ADD, BINARY_CONCAT, BINARY_MATMUL, BINARY_MUL, BINARY_SUB,
    CONV_CONV1D, CONV_CONV2D, CONV_CONVTRANSPOSE2D, LAYER_ACTIVATION, LAYER_BINARY,
    LAYER_CONV, LAYER_EMBEDDING, LAYER_FEATURE_NORM, LAYER_GHOST, LAYER_LINEAR,
    LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT, NORM_BATCH, NORM_GROUP,
    NORM_INSTANCE, NORM_LAYER, NORM_RMS, POOL_ADAPTIVEAVGPOOL2D, POOL_AVGPOOL1D,
    POOL_AVGPOOL2D, POOL_MAXPOOL1D, POOL_MAXPOOL2D, SHIFT_DOWN, SHIFT_LEFT, SHIFT_RIGHT,
    SHIFT_UP,
};
use crate::registry::LayerRegistry;
use crate::resolution_runtime_bridge::runtime_subject_binding_json;
use crate::workspace::{AgentWorkspace, WorkspaceLayerIntrospection};

const LAYER_CATALOG_V1: &str = include_str!("../docs/agent-layer-catalog.v1.json");
const INTROSPECTION_CONTRACT_V1: &str =
    include_str!("../docs/agent-introspection-contract.v1.json");

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn quoted_or_null(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn u8_or_null(value: Option<u8>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn input_contract_json(workspace: &AgentWorkspace) -> String {
    match workspace.input_contract() {
        Some(contract) => format!(
            concat!(
                "{{",
                "\"status\":\"bound\",",
                "\"slot\":0,",
                "\"dtype\":\"f32\",",
                "\"shape\":[{},{},{},{}],",
                "\"layout\":\"{}\",",
                "\"semantics\":\"{}\"",
                "}}"
            ),
            contract.shape[0],
            contract.shape[1],
            contract.shape[2],
            contract.shape[3],
            json_escape(&contract.layout),
            json_escape(&contract.semantics),
        ),
        None => "{\"status\":\"unbound\",\"slot\":0,\"policy\":\"defer_to_runtime\"}"
            .to_string(),
    }
}


fn input_port_json(workspace: &AgentWorkspace) -> String {
    match workspace.input_port_metadata() {
        Some(metadata) => format!(
            concat!(
                "{{",
                "\"status\":\"bound\",",
                "\"slot\":0,",
                "\"role\":\"{}\",",
                "\"provenance\":{{",
                    "\"source\":\"{}\",",
                    "\"revision\":{},",
                    "\"fingerprint\":{}",
                "}}",
                "}}"
            ),
            json_escape(&metadata.role),
            json_escape(&metadata.source),
            metadata.revision,
            if metadata.fingerprint.is_empty() {
                "null".to_string()
            } else {
                format!("\"{}\"", json_escape(&metadata.fingerprint))
            },
        ),
        None => "{\"status\":\"unbound\",\"slot\":0,\"policy\":\"semantic_role_optional\"}"
            .to_string(),
    }
}

fn constructor_name(layer_type: u8, variant: Option<u8>) -> Option<&'static str> {
    match layer_type {
        LAYER_LINEAR => Some("linear"),
        LAYER_EMBEDDING => Some("embedding"),
        LAYER_FEATURE_NORM => Some("featureNorm"),
        LAYER_GHOST => Some("ghost"),
        LAYER_SEBLOCK => Some("seBlock"),
        LAYER_NORM => match variant? {
            NORM_BATCH => Some("batchNorm"),
            NORM_GROUP => Some("groupNorm"),
            NORM_INSTANCE => Some("instanceNorm"),
            NORM_LAYER => Some("layerNorm"),
            NORM_RMS => Some("rmsNorm"),
            _ => None,
        },
        LAYER_CONV => match variant? {
            CONV_CONV1D => Some("conv1d"),
            CONV_CONV2D => Some("conv2d"),
            CONV_CONVTRANSPOSE2D => Some("convTranspose2d"),
            _ => None,
        },
        LAYER_ACTIVATION => match variant? {
            ACT_GELU => Some("gelu"),
            ACT_RELU => Some("relu"),
            ACT_SIGMOID => Some("sigmoid"),
            ACT_TANH => Some("tanh"),
            ACT_HARDSWISH => Some("hardSwish"),
            ACT_LEAKYRELU => Some("leakyRelu"),
            ACT_PRELU => Some("prelu"),
            ACT_SWIGLU => Some("swiGlu"),
            ACT_HARDSIGMOID => Some("hardSigmoid"),
            ACT_SOFTPLUS => Some("softplus"),
            ACT_MISH => Some("mish"),
            ACT_SOFTMAX => Some("softmax"),
            ACT_LOGSOFTMAX => Some("logSoftmax"),
            ACT_GLU => Some("glu"),
            _ => None,
        },
        LAYER_POOL => match variant? {
            POOL_MAXPOOL1D => Some("maxPool1d"),
            POOL_MAXPOOL2D => Some("maxPool2d"),
            POOL_AVGPOOL1D => Some("avgPool1d"),
            POOL_AVGPOOL2D => Some("avgPool2d"),
            POOL_ADAPTIVEAVGPOOL2D => Some("adaptiveAvgPool2d"),
            _ => None,
        },
        LAYER_SHIFT => match variant? {
            SHIFT_UP => Some("shiftUp"),
            SHIFT_DOWN => Some("shiftDown"),
            SHIFT_LEFT => Some("shiftLeft"),
            SHIFT_RIGHT => Some("shiftRight"),
            _ => None,
        },
        LAYER_BINARY => match variant? {
            BINARY_ADD => Some("add"),
            BINARY_SUB => Some("sub"),
            BINARY_MUL => Some("mul"),
            BINARY_MATMUL => Some("matmul"),
            BINARY_CONCAT => Some("concat"),
            _ => None,
        },
        _ => None,
    }
}

fn layer_map(workspace: &AgentWorkspace) -> BTreeMap<u32, WorkspaceLayerIntrospection> {
    workspace
        .introspection_layers()
        .into_iter()
        .map(|layer| (layer.layer_id, layer))
        .collect()
}

/// Return the semantic-introspection contract embedded in the WASM artifact.
#[wasm_bindgen(js_name = introspectionCapabilities)]
pub fn introspection_capabilities() -> String {
    INTROSPECTION_CONTRACT_V1.to_string()
}

/// Return the complete typed AgentLayerSpec constructor catalog.
///
/// The catalog is embedded from a versioned JSON contract so agents can discover
/// all advertised constructor signatures without reading generated JS/TS glue.
#[wasm_bindgen(js_name = agentLayerCatalog)]
pub fn agent_layer_catalog() -> String {
    LAYER_CATALOG_V1.to_string()
}

/// Describe AgentWorkspace as semantic read-only state.
///
/// LayerRegistry is supplied only to report whether initialized workspace metadata
/// is still bound to live execution state. Neither input is mutated.
#[wasm_bindgen(js_name = describeWorkspace)]
pub fn describe_workspace(workspace: &AgentWorkspace, registry: &LayerRegistry) -> String {
    let slots = workspace.introspection_slots();
    let layers = workspace.introspection_layers();
    let (proofs_passed, proofs_failed, proofs_other) = workspace.introspection_proof_counts();
    let attestation_count = workspace.introspection_attestation_count();
    let (receipt_passed, receipt_failed, receipt_other) =
        workspace.introspection_verifier_receipt_counts();
    let custom_tables = workspace.introspection_custom_tables();

    let slots_json = slots
        .iter()
        .map(|slot| {
            let readable = matches!(slot.state.as_str(), "input" | "reserved");
            let releasable = slot.state == "reserved" && !slot.owner.starts_with("layer:");
            format!(
                concat!(
                    "{{",
                    "\"slot\":{},",
                    "\"state\":\"{}\",",
                    "\"owner\":{},",
                    "\"readable\":{},",
                    "\"releasable\":{}",
                    "}}"
                ),
                slot.slot,
                json_escape(&slot.state),
                if slot.owner.is_empty() {
                    "null".to_string()
                } else {
                    format!("\"{}\"", json_escape(&slot.owner))
                },
                if readable { "true" } else { "false" },
                if releasable { "true" } else { "false" },
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let layers_json = layers
        .iter()
        .map(|layer| {
            let registry_present = layer
                .layer_type
                .is_some_and(|layer_type| registry.layer_exists(layer_type, layer.layer_id));
            let constructor = if layer.metadata_valid {
                layer
                    .layer_type
                    .and_then(|layer_type| constructor_name(layer_type, layer.variant))
            } else {
                None
            };
            let fingerprint = layer.layer_type.and_then(|layer_type| {
                if registry_present {
                    registry
                        .layer_init_fingerprint(layer_type, layer.layer_id)
                        .ok()
                } else {
                    None
                }
            });
            format!(
                concat!(
                    "{{",
                    "\"layer_id\":{},",
                    "\"state\":\"{}\",",
                    "\"label\":\"{}\",",
                    "\"layer_type\":{},",
                    "\"variant\":{},",
                    "\"metadata_valid\":{},",
                    "\"constructor\":{},",
                    "\"registry_present\":{},",
                    "\"registry_fingerprint\":{}",
                    "}}"
                ),
                layer.layer_id,
                json_escape(&layer.state),
                json_escape(&layer.label),
                layer
                    .layer_type
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "null".to_string()),
                u8_or_null(layer.variant),
                if layer.metadata_valid { "true" } else { "false" },
                quoted_or_null(constructor),
                if registry_present { "true" } else { "false" },
                fingerprint
                    .as_deref()
                    .map(|value| format!("\"{}\"", json_escape(value)))
                    .unwrap_or_else(|| "null".to_string()),
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    let custom_tables_json = custom_tables
        .iter()
        .map(|table| format!("\"{}\"", json_escape(table)))
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.workspace-description.v1\",",
            "\"projection_only\":true,",
            "\"authority\":{{",
                "\"control\":\"AgentWorkspace\",",
                "\"execution_binding\":\"LayerRegistry\"",
            "}},",
            "\"num_slots\":{},",
            "\"external_input_contract\":{},",
            "\"external_input_port\":{},",
            "\"runtime_subject\":{},",
            "\"runtime_program_bindings\":{{\"count\":{},\"identity_policy\":\"exact_program_identity\"}},",
            "\"slots\":[{}],",
            "\"layers\":[{}],",
            "\"proof_summary\":{{",
                "\"legacy_recordProof_authority\":\"caller_controlled_legacy\",",
                "\"legacy_passed\":{},",
                "\"legacy_failed\":{},",
                "\"legacy_other\":{},",
                "\"attestations\":{},",
                "\"verifier_receipts\":{{",
                    "\"passed\":{},",
                    "\"failed\":{},",
                    "\"other\":{}",
                "}}",
            "}},",
            "\"event_count\":{},",
            "\"custom_tables\":[{}]",
            "}}"
        ),
        workspace.interaction_num_slots(),
        input_contract_json(workspace),
        input_port_json(workspace),
        runtime_subject_binding_json(workspace),
        workspace.runtime_program_binding_count(),
        slots_json,
        layers_json,
        proofs_passed,
        proofs_failed,
        proofs_other,
        attestation_count,
        receipt_passed,
        receipt_failed,
        receipt_other,
        workspace.introspection_event_count(),
        custom_tables_json,
    )
}

/// Describe the semantic graph topology currently held by AgentGraphBuilder.
///
/// Workspace metadata is used only when available to recover constructor/variant
/// labels. Missing metadata is reported as unknown rather than inferred.
/// LayerRegistry is used only to report live execution binding/provenance.
#[wasm_bindgen(js_name = describeGraph)]
pub fn describe_graph(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
) -> Result<String, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "describeGraph: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let layers = layer_map(workspace);
    let steps = builder.introspection_steps();

    let steps_json = steps
        .iter()
        .enumerate()
        .map(
            |(index, (arity, layer_type, layer_id, in_slot, in_slot2, out_slot))| {
                let metadata = layers.get(layer_id);
                let metadata_valid = metadata.is_some_and(|layer| layer.metadata_valid);
                let metadata_matches_type = metadata.is_some_and(|layer| {
                    layer.metadata_valid && layer.layer_type == Some(*layer_type)
                });
                let variant = if metadata_matches_type {
                    metadata.and_then(|layer| layer.variant)
                } else {
                    None
                };
                let constructor = constructor_name(*layer_type, variant);
                let registry_present = registry.layer_exists(*layer_type, *layer_id);
                let fingerprint = if registry_present {
                    registry.layer_init_fingerprint(*layer_type, *layer_id).ok()
                } else {
                    None
                };
                let input_slots = if *arity == 1 {
                    format!("[{in_slot}]")
                } else {
                    format!("[{in_slot},{in_slot2}]")
                };
                format!(
                    concat!(
                        "{{",
                        "\"index\":{},",
                        "\"arity\":{},",
                        "\"constructor\":{},",
                        "\"layer_type\":{},",
                        "\"variant\":{},",
                        "\"layer_id\":{},",
                        "\"input_slots\":{},",
                        "\"output_slot\":{},",
                        "\"workspace_metadata\":{},",
                        "\"metadata_valid\":{},",
                        "\"metadata_type_matches_builder\":{},",
                        "\"registry_present\":{},",
                        "\"registry_fingerprint\":{},",
                        "\"layout_lookup\":{}",
                        "}}"
                    ),
                    index,
                    arity,
                    quoted_or_null(constructor),
                    layer_type,
                    u8_or_null(variant),
                    layer_id,
                    input_slots,
                    out_slot,
                    if metadata.is_some() { "true" } else { "false" },
                    if metadata_valid { "true" } else { "false" },
                    if metadata_matches_type { "true" } else { "false" },
                    if registry_present { "true" } else { "false" },
                    fingerprint
                        .as_deref()
                        .map(|value| format!("\"{}\"", json_escape(value)))
                        .unwrap_or_else(|| "null".to_string()),
                    constructor
                        .map(|name| format!("\"agentLayoutContract.constructors.{}\"", json_escape(name)))
                        .unwrap_or_else(|| "null".to_string()),
                )
            },
        )
        .collect::<Vec<_>>()
        .join(",");

    let written_slots = builder
        .interaction_written_slots()
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.graph-description.v1\",",
            "\"projection_only\":true,",
            "\"authority\":{{",
                "\"topology\":\"AgentGraphBuilder\",",
                "\"semantic_metadata\":\"AgentWorkspace when available\",",
                "\"execution_binding\":\"LayerRegistry\"",
            "}},",
            "\"unknown_metadata_policy\":\"report_null_do_not_infer\",",
            "\"num_slots\":{},",
            "\"external_input_contract\":{},",
            "\"external_input_port\":{},",
            "\"runtime_subject\":{},",
            "\"runtime_program_bindings\":{{\"count\":{},\"identity_policy\":\"exact_program_identity\",\"claim_policy\":\"no_precompile_identity_inference\"}},",
            "\"num_steps\":{},",
            "\"configured_output_slot\":{},",
            "\"written_slots\":[{}],",
            "\"steps\":[{}]",
            "}}"
        ),
        builder.num_slots(),
        input_contract_json(workspace),
        input_port_json(workspace),
        runtime_subject_binding_json(workspace),
        workspace.runtime_program_binding_count(),
        builder.num_steps(),
        builder
            .introspection_output_slot()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string()),
        written_slots,
        steps_json,
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{agent_layer_catalog, describe_graph, describe_workspace};
    use crate::agent::{capability_manifest, AgentGraphBuilder, AgentLayerSpec};
    use crate::contracts::agent_layout_contract;
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn layer_catalog_covers_every_advertised_constructor_and_layout_key() {
        let catalog: serde_json::Value = serde_json::from_str(&agent_layer_catalog()).unwrap();
        let capabilities: serde_json::Value = serde_json::from_str(&capability_manifest()).unwrap();
        let layout: serde_json::Value = serde_json::from_str(&agent_layout_contract()).unwrap();

        let catalog_keys = catalog["constructors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let advertised = capabilities["agent_facade"]["constructors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect::<BTreeSet<_>>();
        let layout_keys = layout["constructors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        assert_eq!(catalog_keys.len(), 41);
        assert_eq!(catalog_keys, advertised);
        assert_eq!(catalog_keys, layout_keys);
    }

    #[test]
    fn workspace_description_is_read_only_and_reports_live_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let spec = AgentLayerSpec::relu(id);
        registry.init_agent_layer(&spec).unwrap();
        workspace.sync_layer(&registry, &spec, "relu".into()).unwrap();

        let before = workspace.snapshot();
        let description: serde_json::Value =
            serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();

        assert_eq!(description["layers"][0]["constructor"], "relu");
        assert_eq!(description["layers"][0]["registry_present"], true);
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn descriptions_expose_semantic_input_port_without_mutating_execution_state() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let registry = LayerRegistry::new();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap();

        let before = workspace.snapshot();
        let workspace_json: serde_json::Value =
            serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
        assert_eq!(workspace_json["external_input_port"]["role"], "observation");
        assert_eq!(
            workspace_json["external_input_port"]["provenance"]["source"],
            "market-feed"
        );
        assert_eq!(workspace.snapshot(), before);

        let builder = AgentGraphBuilder::new(3).unwrap();
        let graph_json: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();
        assert_eq!(graph_json["external_input_port"]["role"], "observation");
        assert_eq!(
            graph_json["external_input_port"]["provenance"]["revision"],
            18
        );
    }

    #[test]
    fn graph_description_exposes_canonical_topology_and_binding() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let spec = AgentLayerSpec::relu(id);
        let out = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu".into(),
        )
        .unwrap();

        let description: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

        assert_eq!(description["steps"][0]["constructor"], "relu");
        assert_eq!(description["steps"][0]["input_slots"][0], 0);
        assert_eq!(description["steps"][0]["output_slot"], out);
        assert_eq!(description["steps"][0]["registry_present"], true);
    }

    #[test]
    fn lower_level_graph_missing_workspace_variant_is_not_guessed() {
        let workspace = AgentWorkspace::new(2).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(77);
        registry.init_agent_layer(&spec).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();

        let description: serde_json::Value =
            serde_json::from_str(&describe_graph(&workspace, &builder, &registry).unwrap()).unwrap();

        assert!(description["steps"][0]["constructor"].is_null());
        assert!(description["steps"][0]["variant"].is_null());
        assert_eq!(description["steps"][0]["workspace_metadata"], false);
        assert_eq!(description["steps"][0]["registry_present"], true);
    }
}
