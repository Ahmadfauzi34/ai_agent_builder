//! Implementasi fasad untuk kontrak agen (`contracts`) — Opsi C Fase 1.
//!
//! Pindahan murni dari `src/contracts.rs`: 7 free function `#[wasm_bindgen]`
//! beserta seluruh item pendukungnya (konstanta skema `include_str!`,
//! `LayoutTag`, `LayoutProfile`, `LayoutCompatibility`, dan helper validasi).
//! Nama export JS tidak berubah (`js_name` dipertahankan byte-identik).
//! Kompatibilitas path lama dijaga via re-export di `src/contracts.rs`
//! (`burn_research::contracts::*`) dan `src/lib.rs` (`burn_research::*`).

use crate::agent::AgentLayerSpec;
use crate::protocol::{
    ACT_PRELU, ACT_SWIGLU, CONV_CONV1D, LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV,
    LAYER_EMBEDDING, LAYER_FEATURE_NORM, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL,
    LAYER_SEBLOCK, LAYER_SHIFT, NORM_BATCH, NORM_GROUP, NORM_INSTANCE, NORM_LAYER, NORM_RMS,
    POOL_AVGPOOL1D, POOL_MAXPOOL1D,
};
use wasm_bindgen::prelude::*;

pub(crate) const AGENT_CONTRACT_SCHEMA_V1: &str =
    include_str!("../../docs/contracts/agent-contracts.v1.json");
pub(crate) const AGENT_LAYOUT_CONTRACT_V1: &str =
    include_str!("../../docs/contracts/agent-layout-contracts.v1.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutTag {
    AnyRank4,
    PreserveInput,
    Dynamic,
    FeatureAxis1Singleton,
    ChannelFirst,
    ChannelFirstSingletonWidth,
    FeatureLast,
    TokenIdsAxis1Singleton,
    SequenceFeatureAxis2SingletonWidth,
}

impl LayoutTag {
    fn as_str(self) -> &'static str {
        match self {
            Self::AnyRank4 => "any_rank4",
            Self::PreserveInput => "preserve_input",
            Self::Dynamic => "dynamic",
            Self::FeatureAxis1Singleton => "feature_axis1_singleton",
            Self::ChannelFirst => "channel_first",
            Self::ChannelFirstSingletonWidth => "channel_first_singleton_width",
            Self::FeatureLast => "feature_last",
            Self::TokenIdsAxis1Singleton => "token_ids_axis1_singleton",
            Self::SequenceFeatureAxis2SingletonWidth => "sequence_feature_axis2_singleton_width",
        }
    }

    fn external_declared(value: &str) -> Result<Option<Self>, String> {
        match value {
            "unknown" => Ok(None),
            "any_rank4" => Ok(Some(Self::AnyRank4)),
            "feature_axis1_singleton" => Ok(Some(Self::FeatureAxis1Singleton)),
            "channel_first" => Ok(Some(Self::ChannelFirst)),
            "channel_first_singleton_width" => Ok(Some(Self::ChannelFirstSingletonWidth)),
            "feature_last" => Ok(Some(Self::FeatureLast)),
            "token_ids_axis1_singleton" => Ok(Some(Self::TokenIdsAxis1Singleton)),
            "sequence_feature_axis2_singleton_width" => {
                Ok(Some(Self::SequenceFeatureAxis2SingletonWidth))
            }
            "preserve_input" | "dynamic" => Err(format!(
                "inputContract: layout {value} is relational/internal and cannot describe an external tensor"
            )),
            _ => Err(format!("inputContract: unknown layout {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LayoutProfile {
    input: LayoutTag,
    output: LayoutTag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutCompatibility {
    Compatible,
    Unknown,
    Incompatible,
}

impl LayoutCompatibility {
    fn as_str(self) -> &'static str {
        match self {
            Self::Compatible => "compatible",
            Self::Unknown => "unknown",
            Self::Incompatible => "incompatible",
        }
    }
}

fn layout_profile_for(layer_type: u8, variant: u8) -> LayoutProfile {
    let input_output = |layout| LayoutProfile {
        input: layout,
        output: layout,
    };

    match layer_type {
        LAYER_LINEAR => input_output(LayoutTag::FeatureAxis1Singleton),
        LAYER_FEATURE_NORM => input_output(LayoutTag::FeatureAxis1Singleton),
        LAYER_NORM => match variant {
            NORM_BATCH | NORM_GROUP | NORM_INSTANCE => input_output(LayoutTag::ChannelFirst),
            NORM_LAYER | NORM_RMS => input_output(LayoutTag::FeatureLast),
            _ => input_output(LayoutTag::Dynamic),
        },
        LAYER_CONV => {
            if variant == CONV_CONV1D {
                input_output(LayoutTag::ChannelFirstSingletonWidth)
            } else {
                input_output(LayoutTag::ChannelFirst)
            }
        }
        LAYER_EMBEDDING => LayoutProfile {
            input: LayoutTag::TokenIdsAxis1Singleton,
            output: LayoutTag::SequenceFeatureAxis2SingletonWidth,
        },
        LAYER_POOL => {
            if matches!(variant, POOL_MAXPOOL1D | POOL_AVGPOOL1D) {
                input_output(LayoutTag::ChannelFirstSingletonWidth)
            } else {
                input_output(LayoutTag::ChannelFirst)
            }
        }
        LAYER_SHIFT | LAYER_GHOST | LAYER_SEBLOCK => input_output(LayoutTag::ChannelFirst),
        LAYER_ACTIVATION => match variant {
            ACT_SWIGLU => input_output(LayoutTag::FeatureLast),
            ACT_PRELU => LayoutProfile {
                input: LayoutTag::Dynamic,
                output: LayoutTag::PreserveInput,
            },
            _ => LayoutProfile {
                input: LayoutTag::AnyRank4,
                output: LayoutTag::PreserveInput,
            },
        },
        LAYER_BINARY => input_output(LayoutTag::Dynamic),
        _ => input_output(LayoutTag::Dynamic),
    }
}

fn layout_profile(spec: &AgentLayerSpec) -> LayoutProfile {
    layout_profile_for(spec.layer_type(), spec.variant())
}

fn compatibility(producer_output: LayoutTag, consumer_input: LayoutTag) -> LayoutCompatibility {
    use LayoutCompatibility::{Compatible, Incompatible, Unknown};
    use LayoutTag::{
        AnyRank4, ChannelFirst, ChannelFirstSingletonWidth, Dynamic, FeatureAxis1Singleton,
        PreserveInput,
    };

    if consumer_input == AnyRank4 {
        return Compatible;
    }
    if consumer_input == Dynamic {
        return Unknown;
    }
    if matches!(producer_output, PreserveInput | Dynamic | AnyRank4) {
        return Unknown;
    }
    if producer_output == consumer_input {
        return Compatible;
    }

    match (producer_output, consumer_input) {
        (FeatureAxis1Singleton, ChannelFirst | ChannelFirstSingletonWidth)
        | (ChannelFirstSingletonWidth, ChannelFirst) => Compatible,
        (ChannelFirst, FeatureAxis1Singleton | ChannelFirstSingletonWidth)
        | (ChannelFirstSingletonWidth, FeatureAxis1Singleton) => Unknown,
        _ => Incompatible,
    }
}

fn validate_layout_profiles(
    producer_profile: LayoutProfile,
    consumer_profile: LayoutProfile,
) -> Result<(), String> {
    let result = compatibility(producer_profile.output, consumer_profile.input);
    if result == LayoutCompatibility::Incompatible {
        return Err(format!(
            "validateAgentLayoutEdge: producer output layout {} is incompatible with consumer input layout {}; implicit relayout is forbidden",
            producer_profile.output.as_str(),
            consumer_profile.input.as_str()
        ));
    }
    Ok(())
}

pub(crate) fn validate_agent_layout_identity_edge(
    producer_layer_type: u8,
    producer_variant: u8,
    consumer: &AgentLayerSpec,
) -> Result<(), String> {
    validate_layout_profiles(
        layout_profile_for(producer_layer_type, producer_variant),
        layout_profile(consumer),
    )
}

fn validate_shape_for_layout(
    shape: [u32; 4],
    layout: LayoutTag,
    context: &str,
) -> Result<(), String> {
    let mismatch = match layout {
        LayoutTag::FeatureAxis1Singleton | LayoutTag::TokenIdsAxis1Singleton => {
            shape[2] != 1 || shape[3] != 1
        }
        LayoutTag::ChannelFirstSingletonWidth | LayoutTag::SequenceFeatureAxis2SingletonWidth => {
            shape[3] != 1
        }
        LayoutTag::AnyRank4 | LayoutTag::ChannelFirst | LayoutTag::FeatureLast => false,
        LayoutTag::PreserveInput | LayoutTag::Dynamic => false,
    };
    if mismatch {
        return Err(format!(
            "{context}: shape [{},{},{},{}] violates layout {}",
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            layout.as_str()
        ));
    }
    Ok(())
}

pub(crate) fn validate_external_input_contract_declaration(
    shape: [u32; 4],
    layout: &str,
) -> Result<(), String> {
    if shape.iter().any(|dim| *dim == 0) {
        return Err(format!(
            "inputContract: every shape dimension must be > 0, got [{},{},{},{}]",
            shape[0], shape[1], shape[2], shape[3]
        ));
    }
    if let Some(layout) = LayoutTag::external_declared(layout)? {
        validate_shape_for_layout(shape, layout, "inputContract")?;
    }
    Ok(())
}

pub(crate) fn validate_external_input_contract_for_spec(
    shape: [u32; 4],
    declared_layout: &str,
    consumer: &AgentLayerSpec,
) -> Result<&'static str, String> {
    validate_external_input_contract_declaration(shape, declared_layout)?;

    let consumer_layout = layout_profile(consumer).input;
    validate_shape_for_layout(shape, consumer_layout, "inputContract.consumer")?;

    let Some(declared) = LayoutTag::external_declared(declared_layout)? else {
        return Ok(if consumer_layout == LayoutTag::Dynamic {
            "unknown"
        } else {
            "shape_compatible_layout_unknown"
        });
    };

    match compatibility(declared, consumer_layout) {
        LayoutCompatibility::Compatible => Ok("compatible"),
        LayoutCompatibility::Unknown => Ok("unknown"),
        LayoutCompatibility::Incompatible => Err(format!(
            "inputContract: declared layout {} is incompatible with consumer input layout {}; implicit relayout is forbidden",
            declared.as_str(),
            consumer_layout.as_str()
        )),
    }
}

/// Return the canonical machine-readable contract manifest used by agents and future fuzzers.
#[wasm_bindgen(js_name = agentContractSchema)]
pub fn agent_contract_schema() -> String {
    AGENT_CONTRACT_SCHEMA_V1.to_string()
}

/// Return the contract schema version without requiring JSON parsing during capability discovery.
#[wasm_bindgen(js_name = agentContractSchemaVersion)]
pub fn agent_contract_schema_version() -> u32 {
    1
}

/// Return the companion tensor-layout contract. Burn tensor/module semantics remain authoritative;
/// this schema only defines how the rank-4 agent adapter interprets axes between layers.
#[wasm_bindgen(js_name = agentLayoutContract)]
pub fn agent_layout_contract() -> String {
    AGENT_LAYOUT_CONTRACT_V1.to_string()
}

#[wasm_bindgen(js_name = agentLayoutContractVersion)]
pub fn agent_layout_contract_version() -> u32 {
    1
}

/// Return the input/output layout profile for one typed layer spec.
#[wasm_bindgen(js_name = agentSpecLayout)]
pub fn agent_spec_layout(spec: &AgentLayerSpec) -> String {
    let profile = layout_profile(spec);
    format!(
        "{{\"input\":\"{}\",\"output\":\"{}\"}}",
        profile.input.as_str(),
        profile.output.as_str()
    )
}

/// Compare a producer's declared output layout with a consumer's declared input layout.
/// `unknown` is intentionally not an error: it means runtime Burn/shape validation is still needed.
#[wasm_bindgen(js_name = agentLayoutCompatibility)]
pub fn agent_layout_compatibility(producer: &AgentLayerSpec, consumer: &AgentLayerSpec) -> String {
    let producer = layout_profile(producer);
    let consumer = layout_profile(consumer);
    compatibility(producer.output, consumer.input)
        .as_str()
        .to_string()
}

/// Reject only known semantic layout mismatches. No implicit transpose/reshape is inserted.
#[wasm_bindgen(js_name = validateAgentLayoutEdge)]
pub fn validate_agent_layout_edge(
    producer: &AgentLayerSpec,
    consumer: &AgentLayerSpec,
) -> Result<(), String> {
    validate_layout_profiles(layout_profile(producer), layout_profile(consumer))
}
