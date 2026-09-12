use wasm_bindgen::prelude::*;

use crate::agent::AgentLayerSpec;
use crate::protocol::{
    ACT_PRELU, ACT_SWIGLU, CONV_CONV1D, LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV,
    LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK,
    LAYER_SHIFT, NORM_BATCH, NORM_GROUP, NORM_INSTANCE, NORM_LAYER, NORM_RMS, POOL_AVGPOOL1D,
    POOL_MAXPOOL1D,
};

const AGENT_CONTRACT_SCHEMA_V1: &str = include_str!("../docs/agent-contracts.v1.json");
const AGENT_LAYOUT_CONTRACT_V1: &str = include_str!("../docs/agent-layout-contracts.v1.json");

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
pub fn agent_layout_compatibility(
    producer: &AgentLayerSpec,
    consumer: &AgentLayerSpec,
) -> String {
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

#[cfg(test)]
mod late_failure_contract_tests;
#[cfg(test)]
mod transactional_init_contract_tests;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        agent_contract_schema, agent_contract_schema_version, agent_layout_compatibility,
        agent_layout_contract, agent_layout_contract_version, agent_spec_layout,
        validate_agent_layout_edge, AGENT_CONTRACT_SCHEMA_V1, AGENT_LAYOUT_CONTRACT_V1,
    };
    use crate::agent::AgentLayerSpec;

    #[test]
    fn exported_contract_schema_is_the_canonical_embedded_file() {
        assert_eq!(agent_contract_schema(), AGENT_CONTRACT_SCHEMA_V1);
        assert_eq!(agent_contract_schema_version(), 1);
    }

    #[test]
    fn schema_carries_p0_p1_boundary_contracts() {
        let schema = agent_contract_schema();
        for required in [
            "burn-research.agent-contracts.v1",
            "validated_init_fingerprint",
            "slot.state:free->reserved",
            "slot.state:reserved->free",
            "error_no_mutation",
            "workspaceInitUnary",
            "workspaceInitBinary",
            "workspaceWireUnary",
            "workspaceWireBinary",
            "workspaceCompile",
            "AgentWorkspace.snapshot",
        ] {
            assert!(schema.contains(required), "missing contract marker: {required}");
        }
    }

    #[test]
    fn schema_exposes_transactional_init_guarantee() {
        let schema = agent_contract_schema();
        assert!(schema.contains("restore_pre_call_workspace"));
        assert!(schema.contains("destroy_newly_initialized_layer"));
        assert!(!schema.contains("registry_post_init_rollback\":\"not_guaranteed"));
    }

    #[test]
    fn schema_exposes_explicit_compatibility_policy() {
        let schema: serde_json::Value = serde_json::from_str(&agent_contract_schema()).unwrap();
        assert_eq!(schema["compatibility"]["unknown_schema_version"], "reject");
        assert_eq!(schema["compatibility"]["unknown_predicate"], "reject");
    }

    #[test]
    fn layout_contract_is_embedded_and_versioned() {
        assert_eq!(agent_layout_contract(), AGENT_LAYOUT_CONTRACT_V1);
        assert_eq!(agent_layout_contract_version(), 1);
        let parsed: serde_json::Value = serde_json::from_str(AGENT_LAYOUT_CONTRACT_V1).unwrap();
        assert_eq!(parsed["policy"]["implicit_relayout"], "forbidden");
    }

    #[test]
    fn layout_contract_covers_every_advertised_constructor() {
        let capabilities: serde_json::Value =
            serde_json::from_str(&crate::agent::capability_manifest()).unwrap();
        let layout: serde_json::Value = serde_json::from_str(AGENT_LAYOUT_CONTRACT_V1).unwrap();

        let advertised = capabilities["agent_facade"]["constructors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect::<BTreeSet<_>>();
        let contracted = layout["constructors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        assert_eq!(contracted, advertised);
    }

    #[test]
    fn linear_to_layer_norm_is_a_known_semantic_mismatch() {
        let linear = AgentLayerSpec::linear(1, 4, 4, true).unwrap();
        let layer_norm = AgentLayerSpec::layer_norm(2, 4, None).unwrap();
        assert_eq!(
            agent_spec_layout(&linear),
            "{\"input\":\"feature_axis1_singleton\",\"output\":\"feature_axis1_singleton\"}"
        );
        assert_eq!(agent_layout_compatibility(&linear, &layer_norm), "incompatible");
        assert!(validate_agent_layout_edge(&linear, &layer_norm).is_err());
    }

    #[test]
    fn linear_to_batch_norm_is_compatible_channel_first_subset() {
        let linear = AgentLayerSpec::linear(1, 4, 4, true).unwrap();
        let batch_norm = AgentLayerSpec::batch_norm(2, 4, None).unwrap();
        assert_eq!(agent_layout_compatibility(&linear, &batch_norm), "compatible");
        assert!(validate_agent_layout_edge(&linear, &batch_norm).is_ok());
    }

    #[test]
    fn conv_to_batch_norm_is_compatible() {
        let conv = AgentLayerSpec::conv2d(1, 3, 8, 3, 3, None, None, None, None).unwrap();
        let batch_norm = AgentLayerSpec::batch_norm(2, 8, None).unwrap();
        assert_eq!(agent_layout_compatibility(&conv, &batch_norm), "compatible");
    }

    #[test]
    fn swiglu_to_layer_norm_uses_the_same_last_feature_layout() {
        let swiglu = AgentLayerSpec::swi_glu(1, 8, 4, true).unwrap();
        let layer_norm = AgentLayerSpec::layer_norm(2, 4, None).unwrap();
        assert_eq!(agent_layout_compatibility(&swiglu, &layer_norm), "compatible");
    }

    #[test]
    fn embedding_output_does_not_silently_relabel_axis2_as_last_feature() {
        let embedding = AgentLayerSpec::embedding(1, 32, 8).unwrap();
        let layer_norm = AgentLayerSpec::layer_norm(2, 8, None).unwrap();
        assert_eq!(agent_layout_compatibility(&embedding, &layer_norm), "incompatible");
        assert!(validate_agent_layout_edge(&embedding, &layer_norm).is_err());
    }

    #[test]
    fn layout_unknown_is_not_overclaimed_as_failure() {
        let relu = AgentLayerSpec::relu(1);
        let layer_norm = AgentLayerSpec::layer_norm(2, 8, None).unwrap();
        assert_eq!(agent_layout_compatibility(&relu, &layer_norm), "unknown");
        assert!(validate_agent_layout_edge(&relu, &layer_norm).is_ok());
    }
}
