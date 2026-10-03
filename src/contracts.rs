//! # Kontrak: `contracts`
//!
//! ## Tanggung jawab
//! Meng-embed kontrak JSON kanonis dari `docs/contracts/` via `include_str!` dan
//! mengekspos validasi skema + `LayoutTag` untuk tata letak layer.
//!
//! ## Invariant
//! - Setiap `include_str!("docs/contracts/*.json")` merujuk ke file yang ada dan
//!   versinya terkunci (`*.v1.json`); perubahan kontrak = file versi baru,
//!   bukan edit diam-diam.
//! - `LayoutTag` adalah satu-satunya sumber kebenaran tag tata letak yang
//!   dipakai validasi.
//!
//! ## Bukan tanggung jawab modul ini
//! - Penegakan runtime atas kontrak → test konformansi di `tests/` dan
//!   audit `scripts/audit_*.mjs`.

pub use crate::facade::contracts::{
    agent_contract_schema, agent_contract_schema_version, agent_layout_compatibility,
    agent_layout_contract, agent_layout_contract_version, agent_spec_layout,
    validate_agent_layout_edge,
};
pub(crate) use crate::facade::contracts::{
    validate_agent_layout_identity_edge, validate_external_input_contract_declaration,
    validate_external_input_contract_for_spec, AGENT_CONTRACT_SCHEMA_V1, AGENT_LAYOUT_CONTRACT_V1,
};

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
            assert!(
                schema.contains(required),
                "missing contract marker: {required}"
            );
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
        assert_eq!(
            agent_layout_compatibility(&linear, &layer_norm),
            "incompatible"
        );
        assert!(validate_agent_layout_edge(&linear, &layer_norm).is_err());
    }

    #[test]
    fn linear_to_batch_norm_is_compatible_channel_first_subset() {
        let linear = AgentLayerSpec::linear(1, 4, 4, true).unwrap();
        let batch_norm = AgentLayerSpec::batch_norm(2, 4, None).unwrap();
        assert_eq!(
            agent_layout_compatibility(&linear, &batch_norm),
            "compatible"
        );
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
        assert_eq!(
            agent_layout_compatibility(&swiglu, &layer_norm),
            "compatible"
        );
    }

    #[test]
    fn embedding_output_does_not_silently_relabel_axis2_as_last_feature() {
        let embedding = AgentLayerSpec::embedding(1, 32, 8).unwrap();
        let layer_norm = AgentLayerSpec::layer_norm(2, 8, None).unwrap();
        assert_eq!(
            agent_layout_compatibility(&embedding, &layer_norm),
            "incompatible"
        );
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
