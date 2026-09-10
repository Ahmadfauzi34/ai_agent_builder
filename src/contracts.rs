use wasm_bindgen::prelude::*;

const AGENT_CONTRACT_SCHEMA_V1: &str = include_str!("../docs/agent-contracts.v1.json");

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

#[cfg(test)]
mod late_failure_contract_tests;

#[cfg(test)]
mod tests {
    use super::{agent_contract_schema, agent_contract_schema_version, AGENT_CONTRACT_SCHEMA_V1};

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
    fn schema_exposes_explicit_compatibility_policy() {
        let schema = agent_contract_schema();
        assert!(schema.contains("\"unknown_schema_version\":\"reject\""));
        assert!(schema.contains("\"unknown_predicate\":\"reject\""));
    }
}
