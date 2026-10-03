pub use crate::facade::ingress::{
    input_contract_capabilities, input_contract_compatibility, workspace_bind_input_contract,
    workspace_clear_input_contract, workspace_input_contract,
};
use wasm_bindgen::prelude::*;

use crate::agent::AgentLayerSpec;
use crate::contracts::{
    validate_external_input_contract_declaration, validate_external_input_contract_for_spec,
};
use crate::workspace::{AgentWorkspace, WorkspaceInputContract};

pub(crate) const INPUT_CONTRACT_V1: &str =
    include_str!("../../docs/contracts/agent-input-contract.v1.json");
pub(crate) const MAX_INPUT_SEMANTICS_BYTES: usize = 512;

pub(crate) fn json_escape(value: &str) -> String {
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

pub(crate) fn contract_json(contract: &WorkspaceInputContract) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.external-input-binding.v1\",",
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
    )
}

#[cfg(test)]
mod tests {
    use super::{
        input_contract_compatibility, workspace_bind_input_contract,
        workspace_clear_input_contract, workspace_input_contract,
    };
    use crate::agent::AgentLayerSpec;
    use crate::workspace::AgentWorkspace;

    #[test]
    fn unknown_layout_still_rejects_shape_proven_incompatible_with_linear() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        workspace_bind_input_contract(
            &mut workspace,
            1,
            2,
            2,
            1,
            "unknown".into(),
            "external-features".into(),
        )
        .unwrap();

        let linear = AgentLayerSpec::linear(1, 2, 1, false).unwrap();
        let result = input_contract_compatibility(&workspace, &linear);
        assert!(result.contains("\"status\":\"incompatible\""));
        assert!(result.contains("violates layout feature_axis1_singleton"));
    }

    #[test]
    fn valid_shape_with_unknown_layout_remains_non_overclaimed() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        workspace_bind_input_contract(
            &mut workspace,
            1,
            2,
            1,
            1,
            "unknown".into(),
            "external-features".into(),
        )
        .unwrap();

        let linear = AgentLayerSpec::linear(1, 2, 1, false).unwrap();
        let result = input_contract_compatibility(&workspace, &linear);
        assert!(result.contains("\"status\":\"shape_compatible_layout_unknown\""));
        assert!(result.contains("\"compatible\":null"));
    }

    #[test]
    fn clear_returns_to_explicit_unbound_state() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        workspace_bind_input_contract(
            &mut workspace,
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            "".into(),
        )
        .unwrap();
        assert!(workspace_input_contract(&workspace).contains("\"status\":\"bound\""));
        assert!(workspace_clear_input_contract(&mut workspace));
        assert!(workspace_input_contract(&workspace).contains("\"status\":\"unbound\""));
    }
}
