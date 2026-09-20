use wasm_bindgen::prelude::*;

use crate::agent::AgentLayerSpec;
use crate::contracts::{
    validate_external_input_contract_declaration, validate_external_input_contract_for_spec,
};
use crate::workspace::{AgentWorkspace, WorkspaceInputContract};

const INPUT_CONTRACT_V1: &str = include_str!("../docs/agent-input-contract.v1.json");
const MAX_INPUT_SEMANTICS_BYTES: usize = 512;

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

fn contract_json(contract: &WorkspaceInputContract) -> String {
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

/// Return the embedded contract for optional external input metadata.
#[wasm_bindgen(js_name = inputContractCapabilities)]
pub fn input_contract_capabilities() -> String {
    INPUT_CONTRACT_V1.to_string()
}

/// Bind optional shape/layout metadata to the canonical external input slot 0.
///
/// This mutates AgentWorkspace metadata only. It does not allocate tensors,
/// initialize layers, alter the graph plan, or mutate LayerRegistry.
#[wasm_bindgen(js_name = workspaceBindInputContract)]
#[allow(clippy::too_many_arguments)]
pub fn workspace_bind_input_contract(
    workspace: &mut AgentWorkspace,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    layout: String,
    semantics: String,
) -> Result<(), String> {
    if semantics.len() > MAX_INPUT_SEMANTICS_BYTES {
        return Err(format!(
            "workspaceBindInputContract: semantics {} bytes exceeds limit {MAX_INPUT_SEMANTICS_BYTES}",
            semantics.len()
        ));
    }

    let shape = [dim0, dim1, dim2, dim3];
    validate_external_input_contract_declaration(shape, &layout)
        .map_err(|err| format!("workspaceBindInputContract: {err}"))?;

    workspace.set_input_contract(WorkspaceInputContract {
        shape,
        layout,
        semantics,
    });
    Ok(())
}

/// Clear optional external input metadata. Runtime execution behavior is unchanged.
#[wasm_bindgen(js_name = workspaceClearInputContract)]
pub fn workspace_clear_input_contract(workspace: &mut AgentWorkspace) -> bool {
    workspace.clear_input_contract_internal()
}

/// Return the current input contract or an explicit unbound marker.
#[wasm_bindgen(js_name = workspaceInputContract)]
pub fn workspace_input_contract(workspace: &AgentWorkspace) -> String {
    match workspace.input_contract() {
        Some(contract) => format!(
            "{{\"status\":\"bound\",\"contract\":{}}}",
            contract_json(contract)
        ),
        None => concat!(
            "{",
            "\"status\":\"unbound\",",
            "\"slot\":0,",
            "\"policy\":\"defer_to_runtime\"",
            "}"
        )
        .to_string(),
    }
}

/// Compare the current optional input contract against one typed consumer spec.
///
/// Unbound and unknown are not failures. Incompatible means the declared
/// shape/layout proves the consumer cannot accept the external input without an
/// explicit transform.
#[wasm_bindgen(js_name = inputContractCompatibility)]
pub fn input_contract_compatibility(
    workspace: &AgentWorkspace,
    consumer: &AgentLayerSpec,
) -> String {
    let Some(contract) = workspace.input_contract() else {
        return concat!(
            "{",
            "\"status\":\"unbound\",",
            "\"compatible\":null,",
            "\"policy\":\"defer_to_runtime\"",
            "}"
        )
        .to_string();
    };

    match validate_external_input_contract_for_spec(contract.shape, &contract.layout, consumer) {
        Ok(result) => format!(
            concat!(
                "{{",
                "\"status\":\"{}\",",
                "\"compatible\":true,",
                "\"consumer_layer_type\":{},",
                "\"consumer_layer_id\":{},",
                "\"contract\":{}",
                "}}"
            ),
            json_escape(result),
            consumer.layer_type(),
            consumer.layer_id(),
            contract_json(contract),
        ),
        Err(message) => format!(
            concat!(
                "{{",
                "\"status\":\"incompatible\",",
                "\"compatible\":false,",
                "\"consumer_layer_type\":{},",
                "\"consumer_layer_id\":{},",
                "\"message\":\"{}\",",
                "\"contract\":{}",
                "}}"
            ),
            consumer.layer_type(),
            consumer.layer_id(),
            json_escape(&message),
            contract_json(contract),
        ),
    }
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
        assert!(result.contains("\"compatible\":true"));
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
