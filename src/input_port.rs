use wasm_bindgen::prelude::*;

use crate::workspace::{AgentWorkspace, WorkspaceInputPortMetadata};

const INPUT_PORT_V1: &str = include_str!("../docs/agent-input-port.v1.json");
const MAX_ROLE_BYTES: usize = 64;
const MAX_SOURCE_BYTES: usize = 256;
const MAX_FINGERPRINT_BYTES: usize = 256;

const CANONICAL_ROLES: [&str; 7] = [
    "observation",
    "state",
    "feature",
    "candidate",
    "parameter",
    "reward",
    "context",
];

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

fn validate_bounded(value: &str, max: usize, context: &str, allow_empty: bool) -> Result<(), String> {
    if !allow_empty && value.is_empty() {
        return Err(format!("{context}: value must be non-empty"));
    }
    if value.len() > max {
        return Err(format!("{context}: {} bytes exceeds limit {max}", value.len()));
    }
    Ok(())
}

pub(crate) fn role_valid(role: &str) -> bool {
    CANONICAL_ROLES.contains(&role) || (
        role.starts_with("x-")
            && role.len() > 2
            && role.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.'))
    )
}

fn metadata_json(metadata: &WorkspaceInputPortMetadata) -> String {
    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-binding.v1\",",
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
        }
    )
}

#[wasm_bindgen(js_name = inputPortCapabilities)]
pub fn input_port_capabilities() -> String {
    INPUT_PORT_V1.to_string()
}

#[wasm_bindgen(js_name = workspaceBindInputPortMetadata)]
pub fn workspace_bind_input_port_metadata(
    workspace: &mut AgentWorkspace,
    role: String,
    source: String,
    revision: u64,
    fingerprint: String,
) -> Result<bool, String> {
    validate_bounded(&role, MAX_ROLE_BYTES, "workspaceBindInputPortMetadata.role", false)?;
    if !role_valid(&role) {
        return Err(format!(
            "workspaceBindInputPortMetadata.role: unsupported role {role}; use canonical role or x- extension namespace"
        ));
    }
    validate_bounded(&source, MAX_SOURCE_BYTES, "workspaceBindInputPortMetadata.source", false)?;
    validate_bounded(
        &fingerprint,
        MAX_FINGERPRINT_BYTES,
        "workspaceBindInputPortMetadata.fingerprint",
        true,
    )?;

    Ok(workspace.set_input_port_metadata(WorkspaceInputPortMetadata {
        role,
        source,
        revision,
        fingerprint,
    }))
}

#[wasm_bindgen(js_name = workspaceClearInputPortMetadata)]
pub fn workspace_clear_input_port_metadata(workspace: &mut AgentWorkspace) -> bool {
    workspace.clear_input_port_metadata_internal()
}

#[wasm_bindgen(js_name = workspaceInputPortMetadata)]
pub fn workspace_input_port_metadata(workspace: &AgentWorkspace) -> String {
    match workspace.input_port_metadata() {
        Some(metadata) => format!(
            "{{\"status\":\"bound\",\"metadata\":{},\"runtime_subject_bound\":{}}}",
            metadata_json(metadata),
            if workspace.runtime_subject_binding().is_some() {
                "true"
            } else {
                "false"
            }
        ),
        None => format!(
            "{{\"status\":\"unbound\",\"slot\":0,\"runtime_subject_bound\":{},\"policy\":\"semantic_role_optional\"}}",
            if workspace.runtime_subject_binding().is_some() {
                "true"
            } else {
                "false"
            }
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        input_port_capabilities, workspace_bind_input_port_metadata,
        workspace_clear_input_port_metadata, workspace_input_port_metadata,
    };
    use crate::workspace::AgentWorkspace;

    #[test]
    fn semantic_port_metadata_is_optional_and_idempotent() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        assert!(workspace_input_port_metadata(&workspace).contains("\"status\":\"unbound\""));

        assert!(workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap());
        assert!(!workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap());

        let json = workspace_input_port_metadata(&workspace);
        assert!(json.contains("\"role\":\"observation\""));
        assert!(json.contains("\"source\":\"market-feed\""));
        assert!(json.contains("\"revision\":18"));
        assert!(json.contains("\"fingerprint\":\"fnv1a64:abcd\""));

        assert!(workspace_clear_input_port_metadata(&mut workspace));
        assert!(!workspace_clear_input_port_metadata(&mut workspace));
    }

    #[test]
    fn role_supports_canonical_and_namespaced_extensions_only() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        assert!(workspace_bind_input_port_metadata(
            &mut workspace,
            "x-market-regime".into(),
            "feature-pipeline".into(),
            1,
            String::new(),
        )
        .is_ok());
        assert!(workspace_bind_input_port_metadata(
            &mut workspace,
            "anything".into(),
            "feature-pipeline".into(),
            1,
            String::new(),
        )
        .is_err());
    }

    #[test]
    fn capability_contract_is_embedded_and_explicitly_metadata_only() {
        let caps: serde_json::Value = serde_json::from_str(&input_port_capabilities()).unwrap();
        assert_eq!(caps["role"], "optional_semantic_input_port");
        assert_eq!(caps["scope"]["execution_effect"], "none");
    }
}
