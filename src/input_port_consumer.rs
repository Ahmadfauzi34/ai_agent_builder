use std::collections::BTreeSet;

use wasm_bindgen::prelude::*;

use crate::input_port::role_valid;
use crate::workspace::AgentWorkspace;

const INPUT_PORT_CONSUMER_V1: &str =
    include_str!("../docs/agent-input-port-consumer.v1.json");
const MAX_CONSUMER_ID_BYTES: usize = 128;
const MAX_ACCEPTED_ROLES: usize = 16;

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

fn bool_json(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn string_array_json(values: &[String]) -> String {
    let body = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

fn parse_roles(value: &str) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value = serde_json::from_str(value)
        .map_err(|error| format!("InputPortConsumerSpec.accepted_roles_json: invalid JSON: {error}"))?;
    let items = parsed
        .as_array()
        .ok_or_else(|| {
            "InputPortConsumerSpec.accepted_roles_json: expected JSON array of role strings"
                .to_string()
        })?;
    if items.is_empty() {
        return Err(
            "InputPortConsumerSpec.accepted_roles_json: at least one role is required".to_string(),
        );
    }
    if items.len() > MAX_ACCEPTED_ROLES {
        return Err(format!(
            "InputPortConsumerSpec.accepted_roles_json: {} roles exceeds limit {MAX_ACCEPTED_ROLES}",
            items.len()
        ));
    }

    let mut unique = BTreeSet::new();
    for value in items {
        let role = value.as_str().ok_or_else(|| {
            "InputPortConsumerSpec.accepted_roles_json: every entry must be a string".to_string()
        })?;
        if !role_valid(role) {
            return Err(format!(
                "InputPortConsumerSpec.accepted_roles_json: invalid role {role}"
            ));
        }
        unique.insert(role.to_string());
    }
    Ok(unique.into_iter().collect())
}

#[wasm_bindgen]
pub struct InputPortConsumerSpec {
    consumer_id: String,
    accepted_roles: Vec<String>,
    allow_extension_roles: bool,
    require_fingerprint: bool,
    minimum_revision: u64,
}

impl InputPortConsumerSpec {
    fn role_matches(&self, role: &str) -> bool {
        self.accepted_roles.iter().any(|value| value == role)
            || (self.allow_extension_roles && role.starts_with("x-"))
    }

    fn json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.input-port-consumer-spec.v1\",",
                "\"consumer_id\":\"{}\",",
                "\"accepted_roles\":{},",
                "\"allow_extension_roles\":{},",
                "\"require_fingerprint\":{},",
                "\"minimum_revision\":{}",
                "}}"
            ),
            json_escape(&self.consumer_id),
            string_array_json(&self.accepted_roles),
            bool_json(self.allow_extension_roles),
            bool_json(self.require_fingerprint),
            self.minimum_revision,
        )
    }
}

#[wasm_bindgen]
impl InputPortConsumerSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(
        consumer_id: String,
        accepted_roles_json: String,
        allow_extension_roles: bool,
        require_fingerprint: bool,
        minimum_revision: u64,
    ) -> Result<InputPortConsumerSpec, String> {
        if consumer_id.is_empty() {
            return Err("InputPortConsumerSpec.consumer_id: value must be non-empty".to_string());
        }
        if consumer_id.len() > MAX_CONSUMER_ID_BYTES {
            return Err(format!(
                "InputPortConsumerSpec.consumer_id: {} bytes exceeds limit {MAX_CONSUMER_ID_BYTES}",
                consumer_id.len()
            ));
        }
        let accepted_roles = parse_roles(&accepted_roles_json)?;
        Ok(Self {
            consumer_id,
            accepted_roles,
            allow_extension_roles,
            require_fingerprint,
            minimum_revision,
        })
    }

    #[wasm_bindgen(js_name = consumerId)]
    pub fn consumer_id(&self) -> String {
        self.consumer_id.clone()
    }

    #[wasm_bindgen(js_name = acceptedRoles)]
    pub fn accepted_roles(&self) -> String {
        string_array_json(&self.accepted_roles)
    }

    #[wasm_bindgen(js_name = allowExtensionRoles)]
    pub fn allow_extension_roles(&self) -> bool {
        self.allow_extension_roles
    }

    #[wasm_bindgen(js_name = requireFingerprint)]
    pub fn require_fingerprint(&self) -> bool {
        self.require_fingerprint
    }

    #[wasm_bindgen(js_name = minimumRevision)]
    pub fn minimum_revision(&self) -> u64 {
        self.minimum_revision
    }

    #[wasm_bindgen(js_name = describe)]
    pub fn describe(&self) -> String {
        self.json()
    }
}

#[wasm_bindgen(js_name = inputPortConsumerCapabilities)]
pub fn input_port_consumer_capabilities() -> String {
    INPUT_PORT_CONSUMER_V1.to_string()
}

#[wasm_bindgen(js_name = inputPortConsumerCompatibility)]
pub fn input_port_consumer_compatibility(
    workspace: &AgentWorkspace,
    consumer: &InputPortConsumerSpec,
) -> String {
    let Some(metadata) = workspace.input_port_metadata() else {
        return format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.input-port-consumer-result.v1\",",
                "\"status\":\"unknown\",",
                "\"compatible\":null,",
                "\"execution_authorized\":false,",
                "\"decision_authority\":\"agent\",",
                "\"consumer\":{},",
                "\"reason\":\"semantic_port_unbound\"",
                "}}"
            ),
            consumer.json(),
        );
    };

    let role_match = consumer.role_matches(&metadata.role);
    let fingerprint_present = !metadata.fingerprint.is_empty();
    let fingerprint_ok = !consumer.require_fingerprint || fingerprint_present;
    let revision_ok =
        consumer.minimum_revision == 0 || metadata.revision >= consumer.minimum_revision;
    let compatible = role_match && fingerprint_ok && revision_ok;

    let mut reasons = Vec::<String>::new();
    if !role_match {
        reasons.push("role_not_accepted".to_string());
    }
    if !fingerprint_ok {
        reasons.push("fingerprint_required".to_string());
    }
    if !revision_ok {
        reasons.push("revision_too_old".to_string());
    }

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-consumer-result.v1\",",
            "\"status\":\"{}\",",
            "\"compatible\":{},",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\",",
            "\"consumer\":{},",
            "\"input_port\":{{",
                "\"slot\":0,",
                "\"role\":\"{}\",",
                "\"provenance\":{{",
                    "\"source\":\"{}\",",
                    "\"revision\":{},",
                    "\"fingerprint_present\":{}",
                "}}",
            "}},",
            "\"predicates\":{{",
                "\"role_match\":{},",
                "\"fingerprint_ok\":{},",
                "\"revision_ok\":{}",
            "}},",
            "\"reasons\":{}",
            "}}"
        ),
        if compatible { "compatible" } else { "incompatible" },
        bool_json(compatible),
        consumer.json(),
        json_escape(&metadata.role),
        json_escape(&metadata.source),
        metadata.revision,
        bool_json(fingerprint_present),
        bool_json(role_match),
        bool_json(fingerprint_ok),
        bool_json(revision_ok),
        string_array_json(&reasons),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        input_port_consumer_capabilities, input_port_consumer_compatibility, InputPortConsumerSpec,
    };
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::workspace::AgentWorkspace;

    #[test]
    fn unbound_port_produces_unknown_not_rejection() {
        let workspace = AgentWorkspace::new(2).unwrap();
        let consumer =
            InputPortConsumerSpec::new("feature-extractor".into(), "[\"observation\"]".into(), false, false, 0)
                .unwrap();
        let result: serde_json::Value =
            serde_json::from_str(&input_port_consumer_compatibility(&workspace, &consumer)).unwrap();
        assert_eq!(result["status"], "unknown");
        assert!(result["compatible"].is_null());
        assert_eq!(result["execution_authorized"], false);
    }

    #[test]
    fn role_and_provenance_predicates_are_operational() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "fnv1a64:abcd".into(),
        )
        .unwrap();

        let compatible = InputPortConsumerSpec::new(
            "feature-extractor".into(),
            "[\"feature\",\"observation\"]".into(),
            false,
            true,
            10,
        )
        .unwrap();
        let result: serde_json::Value =
            serde_json::from_str(&input_port_consumer_compatibility(&workspace, &compatible)).unwrap();
        assert_eq!(result["status"], "compatible");
        assert_eq!(result["predicates"]["role_match"], true);
        assert_eq!(result["predicates"]["fingerprint_ok"], true);
        assert_eq!(result["predicates"]["revision_ok"], true);

        let stale = InputPortConsumerSpec::new(
            "fresh-only".into(),
            "[\"observation\"]".into(),
            false,
            true,
            19,
        )
        .unwrap();
        let stale_result: serde_json::Value =
            serde_json::from_str(&input_port_consumer_compatibility(&workspace, &stale)).unwrap();
        assert_eq!(stale_result["status"], "incompatible");
        assert_eq!(stale_result["predicates"]["revision_ok"], false);
    }

    #[test]
    fn extension_role_policy_is_explicit() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "x-market-regime".into(),
            "regime-model".into(),
            4,
            String::new(),
        )
        .unwrap();

        let strict = InputPortConsumerSpec::new(
            "strict".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        let strict_result: serde_json::Value =
            serde_json::from_str(&input_port_consumer_compatibility(&workspace, &strict)).unwrap();
        assert_eq!(strict_result["status"], "incompatible");

        let extensible = InputPortConsumerSpec::new(
            "extension-aware".into(),
            "[\"observation\"]".into(),
            true,
            false,
            0,
        )
        .unwrap();
        let extensible_result: serde_json::Value =
            serde_json::from_str(&input_port_consumer_compatibility(&workspace, &extensible)).unwrap();
        assert_eq!(extensible_result["status"], "compatible");
    }

    #[test]
    fn capability_keeps_advisory_boundary_explicit() {
        let caps: serde_json::Value =
            serde_json::from_str(&input_port_consumer_capabilities()).unwrap();
        assert_eq!(caps["scope"]["enforcement"], "advisory_preflight_only");
        assert_eq!(caps["compatibility"]["execution_authorized"], false);
        assert_eq!(caps["compatibility"]["decision_authority"], "agent");
    }
}
