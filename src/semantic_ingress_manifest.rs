use wasm_bindgen::prelude::*;

use crate::input_port::role_valid;
use crate::workspace::AgentWorkspace;

const SEMANTIC_INGRESS_MANIFEST_V1: &str =
    include_str!("../docs/semantic-ingress-manifest.v1.json");
const MAX_LOGICAL_PORT_ID_BYTES: usize = 64;
const MAX_SOURCE_BYTES: usize = 256;
const MAX_FINGERPRINT_BYTES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeBacking {
    Slot0,
    Deferred,
}

impl RuntimeBacking {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Slot0 => "graph_input_slot0",
            Self::Deferred => "deferred",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SemanticIngressPort {
    logical_port_id: String,
    role: String,
    source: String,
    revision: u64,
    fingerprint: String,
    required: bool,
    runtime_backing: RuntimeBacking,
}

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

fn validate_text(
    value: &str,
    max_bytes: usize,
    context: &str,
    allow_empty: bool,
) -> Result<(), String> {
    if !allow_empty && value.is_empty() {
        return Err(format!("{context}: value must be non-empty"));
    }
    if value.len() > max_bytes {
        return Err(format!(
            "{context}: {} bytes exceeds limit {max_bytes}",
            value.len()
        ));
    }
    Ok(())
}

fn validate_logical_port_id(value: &str) -> Result<(), String> {
    validate_text(
        value,
        MAX_LOGICAL_PORT_ID_BYTES,
        "SemanticIngressManifest.logical_port_id",
        false,
    )?;
    if !value.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'-' | b'_' | b'.')
    }) {
        return Err(
            "SemanticIngressManifest.logical_port_id: use lowercase ascii letters, digits, '-', '_' or '.'"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_port_fields(
    logical_port_id: &str,
    role: &str,
    source: &str,
    fingerprint: &str,
) -> Result<(), String> {
    validate_logical_port_id(logical_port_id)?;
    if !role_valid(role) {
        return Err(format!(
            "SemanticIngressManifest.role: unsupported role {role}; use canonical role or x- extension namespace"
        ));
    }
    validate_text(
        source,
        MAX_SOURCE_BYTES,
        "SemanticIngressManifest.source",
        false,
    )?;
    validate_text(
        fingerprint,
        MAX_FINGERPRINT_BYTES,
        "SemanticIngressManifest.fingerprint",
        true,
    )?;
    Ok(())
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn length_prefixed(value: &str) -> String {
    format!("{}:{value}", value.len())
}

fn canonical_port(port: &SemanticIngressPort) -> String {
    format!(
        "id={}|role={}|source={}|revision={}|fingerprint={}|required={}|backing={}",
        length_prefixed(&port.logical_port_id),
        length_prefixed(&port.role),
        length_prefixed(&port.source),
        port.revision,
        length_prefixed(&port.fingerprint),
        port.required,
        port.runtime_backing.as_str(),
    )
}

fn port_json(port: &SemanticIngressPort) -> String {
    format!(
        concat!(
            "{{",
            "\"logical_port_id\":\"{}\",",
            "\"role\":\"{}\",",
            "\"provenance\":{{",
                "\"source\":\"{}\",",
                "\"revision\":{},",
                "\"fingerprint\":{}",
            "}},",
            "\"required\":{},",
            "\"runtime_backing\":\"{}\"" ,
            "}}"
        ),
        json_escape(&port.logical_port_id),
        json_escape(&port.role),
        json_escape(&port.source),
        port.revision,
        if port.fingerprint.is_empty() {
            "null".to_string()
        } else {
            format!("\"{}\"", json_escape(&port.fingerprint))
        },
        bool_json(port.required),
        port.runtime_backing.as_str(),
    )
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct SemanticIngressManifest {
    ports: Vec<SemanticIngressPort>,
}

impl SemanticIngressManifest {
    fn add_port(&mut self, port: SemanticIngressPort) -> Result<bool, String> {
        if let Some(existing) = self
            .ports
            .iter()
            .find(|existing| existing.logical_port_id == port.logical_port_id)
        {
            if existing == &port {
                return Ok(false);
            }
            return Err(format!(
                "SemanticIngressManifest: logical port {} already exists with different declaration",
                port.logical_port_id
            ));
        }

        if port.runtime_backing == RuntimeBacking::Slot0
            && self
                .ports
                .iter()
                .any(|existing| existing.runtime_backing == RuntimeBacking::Slot0)
        {
            return Err(
                "SemanticIngressManifest: current graph runtime supports only one runtime-backed external input (slot 0)"
                    .to_string(),
            );
        }

        self.ports.push(port);
        Ok(true)
    }

    fn manifest_fingerprint_internal(&self) -> String {
        let mut canonical_ports = self
            .ports
            .iter()
            .map(canonical_port)
            .collect::<Vec<_>>();
        canonical_ports.sort();
        let canonical = format!(
            "v1|runtime_policy=single_graph_input_slot0|ports={}|{}",
            canonical_ports.len(),
            canonical_ports.join("|")
        );
        fnv1a64(canonical.bytes())
    }

    fn json_internal(&self) -> String {
        let ports = self
            .ports
            .iter()
            .map(port_json)
            .collect::<Vec<_>>()
            .join(",");
        let runtime_backed_count = self
            .ports
            .iter()
            .filter(|port| port.runtime_backing == RuntimeBacking::Slot0)
            .count();
        let deferred_count = self.ports.len().saturating_sub(runtime_backed_count);
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.semantic-ingress-manifest-instance.v1\",",
                "\"manifest_fingerprint\":\"{}\",",
                "\"port_count\":{},",
                "\"runtime_backed_port_count\":{},",
                "\"deferred_port_count\":{},",
                "\"runtime_policy\":\"single_graph_input_slot0\",",
                "\"ports\":[{}],",
                "\"execution_authorized\":false",
                "}}"
            ),
            self.manifest_fingerprint_internal(),
            self.ports.len(),
            runtime_backed_count,
            deferred_count,
            ports,
        )
    }
}

#[wasm_bindgen]
impl SemanticIngressManifest {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SemanticIngressManifest {
        Self { ports: Vec::new() }
    }

    #[wasm_bindgen(js_name = addRuntimeBackedPort)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_runtime_backed_port(
        &mut self,
        logical_port_id: String,
        role: String,
        source: String,
        revision: u64,
        fingerprint: String,
        runtime_slot: u32,
        required: bool,
    ) -> Result<bool, String> {
        if runtime_slot != 0 {
            return Err(format!(
                "SemanticIngressManifest.addRuntimeBackedPort: graph runtime currently backs only slot 0, got slot {runtime_slot}"
            ));
        }
        validate_port_fields(&logical_port_id, &role, &source, &fingerprint)?;
        self.add_port(SemanticIngressPort {
            logical_port_id,
            role,
            source,
            revision,
            fingerprint,
            required,
            runtime_backing: RuntimeBacking::Slot0,
        })
    }

    #[wasm_bindgen(js_name = addDeferredPort)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_deferred_port(
        &mut self,
        logical_port_id: String,
        role: String,
        source: String,
        revision: u64,
        fingerprint: String,
        required: bool,
    ) -> Result<bool, String> {
        validate_port_fields(&logical_port_id, &role, &source, &fingerprint)?;
        self.add_port(SemanticIngressPort {
            logical_port_id,
            role,
            source,
            revision,
            fingerprint,
            required,
            runtime_backing: RuntimeBacking::Deferred,
        })
    }

    #[wasm_bindgen(js_name = portCount)]
    pub fn port_count(&self) -> u32 {
        self.ports.len() as u32
    }

    #[wasm_bindgen(js_name = manifestFingerprint)]
    pub fn manifest_fingerprint(&self) -> String {
        self.manifest_fingerprint_internal()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        self.json_internal()
    }
}

fn workspace_port_status(workspace: &AgentWorkspace, port: &SemanticIngressPort) -> &'static str {
    match port.runtime_backing {
        RuntimeBacking::Deferred => "deferred_no_runtime_backing",
        RuntimeBacking::Slot0 => {
            let Some(metadata) = workspace.input_port_metadata() else {
                return "runtime_input_unbound";
            };
            if metadata.role == port.role
                && metadata.source == port.source
                && metadata.revision == port.revision
                && metadata.fingerprint == port.fingerprint
            {
                "runtime_backing_current"
            } else {
                "runtime_backing_drifted"
            }
        }
    }
}

#[wasm_bindgen(js_name = semanticIngressManifestCapabilities)]
pub fn semantic_ingress_manifest_capabilities() -> String {
    SEMANTIC_INGRESS_MANIFEST_V1.to_string()
}

#[wasm_bindgen(js_name = semanticIngressManifestStatus)]
pub fn semantic_ingress_manifest_status(
    workspace: &AgentWorkspace,
    manifest: &SemanticIngressManifest,
) -> String {
    let mut required_uncovered = 0usize;
    let mut runtime_backed_count = 0usize;
    let mut deferred_count = 0usize;

    let port_status = manifest
        .ports
        .iter()
        .map(|port| {
            let status = workspace_port_status(workspace, port);
            if port.runtime_backing == RuntimeBacking::Slot0 {
                runtime_backed_count += 1;
            } else {
                deferred_count += 1;
            }
            if port.required && status != "runtime_backing_current" {
                required_uncovered += 1;
            }
            format!(
                concat!(
                    "{{",
                    "\"logical_port_id\":\"{}\",",
                    "\"role\":\"{}\",",
                    "\"required\":{},",
                    "\"runtime_backing\":\"{}\",",
                    "\"status\":\"{}\"" ,
                    "}}"
                ),
                json_escape(&port.logical_port_id),
                json_escape(&port.role),
                bool_json(port.required),
                port.runtime_backing.as_str(),
                status,
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-ingress-manifest-status.v1\",",
            "\"manifest_fingerprint\":\"{}\",",
            "\"port_count\":{},",
            "\"runtime_backed_port_count\":{},",
            "\"deferred_port_count\":{},",
            "\"required_uncovered_count\":{},",
            "\"runtime_coverage_complete\":{},",
            "\"ports\":[{}],",
            "\"execution_authorized\":false,",
            "\"mutation\":\"none\"",
            "}}"
        ),
        manifest.manifest_fingerprint_internal(),
        manifest.ports.len(),
        runtime_backed_count,
        deferred_count,
        required_uncovered,
        bool_json(required_uncovered == 0),
        port_status,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        semantic_ingress_manifest_capabilities, semantic_ingress_manifest_status,
        SemanticIngressManifest,
    };
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::workspace::AgentWorkspace;

    #[test]
    fn manifest_can_describe_multi_source_ingress_without_claiming_multi_input_runtime() {
        let mut manifest = SemanticIngressManifest::new();
        assert!(manifest
            .add_runtime_backed_port(
                "observation".into(),
                "observation".into(),
                "market-feed".into(),
                18,
                "obs:18".into(),
                0,
                true,
            )
            .unwrap());
        assert!(manifest
            .add_deferred_port(
                "memory".into(),
                "state".into(),
                "agent-memory".into(),
                4,
                "mem:4".into(),
                true,
            )
            .unwrap());
        assert!(manifest
            .add_deferred_port(
                "objective".into(),
                "context".into(),
                "objective-store".into(),
                2,
                "obj:2".into(),
                false,
            )
            .unwrap());

        let json: serde_json::Value = serde_json::from_str(&manifest.to_json()).unwrap();
        assert_eq!(json["port_count"], 3);
        assert_eq!(json["runtime_backed_port_count"], 1);
        assert_eq!(json["deferred_port_count"], 2);
        assert_eq!(json["execution_authorized"], false);
    }

    #[test]
    fn runtime_backing_is_limited_to_real_graph_input_slot_zero() {
        let mut manifest = SemanticIngressManifest::new();
        assert!(manifest
            .add_runtime_backed_port(
                "observation".into(),
                "observation".into(),
                "sensor".into(),
                1,
                String::new(),
                1,
                true,
            )
            .is_err());

        manifest
            .add_runtime_backed_port(
                "observation".into(),
                "observation".into(),
                "sensor".into(),
                1,
                String::new(),
                0,
                true,
            )
            .unwrap();
        assert!(manifest
            .add_runtime_backed_port(
                "memory".into(),
                "state".into(),
                "memory".into(),
                1,
                String::new(),
                0,
                true,
            )
            .is_err());
    }

    #[test]
    fn status_reports_runtime_current_drift_and_deferred_coverage() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "market-feed".into(),
            18,
            "obs:18".into(),
        )
        .unwrap();

        let mut manifest = SemanticIngressManifest::new();
        manifest
            .add_runtime_backed_port(
                "observation".into(),
                "observation".into(),
                "market-feed".into(),
                18,
                "obs:18".into(),
                0,
                true,
            )
            .unwrap();
        manifest
            .add_deferred_port(
                "memory".into(),
                "state".into(),
                "agent-memory".into(),
                4,
                "mem:4".into(),
                true,
            )
            .unwrap();

        let status: serde_json::Value =
            serde_json::from_str(&semantic_ingress_manifest_status(&workspace, &manifest)).unwrap();
        assert_eq!(status["runtime_coverage_complete"], false);
        assert_eq!(status["required_uncovered_count"], 1);
        assert_eq!(status["ports"][0]["status"], "runtime_backing_current");
        assert_eq!(
            status["ports"][1]["status"],
            "deferred_no_runtime_backing"
        );

        workspace_bind_input_port_metadata(
            &mut workspace,
            "feature".into(),
            "market-feed".into(),
            19,
            "obs:19".into(),
        )
        .unwrap();
        let drifted: serde_json::Value =
            serde_json::from_str(&semantic_ingress_manifest_status(&workspace, &manifest)).unwrap();
        assert_eq!(drifted["ports"][0]["status"], "runtime_backing_drifted");
        assert_eq!(drifted["required_uncovered_count"], 2);
    }

    #[test]
    fn manifest_identity_is_order_independent_for_same_port_set() {
        let mut a = SemanticIngressManifest::new();
        a.add_deferred_port(
            "memory".into(),
            "state".into(),
            "memory".into(),
            1,
            String::new(),
            true,
        )
        .unwrap();
        a.add_runtime_backed_port(
            "observation".into(),
            "observation".into(),
            "sensor".into(),
            1,
            String::new(),
            0,
            true,
        )
        .unwrap();

        let mut b = SemanticIngressManifest::new();
        b.add_runtime_backed_port(
            "observation".into(),
            "observation".into(),
            "sensor".into(),
            1,
            String::new(),
            0,
            true,
        )
        .unwrap();
        b.add_deferred_port(
            "memory".into(),
            "state".into(),
            "memory".into(),
            1,
            String::new(),
            true,
        )
        .unwrap();

        assert_eq!(a.manifest_fingerprint(), b.manifest_fingerprint());
    }

    #[test]
    fn capability_contract_is_explicit_about_deferred_runtime_support() {
        let contract: serde_json::Value =
            serde_json::from_str(semantic_ingress_manifest_capabilities()).unwrap();
        assert_eq!(contract["execution"]["runtime_backed_external_ports"], 1);
        assert_eq!(contract["execution"]["runtime_backed_slot"], 0);
        assert_eq!(contract["execution"]["multi_input_execution"], "deferred");
        assert_eq!(contract["semantics"]["execution_authorized"], false);
    }
}
