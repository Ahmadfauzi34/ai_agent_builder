use wasm_bindgen::prelude::*;

use crate::graph::CompiledMultiInputGraph;
use crate::input_port::role_valid;
use crate::input_port_consumer::InputPortConsumerSpec;
use crate::multi_input_graph::{InputPreflight, MultiInputGraphPlan, MultiInputInputBundle, MultiInputPortContract};
use crate::registry::LayerRegistry;
use crate::semantic_ingress_manifest::validate_logical_port_id;
use crate::WasmTensor;

const CONTRACT: &str = include_str!("../docs/semantic-ingress-manifest.v2.json");
const MAX_SOURCE_BYTES: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backing {
    Slot(u8),
    Deferred,
}

#[derive(Clone, PartialEq, Eq)]
struct LogicalPort {
    id: String,
    role: String,
    expected_source: Option<String>,
    backing: Backing,
    required: bool,
}

#[wasm_bindgen]
pub struct SemanticIngressManifestV2 {
    plan: MultiInputGraphPlan,
    plan_bytes: Vec<u8>,
    ports: Vec<LogicalPort>,
}

struct IngressStatus {
    ready: bool,
    json: String,
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

fn json_string(value: &str) -> String {
    format!("\"{}\"", json_escape(value))
}

fn bool_json(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn bytes_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn logical_port_json(port: &LogicalPort) -> String {
    match port.backing {
        Backing::Slot(slot) => format!(
            "{{\"logical_port_id\":{},\"role\":{},\"backing\":\"graph_input_slot\",\"slot\":{},\"expected_source\":{},\"required\":true}}",
            json_string(&port.id),
            json_string(&port.role),
            slot,
            json_string(port.expected_source.as_deref().unwrap_or("")),
        ),
        Backing::Deferred => format!(
            "{{\"logical_port_id\":{},\"role\":{},\"backing\":\"deferred\",\"required\":{}}}",
            json_string(&port.id),
            json_string(&port.role),
            bool_json(port.required),
        ),
    }
}

impl SemanticIngressManifestV2 {
    fn add_port(&mut self, port: LogicalPort) -> Result<bool, String> {
        if let Some(existing) = self.ports.iter().find(|existing| existing.id == port.id) {
            if existing == &port {
                return Ok(false);
            }
            return Err(format!("SemanticIngressManifestV2: logical port {} has a conflicting declaration", port.id));
        }
        if let Backing::Slot(slot) = port.backing {
            if self.ports.iter().any(|existing| existing.backing == Backing::Slot(slot)) {
                return Err(format!("SemanticIngressManifestV2: graph input slot {slot} is already mapped"));
            }
        }
        self.ports.push(port);
        self.ports.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(true)
    }

    fn runtime_port(&self, slot: u8) -> Option<&LogicalPort> {
        self.ports.iter().find(|port| port.backing == Backing::Slot(slot))
    }

    fn manifest_fingerprint_internal(&self) -> String {
        let mut bytes = b"semantic-ingress-manifest.v2".to_vec();
        bytes.extend_from_slice(&(self.plan_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.plan_bytes);
        for port in &self.ports {
            match port.backing {
                Backing::Slot(slot) => bytes.extend_from_slice(&[0, slot]),
                Backing::Deferred => bytes.push(1),
            }
            for field in [&port.id, &port.role, port.expected_source.as_ref().unwrap_or(&port.id)] {
                bytes.extend_from_slice(&(field.len() as u32).to_le_bytes());
                bytes.extend_from_slice(field.as_bytes());
            }
            bytes.push(u8::from(port.required));
        }
        fingerprint(&bytes)
    }

    fn runtime_port_status(
        &self,
        contract: &MultiInputPortContract,
        bundle: &MultiInputInputBundle,
        input: &InputPreflight,
        bundle_plan_matches: bool,
    ) -> (String, bool) {
        let mapped = self.runtime_port(contract.slot);
        let bound = bundle.bound_input(contract.slot);
        let contract_ready = input.port_checks.iter().any(|(slot, ready)| *slot == contract.slot && *ready);
        let source_matches = mapped.and_then(|port| port.expected_source.as_deref())
            .zip(bound.map(|actual| actual.source.as_str()))
            .is_some_and(|(expected, actual)| expected == actual);
        let state = if mapped.is_none() {
            "runtime_backing_missing"
        } else if !bundle_plan_matches {
            "bundle_plan_mismatch"
        } else if bound.is_none() {
            "runtime_input_unbound"
        } else if !source_matches {
            "source_mismatch"
        } else if !contract_ready {
            "contract_mismatch"
        } else {
            "runtime_backing_current"
        };
        let ready = state == "runtime_backing_current";
        let json = format!(
            "{{\"logical_port_id\":{},\"slot\":{},\"role\":{},\"shape\":[{},{},{},{}],\"layout\":{},\"expected_source\":{},\"actual_source\":{},\"revision\":{},\"fingerprint_present\":{},\"input_value_fingerprint\":{},\"contract_ready\":{},\"source_matches\":{},\"status\":{},\"required\":true}}",
            mapped.map_or("null".to_string(), |port| json_string(&port.id)),
            contract.slot,
            json_string(&contract.role),
            contract.shape[0], contract.shape[1], contract.shape[2], contract.shape[3],
            json_string(&contract.layout),
            mapped.and_then(|port| port.expected_source.as_deref()).map_or("null".to_string(), json_string),
            bound.map_or("null".to_string(), |actual| json_string(&actual.source)),
            bound.map_or("null".to_string(), |actual| actual.revision.to_string()),
            bool_json(bound.is_some_and(|actual| !actual.fingerprint.is_empty())),
            bound.map_or("null".to_string(), |actual| json_string(&actual.value_fingerprint)),
            bool_json(contract_ready),
            bool_json(source_matches),
            json_string(state),
        );
        (json, ready)
    }

    fn status_internal(
        &self,
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        bundle: &MultiInputInputBundle,
    ) -> IngressStatus {
        let input = bundle.input_preflight(&self.plan);
        let bundle_plan_matches = bundle.matches_plan_internal(&self.plan);
        let graph_plan_matches = graph.input_plan_v1() == self.plan_bytes;
        let registry_binding_current = graph.validate_registry_binding(registry).is_ok();
        let mut runtime_coverage_complete = input.ready && bundle_plan_matches;
        let mut ports = Vec::with_capacity(self.ports.len().max(self.plan.ports().len()));
        for contract in self.plan.ports() {
            let (report, ready) = self.runtime_port_status(contract, bundle, &input, bundle_plan_matches);
            runtime_coverage_complete &= ready;
            ports.push(report);
        }
        for port in self.ports.iter().filter(|port| port.backing == Backing::Deferred) {
            runtime_coverage_complete &= !port.required;
            ports.push(format!(
                "{{\"logical_port_id\":{},\"role\":{},\"status\":\"deferred_no_runtime_backing\",\"required\":{}}}",
                json_string(&port.id), json_string(&port.role), bool_json(port.required),
            ));
        }
        let ready = runtime_coverage_complete && graph_plan_matches && registry_binding_current;
        let json = format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-ingress-manifest-status.v2\",\"manifest_fingerprint\":{},\"plan_fingerprint\":{},\"graph_plan_matches\":{},\"bundle_plan_matches\":{},\"registry_binding_current\":{},\"runtime_coverage_complete\":{},\"ready\":{},\"execution_authorized\":false,\"mutation\":\"none\",\"ports\":[{}],\"graph_preflight\":{}}}",
            json_string(&self.manifest_fingerprint_internal()),
            json_string(&self.plan.fingerprint_internal().unwrap_or_default()),
            bool_json(graph_plan_matches),
            bool_json(bundle_plan_matches),
            bool_json(registry_binding_current),
            bool_json(runtime_coverage_complete),
            bool_json(ready),
            ports.join(","),
            graph.preflight(registry, bundle),
        );
        IngressStatus { ready, json }
    }
}

#[wasm_bindgen]
impl SemanticIngressManifestV2 {
    #[wasm_bindgen(constructor)]
    pub fn new(plan: &MultiInputGraphPlan) -> Result<SemanticIngressManifestV2, String> {
        let plan_bytes = plan.validate_for_compile()?;
        Ok(Self { plan: plan.clone(), plan_bytes, ports: Vec::new() })
    }

    #[wasm_bindgen(js_name = addRuntimePort)]
    pub fn add_runtime_port(&mut self, logical_port_id: String, slot: u8, expected_source: String) -> Result<bool, String> {
        validate_logical_port_id(&logical_port_id)?;
        if expected_source.is_empty() || expected_source.len() > MAX_SOURCE_BYTES {
            return Err(format!("SemanticIngressManifestV2.expected_source: value must be 1..={MAX_SOURCE_BYTES} bytes"));
        }
        let contract = self.plan.ports().iter().find(|port| port.slot == slot)
            .ok_or_else(|| format!("SemanticIngressManifestV2: slot {slot} is not declared in the multi-input graph plan"))?;
        self.add_port(LogicalPort {
            id: logical_port_id,
            role: contract.role.clone(),
            expected_source: Some(expected_source),
            backing: Backing::Slot(slot),
            required: true,
        })
    }

    #[wasm_bindgen(js_name = addDeferredPort)]
    pub fn add_deferred_port(&mut self, logical_port_id: String, role: String, required: bool) -> Result<bool, String> {
        validate_logical_port_id(&logical_port_id)?;
        if !role_valid(&role) {
            return Err(format!("SemanticIngressManifestV2: invalid deferred role {role}"));
        }
        self.add_port(LogicalPort { id: logical_port_id, role, expected_source: None, backing: Backing::Deferred, required })
    }

    #[wasm_bindgen(js_name = manifestFingerprint)]
    pub fn manifest_fingerprint(&self) -> String {
        self.manifest_fingerprint_internal()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-ingress-manifest-instance.v2\",\"plan_hex\":{},\"manifest_fingerprint\":{},\"ports\":[{}],\"execution_authorized\":false}}",
            json_string(&bytes_hex(&self.plan_bytes)),
            json_string(&self.manifest_fingerprint_internal()),
            self.ports.iter().map(logical_port_json).collect::<Vec<_>>().join(","),
        )
    }

    #[wasm_bindgen(js_name = inputPortStatus)]
    pub fn input_port_status(&self, slot: u8, bundle: &MultiInputInputBundle) -> Result<String, String> {
        let contract = self.plan.ports().iter().find(|port| port.slot == slot)
            .ok_or_else(|| format!("SemanticIngressManifestV2.inputPortStatus: undeclared slot {slot}"))?;
        let matches = bundle.matches_plan_internal(&self.plan);
        let input = bundle.input_preflight(&self.plan);
        let (port, _) = self.runtime_port_status(contract, bundle, &input, matches);
        Ok(format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-input-port-status.v2\",\"bundle_plan_matches\":{},\"execution_authorized\":false,\"port\":{}}}",
            bool_json(matches), port,
        ))
    }

    #[wasm_bindgen(js_name = consumerCompatibility)]
    pub fn consumer_compatibility(&self, slot: u8, bundle: &MultiInputInputBundle, consumer: &InputPortConsumerSpec) -> Result<String, String> {
        let contract = self.plan.ports().iter().find(|port| port.slot == slot)
            .ok_or_else(|| format!("SemanticIngressManifestV2.consumerCompatibility: undeclared slot {slot}"))?;
        let mapped = self.runtime_port(slot).ok_or_else(|| format!("SemanticIngressManifestV2.consumerCompatibility: slot {slot} has no logical port mapping"))?;
        let input = bundle.input_preflight(&self.plan);
        let bundle_plan_matches = bundle.matches_plan_internal(&self.plan);
        let (port_status, port_ready) = self.runtime_port_status(contract, bundle, &input, bundle_plan_matches);
        let bound = bundle.bound_input(slot);
        let role_match = bound.is_some_and(|actual| consumer.role_matches(&actual.role));
        let fingerprint_ok = bound.is_some_and(|actual| !consumer.require_fingerprint_value() || !actual.fingerprint.is_empty());
        let revision_ok = bound.is_some_and(|actual| actual.revision >= consumer.minimum_revision_value());
        let compatible = port_ready && role_match && fingerprint_ok && revision_ok;
        let status = if bound.is_none() { "unknown" } else if compatible { "compatible" } else { "incompatible" };
        Ok(format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-input-consumer-compatibility.v2\",\"logical_port_id\":{},\"consumer\":{},\"input_port\":{},\"status\":{},\"compatible\":{},\"predicates\":{{\"port_ready\":{},\"role_match\":{},\"fingerprint_ok\":{},\"revision_ok\":{}}},\"execution_authorized\":false,\"decision_authority\":\"agent\"}}",
            json_string(&mapped.id),
            consumer.json(),
            port_status,
            json_string(status),
            if bound.is_none() { "null" } else { bool_json(compatible) },
            bool_json(port_ready), bool_json(role_match), bool_json(fingerprint_ok), bool_json(revision_ok),
        ))
    }

    #[wasm_bindgen(js_name = status)]
    pub fn status(&self, registry: &LayerRegistry, graph: &CompiledMultiInputGraph, bundle: &MultiInputInputBundle) -> String {
        self.status_internal(registry, graph, bundle).json
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(&self, registry: &LayerRegistry, graph: &CompiledMultiInputGraph, bundle: &MultiInputInputBundle) -> Result<WasmTensor, String> {
        if !self.status_internal(registry, graph, bundle).ready {
            return Err("SemanticIngressManifestV2.run: ingress or graph preflight failed; execution was not started".into());
        }
        graph.run(registry, bundle)
    }

    #[wasm_bindgen(js_name = verifyFlat)]
    pub fn verify_flat(&self, registry: &LayerRegistry, graph: &CompiledMultiInputGraph, bundle: &MultiInputInputBundle, candidate: &[f32], abs_tol: f64, rel_tol: f64) -> Result<String, String> {
        let status = self.status_internal(registry, graph, bundle);
        if !status.ready {
            return Err("SemanticIngressManifestV2.verifyFlat: ingress or graph preflight failed; execution was not started".into());
        }
        let verification = graph.verify_flat(registry, bundle, candidate, abs_tol, rel_tol)?;
        Ok(format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-ingress-verification.v2\",\"ingress\":{},\"reference\":{}}}",
            status.json, verification,
        ))
    }
}

#[wasm_bindgen(js_name = semanticIngressManifestV2Capabilities)]
pub fn semantic_ingress_manifest_v2_capabilities() -> String {
    CONTRACT.to_string()
}

#[cfg(test)]
mod tests {
    use super::SemanticIngressManifestV2;
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::multi_input_graph::{MultiInputGraphPlan, MultiInputInputBundle};
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn setup() -> (LayerRegistry, MultiInputGraphPlan, crate::graph::CompiledMultiInputGraph) {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(21);
        registry.init_agent_layer(&add).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
        plan.add_input_port(0, "observation".into(), 1, 2, 1, 1, "feature_axis1_singleton".into(), true, 2).unwrap();
        plan.add_input_port(1, "state".into(), 1, 2, 1, 1, "feature_axis1_singleton".into(), true, 3).unwrap();
        let graph = registry.compile_multi_input_graph(&plan).unwrap();
        (registry, plan, graph)
    }

    fn status(manifest: &SemanticIngressManifestV2, registry: &LayerRegistry, graph: &crate::graph::CompiledMultiInputGraph, bundle: &MultiInputInputBundle) -> serde_json::Value {
        serde_json::from_str(&manifest.status(registry, graph, bundle)).unwrap()
    }

    #[test]
    fn source_and_port_coverage_are_real_run_gates() {
        let (registry, plan, graph) = setup();
        let mut manifest = SemanticIngressManifestV2::new(&plan).unwrap();
        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        assert!(!status(&manifest, &registry, &graph, &bundle)["ready"].as_bool().unwrap());
        assert!(manifest.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap());
        assert!(!manifest.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap());
        assert!(manifest.add_runtime_port("memory".into(), 1, "memory".into()).unwrap());
        assert!(manifest.add_runtime_port("other".into(), 1, "memory".into()).is_err());
        assert!(manifest.add_runtime_port("ghost".into(), 2, "sensor".into()).is_err());

        let left = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let right = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        bundle.bind_input(0, &left, "observation".into(), "feature_axis1_singleton".into(), "sensor".into(), 2, "obs".into()).unwrap();
        bundle.bind_input(1, &right, "state".into(), "feature_axis1_singleton".into(), "wrong-source".into(), 3, "state".into()).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&graph.preflight(&registry, &bundle)).unwrap()["ready"].as_bool().unwrap());
        let mismatch = status(&manifest, &registry, &graph, &bundle);
        assert_eq!(mismatch["ports"][1]["status"], "source_mismatch");
        assert_eq!(mismatch["ready"], false);
        assert!(manifest.run(&registry, &graph, &bundle).is_err());

        bundle.clear_input(1);
        bundle.bind_input(1, &right, "state".into(), "feature_axis1_singleton".into(), "memory".into(), 3, "state".into()).unwrap();
        let ready = status(&manifest, &registry, &graph, &bundle);
        assert_eq!(ready["ready"], true);
        assert_eq!(ready["execution_authorized"], false);
        assert_eq!(manifest.run(&registry, &graph, &bundle).unwrap().to_array(), vec![4.0, 6.0]);
        let verified: serde_json::Value = serde_json::from_str(&manifest.verify_flat(&registry, &graph, &bundle, &[4.0, 6.0], 1e-6, 1e-6).unwrap()).unwrap();
        assert_eq!(verified["reference"]["verification"]["passed"], true);
        assert_eq!(verified["ingress"]["ready"], true);
    }

    #[test]
    fn deferred_and_consumer_status_preserve_ignorance_and_policy() {
        let (registry, plan, graph) = setup();
        let mut manifest = SemanticIngressManifestV2::new(&plan).unwrap();
        manifest.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap();
        manifest.add_runtime_port("memory".into(), 1, "memory".into()).unwrap();
        manifest.add_deferred_port("objective".into(), "x-objective".into(), false).unwrap();
        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        let consumer = InputPortConsumerSpec::new("state-consumer".into(), "[\"state\"]".into(), false, true, 3).unwrap();
        let unknown: serde_json::Value = serde_json::from_str(&manifest.consumer_compatibility(1, &bundle, &consumer).unwrap()).unwrap();
        assert_eq!(unknown["status"], "unknown");
        assert!(unknown["compatible"].is_null());

        let left = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let right = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        bundle.bind_input(0, &left, "observation".into(), "feature_axis1_singleton".into(), "sensor".into(), 2, "obs".into()).unwrap();
        bundle.bind_input(1, &right, "state".into(), "feature_axis1_singleton".into(), "memory".into(), 3, "state".into()).unwrap();
        assert_eq!(status(&manifest, &registry, &graph, &bundle)["ready"], true);
        let compatible: serde_json::Value = serde_json::from_str(&manifest.consumer_compatibility(1, &bundle, &consumer).unwrap()).unwrap();
        assert_eq!(compatible["status"], "compatible");
        let wrong = InputPortConsumerSpec::new("observation-only".into(), "[\"observation\"]".into(), false, true, 3).unwrap();
        let incompatible: serde_json::Value = serde_json::from_str(&manifest.consumer_compatibility(1, &bundle, &wrong).unwrap()).unwrap();
        assert_eq!(incompatible["status"], "incompatible");

        let mut required = SemanticIngressManifestV2::new(&plan).unwrap();
        required.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap();
        required.add_runtime_port("memory".into(), 1, "memory".into()).unwrap();
        required.add_deferred_port("objective".into(), "x-objective".into(), true).unwrap();
        assert_eq!(status(&required, &registry, &graph, &bundle)["runtime_coverage_complete"], false);
        assert!(required.run(&registry, &graph, &bundle).is_err());
    }

    #[test]
    fn exact_plan_and_registry_drift_reject_manifest_execution() {
        let (mut registry, plan, graph) = setup();
        let mut manifest = SemanticIngressManifestV2::new(&plan).unwrap();
        manifest.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap();
        manifest.add_runtime_port("memory".into(), 1, "memory".into()).unwrap();
        let mut reversed = SemanticIngressManifestV2::new(&plan).unwrap();
        reversed.add_runtime_port("memory".into(), 1, "memory".into()).unwrap();
        reversed.add_runtime_port("observation".into(), 0, "sensor".into()).unwrap();
        assert_eq!(manifest.manifest_fingerprint(), reversed.manifest_fingerprint());

        let replacement = AgentLayerSpec::sub(21);
        registry.init_agent_layer(&replacement).unwrap();
        let bundle = MultiInputInputBundle::new(&plan).unwrap();
        assert_eq!(status(&manifest, &registry, &graph, &bundle)["registry_binding_current"], false);
        assert!(manifest.run(&registry, &graph, &bundle).is_err());

        let other_layer = AgentLayerSpec::sub(22);
        registry.init_agent_layer(&other_layer).unwrap();
        let mut other = AgentGraphBuilder::new(3).unwrap();
        other.add_binary(&other_layer, 0, 1, 2).unwrap();
        other.set_output(2).unwrap();
        let mut other_plan = MultiInputGraphPlan::new(&other).unwrap();
        other_plan.add_input_port(0, "observation".into(), 1, 2, 1, 1, "feature_axis1_singleton".into(), true, 2).unwrap();
        other_plan.add_input_port(1, "state".into(), 1, 2, 1, 1, "feature_axis1_singleton".into(), true, 3).unwrap();
        let other_graph = registry.compile_multi_input_graph(&other_plan).unwrap();
        assert_eq!(status(&manifest, &registry, &other_graph, &bundle)["graph_plan_matches"], false);
        let other_bundle = MultiInputInputBundle::new(&other_plan).unwrap();
        assert_eq!(status(&manifest, &registry, &other_graph, &other_bundle)["bundle_plan_matches"], false);
    }
}
