//! Fasad WASM tunggal — domain `semantic` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::semantic::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentGraphSemanticLifecycleTransition;
use crate::graph::multi_input_graph::MultiInputGraphPlan;
use crate::graph::multi_input_graph::MultiInputInputBundle;
use crate::graph::CompiledGraph;
use crate::graph::CompiledMultiInputGraph;
use crate::graph::TracedMultiInputRun;
use crate::ingress::input_port::role_valid;
use crate::ingress::input_port_consumer::InputPortConsumerSpec;
use crate::registry::LayerRegistry;
use crate::semantic::semantic_execution_context::semantic_execution_context_for;
use crate::semantic::semantic_execution_context::SEMANTIC_EXECUTION_CONTEXT_V1;
use crate::semantic::semantic_ingress_manifest::bool_json;
use crate::semantic::semantic_ingress_manifest::json_escape;
use crate::semantic::semantic_ingress_manifest::validate_logical_port_id;
use crate::semantic::semantic_ingress_manifest::workspace_port_status;
use crate::semantic::semantic_ingress_manifest::RuntimeBacking;
use crate::semantic::semantic_ingress_manifest::SemanticIngressManifest;
use crate::semantic::semantic_ingress_manifest::SEMANTIC_INGRESS_MANIFEST_V1;
use crate::semantic::semantic_ingress_manifest_v2::bytes_hex;
use crate::semantic::semantic_ingress_manifest_v2::json_string;
use crate::semantic::semantic_ingress_manifest_v2::logical_port_json;
use crate::semantic::semantic_ingress_manifest_v2::Backing;
use crate::semantic::semantic_ingress_manifest_v2::LogicalPort;
use crate::semantic::semantic_ingress_manifest_v2::SemanticIngressManifestV2;
use crate::semantic::semantic_ingress_manifest_v2::CONTRACT;
use crate::semantic::semantic_ingress_manifest_v2::MAX_SOURCE_BYTES;
use crate::semantic::semantic_lifecycle::parse_roles;
use crate::semantic::semantic_lifecycle::resolve_input_lineage;
use crate::semantic::semantic_lifecycle::semantic_lifecycle_identity_json;
use crate::semantic::semantic_lifecycle::string_array_json;
use crate::semantic::semantic_lifecycle::transition_fingerprint;
use crate::semantic::semantic_lifecycle::transition_record_json;
use crate::semantic::semantic_lifecycle::validate_role;
use crate::semantic::semantic_lifecycle::SemanticTransitionSpec;
use crate::semantic::semantic_lifecycle::MAX_TRANSITION_ID_BYTES;
use crate::semantic::semantic_lifecycle::SEMANTIC_LIFECYCLE_V1;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

#[wasm_bindgen(js_name = semanticExecutionContextCapabilities)]
pub fn semantic_execution_context_capabilities() -> String {
    SEMANTIC_EXECUTION_CONTEXT_V1.to_string()
}

#[wasm_bindgen(js_name = semanticExecutionContext)]
pub fn semantic_execution_context(
    builder: &AgentGraphBuilder,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    semantic_execution_context_for(builder, graph, registry).map(|context| context.json())
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
                    "\"status\":\"{}\"",
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

#[wasm_bindgen(js_name = semanticIngressManifestV2Capabilities)]
pub fn semantic_ingress_manifest_v2_capabilities() -> String {
    CONTRACT.to_string()
}

#[wasm_bindgen(js_name = semanticLifecycleCapabilities)]
pub fn semantic_lifecycle_capabilities() -> String {
    SEMANTIC_LIFECYCLE_V1.to_string()
}

#[wasm_bindgen(js_name = bindSemanticLifecycleTransition)]
pub fn bind_semantic_lifecycle_transition(
    workspace: &AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    spec: &SemanticTransitionSpec,
    step_index: u32,
) -> Result<bool, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "bindSemanticLifecycleTransition: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let index = usize::try_from(step_index)
        .map_err(|_| "bindSemanticLifecycleTransition: step index conversion failed".to_string())?;
    let steps = builder.introspection_steps();
    let Some((arity, _, _, in_slot, in_slot2, out_slot)) = steps.get(index).copied() else {
        return Err(format!(
            "bindSemanticLifecycleTransition: step index {step_index} is outside num_steps {}",
            steps.len()
        ));
    };

    if spec.input_roles.len() != usize::from(arity) {
        return Err(format!(
            "bindSemanticLifecycleTransition: spec has {} input roles but step {step_index} arity is {arity}",
            spec.input_roles.len()
        ));
    }

    let mut inputs = Vec::with_capacity(usize::from(arity));
    if arity == 1 {
        inputs.push(resolve_input_lineage(
            workspace, builder, step_index, "input", in_slot,
        )?);
    } else {
        inputs.push(resolve_input_lineage(
            workspace, builder, step_index, "left", in_slot,
        )?);
        inputs.push(resolve_input_lineage(
            workspace, builder, step_index, "right", in_slot2,
        )?);
    }

    for (position, (declared, actual)) in spec
        .input_roles
        .iter()
        .zip(inputs.iter().map(|input| &input.role))
        .enumerate()
    {
        if declared != actual {
            return Err(format!(
                "bindSemanticLifecycleTransition: input role mismatch at position {position}; declared {declared}, lineage resolves to {actual}"
            ));
        }
    }

    let fingerprint = transition_fingerprint(
        step_index,
        &spec.transition_id,
        &inputs,
        out_slot,
        &spec.output_role,
    );

    builder.bind_semantic_lifecycle_transition(AgentGraphSemanticLifecycleTransition {
        step_index,
        transition_id: spec.transition_id.clone(),
        inputs,
        output_slot: out_slot,
        output_role: spec.output_role.clone(),
        transition_fingerprint: fingerprint,
    })
}

#[wasm_bindgen(js_name = semanticLifecycleIdentity)]
pub fn semantic_lifecycle_identity(builder: &AgentGraphBuilder) -> String {
    semantic_lifecycle_identity_json(builder)
}

#[wasm_bindgen(js_name = semanticLifecycleTransition)]
pub fn semantic_lifecycle_transition(
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<String, String> {
    let index = usize::try_from(step_index)
        .map_err(|_| "semanticLifecycleTransition: step index conversion failed".to_string())?;
    if index >= builder.introspection_steps().len() {
        return Err(format!(
            "semanticLifecycleTransition: step index {step_index} is outside num_steps {}",
            builder.num_steps()
        ));
    }

    match builder.semantic_lifecycle_transition(step_index) {
        Some(transition) => Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",\"status\":\"bound\",\"transition\":{}}}",
            transition_record_json(transition)
        )),
        None => Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.semantic-lifecycle-transition-status.v1\",\"status\":\"unbound\",\"step_index\":{step_index}}}"
        )),
    }
}

#[wasm_bindgen(js_name = semanticLifecycleProjection)]
pub fn semantic_lifecycle_projection(builder: &AgentGraphBuilder) -> String {
    let transitions = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(transition_record_json)
        .collect::<Vec<_>>()
        .join(",");

    let bound = builder
        .semantic_lifecycle_transitions()
        .iter()
        .map(|transition| transition.step_index)
        .collect::<std::collections::BTreeSet<_>>();
    let unbound = (0..builder.num_steps())
        .filter(|step_index| !bound.contains(step_index))
        .map(|step_index| step_index.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.semantic-lifecycle-projection.v1\",",
            "\"projection_only\":true,",
            "\"transition_count\":{},",
            "\"coverage_complete\":{},",
            "\"unbound_step_indices\":[{}],",
            "\"identity\":{},",
            "\"transitions\":[{}]",
            "}}"
        ),
        builder.semantic_lifecycle_transitions().len(),
        if builder.semantic_lifecycle_transitions().len() == builder.num_steps() as usize {
            "true"
        } else {
            "false"
        },
        unbound,
        semantic_lifecycle_identity_json(builder),
        transitions,
    )
}

#[wasm_bindgen]
impl SemanticIngressManifestV2 {
    #[wasm_bindgen(constructor)]
    pub fn new(plan: &MultiInputGraphPlan) -> Result<SemanticIngressManifestV2, String> {
        let plan_bytes = plan.validate_for_compile()?;
        Ok(Self {
            plan: plan.clone(),
            plan_bytes,
            ports: Vec::new(),
        })
    }

    #[wasm_bindgen(js_name = addRuntimePort)]
    pub fn add_runtime_port(
        &mut self,
        logical_port_id: String,
        slot: u8,
        expected_source: String,
    ) -> Result<bool, String> {
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
    pub fn add_deferred_port(
        &mut self,
        logical_port_id: String,
        role: String,
        required: bool,
    ) -> Result<bool, String> {
        validate_logical_port_id(&logical_port_id)?;
        if !role_valid(&role) {
            return Err(format!(
                "SemanticIngressManifestV2: invalid deferred role {role}"
            ));
        }
        self.add_port(LogicalPort {
            id: logical_port_id,
            role,
            expected_source: None,
            backing: Backing::Deferred,
            required,
        })
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
    pub fn input_port_status(
        &self,
        slot: u8,
        bundle: &MultiInputInputBundle,
    ) -> Result<String, String> {
        let contract = self
            .plan
            .ports()
            .iter()
            .find(|port| port.slot == slot)
            .ok_or_else(|| {
                format!("SemanticIngressManifestV2.inputPortStatus: undeclared slot {slot}")
            })?;
        let matches = bundle.matches_plan_internal(&self.plan);
        let input = bundle.input_preflight(&self.plan);
        let (port, _) = self.runtime_port_status(contract, bundle, &input, matches);
        Ok(format!(
            "{{\"schema_version\":2,\"schema_id\":\"burn-research.semantic-input-port-status.v2\",\"bundle_plan_matches\":{},\"execution_authorized\":false,\"port\":{}}}",
            bool_json(matches), port,
        ))
    }

    #[wasm_bindgen(js_name = consumerCompatibility)]
    pub fn consumer_compatibility(
        &self,
        slot: u8,
        bundle: &MultiInputInputBundle,
        consumer: &InputPortConsumerSpec,
    ) -> Result<String, String> {
        let contract = self
            .plan
            .ports()
            .iter()
            .find(|port| port.slot == slot)
            .ok_or_else(|| {
                format!("SemanticIngressManifestV2.consumerCompatibility: undeclared slot {slot}")
            })?;
        let mapped = self.runtime_port(slot).ok_or_else(|| format!("SemanticIngressManifestV2.consumerCompatibility: slot {slot} has no logical port mapping"))?;
        let input = bundle.input_preflight(&self.plan);
        let bundle_plan_matches = bundle.matches_plan_internal(&self.plan);
        let (port_status, port_ready) =
            self.runtime_port_status(contract, bundle, &input, bundle_plan_matches);
        let bound = bundle.bound_input(slot);
        let role_match = bound.is_some_and(|actual| consumer.role_matches(&actual.role));
        let fingerprint_ok = bound.is_some_and(|actual| {
            !consumer.require_fingerprint_value() || !actual.fingerprint.is_empty()
        });
        let revision_ok =
            bound.is_some_and(|actual| actual.revision >= consumer.minimum_revision_value());
        let compatible = port_ready && role_match && fingerprint_ok && revision_ok;
        let status = if bound.is_none() {
            "unknown"
        } else if compatible {
            "compatible"
        } else {
            "incompatible"
        };
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
    pub fn status(
        &self,
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        bundle: &MultiInputInputBundle,
    ) -> String {
        self.status_internal(registry, graph, bundle).json
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        bundle: &MultiInputInputBundle,
    ) -> Result<WasmTensor, String> {
        if !self.status_internal(registry, graph, bundle).ready {
            return Err("SemanticIngressManifestV2.run: ingress or graph preflight failed; execution was not started".into());
        }
        graph.run(registry, bundle)
    }

    #[wasm_bindgen(js_name = runWithTrace)]
    pub fn run_with_trace(
        &self,
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        bundle: &MultiInputInputBundle,
        start_step: u32,
        max_steps: u32,
        max_tensor_bytes: u32,
    ) -> Result<TracedMultiInputRun, String> {
        if !self.status_internal(registry, graph, bundle).ready {
            return Err("SemanticIngressManifestV2.runWithTrace: ingress or graph preflight failed; execution was not started".into());
        }
        graph.run_with_trace(registry, bundle, start_step, max_steps, max_tensor_bytes)
    }

    #[wasm_bindgen(js_name = verifyFlat)]
    pub fn verify_flat(
        &self,
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        bundle: &MultiInputInputBundle,
        candidate: &[f32],
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
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

#[wasm_bindgen]
impl SemanticTransitionSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(
        transition_id: String,
        input_roles_json: String,
        output_role: String,
    ) -> Result<SemanticTransitionSpec, String> {
        if transition_id.is_empty() {
            return Err(
                "SemanticTransitionSpec.transition_id: value must be non-empty".to_string(),
            );
        }
        if transition_id.len() > MAX_TRANSITION_ID_BYTES {
            return Err(format!(
                "SemanticTransitionSpec.transition_id: {} bytes exceeds limit {MAX_TRANSITION_ID_BYTES}",
                transition_id.len()
            ));
        }
        let input_roles = parse_roles(&input_roles_json)?;
        validate_role(&output_role, "SemanticTransitionSpec.output_role")?;
        Ok(Self {
            transition_id,
            input_roles,
            output_role,
        })
    }

    #[wasm_bindgen(js_name = transitionId)]
    pub fn transition_id(&self) -> String {
        self.transition_id.clone()
    }

    #[wasm_bindgen(js_name = inputRoles)]
    pub fn input_roles(&self) -> String {
        string_array_json(&self.input_roles)
    }

    #[wasm_bindgen(js_name = outputRole)]
    pub fn output_role(&self) -> String {
        self.output_role.clone()
    }

    #[wasm_bindgen(js_name = describe)]
    pub fn describe(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.semantic-transition-spec.v1\",",
                "\"transition_id\":\"{}\",",
                "\"input_roles\":{},",
                "\"output_role\":\"{}\"",
                "}}"
            ),
            json_escape(&self.transition_id),
            string_array_json(&self.input_roles),
            json_escape(&self.output_role),
        )
    }
}
