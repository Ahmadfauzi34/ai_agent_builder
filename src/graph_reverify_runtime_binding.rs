use crate::agent_response_intent::AgentResponseIntent;
use crate::coprocessor::validate_tolerance;
use crate::graph::CompiledGraph;
use crate::proof_provenance::{f32_fingerprint, tensor_fingerprint, validate_label};
use crate::registry::LayerRegistry;
use crate::response_dispatch_requirements::response_dispatch_requirements;
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

const GRAPH_REVERIFY_RUNTIME_BINDING_V1: &str =
    include_str!("../docs/graph-reverify-runtime-binding.v1.json");

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

fn shape_canonical(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn binding_fingerprint(
    intent: &AgentResponseIntent,
    dispatch_fingerprint: &str,
    requirements_fingerprint: &str,
    program_identity: &str,
    input_shape: &[usize],
    input_fingerprint: &str,
    candidate_fingerprint: &str,
    candidate_len: usize,
    abs_tol_bits: u64,
    rel_tol_bits: u64,
    label: &str,
) -> String {
    let canonical = format!(
        concat!(
            "v1|intent={}|evidence={}|dispatch={}|requirements={}|program={}|",
            "shape={}|input={}|candidate={}|candidate_len={}|",
            "abs_tol_bits={:016x}|rel_tol_bits={:016x}|label={}|"
        ),
        length_prefixed(intent.response_intent_fingerprint()),
        length_prefixed(intent.evidence_fingerprint()),
        length_prefixed(dispatch_fingerprint),
        length_prefixed(requirements_fingerprint),
        length_prefixed(program_identity),
        shape_canonical(input_shape),
        length_prefixed(input_fingerprint),
        length_prefixed(candidate_fingerprint),
        candidate_len,
        abs_tol_bits,
        rel_tol_bits,
        length_prefixed(label),
    );
    fnv1a64(canonical.bytes())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphReverifyRuntimeBinding {
    source_entry_index: usize,
    source_evidence_fingerprint: String,
    response_intent_fingerprint: String,
    dispatch_fingerprint: String,
    requirements_fingerprint: String,
    program_identity: String,
    input_shape: Vec<usize>,
    input_fingerprint: String,
    candidate_fingerprint: String,
    candidate_len: usize,
    abs_tol_bits: u64,
    rel_tol_bits: u64,
    label: String,
    binding_fingerprint: String,
}

impl GraphReverifyRuntimeBinding {
    pub fn source_entry_index(&self) -> usize {
        self.source_entry_index
    }

    pub fn source_evidence_fingerprint(&self) -> &str {
        &self.source_evidence_fingerprint
    }

    pub fn response_intent_fingerprint(&self) -> &str {
        &self.response_intent_fingerprint
    }

    pub fn dispatch_fingerprint(&self) -> &str {
        &self.dispatch_fingerprint
    }

    pub fn requirements_fingerprint(&self) -> &str {
        &self.requirements_fingerprint
    }

    pub fn program_identity(&self) -> &str {
        &self.program_identity
    }

    pub fn input_fingerprint(&self) -> &str {
        &self.input_fingerprint
    }

    pub fn candidate_fingerprint(&self) -> &str {
        &self.candidate_fingerprint
    }

    pub fn candidate_len(&self) -> usize {
        self.candidate_len
    }

    pub fn abs_tol(&self) -> f64 {
        f64::from_bits(self.abs_tol_bits)
    }

    pub fn rel_tol(&self) -> f64 {
        f64::from_bits(self.rel_tol_bits)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn binding_fingerprint(&self) -> &str {
        &self.binding_fingerprint
    }

    pub fn to_json(&self) -> String {
        let shape = self
            .input_shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.graph-reverify-runtime-binding.v1\",",
                "\"role\":\"real_handle_derived_graph_reverify_binding_nonexecuting\",",
                "\"source_entry_index\":{},",
                "\"source_evidence_fingerprint\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"dispatch_fingerprint\":\"{}\",",
                "\"requirements_fingerprint\":\"{}\",",
                "\"authority\":\"runtime_verifier\",",
                "\"operation\":\"CompiledGraph.verifyFlat\",",
                "\"runtime\":{{",
                    "\"program_identity\":{},",
                    "\"input_shape\":[{}],",
                    "\"input_fingerprint\":\"{}\",",
                    "\"candidate_fingerprint\":\"{}\",",
                    "\"candidate_len\":{},",
                    "\"abs_tol\":{},",
                    "\"rel_tol\":{},",
                    "\"label\":\"{}\"",
                "}},",
                "\"binding_fingerprint\":\"{}\",",
                "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
                "\"execution_authorized\":false,",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"none\"",
                "}}"
            ),
            self.source_entry_index,
            json_escape(&self.source_evidence_fingerprint),
            json_escape(&self.response_intent_fingerprint),
            json_escape(&self.dispatch_fingerprint),
            json_escape(&self.requirements_fingerprint),
            self.program_identity,
            shape,
            json_escape(&self.input_fingerprint),
            json_escape(&self.candidate_fingerprint),
            self.candidate_len,
            self.abs_tol(),
            self.rel_tol(),
            json_escape(&self.label),
            json_escape(&self.binding_fingerprint),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphReverifyRuntimeBindingPreflight {
    pub ready: bool,
    pub status: String,
    pub requirements_current: bool,
    pub workspace_subject_matches: bool,
    pub source_evidence_matches: bool,
    pub program_identity_matches: bool,
    pub registry_binding_valid: bool,
    pub runtime_program_bound: bool,
    pub input_matches: bool,
    pub candidate_matches: bool,
    pub tolerance_matches: bool,
    pub label_matches: bool,
    pub binding_fingerprint_matches: bool,
    pub execution_authorized: bool,
    pub mutation: String,
}

impl GraphReverifyRuntimeBindingPreflight {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.graph-reverify-runtime-binding-preflight.v1\",",
                "\"ready\":{},",
                "\"status\":\"{}\",",
                "\"checks\":{{",
                    "\"requirements_current\":{},",
                    "\"workspace_subject_matches\":{},",
                    "\"source_evidence_matches\":{},",
                    "\"program_identity_matches\":{},",
                    "\"registry_binding_valid\":{},",
                    "\"runtime_program_bound\":{},",
                    "\"input_matches\":{},",
                    "\"candidate_matches\":{},",
                    "\"tolerance_matches\":{},",
                    "\"label_matches\":{},",
                    "\"binding_fingerprint_matches\":{}",
                "}},",
                "\"execution_authorized\":false,",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"none\"",
                "}}"
            ),
            self.ready,
            json_escape(&self.status),
            self.requirements_current,
            self.workspace_subject_matches,
            self.source_evidence_matches,
            self.program_identity_matches,
            self.registry_binding_valid,
            self.runtime_program_bound,
            self.input_matches,
            self.candidate_matches,
            self.tolerance_matches,
            self.label_matches,
            self.binding_fingerprint_matches,
        )
    }
}

fn current_graph_requirements(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> Result<(String, String), String> {
    if intent.selected_action() != EvidenceResponseAction::Reverify {
        return Err(format!(
            "GraphReverifyRuntimeBinding: expected reverify intent, got {}",
            intent.selected_action().as_str()
        ));
    }
    if intent.evidence_kind() != "graph_verifier_receipt" {
        return Err(format!(
            "GraphReverifyRuntimeBinding: expected graph_verifier_receipt, got {}",
            intent.evidence_kind()
        ));
    }

    let requirements = response_dispatch_requirements(inbox, intent);
    if !requirements.ready {
        return Err(format!(
            "GraphReverifyRuntimeBinding: dispatch requirements are not current: {}",
            requirements.status
        ));
    }
    if requirements.authority.as_deref() != Some("runtime_verifier")
        || requirements.operation.as_deref() != Some("CompiledGraph.verifyFlat")
        || requirements.executor_contract.as_deref() != Some("CompiledGraph.verifyFlat")
        || requirements.payload_mode.as_deref() != Some("typed_runtime_verifier_inputs")
    {
        return Err("GraphReverifyRuntimeBinding: graph verifier route mismatch".to_string());
    }

    let dispatch_fingerprint = requirements
        .dispatch_fingerprint
        .ok_or_else(|| "GraphReverifyRuntimeBinding: dispatch fingerprint missing".to_string())?;
    let requirements_fingerprint = requirements.requirements_fingerprint.ok_or_else(|| {
        "GraphReverifyRuntimeBinding: requirements fingerprint missing".to_string()
    })?;
    Ok((dispatch_fingerprint, requirements_fingerprint))
}

fn validate_candidate(candidate: &[f32]) -> Result<(), String> {
    if candidate.is_empty() {
        return Err("GraphReverifyRuntimeBinding: candidate must be non-empty".to_string());
    }
    if let Some((index, value)) = candidate
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "GraphReverifyRuntimeBinding: candidate contains non-finite value at index {index}: {value}"
        ));
    }
    Ok(())
}

pub fn graph_reverify_runtime_binding_capabilities() -> &'static str {
    GRAPH_REVERIFY_RUNTIME_BINDING_V1
}

pub fn create_graph_reverify_runtime_binding(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    workspace: &AgentWorkspace,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: impl Into<String>,
) -> Result<GraphReverifyRuntimeBinding, String> {
    let (dispatch_fingerprint, requirements_fingerprint) =
        current_graph_requirements(inbox, intent)?;

    if !inbox.target_matches_workspace(workspace) {
        return Err(
            "GraphReverifyRuntimeBinding: workspace runtime subject does not match evidence inbox"
                .to_string(),
        );
    }

    let evidence = inbox
        .observations()
        .get(intent.entry_index())
        .ok_or_else(|| "GraphReverifyRuntimeBinding: source evidence entry missing".to_string())?;
    let source_program_identity = evidence.graph_program_identity().ok_or_else(|| {
        "GraphReverifyRuntimeBinding: source evidence has no graph program identity".to_string()
    })?;

    graph.validate_registry_binding(registry)?;
    let program_identity = graph.program_identity();
    if source_program_identity != program_identity {
        return Err(
            "GraphReverifyRuntimeBinding: graph programIdentity does not match selected evidence"
                .to_string(),
        );
    }
    workspace.require_runtime_program_identity_if_bound(
        &program_identity,
        "GraphReverifyRuntimeBinding",
    )?;

    validate_candidate(candidate)?;
    validate_tolerance(abs_tol, rel_tol)?;
    let label = label.into();
    validate_label(&label, "GraphReverifyRuntimeBinding")?;

    let input_shape = input.shape();
    let input_fingerprint = tensor_fingerprint(input);
    let candidate_fingerprint = f32_fingerprint(candidate);
    let abs_tol_bits = abs_tol.to_bits();
    let rel_tol_bits = rel_tol.to_bits();
    let fingerprint = binding_fingerprint(
        intent,
        &dispatch_fingerprint,
        &requirements_fingerprint,
        &program_identity,
        &input_shape,
        &input_fingerprint,
        &candidate_fingerprint,
        candidate.len(),
        abs_tol_bits,
        rel_tol_bits,
        &label,
    );

    Ok(GraphReverifyRuntimeBinding {
        source_entry_index: intent.entry_index(),
        source_evidence_fingerprint: intent.evidence_fingerprint().to_string(),
        response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
        dispatch_fingerprint,
        requirements_fingerprint,
        program_identity,
        input_shape,
        input_fingerprint,
        candidate_fingerprint,
        candidate_len: candidate.len(),
        abs_tol_bits,
        rel_tol_bits,
        label,
        binding_fingerprint: fingerprint,
    })
}

pub fn preflight_graph_reverify_runtime_binding(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    binding: &GraphReverifyRuntimeBinding,
    workspace: &AgentWorkspace,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: &str,
) -> GraphReverifyRuntimeBindingPreflight {
    let current_requirements = current_graph_requirements(inbox, intent).ok();
    let requirements_current = current_requirements.as_ref().is_some_and(
        |(dispatch, requirements)| {
            dispatch == &binding.dispatch_fingerprint
                && requirements == &binding.requirements_fingerprint
        },
    );
    let workspace_subject_matches = inbox.target_matches_workspace(workspace);

    let source_evidence_matches = inbox
        .observations()
        .get(binding.source_entry_index)
        .is_some_and(|evidence| {
            binding.source_entry_index == intent.entry_index()
                && binding.source_evidence_fingerprint == intent.evidence_fingerprint()
                && evidence.kind() == "graph_verifier_receipt"
                && evidence.graph_program_identity() == Some(binding.program_identity.as_str())
        });

    let current_program_identity = graph.program_identity();
    let program_identity_matches = current_program_identity == binding.program_identity;
    let registry_binding_valid = graph.validate_registry_binding(registry).is_ok();
    let runtime_program_bound = workspace
        .require_runtime_program_identity_if_bound(
            &current_program_identity,
            "GraphReverifyRuntimeBindingPreflight",
        )
        .is_ok();

    let current_input_fingerprint = tensor_fingerprint(input);
    let input_matches = input.shape() == binding.input_shape
        && current_input_fingerprint == binding.input_fingerprint;

    let candidate_valid = validate_candidate(candidate).is_ok();
    let current_candidate_fingerprint = if candidate_valid {
        Some(f32_fingerprint(candidate))
    } else {
        None
    };
    let candidate_matches = candidate_valid
        && candidate.len() == binding.candidate_len
        && current_candidate_fingerprint.as_deref()
            == Some(binding.candidate_fingerprint.as_str());

    let tolerance_matches = validate_tolerance(abs_tol, rel_tol).is_ok()
        && abs_tol.to_bits() == binding.abs_tol_bits
        && rel_tol.to_bits() == binding.rel_tol_bits;
    let label_matches = validate_label(label, "GraphReverifyRuntimeBindingPreflight").is_ok()
        && label == binding.label;

    let binding_fingerprint_matches = current_requirements.as_ref().is_some_and(
        |(dispatch, requirements)| {
            binding_fingerprint(
                intent,
                dispatch,
                requirements,
                &current_program_identity,
                &input.shape(),
                &current_input_fingerprint,
                current_candidate_fingerprint.as_deref().unwrap_or("invalid"),
                candidate.len(),
                abs_tol.to_bits(),
                rel_tol.to_bits(),
                label,
            ) == binding.binding_fingerprint
        },
    );

    let ready = requirements_current
        && workspace_subject_matches
        && source_evidence_matches
        && program_identity_matches
        && registry_binding_valid
        && runtime_program_bound
        && input_matches
        && candidate_matches
        && tolerance_matches
        && label_matches
        && binding_fingerprint_matches;

    GraphReverifyRuntimeBindingPreflight {
        ready,
        status: if ready {
            "ready_nonexecuting".to_string()
        } else {
            "closed:runtime_binding_or_upstream_drift".to_string()
        },
        requirements_current,
        workspace_subject_matches,
        source_evidence_matches,
        program_identity_matches,
        registry_binding_valid,
        runtime_program_bound,
        input_matches,
        candidate_matches,
        tolerance_matches,
        label_matches,
        binding_fingerprint_matches,
        execution_authorized: false,
        mutation: "none".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        create_graph_reverify_runtime_binding, graph_reverify_runtime_binding_capabilities,
        preflight_graph_reverify_runtime_binding,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::proof_provenance::workspace_verify_graph_receipt;
    use crate::protocol::LAYER_ACTIVATION;
    use crate::registry::LayerRegistry;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_runtime_bridge::{
        bind_runtime_subject, workspace_compile_for_runtime_subject, RuntimeSubjectProjection,
    };
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};
    use crate::workspace::AgentWorkspace;
    use crate::WasmTensor;

    fn fixture() -> (
        ResolutionEvidenceInbox,
        crate::agent_response_intent::AgentResponseIntent,
        AgentWorkspace,
        AgentGraphBuilder,
        LayerRegistry,
        crate::graph::CompiledGraph,
        WasmTensor,
        Vec<f32>,
    ) {
        let mut review = ResolutionReviewSession::new("intent-graph-reverify-binding").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-graph-reverify-binding".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-graph-reverify-binding".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-graph-reverify-binding".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "customer".to_string(),
            fields: Vec::new(),
        };

        let mut workspace = AgentWorkspace::new(2).unwrap();
        bind_runtime_subject(&mut workspace, &projection).unwrap();

        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::activation(7, 0).unwrap();
        assert_eq!(spec.layer_type(), LAYER_ACTIVATION);
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph =
            workspace_compile_for_runtime_subject(&mut workspace, &builder, &registry, 1).unwrap();

        let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
        let candidate = vec![0.0, 3.0];

        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let source_receipt = workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "source-failed".to_string(),
        )
        .unwrap();
        let source = RuntimeEvidence::from_graph_verifier_receipt_json(&source_receipt).unwrap();
        assert!(inbox.record(source).unwrap());

        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        (inbox, intent, workspace, builder, registry, graph, input, candidate)
    }

    #[test]
    fn real_runtime_handles_bind_without_executing_graph_again() {
        let (inbox, intent, workspace, _builder, registry, graph, input, candidate) = fixture();
        let receipt_count_before: serde_json::Value =
            serde_json::from_str(&workspace.verifier_receipts()).unwrap();

        let binding = create_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeat-check",
        )
        .unwrap();
        let preflight = preflight_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &binding,
            &workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeat-check",
        );
        assert!(preflight.ready);
        assert!(!preflight.execution_authorized);

        let receipt_count_after: serde_json::Value =
            serde_json::from_str(&workspace.verifier_receipts()).unwrap();
        assert_eq!(receipt_count_before, receipt_count_after);
    }

    #[test]
    fn candidate_drift_closes_preflight() {
        let (inbox, intent, workspace, _builder, registry, graph, input, candidate) = fixture();
        let binding = create_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeat-check",
        )
        .unwrap();

        let drifted = vec![0.0, 4.0];
        let preflight = preflight_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &binding,
            &workspace,
            &graph,
            &registry,
            &input,
            &drifted,
            0.0,
            0.0,
            "repeat-check",
        );
        assert!(!preflight.ready);
        assert!(!preflight.candidate_matches);
        assert!(!preflight.binding_fingerprint_matches);
    }

    #[test]
    fn binding_rejects_graph_that_does_not_match_selected_evidence() {
        let (inbox, intent, workspace, _builder, mut registry, _graph, input, candidate) = fixture();
        let other_spec = AgentLayerSpec::activation(9, 1).unwrap();
        registry.init_agent_layer(&other_spec).unwrap();
        let mut other_builder = AgentGraphBuilder::new(2).unwrap();
        other_builder.add_unary(&other_spec, 0, 1).unwrap();
        other_builder.set_output(1).unwrap();
        let other_graph = other_builder.compile(&registry).unwrap();

        let result = create_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &workspace,
            &other_graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeat-check",
        );
        assert!(result.is_err());
    }

    #[test]
    fn capability_contract_is_nonexecuting_and_real_handle_derived() {
        let contract: serde_json::Value =
            serde_json::from_str(graph_reverify_runtime_binding_capabilities()).unwrap();
        assert_eq!(contract["execution_authorized"], false);
        assert_eq!(contract["execution_effect"], "none");
        assert_eq!(contract["binding_source"], "real_runtime_handles");
    }
}
