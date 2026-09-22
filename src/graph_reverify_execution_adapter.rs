use crate::agent_response_intent::AgentResponseIntent;
use crate::graph::CompiledGraph;
use crate::graph_reverify_runtime_binding::{
    preflight_graph_reverify_runtime_binding, GraphReverifyRuntimeBinding,
};
use crate::proof_provenance::workspace_verify_graph_receipt;
use crate::registry::LayerRegistry;
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;
#[cfg(not(target_arch = "wasm32"))]
use crate::runtime_resolution_evidence::RuntimeEvidence;
use crate::workspace::AgentWorkspace;
use crate::WasmTensor;

const GRAPH_REVERIFY_EXECUTION_ADAPTER_V1: &str =
    include_str!("../docs/graph-reverify-execution-adapter.v1.json");

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

fn receipt_total(counts: (usize, usize, usize)) -> usize {
    counts.0.saturating_add(counts.1).saturating_add(counts.2)
}

fn execution_fingerprint(
    binding_fingerprint: &str,
    response_intent_fingerprint: &str,
    dispatch_fingerprint: &str,
    verifier_receipt_id: u32,
    verifier_outcome: &str,
    verifier_receipt_json: &str,
) -> String {
    let canonical = format!(
        concat!(
            "v1|binding={}|intent={}|dispatch={}|receipt_id={}|",
            "outcome={}|receipt={}|"
        ),
        length_prefixed(binding_fingerprint),
        length_prefixed(response_intent_fingerprint),
        length_prefixed(dispatch_fingerprint),
        verifier_receipt_id,
        length_prefixed(verifier_outcome),
        length_prefixed(verifier_receipt_json),
    );
    fnv1a64(canonical.bytes())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphReverifyExecutionResult {
    source_entry_index: usize,
    source_evidence_fingerprint: String,
    binding_fingerprint: String,
    response_intent_fingerprint: String,
    dispatch_fingerprint: String,
    execution_fingerprint: String,
    verifier_receipt_id: u32,
    verifier_outcome: String,
    verifier_receipt_json: String,
    before_verifier_receipt_count: usize,
    after_verifier_receipt_count: usize,
}

impl GraphReverifyExecutionResult {
    pub fn source_entry_index(&self) -> usize {
        self.source_entry_index
    }

    pub fn source_evidence_fingerprint(&self) -> &str {
        &self.source_evidence_fingerprint
    }

    pub fn binding_fingerprint(&self) -> &str {
        &self.binding_fingerprint
    }

    pub fn response_intent_fingerprint(&self) -> &str {
        &self.response_intent_fingerprint
    }

    pub fn dispatch_fingerprint(&self) -> &str {
        &self.dispatch_fingerprint
    }

    pub fn execution_fingerprint(&self) -> &str {
        &self.execution_fingerprint
    }

    pub fn verifier_receipt_id(&self) -> u32 {
        self.verifier_receipt_id
    }

    pub fn verifier_outcome(&self) -> &str {
        &self.verifier_outcome
    }

    pub fn verifier_receipt_json(&self) -> &str {
        &self.verifier_receipt_json
    }

    pub fn before_verifier_receipt_count(&self) -> usize {
        self.before_verifier_receipt_count
    }

    pub fn after_verifier_receipt_count(&self) -> usize {
        self.after_verifier_receipt_count
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.graph-reverify-execution-result.v1\",",
                "\"role\":\"correlated_graph_reverify_execution_result\",",
                "\"source_entry_index\":{},",
                "\"source_evidence_fingerprint\":\"{}\",",
                "\"binding_fingerprint\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"dispatch_fingerprint\":\"{}\",",
                "\"execution_fingerprint\":\"{}\",",
                "\"execution_trigger\":\"explicit_caller_invocation\",",
                "\"authorization_claim\":\"none\",",
                "\"canonical_surface\":\"workspaceVerifyGraphReceipt\",",
                "\"canonical_authority\":\"wasm_verifier\",",
                "\"operation\":\"CompiledGraph.verifyFlat\",",
                "\"verifier_receipt_id\":{},",
                "\"verifier_outcome\":\"{}\",",
                "\"verifier_receipt\":{},",
                "\"before_verifier_receipt_count\":{},",
                "\"after_verifier_receipt_count\":{},",
                "\"mutation\":\"workspace_verifier_receipt_append\",",
                "\"execution_performed\":true,",
                "\"automatic_action_selection\":false",
                "}}"
            ),
            self.source_entry_index,
            json_escape(&self.source_evidence_fingerprint),
            json_escape(&self.binding_fingerprint),
            json_escape(&self.response_intent_fingerprint),
            json_escape(&self.dispatch_fingerprint),
            json_escape(&self.execution_fingerprint),
            self.verifier_receipt_id,
            json_escape(&self.verifier_outcome),
            self.verifier_receipt_json,
            self.before_verifier_receipt_count,
            self.after_verifier_receipt_count,
        )
    }
}

pub fn graph_reverify_execution_adapter_capabilities() -> &'static str {
    GRAPH_REVERIFY_EXECUTION_ADAPTER_V1
}

#[allow(clippy::too_many_arguments)]
pub fn execute_graph_reverify_binding(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    binding: &GraphReverifyRuntimeBinding,
    workspace: &mut AgentWorkspace,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    input: &WasmTensor,
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
    label: &str,
) -> Result<GraphReverifyExecutionResult, String> {
    if intent.selected_action() != EvidenceResponseAction::Reverify {
        return Err(
            "GraphReverifyExecutionAdapter: only reverify intents are supported".to_string(),
        );
    }

    let preflight = preflight_graph_reverify_runtime_binding(
        inbox,
        intent,
        binding,
        workspace,
        graph,
        registry,
        input,
        candidate,
        abs_tol,
        rel_tol,
        label,
    );
    if !preflight.ready {
        return Err(format!(
            "GraphReverifyExecutionAdapter: runtime binding preflight not ready: {}",
            preflight.status
        ));
    }

    let before_next_receipt_id = workspace.next_verifier_receipt_id();
    let expected_after_next_receipt_id = before_next_receipt_id.checked_add(1).ok_or_else(|| {
        "GraphReverifyExecutionAdapter: verifier receipt allocator exhausted".to_string()
    })?;
    let before_counts = workspace.introspection_verifier_receipt_counts();
    let before_total = receipt_total(before_counts);
    let rollback = workspace.clone();

    let verifier_receipt_json = match workspace_verify_graph_receipt(
        workspace,
        graph,
        registry,
        input,
        candidate,
        abs_tol,
        rel_tol,
        label.to_string(),
    ) {
        Ok(receipt) => receipt,
        Err(error) => {
            *workspace = rollback;
            return Err(format!(
                "GraphReverifyExecutionAdapter: canonical workspaceVerifyGraphReceipt rejected after preflight: {error}"
            ));
        }
    };

    let after_counts = workspace.introspection_verifier_receipt_counts();
    let after_total = receipt_total(after_counts);
    let after_next_receipt_id = workspace.next_verifier_receipt_id();

    let passed_delta = after_counts.0 == before_counts.0.saturating_add(1)
        && after_counts.1 == before_counts.1
        && after_counts.2 == before_counts.2;
    let failed_delta = after_counts.1 == before_counts.1.saturating_add(1)
        && after_counts.0 == before_counts.0
        && after_counts.2 == before_counts.2;

    let postcondition_ok = after_total == before_total.saturating_add(1)
        && after_next_receipt_id == expected_after_next_receipt_id
        && (passed_delta ^ failed_delta);

    if !postcondition_ok {
        *workspace = rollback;
        return Err(
            "GraphReverifyExecutionAdapter: verifier receipt postcondition mismatch; workspace rolled back"
                .to_string(),
        );
    }

    let verifier_outcome = if passed_delta { "passed" } else { "failed" };
    let fingerprint = execution_fingerprint(
        binding.binding_fingerprint(),
        intent.response_intent_fingerprint(),
        binding.dispatch_fingerprint(),
        before_next_receipt_id,
        verifier_outcome,
        &verifier_receipt_json,
    );

    Ok(GraphReverifyExecutionResult {
        source_entry_index: binding.source_entry_index(),
        source_evidence_fingerprint: binding.source_evidence_fingerprint().to_string(),
        binding_fingerprint: binding.binding_fingerprint().to_string(),
        response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
        dispatch_fingerprint: binding.dispatch_fingerprint().to_string(),
        execution_fingerprint: fingerprint,
        verifier_receipt_id: before_next_receipt_id,
        verifier_outcome: verifier_outcome.to_string(),
        verifier_receipt_json,
        before_verifier_receipt_count: before_total,
        after_verifier_receipt_count: after_total,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn rejoin_graph_reverify_execution_result(
    inbox: &mut ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    binding: &GraphReverifyRuntimeBinding,
    result: &GraphReverifyExecutionResult,
) -> Result<bool, String> {
    const CONTEXT: &str = "GraphReverifyExecutionAdapter.rejoin";

    if intent.selected_action() != EvidenceResponseAction::Reverify
        || intent.evidence_kind() != "graph_verifier_receipt"
    {
        return Err(format!(
            "{CONTEXT}: current intent is not a graph reverify intent"
        ));
    }
    if binding.response_intent_fingerprint() != intent.response_intent_fingerprint()
        || result.response_intent_fingerprint != intent.response_intent_fingerprint()
        || result.binding_fingerprint != binding.binding_fingerprint()
        || result.dispatch_fingerprint != binding.dispatch_fingerprint()
        || result.source_entry_index != binding.source_entry_index()
        || result.source_evidence_fingerprint != binding.source_evidence_fingerprint()
    {
        return Err(format!("{CONTEXT}: execution correlation mismatch"));
    }

    let expected_fingerprint = execution_fingerprint(
        binding.binding_fingerprint(),
        intent.response_intent_fingerprint(),
        binding.dispatch_fingerprint(),
        result.verifier_receipt_id,
        &result.verifier_outcome,
        &result.verifier_receipt_json,
    );
    if expected_fingerprint != result.execution_fingerprint {
        return Err(format!("{CONTEXT}: execution fingerprint mismatch"));
    }

    let evidence = RuntimeEvidence::from_graph_verifier_receipt_json(
        &result.verifier_receipt_json,
    )?;
    if evidence.kind() != "graph_verifier_receipt"
        || evidence.source_authority() != "wasm_verifier"
        || evidence.evidence_authority() != "observation_only"
        || evidence.outcome() != result.verifier_outcome
    {
        return Err(format!("{CONTEXT}: canonical verifier evidence mismatch"));
    }

    inbox.record(evidence)
}

#[cfg(test)]
mod tests {
    use super::{
        execute_graph_reverify_binding, graph_reverify_execution_adapter_capabilities,
        rejoin_graph_reverify_execution_result,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::graph_reverify_runtime_binding::create_graph_reverify_runtime_binding;
    use crate::proof_provenance::workspace_verify_graph_receipt;
    use crate::registry::LayerRegistry;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_runtime_bridge::{
        bind_runtime_subject_projection, RuntimeSubjectProjection,
    };
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_compile_for_runtime_subject;
    use crate::WasmTensor;

    fn fixture() -> (
        ResolutionEvidenceInbox,
        crate::agent_response_intent::AgentResponseIntent,
        AgentWorkspace,
        LayerRegistry,
        crate::graph::CompiledGraph,
        WasmTensor,
    ) {
        let mut review = ResolutionReviewSession::new("intent-graph-reverify-execution").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-graph-reverify-execution".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-graph-reverify-execution".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-graph-reverify-execution".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "customer".to_string(),
            fields: Vec::new(),
        };

        let mut workspace = AgentWorkspace::new(2).unwrap();
        bind_runtime_subject_projection(&mut workspace, &projection).unwrap();

        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(7);
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph =
            workspace_compile_for_runtime_subject(&mut workspace, &builder, &registry, 1).unwrap();
        let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);

        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let source_receipt = workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &[0.0, 3.0],
            0.0,
            0.0,
            "source-failed".to_string(),
        )
        .unwrap();
        let source = RuntimeEvidence::from_graph_verifier_receipt_json(&source_receipt).unwrap();
        assert_eq!(source.outcome(), "failed");
        assert!(inbox.record(source).unwrap());

        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        (inbox, intent, workspace, registry, graph, input)
    }

    #[test]
    fn explicit_execution_calls_canonical_verifier_once_and_rejoins_separately() {
        let (mut inbox, intent, mut workspace, registry, graph, input) = fixture();
        let corrected = vec![0.0, 2.0];
        let binding = create_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &workspace,
            &graph,
            &registry,
            &input,
            &corrected,
            0.0,
            0.0,
            "reverify-corrected",
        )
        .unwrap();

        let inbox_before = inbox.len();
        let result = execute_graph_reverify_binding(
            &inbox,
            &intent,
            &binding,
            &mut workspace,
            &graph,
            &registry,
            &input,
            &corrected,
            0.0,
            0.0,
            "reverify-corrected",
        )
        .unwrap();

        assert_eq!(result.verifier_outcome(), "passed");
        assert_eq!(
            result.after_verifier_receipt_count(),
            result.before_verifier_receipt_count() + 1
        );
        assert_eq!(inbox.len(), inbox_before);
        assert!(result.verifier_receipt_json().contains("\"authority\":\"wasm_verifier\""));
        assert!(result
            .verifier_receipt_json()
            .contains("\"verifier\":\"CompiledGraph.verifyFlat\""));

        assert!(rejoin_graph_reverify_execution_result(
            &mut inbox,
            &intent,
            &binding,
            &result,
        )
        .unwrap());
        assert_eq!(inbox.len(), inbox_before + 1);

        let json = inbox.to_json();
        assert!(json.contains("\"graph_verifier_passed\":1"));
        assert!(json.contains("\"graph_verifier_failed\":1"));
        assert!(json.contains("\"action_selected\":false"));
        assert!(json.contains("\"state_transition\":\"none\""));
    }

    #[test]
    fn failed_numerical_reverify_is_successful_execution_and_observation_only_evidence() {
        let (mut inbox, intent, mut workspace, registry, graph, input) = fixture();
        let still_wrong = vec![0.0, 4.0];
        let binding = create_graph_reverify_runtime_binding(
            &inbox,
            &intent,
            &workspace,
            &graph,
            &registry,
            &input,
            &still_wrong,
            0.0,
            0.0,
            "reverify-failed",
        )
        .unwrap();

        let result = execute_graph_reverify_binding(
            &inbox,
            &intent,
            &binding,
            &mut workspace,
            &graph,
            &registry,
            &input,
            &still_wrong,
            0.0,
            0.0,
            "reverify-failed",
        )
        .unwrap();
        assert_eq!(result.verifier_outcome(), "failed");

        assert!(rejoin_graph_reverify_execution_result(
            &mut inbox,
            &intent,
            &binding,
            &result,
        )
        .unwrap());

        let json = inbox.to_json();
        assert!(json.contains("\"graph_verifier_failed\":2"));
        assert!(json.contains("\"interpretation_required\":true"));
        assert!(json.contains("\"action_selected\":false"));
        assert!(json.contains("\"revision_created\":false"));
    }

    #[test]
    fn runtime_drift_is_rejected_before_canonical_execution_without_workspace_mutation() {
        let (inbox, intent, mut workspace, registry, graph, input) = fixture();
        let candidate = vec![0.0, 2.0];
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
            "reverify-drift",
        )
        .unwrap();

        let before = workspace.snapshot();
        let drifted = vec![0.0, 2.5];
        let result = execute_graph_reverify_binding(
            &inbox,
            &intent,
            &binding,
            &mut workspace,
            &graph,
            &registry,
            &input,
            &drifted,
            0.0,
            0.0,
            "reverify-drift",
        );
        assert!(result.is_err());
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn repeated_explicit_reverify_is_allowed_and_allocates_distinct_receipts() {
        let (inbox, intent, mut workspace, registry, graph, input) = fixture();
        let candidate = vec![0.0, 2.0];
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
            "repeatable-reverify",
        )
        .unwrap();

        let first = execute_graph_reverify_binding(
            &inbox,
            &intent,
            &binding,
            &mut workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeatable-reverify",
        )
        .unwrap();
        let second = execute_graph_reverify_binding(
            &inbox,
            &intent,
            &binding,
            &mut workspace,
            &graph,
            &registry,
            &input,
            &candidate,
            0.0,
            0.0,
            "repeatable-reverify",
        )
        .unwrap();

        assert_eq!(second.verifier_receipt_id(), first.verifier_receipt_id() + 1);
        assert_ne!(first.execution_fingerprint(), second.execution_fingerprint());
    }

    #[test]
    fn capability_contract_preserves_canonical_verifier_and_agent_authority() {
        let contract: serde_json::Value =
            serde_json::from_str(graph_reverify_execution_adapter_capabilities()).unwrap();
        assert_eq!(
            contract["execution"]["canonical_surface"],
            "workspaceVerifyGraphReceipt"
        );
        assert_eq!(contract["execution"]["canonical_authority"], "wasm_verifier");
        assert_eq!(contract["semantics"]["automatic_action_selection"], false);
        assert_eq!(contract["rejoin"]["evidence_authority"], "observation_only");
    }
}
