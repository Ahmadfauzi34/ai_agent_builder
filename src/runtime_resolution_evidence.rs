use wasm_bindgen::prelude::*;

use crate::authorization::{AuthorizationPolicy, AuthorizationSnapshot};
use crate::effective_spec::ApprovedEffectiveSpec;
use crate::resolution::ResolutionSnapshot;
use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
use crate::workspace::AgentWorkspace;

const RUNTIME_RESOLUTION_EVIDENCE_V1: &str =
    include_str!("../docs/runtime-resolution-evidence.v1.json");

pub const MAX_RUNTIME_EVIDENCE_ENTRIES: usize = 64;
const MAX_SHORT_FIELD_BYTES: usize = 256;
const MAX_DETAIL_BYTES: usize = 4096;
const MAX_PROGRAM_IDENTITY_BYTES: usize = 16_384;
const MAX_SEMANTIC_EXECUTION_CONTEXT_BYTES: usize = 32_768;

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

fn validate_nonempty_bounded(
    value: &str,
    max_bytes: usize,
    context: &str,
) -> Result<(), String> {
    if value.trim().is_empty() {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEvidenceSubject {
    pub intent_id: String,
    pub workflow_revision: u64,
    pub approval_id: String,
    pub subject_kind: String,
    pub subject_identity: String,
    pub authorization_policy_id: String,
    pub authorization_policy_revision: u64,
    pub authorization_is_revision: bool,
}

impl RuntimeEvidenceSubject {
    pub fn from_projection(projection: &RuntimeSubjectProjection) -> Self {
        Self {
            intent_id: projection.intent_id.clone(),
            workflow_revision: projection.workflow_revision,
            approval_id: projection.approval_id.clone(),
            subject_kind: projection.subject_kind.clone(),
            subject_identity: projection.subject_identity.clone(),
            authorization_policy_id: projection.authorization_policy_id.clone(),
            authorization_policy_revision: projection.authorization_policy_revision,
            authorization_is_revision: projection.authorization_is_revision,
        }
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"intent_id\":\"{}\",",
                "\"workflow_revision\":{},",
                "\"approval_id\":\"{}\",",
                "\"subject_kind\":\"{}\",",
                "\"subject_identity\":\"{}\",",
                "\"authorization_policy_id\":\"{}\",",
                "\"authorization_policy_revision\":{},",
                "\"authorization_is_revision\":{}",
                "}}"
            ),
            json_escape(&self.intent_id),
            self.workflow_revision,
            json_escape(&self.approval_id),
            json_escape(&self.subject_kind),
            json_escape(&self.subject_identity),
            json_escape(&self.authorization_policy_id),
            self.authorization_policy_revision,
            self.authorization_is_revision,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeEvidencePayload {
    AgentFault {
        code: String,
        class: String,
        operation: String,
        predicate: String,
        recoverable: bool,
        message: String,
    },
    GraphVerifierReceipt {
        receipt_id: u32,
        label: String,
        program_identity: String,
        semantic_execution_context: Option<String>,
        semantic_context_fingerprint: Option<String>,
        passed: bool,
        detail: String,
    },
    MathProgramVerifierReceipt {
        receipt_id: u32,
        label: String,
        program_identity: String,
        passed: bool,
        detail: String,
    },
    DirectMathVerifierReceipt {
        receipt_id: u32,
        operation_id: String,
        label: String,
        reference_program_identity: String,
        passed: bool,
        detail: String,
    },
    VectorVerifierReceipt {
        receipt_id: u32,
        label: String,
        passed: bool,
        detail: String,
    },
    RevisionDispatchExecutionReceipt {
        receipt_fingerprint: String,
        request_fingerprint: String,
        response_intent_fingerprint: String,
        dispatch_fingerprint: String,
        lineage_id: String,
        revision_id: String,
        revision_key: String,
        parent_revision_id: Option<String>,
        before_revision_count: usize,
        after_revision_count: usize,
    },
}

impl RuntimeEvidencePayload {
    pub fn source_authority(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "agent_fault_preflight",
            Self::GraphVerifierReceipt { .. }
            | Self::MathProgramVerifierReceipt { .. }
            | Self::DirectMathVerifierReceipt { .. } => "wasm_verifier",
            Self::VectorVerifierReceipt { .. } => "wasm_comparator",
            Self::RevisionDispatchExecutionReceipt { .. } => "ResolutionRevisionChain",
        }
    }

    pub fn evidence_authority(&self) -> &'static str {
        "observation_only"
    }

    pub fn transport_integrity(&self) -> &'static str {
        match self {
            Self::RevisionDispatchExecutionReceipt { .. } => "native_typed_correlated",
            _ => "host_structured_unverified",
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "agent_fault",
            Self::GraphVerifierReceipt { .. } => "graph_verifier_receipt",
            Self::MathProgramVerifierReceipt { .. } => "math_program_verifier_receipt",
            Self::DirectMathVerifierReceipt { .. } => "direct_math_verifier_receipt",
            Self::VectorVerifierReceipt { .. } => "vector_verifier_receipt",
            Self::RevisionDispatchExecutionReceipt { .. } => "revision_dispatch_execution_receipt",
        }
    }

    pub fn outcome(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "fault",
            Self::GraphVerifierReceipt { passed, .. }
            | Self::MathProgramVerifierReceipt { passed, .. }
            | Self::DirectMathVerifierReceipt { passed, .. }
            | Self::VectorVerifierReceipt { passed, .. } => {
                if *passed {
                    "passed"
                } else {
                    "failed"
                }
            }
            Self::RevisionDispatchExecutionReceipt { .. } => "committed",
        }
    }

    fn to_json(&self) -> String {
        match self {
            Self::AgentFault {
                code,
                class,
                operation,
                predicate,
                recoverable,
                message,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"agent_fault\",",
                    "\"evidence_authority\":\"observation_only\",",
                    "\"source_authority\":\"agent_fault_preflight\",",
                    "\"transport_integrity\":\"host_structured_unverified\",",
                    "\"outcome\":\"fault\",",
                    "\"code\":\"{}\",",
                    "\"class\":\"{}\",",
                    "\"operation\":\"{}\",",
                    "\"predicate\":\"{}\",",
                    "\"recoverable\":{},",
                    "\"message\":\"{}\"",
                    "}}"
                ),
                json_escape(code),
                json_escape(class),
                json_escape(operation),
                json_escape(predicate),
                recoverable,
                json_escape(message),
            ),
            Self::GraphVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                semantic_execution_context,
                semantic_context_fingerprint,
                passed,
                detail,
            } => {
                let semantic = match (
                    semantic_execution_context.as_deref(),
                    semantic_context_fingerprint.as_deref(),
                ) {
                    (Some(context), Some(fingerprint)) => format!(
                        ",\"semantic_execution_context\":\"{}\",\"semantic_context_fingerprint\":\"{}\"",
                        json_escape(context),
                        json_escape(fingerprint),
                    ),
                    _ => String::new(),
                };
                format!(
                    concat!(
                        "{{",
                        "\"kind\":\"graph_verifier_receipt\",",
                        "\"evidence_authority\":\"observation_only\",",
                        "\"source_authority\":\"wasm_verifier\",",
                        "\"transport_integrity\":\"host_structured_unverified\",",
                        "\"reference_authority\":\"burn_compiled_graph\",",
                        "\"outcome\":\"{}\",",
                        "\"receipt_id\":{},",
                        "\"label\":\"{}\",",
                        "\"program_identity\":\"{}\",",
                        "\"passed\":{},",
                        "\"detail\":\"{}\"",
                        "{}",
                        "}}"
                    ),
                    if *passed { "passed" } else { "failed" },
                    receipt_id,
                    json_escape(label),
                    json_escape(program_identity),
                    passed,
                    json_escape(detail),
                    semantic,
                )
            },
            Self::MathProgramVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                passed,
                detail,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"math_program_verifier_receipt\",",
                    "\"evidence_authority\":\"observation_only\",",
                    "\"source_authority\":\"wasm_verifier\",",
                    "\"transport_integrity\":\"host_structured_unverified\",",
                    "\"reference_authority\":\"burn_math_program\",",
                    "\"outcome\":\"{}\",",
                    "\"receipt_id\":{},",
                    "\"label\":\"{}\",",
                    "\"program_identity\":\"{}\",",
                    "\"passed\":{},",
                    "\"detail\":\"{}\"",
                    "}}"
                ),
                if *passed { "passed" } else { "failed" },
                receipt_id,
                json_escape(label),
                json_escape(program_identity),
                passed,
                json_escape(detail),
            ),
            Self::DirectMathVerifierReceipt {
                receipt_id,
                operation_id,
                label,
                reference_program_identity,
                passed,
                detail,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"direct_math_verifier_receipt\",",
                    "\"evidence_authority\":\"observation_only\",",
                    "\"source_authority\":\"wasm_verifier\",",
                    "\"transport_integrity\":\"host_structured_unverified\",",
                    "\"candidate_authority\":\"burn_direct_math\",",
                    "\"reference_authority\":\"burn_math_program\",",
                    "\"reference_program_generation\":\"v9\",",
                    "\"outcome\":\"{}\",",
                    "\"receipt_id\":{},",
                    "\"operation_id\":\"{}\",",
                    "\"label\":\"{}\",",
                    "\"reference_program_identity\":\"{}\",",
                    "\"passed\":{},",
                    "\"detail\":\"{}\"",
                    "}}"
                ),
                if *passed { "passed" } else { "failed" },
                receipt_id,
                json_escape(operation_id),
                json_escape(label),
                json_escape(reference_program_identity),
                passed,
                json_escape(detail),
            ),
            Self::VectorVerifierReceipt {
                receipt_id,
                label,
                passed,
                detail,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"vector_verifier_receipt\",",
                    "\"evidence_authority\":\"observation_only\",",
                    "\"source_authority\":\"wasm_comparator\",",
                    "\"transport_integrity\":\"host_structured_unverified\",",
                    "\"reference_authority\":\"caller_supplied\",",
                    "\"outcome\":\"{}\",",
                    "\"receipt_id\":{},",
                    "\"label\":\"{}\",",
                    "\"passed\":{},",
                    "\"detail\":\"{}\"",
                    "}}"
                ),
                if *passed { "passed" } else { "failed" },
                receipt_id,
                json_escape(label),
                passed,
                json_escape(detail),
            ),
            Self::RevisionDispatchExecutionReceipt {
                receipt_fingerprint,
                request_fingerprint,
                response_intent_fingerprint,
                dispatch_fingerprint,
                lineage_id,
                revision_id,
                revision_key,
                parent_revision_id,
                before_revision_count,
                after_revision_count,
            } => {
                let parent_revision_id = parent_revision_id
                    .as_deref()
                    .map(|value| format!("\"{}\"", json_escape(value)))
                    .unwrap_or_else(|| "null".to_string());
                format!(
                    concat!(
                        "{{",
                        "\"kind\":\"revision_dispatch_execution_receipt\",",
                        "\"evidence_authority\":\"observation_only\",",
                        "\"source_authority\":\"ResolutionRevisionChain\",",
                        "\"source_operation\":\"open_revision\",",
                        "\"transport_integrity\":\"native_typed_correlated\",",
                        "\"outcome\":\"committed\",",
                        "\"receipt_fingerprint\":\"{}\",",
                        "\"request_fingerprint\":\"{}\",",
                        "\"response_intent_fingerprint\":\"{}\",",
                        "\"dispatch_fingerprint\":\"{}\",",
                        "\"lineage_id\":\"{}\",",
                        "\"revision_id\":\"{}\",",
                        "\"revision_key\":\"{}\",",
                        "\"parent_revision_id\":{},",
                        "\"before_revision_count\":{},",
                        "\"after_revision_count\":{},",
                        "\"execution_trigger\":\"explicit_caller_invocation\",",
                        "\"authorization_claim\":\"none\",",
                        "\"source_mutation\":\"committed\"",
                        "}}"
                    ),
                    json_escape(receipt_fingerprint),
                    json_escape(request_fingerprint),
                    json_escape(response_intent_fingerprint),
                    json_escape(dispatch_fingerprint),
                    json_escape(lineage_id),
                    json_escape(revision_id),
                    json_escape(revision_key),
                    parent_revision_id,
                    before_revision_count,
                    after_revision_count,
                )
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEvidence {
    subject: Option<RuntimeEvidenceSubject>,
    payload: RuntimeEvidencePayload,
}

impl RuntimeEvidence {
    pub fn subject(&self) -> Option<&RuntimeEvidenceSubject> {
        self.subject.as_ref()
    }

    pub fn source_authority(&self) -> &'static str {
        self.payload.source_authority()
    }

    pub fn evidence_authority(&self) -> &'static str {
        self.payload.evidence_authority()
    }

    pub fn transport_integrity(&self) -> &'static str {
        self.payload.transport_integrity()
    }

    pub fn kind(&self) -> &'static str {
        self.payload.kind()
    }

    pub fn outcome(&self) -> &'static str {
        self.payload.outcome()
    }

    pub(crate) fn graph_program_identity(&self) -> Option<&str> {
        match &self.payload {
            RuntimeEvidencePayload::GraphVerifierReceipt {
                program_identity, ..
            } => Some(program_identity.as_str()),
            _ => None,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn parse_json_object(
        raw: &str,
        context: &str,
    ) -> Result<serde_json::Value, String> {
        let value: serde_json::Value =
            serde_json::from_str(raw).map_err(|err| format!("{context}: invalid JSON: {err}"))?;
        if !value.is_object() {
            return Err(format!("{context}: root must be a JSON object"));
        }
        Ok(value)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn require_string<'a>(
        value: &'a serde_json::Value,
        field: &str,
        context: &str,
    ) -> Result<&'a str, String> {
        value
            .get(field)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{context}: missing or non-string field {field}"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn require_bool(
        value: &serde_json::Value,
        field: &str,
        context: &str,
    ) -> Result<bool, String> {
        value
            .get(field)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| format!("{context}: missing or non-boolean field {field}"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn require_u64(
        value: &serde_json::Value,
        field: &str,
        context: &str,
    ) -> Result<u64, String> {
        value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("{context}: missing or non-u64 field {field}"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn require_exact_string(
        value: &serde_json::Value,
        field: &str,
        expected: &str,
        context: &str,
    ) -> Result<(), String> {
        let actual = Self::require_string(value, field, context)?;
        if actual != expected {
            return Err(format!(
                "{context}: field {field} must be {expected}, got {actual}"
            ));
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn canonical_graph_program_identity_json(
        value: &serde_json::Value,
        context: &str,
    ) -> Result<String, String> {
        Self::require_exact_string(
            value,
            "schema",
            "burn-research.program-identity.v1",
            context,
        )?;
        let plan_hex = Self::require_string(value, "plan_hex", context)?;
        let fingerprints = value
            .get("layer_init_fingerprints")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!("{context}: program_identity.layer_init_fingerprints must be an array")
            })?;
        let fingerprints = fingerprints
            .iter()
            .enumerate()
            .map(|(index, fingerprint)| {
                fingerprint
                    .as_str()
                    .map(|fingerprint| format!("\"{}\"", json_escape(fingerprint)))
                    .ok_or_else(|| {
                        format!(
                            "{context}: program_identity.layer_init_fingerprints[{index}] must be a string"
                        )
                    })
            })
            .collect::<Result<Vec<_>, String>>()?
            .join(",");

        Ok(format!(
            "{{\"schema\":\"burn-research.program-identity.v1\",\"plan_hex\":\"{}\",\"layer_init_fingerprints\":[{}]}}",
            json_escape(plan_hex),
            fingerprints,
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn runtime_subject_from_receipt(
        value: &serde_json::Value,
        context: &str,
    ) -> Result<Option<RuntimeEvidenceSubject>, String> {
        let subject = value
            .get("runtime_subject")
            .ok_or_else(|| format!("{context}: missing runtime_subject"))?;
        let status = Self::require_string(subject, "status", context)?;
        if status == "unbound" {
            return Ok(None);
        }
        if status != "bound" {
            return Err(format!(
                "{context}: runtime_subject.status must be bound|unbound, got {status}"
            ));
        }

        Ok(Some(RuntimeEvidenceSubject {
            intent_id: Self::require_string(subject, "intent_id", context)?.to_string(),
            workflow_revision: Self::require_u64(subject, "workflow_revision", context)?,
            approval_id: Self::require_string(subject, "approval_id", context)?.to_string(),
            subject_kind: Self::require_string(subject, "subject_kind", context)?.to_string(),
            subject_identity: Self::require_string(subject, "subject_identity", context)?
                .to_string(),
            authorization_policy_id: Self::require_string(
                subject,
                "authorization_policy_id",
                context,
            )?
            .to_string(),
            authorization_policy_revision: Self::require_u64(
                subject,
                "authorization_policy_revision",
                context,
            )?,
            authorization_is_revision: Self::require_bool(
                subject,
                "authorization_is_revision",
                context,
            )?,
        }))
    }

    /// Adapt the exact structured AgentFault envelope emitted by the WASM preflight surface.
    ///
    /// This parser never inspects legacy error text to choose an evidence class. The payload class
    /// is fixed by this entry point and the source schema id is validated before projection.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_bound_agent_fault_envelope(
        projection: &RuntimeSubjectProjection,
        envelope_json: &str,
    ) -> Result<Self, String> {
        const CONTEXT: &str = "RuntimeEvidence.from_bound_agent_fault_envelope";
        let envelope = Self::parse_json_object(envelope_json, CONTEXT)?;
        Self::require_exact_string(
            &envelope,
            "schema_id",
            "burn-research.agent-fault.v1",
            CONTEXT,
        )?;
        Self::require_exact_string(&envelope, "status", "fault", CONTEXT)?;
        let fault = envelope
            .get("fault")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing fault object"))?;

        if let Some(mutation) = fault.get("mutation").and_then(serde_json::Value::as_str) {
            if mutation != "none" {
                return Err(format!(
                    "{CONTEXT}: AgentFault mutation must be none, got {mutation}"
                ));
            }
        }

        Self::bound_agent_fault(
            projection,
            Self::require_string(fault, "code", CONTEXT)?,
            Self::require_string(fault, "class", CONTEXT)?,
            Self::require_string(fault, "operation", CONTEXT)?,
            Self::require_string(fault, "predicate", CONTEXT)?,
            Self::require_bool(fault, "recoverable", CONTEXT)?,
            Self::require_string(fault, "message", CONTEXT)?,
        )
    }

    /// Adapt the exact structured Burn-backed graph verifier receipt emitted by WASM.
    ///
    /// The runtime subject is read from the receipt itself. The result object is retained as a
    /// compact structured detail string; the original receipt remains the authoritative artifact.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_graph_verifier_receipt_json(receipt_json: &str) -> Result<Self, String> {
        const CONTEXT: &str = "RuntimeEvidence.from_graph_verifier_receipt_json";
        let receipt = Self::parse_json_object(receipt_json, CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "schema_id",
            "burn-research.verifier-receipt.v1",
            CONTEXT,
        )?;
        Self::require_exact_string(&receipt, "authority", "wasm_verifier", CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "verifier",
            "CompiledGraph.verifyFlat",
            CONTEXT,
        )?;
        Self::require_exact_string(
            &receipt,
            "reference_authority",
            "burn_compiled_graph",
            CONTEXT,
        )?;

        let subject = Self::runtime_subject_from_receipt(&receipt, CONTEXT)?;
        let receipt_id = Self::require_u64(&receipt, "receipt_id", CONTEXT)?;
        let receipt_id = u32::try_from(receipt_id)
            .map_err(|_| format!("{CONTEXT}: receipt_id exceeds u32"))?;
        let label = Self::require_string(&receipt, "label", CONTEXT)?;
        let program_identity = receipt
            .get("program_identity")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing program_identity object"))?;
        Self::require_exact_string(
            program_identity,
            "schema",
            "burn-research.program-identity.v1",
            CONTEXT,
        )?;
        let canonical_program_identity =
            Self::canonical_graph_program_identity_json(program_identity, CONTEXT)?;
        let result = receipt
            .get("result")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing result object"))?;
        let passed = Self::require_bool(result, "passed", CONTEXT)?;

        match (
            receipt.get("semantic_execution_context"),
            receipt.get("semantic_context_fingerprint"),
        ) {
            (None, None) => Self::graph_verifier_receipt(
                subject,
                receipt_id,
                label,
                canonical_program_identity.clone(),
                passed,
                result.to_string(),
            ),
            (Some(semantic_context), Some(receipt_fingerprint)) => {
                if !semantic_context.is_object() {
                    return Err(format!(
                        "{CONTEXT}: semantic_execution_context must be a JSON object"
                    ));
                }
                Self::require_exact_string(
                    semantic_context,
                    "schema_id",
                    "burn-research.semantic-execution-context.v1",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    semantic_context,
                    "program_identity_effect",
                    "none",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    semantic_context,
                    "execution_effect",
                    "none",
                    CONTEXT,
                )?;
                if !Self::require_bool(semantic_context, "projection_only", CONTEXT)? {
                    return Err(format!(
                        "{CONTEXT}: semantic_execution_context.projection_only must be true"
                    ));
                }
                Self::require_exact_string(
                    semantic_context,
                    "fingerprint_algorithm",
                    "fnv1a64_noncryptographic",
                    CONTEXT,
                )?;
                let context_authority = semantic_context
                    .get("authority")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        format!("{CONTEXT}: semantic_execution_context missing authority object")
                    })?;
                Self::require_exact_string(
                    context_authority,
                    "execution_identity",
                    "CompiledGraph.programIdentity",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    context_authority,
                    "semantic_graph",
                    "AgentGraphBuilder.semanticGraphIdentity",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    context_authority,
                    "semantic_lifecycle",
                    "AgentGraphBuilder.semanticLifecycleIdentity",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    context_authority,
                    "context",
                    "derived_projection",
                    CONTEXT,
                )?;

                let context_program_identity = semantic_context
                    .get("program_identity")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        format!(
                            "{CONTEXT}: semantic_execution_context missing program_identity object"
                        )
                    })?;
                if context_program_identity != program_identity {
                    return Err(format!(
                        "{CONTEXT}: semantic execution context program_identity does not match receipt program_identity"
                    ));
                }

                let semantic_graph_identity = semantic_context
                    .get("semantic_graph_identity")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        format!(
                            "{CONTEXT}: semantic_execution_context missing semantic_graph_identity object"
                        )
                    })?;
                Self::require_exact_string(
                    semantic_graph_identity,
                    "schema_id",
                    "burn-research.semantic-graph-identity.v1",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    semantic_graph_identity,
                    "execution_program_identity_effect",
                    "none",
                    CONTEXT,
                )?;

                let semantic_lifecycle_identity = semantic_context
                    .get("semantic_lifecycle_identity")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        format!(
                            "{CONTEXT}: semantic_execution_context missing semantic_lifecycle_identity object"
                        )
                    })?;
                Self::require_exact_string(
                    semantic_lifecycle_identity,
                    "schema_id",
                    "burn-research.semantic-lifecycle-identity.v1",
                    CONTEXT,
                )?;
                Self::require_exact_string(
                    semantic_lifecycle_identity,
                    "execution_program_identity_effect",
                    "none",
                    CONTEXT,
                )?;

                let lifecycle_base = semantic_lifecycle_identity
                    .get("base_semantic_graph_identity")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        format!(
                            "{CONTEXT}: semantic lifecycle identity missing base_semantic_graph_identity object"
                        )
                    })?;
                if lifecycle_base != semantic_graph_identity {
                    return Err(format!(
                        "{CONTEXT}: semantic lifecycle base graph identity does not match semantic_graph_identity"
                    ));
                }

                let context_fingerprint =
                    Self::require_string(semantic_context, "context_fingerprint", CONTEXT)?;
                let receipt_fingerprint = receipt_fingerprint.as_str().ok_or_else(|| {
                    format!("{CONTEXT}: semantic_context_fingerprint must be a string")
                })?;
                if receipt_fingerprint != context_fingerprint {
                    return Err(format!(
                        "{CONTEXT}: semantic_context_fingerprint does not match semantic_execution_context.context_fingerprint"
                    ));
                }

                Self::semantic_graph_verifier_receipt(
                    subject,
                    receipt_id,
                    label,
                    canonical_program_identity,
                    semantic_context.to_string(),
                    receipt_fingerprint,
                    passed,
                    result.to_string(),
                )
            }
            _ => Err(format!(
                "{CONTEXT}: semantic_execution_context and semantic_context_fingerprint must be present together"
            )),
        }
    }

    /// Adapt the exact structured MathProgram verifier receipt emitted by WASM.
    ///
    /// The receipt must preserve the MathProgram verifier/reference authority tuple exactly.
    /// The original runtime receipt remains authoritative; this adapter creates observation-only
    /// reverse evidence and does not authenticate host transport.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_math_program_verifier_receipt_json(
        receipt_json: &str,
    ) -> Result<Self, String> {
        const CONTEXT: &str =
            "RuntimeEvidence.from_math_program_verifier_receipt_json";
        let receipt = Self::parse_json_object(receipt_json, CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "schema_id",
            "burn-research.verifier-receipt.v1",
            CONTEXT,
        )?;
        Self::require_exact_string(&receipt, "authority", "wasm_verifier", CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "verifier",
            "MathProgram.verifyFlat",
            CONTEXT,
        )?;
        Self::require_exact_string(
            &receipt,
            "reference_authority",
            "burn_math_program",
            CONTEXT,
        )?;

        let subject = Self::runtime_subject_from_receipt(&receipt, CONTEXT)?;
        let receipt_id = Self::require_u64(&receipt, "receipt_id", CONTEXT)?;
        let receipt_id = u32::try_from(receipt_id)
            .map_err(|_| format!("{CONTEXT}: receipt_id exceeds u32"))?;
        let label = Self::require_string(&receipt, "label", CONTEXT)?;
        let program_identity = receipt
            .get("program_identity")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing program_identity object"))?;
        Self::require_exact_string(
            program_identity,
            "schema",
            "burn-research.math-program-identity.v1",
            CONTEXT,
        )?;
        let result = receipt
            .get("result")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing result object"))?;
        let passed = Self::require_bool(result, "passed", CONTEXT)?;

        Self::math_program_verifier_receipt(
            subject,
            receipt_id,
            label,
            program_identity.to_string(),
            passed,
            result.to_string(),
        )
    }

    /// Adapt the exact structured direct-math verifier receipt emitted by WASM.
    ///
    /// This adapter fixes the full authority class. A caller cannot relabel graph,
    /// MathProgram-only, or vector-comparator receipts as direct-math evidence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_direct_math_verifier_receipt_json(
        receipt_json: &str,
    ) -> Result<Self, String> {
        const CONTEXT: &str =
            "RuntimeEvidence.from_direct_math_verifier_receipt_json";
        let receipt = Self::parse_json_object(receipt_json, CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "schema_id",
            "burn-research.verifier-receipt.v1",
            CONTEXT,
        )?;
        Self::require_exact_string(&receipt, "authority", "wasm_verifier", CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "verifier",
            "DirectMath.verifyAgainstMathProgramV9",
            CONTEXT,
        )?;
        Self::require_exact_string(
            &receipt,
            "candidate_authority",
            "burn_direct_math",
            CONTEXT,
        )?;
        Self::require_exact_string(
            &receipt,
            "reference_authority",
            "burn_math_program",
            CONTEXT,
        )?;
        Self::require_exact_string(
            &receipt,
            "reference_program_generation",
            "v9",
            CONTEXT,
        )?;

        let subject = Self::runtime_subject_from_receipt(&receipt, CONTEXT)?;
        let receipt_id = Self::require_u64(&receipt, "receipt_id", CONTEXT)?;
        let receipt_id = u32::try_from(receipt_id)
            .map_err(|_| format!("{CONTEXT}: receipt_id exceeds u32"))?;
        let operation_id = Self::require_string(&receipt, "operation_id", CONTEXT)?;
        let label = Self::require_string(&receipt, "label", CONTEXT)?;
        let reference_program_identity = receipt
            .get("program_identity")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing program_identity object"))?;
        Self::require_exact_string(
            reference_program_identity,
            "schema",
            "burn-research.math-program-identity.v1",
            CONTEXT,
        )?;
        let result = receipt
            .get("result")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing result object"))?;
        let passed = Self::require_bool(result, "passed", CONTEXT)?;

        Self::direct_math_verifier_receipt(
            subject,
            receipt_id,
            operation_id,
            label,
            reference_program_identity.to_string(),
            passed,
            result.to_string(),
        )
    }

    /// Adapt the exact structured vector-comparator receipt emitted by WASM.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_vector_verifier_receipt_json(receipt_json: &str) -> Result<Self, String> {
        const CONTEXT: &str = "RuntimeEvidence.from_vector_verifier_receipt_json";
        let receipt = Self::parse_json_object(receipt_json, CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "schema_id",
            "burn-research.verifier-receipt.v1",
            CONTEXT,
        )?;
        Self::require_exact_string(&receipt, "authority", "wasm_comparator", CONTEXT)?;
        Self::require_exact_string(&receipt, "verifier", "mathVerifyVectors", CONTEXT)?;
        Self::require_exact_string(
            &receipt,
            "reference_authority",
            "caller_supplied",
            CONTEXT,
        )?;

        let subject = Self::runtime_subject_from_receipt(&receipt, CONTEXT)?;
        let receipt_id = Self::require_u64(&receipt, "receipt_id", CONTEXT)?;
        let receipt_id = u32::try_from(receipt_id)
            .map_err(|_| format!("{CONTEXT}: receipt_id exceeds u32"))?;
        let label = Self::require_string(&receipt, "label", CONTEXT)?;
        let result = receipt
            .get("result")
            .filter(|value| value.is_object())
            .ok_or_else(|| format!("{CONTEXT}: missing result object"))?;
        let passed = Self::require_bool(result, "passed", CONTEXT)?;

        Self::vector_verifier_receipt(subject, receipt_id, label, passed, result.to_string())
    }

    pub fn agent_fault(
        subject: Option<RuntimeEvidenceSubject>,
        code: impl Into<String>,
        class: impl Into<String>,
        operation: impl Into<String>,
        predicate: impl Into<String>,
        recoverable: bool,
        message: impl Into<String>,
    ) -> Result<Self, String> {
        let code = code.into();
        let class = class.into();
        let operation = operation.into();
        let predicate = predicate.into();
        let message = message.into();

        validate_nonempty_bounded(&code, MAX_SHORT_FIELD_BYTES, "RuntimeEvidence.agent_fault.code")?;
        validate_nonempty_bounded(
            &class,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.class",
        )?;
        validate_nonempty_bounded(
            &operation,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.operation",
        )?;
        validate_nonempty_bounded(
            &predicate,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.predicate",
        )?;
        validate_nonempty_bounded(
            &message,
            MAX_DETAIL_BYTES,
            "RuntimeEvidence.agent_fault.message",
        )?;

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::AgentFault {
                code,
                class,
                operation,
                predicate,
                recoverable,
                message,
            },
        })
    }

    fn graph_verifier_receipt_internal(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: String,
        program_identity: String,
        semantic_execution_context: Option<String>,
        semantic_context_fingerprint: Option<String>,
        passed: bool,
        detail: String,
        context: &str,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err(format!("{context}: receipt id must be > 0"));
        }
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            &format!("{context}.label"),
        )?;
        validate_nonempty_bounded(
            &program_identity,
            MAX_PROGRAM_IDENTITY_BYTES,
            &format!("{context}.program_identity"),
        )?;
        match (
            semantic_execution_context.as_deref(),
            semantic_context_fingerprint.as_deref(),
        ) {
            (None, None) => {}
            (Some(semantic_context), Some(fingerprint)) => {
                validate_nonempty_bounded(
                    semantic_context,
                    MAX_SEMANTIC_EXECUTION_CONTEXT_BYTES,
                    &format!("{context}.semantic_execution_context"),
                )?;
                validate_nonempty_bounded(
                    fingerprint,
                    MAX_SHORT_FIELD_BYTES,
                    &format!("{context}.semantic_context_fingerprint"),
                )?;
            }
            _ => {
                return Err(format!(
                    "{context}: semantic_execution_context and semantic_context_fingerprint must be supplied together"
                ))
            }
        }
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "{context}.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::GraphVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                semantic_execution_context,
                semantic_context_fingerprint,
                passed,
                detail,
            },
        })
    }

    pub fn graph_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::graph_verifier_receipt_internal(
            subject,
            receipt_id,
            label.into(),
            program_identity.into(),
            None,
            None,
            passed,
            detail.into(),
            "RuntimeEvidence.graph_verifier_receipt",
        )
    }

    pub fn semantic_graph_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        semantic_execution_context: impl Into<String>,
        semantic_context_fingerprint: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::graph_verifier_receipt_internal(
            subject,
            receipt_id,
            label.into(),
            program_identity.into(),
            Some(semantic_execution_context.into()),
            Some(semantic_context_fingerprint.into()),
            passed,
            detail.into(),
            "RuntimeEvidence.semantic_graph_verifier_receipt",
        )
    }

    pub fn math_program_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err(
                "RuntimeEvidence.math_program_verifier_receipt: receipt id must be > 0"
                    .to_string(),
            );
        }
        let label = label.into();
        let program_identity = program_identity.into();
        let detail = detail.into();
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.math_program_verifier_receipt.label",
        )?;
        validate_nonempty_bounded(
            &program_identity,
            MAX_PROGRAM_IDENTITY_BYTES,
            "RuntimeEvidence.math_program_verifier_receipt.program_identity",
        )?;
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "RuntimeEvidence.math_program_verifier_receipt.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::MathProgramVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                passed,
                detail,
            },
        })
    }

    pub fn direct_math_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        operation_id: impl Into<String>,
        label: impl Into<String>,
        reference_program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err(
                "RuntimeEvidence.direct_math_verifier_receipt: receipt id must be > 0"
                    .to_string(),
            );
        }
        let operation_id = operation_id.into();
        let label = label.into();
        let reference_program_identity = reference_program_identity.into();
        let detail = detail.into();

        validate_nonempty_bounded(
            &operation_id,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.direct_math_verifier_receipt.operation_id",
        )?;
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.direct_math_verifier_receipt.label",
        )?;
        validate_nonempty_bounded(
            &reference_program_identity,
            MAX_PROGRAM_IDENTITY_BYTES,
            "RuntimeEvidence.direct_math_verifier_receipt.reference_program_identity",
        )?;
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "RuntimeEvidence.direct_math_verifier_receipt.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::DirectMathVerifierReceipt {
                receipt_id,
                operation_id,
                label,
                reference_program_identity,
                passed,
                detail,
            },
        })
    }

    pub fn vector_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err("RuntimeEvidence.vector_verifier_receipt: receipt id must be > 0".to_string());
        }
        let label = label.into();
        let detail = detail.into();
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.vector_verifier_receipt.label",
        )?;
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "RuntimeEvidence.vector_verifier_receipt.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::VectorVerifierReceipt {
                receipt_id,
                label,
                passed,
                detail,
            },
        })
    }

    pub fn bound_agent_fault(
        projection: &RuntimeSubjectProjection,
        code: impl Into<String>,
        class: impl Into<String>,
        operation: impl Into<String>,
        predicate: impl Into<String>,
        recoverable: bool,
        message: impl Into<String>,
    ) -> Result<Self, String> {
        Self::agent_fault(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            code,
            class,
            operation,
            predicate,
            recoverable,
            message,
        )
    }

    pub fn bound_graph_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::graph_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            program_identity,
            passed,
            detail,
        )
    }

    pub fn bound_semantic_graph_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        semantic_execution_context: impl Into<String>,
        semantic_context_fingerprint: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::semantic_graph_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            program_identity,
            semantic_execution_context,
            semantic_context_fingerprint,
            passed,
            detail,
        )
    }

    pub fn bound_math_program_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::math_program_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            program_identity,
            passed,
            detail,
        )
    }

    pub fn bound_direct_math_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        operation_id: impl Into<String>,
        label: impl Into<String>,
        reference_program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::direct_math_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            operation_id,
            label,
            reference_program_identity,
            passed,
            detail,
        )
    }

    pub fn bound_vector_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::vector_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            passed,
            detail,
        )
    }

    pub(crate) fn bound_revision_dispatch_execution_receipt_for_inbox(
        inbox: &ResolutionEvidenceInbox,
        receipt_fingerprint: impl Into<String>,
        request_fingerprint: impl Into<String>,
        response_intent_fingerprint: impl Into<String>,
        dispatch_fingerprint: impl Into<String>,
        lineage_id: impl Into<String>,
        revision_id: impl Into<String>,
        revision_key: impl Into<String>,
        parent_revision_id: Option<String>,
        before_revision_count: usize,
        after_revision_count: usize,
    ) -> Result<Self, String> {
        const CONTEXT: &str =
            "RuntimeEvidence.bound_revision_dispatch_execution_receipt_for_inbox";
        if after_revision_count != before_revision_count.saturating_add(1) {
            return Err(format!(
                "{CONTEXT}: revision count delta must be exactly one, got {before_revision_count}->{after_revision_count}"
            ));
        }
        let receipt_fingerprint = receipt_fingerprint.into();
        let request_fingerprint = request_fingerprint.into();
        let response_intent_fingerprint = response_intent_fingerprint.into();
        let dispatch_fingerprint = dispatch_fingerprint.into();
        let lineage_id = lineage_id.into();
        let revision_id = revision_id.into();
        let revision_key = revision_key.into();
        for (value, field) in [
            (&receipt_fingerprint, "receipt_fingerprint"),
            (&request_fingerprint, "request_fingerprint"),
            (&response_intent_fingerprint, "response_intent_fingerprint"),
            (&dispatch_fingerprint, "dispatch_fingerprint"),
            (&lineage_id, "lineage_id"),
            (&revision_id, "revision_id"),
            (&revision_key, "revision_key"),
        ] {
            validate_nonempty_bounded(
                value,
                MAX_SHORT_FIELD_BYTES,
                &format!("{CONTEXT}.{field}"),
            )?;
        }
        if let Some(parent_revision_id) = parent_revision_id.as_deref() {
            validate_nonempty_bounded(
                parent_revision_id,
                MAX_SHORT_FIELD_BYTES,
                &format!("{CONTEXT}.parent_revision_id"),
            )?;
        }
        Ok(Self {
            subject: Some(inbox.target.clone()),
            payload: RuntimeEvidencePayload::RevisionDispatchExecutionReceipt {
                receipt_fingerprint,
                request_fingerprint,
                response_intent_fingerprint,
                dispatch_fingerprint,
                lineage_id,
                revision_id,
                revision_key,
                parent_revision_id,
                before_revision_count,
                after_revision_count,
            },
        })
    }

    pub fn to_json(&self) -> String {
        let subject = self
            .subject
            .as_ref()
            .map(RuntimeEvidenceSubject::to_json)
            .unwrap_or_else(|| "{\"status\":\"unbound\"}".to_string());
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.runtime-evidence.v1\",",
                "\"subject\":{},",
                "\"payload\":{}",
                "}}"
            ),
            subject,
            self.payload.to_json(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejoinStatus {
    Exact,
    Unbound,
    ForeignIntent,
    StaleRuntimeRevision,
    FutureRuntimeRevision,
    ApprovalMismatch,
    SubjectMismatch,
    AuthorizationMismatch,
}

impl RejoinStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Unbound => "unbound",
            Self::ForeignIntent => "foreign_intent",
            Self::StaleRuntimeRevision => "stale_runtime_revision",
            Self::FutureRuntimeRevision => "future_runtime_revision",
            Self::ApprovalMismatch => "approval_mismatch",
            Self::SubjectMismatch => "subject_mismatch",
            Self::AuthorizationMismatch => "authorization_mismatch",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionEvidenceInbox {
    target: RuntimeEvidenceSubject,
    entries: Vec<RuntimeEvidence>,
}

impl ResolutionEvidenceInbox {
    pub fn from_authorized(
        resolution: &ResolutionSnapshot,
        approved: &ApprovedEffectiveSpec,
        policy: &AuthorizationPolicy,
        authorization: &AuthorizationSnapshot,
    ) -> Result<Self, String> {
        let projection =
            RuntimeSubjectProjection::from_authorized(approved, policy, authorization)?;
        Self::new(resolution, &projection)
    }

    pub fn new(
        resolution: &ResolutionSnapshot,
        projection: &RuntimeSubjectProjection,
    ) -> Result<Self, String> {
        if !resolution.compile_eligible() {
            return Err(
                "ResolutionEvidenceInbox: ResolutionSnapshot must be compile-eligible".to_string(),
            );
        }
        if resolution.intent_id != projection.intent_id {
            return Err(format!(
                "ResolutionEvidenceInbox: resolution intent {} does not match runtime intent {}",
                resolution.intent_id, projection.intent_id
            ));
        }
        if resolution.revision != projection.workflow_revision {
            return Err(format!(
                "ResolutionEvidenceInbox: resolution revision {} does not match runtime revision {}",
                resolution.revision, projection.workflow_revision
            ));
        }

        Ok(Self {
            target: RuntimeEvidenceSubject::from_projection(projection),
            entries: Vec::new(),
        })
    }

    pub fn classify(&self, evidence: &RuntimeEvidence) -> RejoinStatus {
        let Some(subject) = &evidence.subject else {
            return RejoinStatus::Unbound;
        };
        if subject.intent_id != self.target.intent_id {
            return RejoinStatus::ForeignIntent;
        }
        if subject.workflow_revision < self.target.workflow_revision {
            return RejoinStatus::StaleRuntimeRevision;
        }
        if subject.workflow_revision > self.target.workflow_revision {
            return RejoinStatus::FutureRuntimeRevision;
        }
        if subject.approval_id != self.target.approval_id {
            return RejoinStatus::ApprovalMismatch;
        }
        if subject.subject_kind != self.target.subject_kind
            || subject.subject_identity != self.target.subject_identity
        {
            return RejoinStatus::SubjectMismatch;
        }
        if subject.authorization_policy_id != self.target.authorization_policy_id
            || subject.authorization_policy_revision != self.target.authorization_policy_revision
            || subject.authorization_is_revision != self.target.authorization_is_revision
        {
            return RejoinStatus::AuthorizationMismatch;
        }
        RejoinStatus::Exact
    }

    pub fn record(&mut self, evidence: RuntimeEvidence) -> Result<bool, String> {
        let status = self.classify(&evidence);
        if status != RejoinStatus::Exact {
            return Err(format!(
                "ResolutionEvidenceInbox: runtime evidence rejoin status {} is not admissible",
                status.as_str()
            ));
        }
        if self.entries.iter().any(|existing| existing == &evidence) {
            return Ok(false);
        }
        if self.entries.len() >= MAX_RUNTIME_EVIDENCE_ENTRIES {
            return Err(format!(
                "ResolutionEvidenceInbox: evidence limit {MAX_RUNTIME_EVIDENCE_ENTRIES} reached"
            ));
        }
        self.entries.push(evidence);
        Ok(true)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn observations(&self) -> &[RuntimeEvidence] {
        &self.entries
    }

    pub(crate) fn target_intent_id(&self) -> &str {
        &self.target.intent_id
    }

    pub(crate) fn target_workflow_revision(&self) -> u64 {
        self.target.workflow_revision
    }

    pub(crate) fn target_approval_id(&self) -> &str {
        &self.target.approval_id
    }

    pub(crate) fn target_matches_workspace(&self, workspace: &AgentWorkspace) -> bool {
        let Some(binding) = workspace.runtime_subject_binding() else {
            return false;
        };
        binding.intent_id == self.target.intent_id
            && binding.workflow_revision == self.target.workflow_revision
            && binding.approval_id == self.target.approval_id
            && binding.subject_kind == self.target.subject_kind
            && binding.subject_identity == self.target.subject_identity
            && binding.authorization_policy_id == self.target.authorization_policy_id
            && binding.authorization_policy_revision == self.target.authorization_policy_revision
            && binding.authorization_is_revision == self.target.authorization_is_revision
    }

    pub fn to_json(&self) -> String {
        let mut fault_count = 0usize;
        let mut graph_passed = 0usize;
        let mut graph_failed = 0usize;
        let mut graph_semantic_context = 0usize;
        let mut math_program_passed = 0usize;
        let mut math_program_failed = 0usize;
        let mut direct_math_passed = 0usize;
        let mut direct_math_failed = 0usize;
        let mut vector_passed = 0usize;
        let mut vector_failed = 0usize;
        let mut revision_dispatch_committed = 0usize;

        for evidence in &self.entries {
            match &evidence.payload {
                RuntimeEvidencePayload::AgentFault { .. } => fault_count += 1,
                RuntimeEvidencePayload::GraphVerifierReceipt {
                    semantic_execution_context,
                    passed,
                    ..
                } => {
                    if semantic_execution_context.is_some() {
                        graph_semantic_context += 1;
                    }
                    if *passed {
                        graph_passed += 1;
                    } else {
                        graph_failed += 1;
                    }
                }
                RuntimeEvidencePayload::MathProgramVerifierReceipt { passed, .. } => {
                    if *passed {
                        math_program_passed += 1;
                    } else {
                        math_program_failed += 1;
                    }
                }
                RuntimeEvidencePayload::DirectMathVerifierReceipt { passed, .. } => {
                    if *passed {
                        direct_math_passed += 1;
                    } else {
                        direct_math_failed += 1;
                    }
                }
                RuntimeEvidencePayload::VectorVerifierReceipt { passed, .. } => {
                    if *passed {
                        vector_passed += 1;
                    } else {
                        vector_failed += 1;
                    }
                }
                RuntimeEvidencePayload::RevisionDispatchExecutionReceipt { .. } => {
                    revision_dispatch_committed += 1;
                }
            }
        }

        let entries = self
            .entries
            .iter()
            .map(RuntimeEvidence::to_json)
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.resolution-evidence-inbox.v1\",",
                "\"target\":{},",
                "\"entry_count\":{},",
                "\"max_entries\":{},",
                "\"summary\":{{",
                    "\"agent_faults\":{},",
                    "\"graph_verifier_passed\":{},",
                    "\"graph_verifier_failed\":{},",
                    "\"graph_verifier_with_semantic_context\":{},",
                    "\"math_program_verifier_passed\":{},",
                    "\"math_program_verifier_failed\":{},",
                    "\"direct_math_verifier_passed\":{},",
                    "\"direct_math_verifier_failed\":{},",
                    "\"vector_verifier_passed\":{},",
                    "\"vector_verifier_failed\":{},",
                    "\"revision_dispatch_committed\":{}",
                "}},",
                "\"resolution_effect\":{{",
                    "\"diagnostic_created\":false,",
                    "\"state_transition\":\"none\",",
                    "\"revision_created\":false,",
                    "\"action_selected\":false,",
                    "\"interpretation_required\":true",
                "}},",
                "\"entries\":[{}]",
                "}}"
            ),
            self.target.to_json(),
            self.entries.len(),
            MAX_RUNTIME_EVIDENCE_ENTRIES,
            fault_count,
            graph_passed,
            graph_failed,
            graph_semantic_context,
            math_program_passed,
            math_program_failed,
            direct_math_passed,
            direct_math_failed,
            vector_passed,
            vector_failed,
            revision_dispatch_committed,
            entries,
        )
    }
}

#[wasm_bindgen(js_name = runtimeResolutionEvidenceCapabilities)]
pub fn runtime_resolution_evidence_capabilities() -> String {
    RUNTIME_RESOLUTION_EVIDENCE_V1.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        RejoinStatus, ResolutionEvidenceInbox, RuntimeEvidence, RuntimeEvidenceSubject,
        MAX_RUNTIME_EVIDENCE_ENTRIES,
    };
    use crate::resolution::ResolutionWorkflow;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;

    fn resolved_snapshot(intent_id: &str) -> crate::resolution::ResolutionSnapshot {
        let mut workflow = ResolutionWorkflow::new(intent_id).unwrap();
        workflow.submit().unwrap();
        workflow.finalize_resolution().unwrap();
        workflow.snapshot()
    }

    fn projection(
        intent_id: &str,
        workflow_revision: u64,
        subject_identity: &str,
    ) -> RuntimeSubjectProjection {
        RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: intent_id.to_string(),
            workflow_revision,
            approval_id: "approval-1".to_string(),
            subject_kind: "effective-spec".to_string(),
            subject_identity: subject_identity.to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: subject_identity.to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy".to_string(),
            authorization_policy_revision: 3,
            authorization_is_revision: false,
            approver: "owner".to_string(),
            fields: Vec::new(),
        }
    }

    fn semantic_graph_receipt(
        projection: &RuntimeSubjectProjection,
    ) -> serde_json::Value {
        let program_identity = serde_json::json!({
            "schema": "burn-research.program-identity.v1",
            "plan_hex": "0100000000000100000001000000",
            "layer_init_fingerprints": ["type=01;id=1;variant=ff;flags=00;payload="]
        });
        let semantic_graph_identity = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.semantic-graph-identity.v1",
            "topology_authority": "AgentGraphBuilder",
            "semantic_binding_count": 1,
            "execution_program_identity_effect": "none",
            "fingerprint": "fnv1a64:graph"
        });
        let semantic_lifecycle_identity = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.semantic-lifecycle-identity.v1",
            "base_semantic_graph_identity": semantic_graph_identity.clone(),
            "transition_count": 1,
            "execution_program_identity_effect": "none",
            "fingerprint": "fnv1a64:lifecycle"
        });
        let semantic_execution_context = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.semantic-execution-context.v1",
            "projection_only": true,
            "authority": {
                "execution_identity": "CompiledGraph.programIdentity",
                "semantic_graph": "AgentGraphBuilder.semanticGraphIdentity",
                "semantic_lifecycle": "AgentGraphBuilder.semanticLifecycleIdentity",
                "context": "derived_projection"
            },
            "program_identity": program_identity.clone(),
            "semantic_graph_identity": semantic_graph_identity,
            "semantic_lifecycle_identity": semantic_lifecycle_identity,
            "lifecycle_coverage_complete": true,
            "context_fingerprint": "fnv1a64:semantic-context",
            "fingerprint_algorithm": "fnv1a64_noncryptographic",
            "program_identity_effect": "none",
            "execution_effect": "none"
        });

        serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 17,
            "authority": "wasm_verifier",
            "verifier": "CompiledGraph.verifyFlat",
            "reference_authority": "burn_compiled_graph",
            "label": "semantic-graph-check",
            "fingerprint_algorithm": "fnv1a64_noncryptographic",
            "program_identity": program_identity,
            "program_identity_fingerprint": "fnv1a64:program",
            "mutable_state_in_program_identity": false,
            "runtime_subject": {
                "status": "bound",
                "intent_id": projection.intent_id.clone(),
                "workflow_revision": projection.workflow_revision,
                "approval_id": projection.approval_id.clone(),
                "subject_kind": projection.subject_kind.clone(),
                "subject_identity": projection.subject_identity.clone(),
                "authorization_policy_id": projection.authorization_policy_id.clone(),
                "authorization_policy_revision": projection.authorization_policy_revision,
                "authorization_is_revision": projection.authorization_is_revision
            },
            "semantic_execution_context": semantic_execution_context,
            "semantic_context_fingerprint": "fnv1a64:semantic-context",
            "input_fingerprint": "fnv1a64:input",
            "reference_fingerprint": "fnv1a64:reference",
            "candidate_fingerprint": "fnv1a64:candidate",
            "tolerances": {"abs": 0.0, "rel": 0.0},
            "result": {
                "schema_version": 1,
                "metric": "max_abs",
                "passed": true,
                "checked": 2,
                "finite_checked": 2,
                "max_abs": 0.0,
                "max_rel": 0.0
            }
        })
    }

    #[test]
    fn exact_fault_rejoins_without_mutating_resolution_state() {
        let snapshot = resolved_snapshot("intent-a");
        let before = snapshot.clone();
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let fault = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_LAYOUT_PREFLIGHT",
            "semantic_precondition",
            "workspaceInitUnary",
            "layout.compatible",
            true,
            "known incompatible layout",
        )
        .unwrap();

        assert_eq!(inbox.classify(&fault), RejoinStatus::Exact);
        assert!(inbox.record(fault).unwrap());
        assert_eq!(inbox.len(), 1);
        assert_eq!(snapshot, before);

        let json = inbox.to_json();
        assert!(json.contains("\"agent_faults\":1"));
        assert!(json.contains("\"diagnostic_created\":false"));
        assert!(json.contains("\"state_transition\":\"none\""));
    }

    #[test]
    fn failed_verifier_is_evidence_not_automatic_resolution_failure() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "graph-check",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"abc\"}",
            false,
            "candidate mismatch",
        )
        .unwrap();

        assert!(inbox.record(evidence).unwrap());
        let json = inbox.to_json();
        assert!(json.contains("\"graph_verifier_failed\":1"));
        assert!(json.contains("\"interpretation_required\":true"));
        assert!(snapshot.compile_eligible());
    }

    #[test]
    fn semantic_graph_receipt_rejoins_with_lineage_as_observation_only() {
        let snapshot = resolved_snapshot("intent-a");
        let before = snapshot.clone();
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let receipt = semantic_graph_receipt(&projection);

        let evidence =
            RuntimeEvidence::from_graph_verifier_receipt_json(&receipt.to_string()).unwrap();
        assert_eq!(evidence.kind(), "graph_verifier_receipt");
        assert_eq!(evidence.source_authority(), "wasm_verifier");
        assert_eq!(evidence.evidence_authority(), "observation_only");
        assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
        assert_eq!(evidence.outcome(), "passed");

        let evidence_json: serde_json::Value =
            serde_json::from_str(&evidence.to_json()).unwrap();
        assert_eq!(
            evidence_json["payload"]["semantic_context_fingerprint"],
            "fnv1a64:semantic-context"
        );
        let transported_context = evidence_json["payload"]["semantic_execution_context"]
            .as_str()
            .unwrap();
        let transported_context: serde_json::Value =
            serde_json::from_str(transported_context).unwrap();
        assert_eq!(
            transported_context["schema_id"],
            "burn-research.semantic-execution-context.v1"
        );
        assert_eq!(
            transported_context["semantic_graph_identity"]["fingerprint"],
            "fnv1a64:graph"
        );
        assert_eq!(
            transported_context["semantic_lifecycle_identity"]["fingerprint"],
            "fnv1a64:lifecycle"
        );

        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();
        assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
        assert!(inbox.record(evidence).unwrap());
        assert_eq!(snapshot, before);

        let inbox_json: serde_json::Value =
            serde_json::from_str(&inbox.to_json()).unwrap();
        assert_eq!(
            inbox_json["summary"]["graph_verifier_with_semantic_context"],
            1
        );
        assert_eq!(inbox_json["resolution_effect"]["diagnostic_created"], false);
        assert_eq!(inbox_json["resolution_effect"]["state_transition"], "none");
        assert_eq!(inbox_json["resolution_effect"]["revision_created"], false);
        assert_eq!(inbox_json["resolution_effect"]["action_selected"], false);
        assert_eq!(
            inbox_json["entries"][0]["payload"]["semantic_context_fingerprint"],
            "fnv1a64:semantic-context"
        );
    }

    #[test]
    fn semantic_graph_receipt_validation_fails_closed_on_partial_or_mismatched_context() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");

        let mut partial = semantic_graph_receipt(&projection);
        partial
            .as_object_mut()
            .unwrap()
            .remove("semantic_context_fingerprint");
        let err =
            RuntimeEvidence::from_graph_verifier_receipt_json(&partial.to_string()).unwrap_err();
        assert!(err.contains("must be present together"));

        let mut bad_fingerprint = semantic_graph_receipt(&projection);
        bad_fingerprint["semantic_context_fingerprint"] =
            serde_json::json!("fnv1a64:different");
        let err = RuntimeEvidence::from_graph_verifier_receipt_json(
            &bad_fingerprint.to_string(),
        )
        .unwrap_err();
        assert!(err.contains("semantic_context_fingerprint does not match"));

        let mut bad_program = semantic_graph_receipt(&projection);
        bad_program["semantic_execution_context"]["program_identity"]["plan_hex"] =
            serde_json::json!("different");
        let err =
            RuntimeEvidence::from_graph_verifier_receipt_json(&bad_program.to_string()).unwrap_err();
        assert!(err.contains("program_identity does not match"));

        let mut bad_lineage = semantic_graph_receipt(&projection);
        bad_lineage["semantic_execution_context"]["semantic_lifecycle_identity"]
            ["base_semantic_graph_identity"]["fingerprint"] =
            serde_json::json!("fnv1a64:different-graph");
        let err = RuntimeEvidence::from_graph_verifier_receipt_json(
            &bad_lineage.to_string(),
        )
        .unwrap_err();
        assert!(err.contains("base graph identity does not match"));
    }

    #[test]
    fn legacy_graph_receipt_remains_semantic_context_free() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut receipt = semantic_graph_receipt(&projection);
        receipt
            .as_object_mut()
            .unwrap()
            .remove("semantic_execution_context");
        receipt
            .as_object_mut()
            .unwrap()
            .remove("semantic_context_fingerprint");

        let evidence =
            RuntimeEvidence::from_graph_verifier_receipt_json(&receipt.to_string()).unwrap();
        let evidence_json: serde_json::Value =
            serde_json::from_str(&evidence.to_json()).unwrap();
        assert!(evidence_json["payload"]
            .get("semantic_execution_context")
            .is_none());
        assert!(evidence_json["payload"]
            .get("semantic_context_fingerprint")
            .is_none());

        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();
        assert!(inbox.record(evidence).unwrap());
        let inbox_json: serde_json::Value =
            serde_json::from_str(&inbox.to_json()).unwrap();
        assert_eq!(
            inbox_json["summary"]["graph_verifier_with_semantic_context"],
            0
        );
    }

    #[test]
    fn stale_foreign_and_unbound_evidence_are_classified_and_rejected() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let mut stale_subject = RuntimeEvidenceSubject::from_projection(&projection);
        stale_subject.workflow_revision -= 1;
        let stale = RuntimeEvidence::agent_fault(
            Some(stale_subject),
            "E_STALE",
            "control_precondition",
            "op",
            "revision.match",
            true,
            "stale",
        )
        .unwrap();
        assert_eq!(inbox.classify(&stale), RejoinStatus::StaleRuntimeRevision);
        assert!(inbox.record(stale).is_err());

        let mut foreign_subject = RuntimeEvidenceSubject::from_projection(&projection);
        foreign_subject.intent_id = "intent-b".to_string();
        let foreign = RuntimeEvidence::agent_fault(
            Some(foreign_subject),
            "E_FOREIGN",
            "control_precondition",
            "op",
            "intent.match",
            true,
            "foreign",
        )
        .unwrap();
        assert_eq!(inbox.classify(&foreign), RejoinStatus::ForeignIntent);
        assert!(inbox.record(foreign).is_err());

        let unbound = RuntimeEvidence::agent_fault(
            None,
            "E_UNBOUND",
            "control_precondition",
            "op",
            "subject.bound",
            true,
            "unbound",
        )
        .unwrap();
        assert_eq!(inbox.classify(&unbound), RejoinStatus::Unbound);
        assert!(inbox.record(unbound).is_err());
        assert!(inbox.is_empty());
    }

    #[test]
    fn approval_subject_and_authorization_mismatch_remain_distinct() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let mut approval_subject = RuntimeEvidenceSubject::from_projection(&projection);
        approval_subject.approval_id = "approval-other".to_string();
        let approval = RuntimeEvidence::agent_fault(
            Some(approval_subject),
            "E_APPROVAL",
            "control_precondition",
            "op",
            "approval.match",
            true,
            "approval mismatch",
        )
        .unwrap();
        assert_eq!(inbox.classify(&approval), RejoinStatus::ApprovalMismatch);

        let mut semantic_subject = RuntimeEvidenceSubject::from_projection(&projection);
        semantic_subject.subject_identity = "spec-other".to_string();
        let subject = RuntimeEvidence::agent_fault(
            Some(semantic_subject),
            "E_SUBJECT",
            "control_precondition",
            "op",
            "subject.match",
            true,
            "subject mismatch",
        )
        .unwrap();
        assert_eq!(inbox.classify(&subject), RejoinStatus::SubjectMismatch);

        let mut authorization_subject = RuntimeEvidenceSubject::from_projection(&projection);
        authorization_subject.authorization_policy_revision = 4;
        let authorization = RuntimeEvidence::agent_fault(
            Some(authorization_subject),
            "E_AUTH",
            "control_precondition",
            "op",
            "authorization.match",
            true,
            "authorization mismatch",
        )
        .unwrap();
        assert_eq!(
            inbox.classify(&authorization),
            RejoinStatus::AuthorizationMismatch
        );
    }

    #[test]
    fn inbox_requires_compile_eligible_exact_resolution_revision() {
        let mut workflow = ResolutionWorkflow::new("intent-a").unwrap();
        workflow.submit().unwrap();
        let submitted = workflow.snapshot();
        let submitted_projection = projection("intent-a", submitted.revision, "spec-a");
        assert!(ResolutionEvidenceInbox::new(&submitted, &submitted_projection).is_err());

        workflow.finalize_resolution().unwrap();
        let resolved = workflow.snapshot();
        let stale_projection = projection("intent-a", resolved.revision - 1, "spec-a");
        assert!(ResolutionEvidenceInbox::new(&resolved, &stale_projection).is_err());
    }

    #[test]
    fn structured_math_program_receipt_rejoins_as_observation_only() {
        let snapshot = resolved_snapshot("intent-a");
        let before = snapshot.clone();
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let receipt = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 7,
            "authority": "wasm_verifier",
            "verifier": "MathProgram.verifyFlat",
            "reference_authority": "burn_math_program",
            "label": "math-program-check",
            "fingerprint_algorithm": "fnv1a64_noncryptographic",
            "program_plan_version": 9,
            "program_identity": {
                "schema": "burn-research.math-program-identity.v1",
                "plan_hex": "42524d5009"
            },
            "program_identity_fingerprint": "fnv1a64:probe-program",
            "mutable_state_in_program_identity": false,
            "runtime_subject": {
                "status": "bound",
                "intent_id": projection.intent_id.clone(),
                "workflow_revision": projection.workflow_revision,
                "approval_id": projection.approval_id.clone(),
                "subject_kind": projection.subject_kind.clone(),
                "subject_identity": projection.subject_identity.clone(),
                "authorization_policy_id": projection.authorization_policy_id.clone(),
                "authorization_policy_revision": projection.authorization_policy_revision,
                "authorization_is_revision": projection.authorization_is_revision
            },
            "input_count": 2,
            "input_fingerprint": "fnv1a64:probe-input",
            "reference_fingerprint": "fnv1a64:probe-reference",
            "candidate_fingerprint": "fnv1a64:probe-candidate",
            "tolerances": {"abs": 0.0, "rel": 0.0},
            "result": {
                "passed": false,
                "len": 3,
                "max_abs_error": 1.0,
                "max_rel_error": 1.0,
                "rmse": 0.5773502691896257,
                "first_failure": 1
            }
        })
        .to_string();

        let evidence =
            RuntimeEvidence::from_math_program_verifier_receipt_json(&receipt).unwrap();
        assert_eq!(evidence.source_authority(), "wasm_verifier");
        assert_eq!(evidence.evidence_authority(), "observation_only");
        assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
        assert_eq!(evidence.kind(), "math_program_verifier_receipt");
        assert_eq!(evidence.outcome(), "failed");
        assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
        assert!(inbox.record(evidence).unwrap());

        let json = inbox.to_json();
        assert!(json.contains("\"math_program_verifier_failed\":1"));
        assert!(json.contains("\"reference_authority\":\"burn_math_program\""));
        assert!(json.contains("\"diagnostic_created\":false"));
        assert!(json.contains("\"state_transition\":\"none\""));
        assert_eq!(snapshot, before);
    }

    #[test]
    fn math_program_adapter_fails_closed_on_authority_class_escalation() {
        let graph_receipt = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 1,
            "authority": "wasm_verifier",
            "verifier": "CompiledGraph.verifyFlat",
            "reference_authority": "burn_compiled_graph",
            "label": "graph",
            "program_identity": {
                "schema": "burn-research.program-identity.v1",
                "plan_hex": "graph"
            },
            "runtime_subject": {"status": "unbound"},
            "result": {"passed": true}
        })
        .to_string();
        assert!(
            RuntimeEvidence::from_math_program_verifier_receipt_json(&graph_receipt)
                .is_err()
        );

        let vector_receipt = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 1,
            "authority": "wasm_comparator",
            "verifier": "mathVerifyVectors",
            "reference_authority": "caller_supplied",
            "label": "vector",
            "runtime_subject": {"status": "unbound"},
            "result": {"passed": true}
        })
        .to_string();
        assert!(
            RuntimeEvidence::from_math_program_verifier_receipt_json(&vector_receipt)
                .is_err()
        );

        let forged_identity = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 1,
            "authority": "wasm_verifier",
            "verifier": "MathProgram.verifyFlat",
            "reference_authority": "burn_math_program",
            "label": "forged",
            "program_identity": {
                "schema": "burn-research.program-identity.v1",
                "plan_hex": "wrong-schema"
            },
            "runtime_subject": {"status": "unbound"},
            "result": {"passed": true}
        })
        .to_string();
        assert!(
            RuntimeEvidence::from_math_program_verifier_receipt_json(&forged_identity)
                .is_err()
        );
    }

    #[test]
    fn structured_direct_math_receipt_rejoins_as_observation_only() {
        let snapshot = resolved_snapshot("intent-direct");
        let before = snapshot.clone();
        let projection = projection("intent-direct", snapshot.revision, "spec-direct");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let receipt = serde_json::json!({
            "schema_version": 1,
            "schema_id": "burn-research.verifier-receipt.v1",
            "receipt_id": 11,
            "authority": "wasm_verifier",
            "verifier": "DirectMath.verifyAgainstMathProgramV9",
            "reference_authority": "burn_math_program",
            "candidate_authority": "burn_direct_math",
            "operation_id": "numeric.add",
            "label": "direct-add",
            "fingerprint_algorithm": "fnv1a64_noncryptographic",
            "reference_program_generation": "v9",
            "program_identity": {
                "schema": "burn-research.math-program-identity.v1",
                "plan_hex": "42524d5009"
            },
            "program_identity_fingerprint": "fnv1a64:direct-program",
            "mutable_state_in_program_identity": false,
            "runtime_subject": {
                "status": "bound",
                "intent_id": projection.intent_id.clone(),
                "workflow_revision": projection.workflow_revision,
                "approval_id": projection.approval_id.clone(),
                "subject_kind": projection.subject_kind.clone(),
                "subject_identity": projection.subject_identity.clone(),
                "authorization_policy_id": projection.authorization_policy_id.clone(),
                "authorization_policy_revision": projection.authorization_policy_revision,
                "authorization_is_revision": projection.authorization_is_revision
            },
            "input_count": 2,
            "input_fingerprint": "fnv1a64:direct-input",
            "reference_fingerprint": "fnv1a64:direct-reference",
            "candidate_fingerprint": "fnv1a64:direct-candidate",
            "tolerances": {"abs": 0.0, "rel": 0.0},
            "result": {
                "passed": true,
                "len": 2,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "rmse": 0.0,
                "first_failure": null
            }
        })
        .to_string();

        let evidence =
            RuntimeEvidence::from_direct_math_verifier_receipt_json(&receipt).unwrap();

        assert_eq!(evidence.source_authority(), "wasm_verifier");
        assert_eq!(evidence.evidence_authority(), "observation_only");
        assert_eq!(evidence.transport_integrity(), "host_structured_unverified");
        assert_eq!(evidence.kind(), "direct_math_verifier_receipt");
        assert_eq!(evidence.outcome(), "passed");
        assert_eq!(inbox.classify(&evidence), RejoinStatus::Exact);
        assert!(inbox.record(evidence).unwrap());

        let json = inbox.to_json();
        assert!(json.contains("\"direct_math_verifier_passed\":1"));
        assert!(json.contains("\"candidate_authority\":\"burn_direct_math\""));
        assert!(json.contains("\"reference_authority\":\"burn_math_program\""));
        assert!(json.contains("\"reference_program_generation\":\"v9\""));
        assert!(json.contains("\"diagnostic_created\":false"));
        assert!(json.contains("\"state_transition\":\"none\""));
        assert_eq!(snapshot, before);
    }

    #[test]
    fn direct_math_adapter_fails_closed_on_authority_class_escalation() {
        fn base_receipt() -> serde_json::Value {
            serde_json::json!({
                "schema_version": 1,
                "schema_id": "burn-research.verifier-receipt.v1",
                "receipt_id": 1,
                "authority": "wasm_verifier",
                "verifier": "DirectMath.verifyAgainstMathProgramV9",
                "reference_authority": "burn_math_program",
                "candidate_authority": "burn_direct_math",
                "operation_id": "numeric.add",
                "label": "direct",
                "reference_program_generation": "v9",
                "program_identity": {
                    "schema": "burn-research.math-program-identity.v1",
                    "plan_hex": "42524d5009"
                },
                "runtime_subject": {"status": "unbound"},
                "result": {"passed": true}
            })
        }

        let mut wrong_verifier = base_receipt();
        wrong_verifier["verifier"] =
            serde_json::Value::String("MathProgram.verifyFlat".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(
                &wrong_verifier.to_string()
            )
            .is_err()
        );

        let mut wrong_candidate = base_receipt();
        wrong_candidate["candidate_authority"] =
            serde_json::Value::String("caller_supplied".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(
                &wrong_candidate.to_string()
            )
            .is_err()
        );

        let mut wrong_reference = base_receipt();
        wrong_reference["reference_authority"] =
            serde_json::Value::String("burn_compiled_graph".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(
                &wrong_reference.to_string()
            )
            .is_err()
        );

        let mut wrong_generation = base_receipt();
        wrong_generation["reference_program_generation"] =
            serde_json::Value::String("v8".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(
                &wrong_generation.to_string()
            )
            .is_err()
        );

        let mut wrong_identity = base_receipt();
        wrong_identity["program_identity"]["schema"] =
            serde_json::Value::String("burn-research.program-identity.v1".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(
                &wrong_identity.to_string()
            )
            .is_err()
        );

        let mut vector = base_receipt();
        vector["authority"] = serde_json::Value::String("wasm_comparator".into());
        vector["verifier"] = serde_json::Value::String("mathVerifyVectors".into());
        vector["reference_authority"] =
            serde_json::Value::String("caller_supplied".into());
        assert!(
            RuntimeEvidence::from_direct_math_verifier_receipt_json(&vector.to_string())
                .is_err()
        );
    }

    #[test]
    fn duplicate_is_idempotent_and_inbox_is_bounded() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let first = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_0",
            "control_precondition",
            "op",
            "p",
            true,
            "detail",
        )
        .unwrap();
        assert!(inbox.record(first.clone()).unwrap());
        assert!(!inbox.record(first).unwrap());

        for index in 1..MAX_RUNTIME_EVIDENCE_ENTRIES {
            let evidence = RuntimeEvidence::bound_agent_fault(
                &projection,
                format!("E_{index}"),
                "control_precondition",
                "op",
                "p",
                true,
                format!("detail-{index}"),
            )
            .unwrap();
            assert!(inbox.record(evidence).unwrap());
        }
        assert_eq!(inbox.len(), MAX_RUNTIME_EVIDENCE_ENTRIES);

        let overflow = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_OVERFLOW",
            "control_precondition",
            "op",
            "p",
            true,
            "overflow",
        )
        .unwrap();
        assert!(inbox.record(overflow).is_err());
        assert_eq!(inbox.len(), MAX_RUNTIME_EVIDENCE_ENTRIES);
    }
}
