use wasm_bindgen::prelude::*;

use crate::authorization::{
    AuthorizationPolicy, AuthorizationSnapshot, AUTHORIZATION_SNAPSHOT_SCHEMA,
};
use crate::effective_spec::{
    ApprovedEffectiveSpec, EffectiveFieldOrigin, EffectiveSpecApprovalEvidence,
    EFFECTIVE_SPEC_SCHEMA, EFFECTIVE_SPEC_SUBJECT_KIND,
};
use crate::workspace::{AgentWorkspace, WorkspaceRuntimeSubjectBinding};

const BRIDGE_CONTRACT_V1: &str = include_str!("../docs/resolution-runtime-bridge.v1.json");

const MAX_ID_BYTES: usize = 512;
const MAX_SUBJECT_KIND_BYTES: usize = 128;
const MAX_SUBJECT_IDENTITY_BYTES: usize = 4096;

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
pub struct RuntimeProjectedField {
    pub key: String,
    pub value: String,
    pub origin: String,
    pub source_approval_id: Option<String>,
    pub source_spec_identity: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSubjectProjection {
    pub schema: String,
    pub intent_id: String,
    pub workflow_revision: u64,
    pub approval_id: String,
    pub subject_kind: String,
    pub subject_identity: String,
    pub effective_spec_schema: String,
    pub effective_spec_identity: String,
    pub authorization_schema: String,
    pub authorization_policy_id: String,
    pub authorization_policy_revision: u64,
    pub authorization_is_revision: bool,
    pub approver: String,
    pub fields: Vec<RuntimeProjectedField>,
}

impl RuntimeSubjectProjection {
    pub fn from_authorized(
        approved: &ApprovedEffectiveSpec,
        policy: &AuthorizationPolicy,
        authorization: &AuthorizationSnapshot,
    ) -> Result<Self, String> {
        match &approved.evidence {
            EffectiveSpecApprovalEvidence::Root(bound) => {
                policy.validate_authorization(authorization, bound)?;
            }
            EffectiveSpecApprovalEvidence::Revision(bound) => {
                policy.validate_revision_authorization(authorization, bound)?;
            }
        }

        if approved.spec.schema != EFFECTIVE_SPEC_SCHEMA {
            return Err("ResolutionRuntimeBridge: unsupported effective-spec schema".to_string());
        }
        if authorization.schema() != AUTHORIZATION_SNAPSHOT_SCHEMA {
            return Err("ResolutionRuntimeBridge: unsupported authorization schema".to_string());
        }
        if authorization.policy_id() != policy.policy_id()
            || authorization.policy_revision() != policy.revision()
        {
            return Err(
                "ResolutionRuntimeBridge: authorization is stale for the current policy"
                    .to_string(),
            );
        }

        let (subject, intent_id, workflow_revision, evidence_is_revision) =
            match &approved.evidence {
                EffectiveSpecApprovalEvidence::Root(bound) => (
                    &bound.subject,
                    bound.approval.intent_id.as_str(),
                    bound.approval.workflow_revision,
                    false,
                ),
                EffectiveSpecApprovalEvidence::Revision(bound) => (
                    &bound.subject,
                    bound.approval.approval.intent_id.as_str(),
                    bound.approval.approval.workflow_revision,
                    true,
                ),
            };

        if subject.kind() != EFFECTIVE_SPEC_SUBJECT_KIND {
            return Err(format!(
                "ResolutionRuntimeBridge: expected subject kind {EFFECTIVE_SPEC_SUBJECT_KIND}, got {}",
                subject.kind()
            ));
        }
        if subject.identity() != approved.spec.identity.as_str() {
            return Err(
                "ResolutionRuntimeBridge: approval subject identity does not match EffectiveSpec.identity"
                    .to_string(),
            );
        }
        if authorization.subject() != subject {
            return Err(
                "ResolutionRuntimeBridge: authorization subject does not match approved subject"
                    .to_string(),
            );
        }
        if authorization.approval_id() != approved.approval_id() {
            return Err(
                "ResolutionRuntimeBridge: authorization approval id does not match ApprovedEffectiveSpec"
                    .to_string(),
            );
        }
        if authorization.intent_id() != intent_id
            || authorization.workflow_revision() != workflow_revision
        {
            return Err(
                "ResolutionRuntimeBridge: authorization intent/revision does not match approved resolution"
                    .to_string(),
            );
        }
        if authorization.is_revision() != evidence_is_revision {
            return Err(
                "ResolutionRuntimeBridge: root/revision authorization class mismatch".to_string(),
            );
        }

        validate_nonempty_bounded(intent_id, MAX_ID_BYTES, "ResolutionRuntimeBridge.intent_id")?;
        validate_nonempty_bounded(
            approved.approval_id(),
            MAX_ID_BYTES,
            "ResolutionRuntimeBridge.approval_id",
        )?;
        validate_nonempty_bounded(
            subject.kind(),
            MAX_SUBJECT_KIND_BYTES,
            "ResolutionRuntimeBridge.subject_kind",
        )?;
        validate_nonempty_bounded(
            subject.identity(),
            MAX_SUBJECT_IDENTITY_BYTES,
            "ResolutionRuntimeBridge.subject_identity",
        )?;
        validate_nonempty_bounded(
            authorization.policy_id(),
            MAX_ID_BYTES,
            "ResolutionRuntimeBridge.authorization_policy_id",
        )?;

        let fields = approved
            .spec
            .fields
            .iter()
            .map(|field| {
                let (origin, source_approval_id, source_spec_identity) = match &field.origin {
                    EffectiveFieldOrigin::DeclaredHere => ("declared_here", None, None),
                    EffectiveFieldOrigin::InheritedFrom {
                        approval_id,
                        spec_identity,
                    } => (
                        "inherited",
                        Some(approval_id.clone()),
                        Some(spec_identity.clone()),
                    ),
                    EffectiveFieldOrigin::OverriddenFrom {
                        approval_id,
                        spec_identity,
                    } => (
                        "overridden",
                        Some(approval_id.clone()),
                        Some(spec_identity.clone()),
                    ),
                };
                RuntimeProjectedField {
                    key: field.key.clone(),
                    value: field.value.clone(),
                    origin: origin.to_string(),
                    source_approval_id,
                    source_spec_identity,
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: intent_id.to_string(),
            workflow_revision,
            approval_id: approved.approval_id().to_string(),
            subject_kind: subject.kind().to_string(),
            subject_identity: subject.identity().to_string(),
            effective_spec_schema: approved.spec.schema.clone(),
            effective_spec_identity: approved.spec.identity.clone(),
            authorization_schema: authorization.schema().to_string(),
            authorization_policy_id: authorization.policy_id().to_string(),
            authorization_policy_revision: authorization.policy_revision(),
            authorization_is_revision: authorization.is_revision(),
            approver: authorization.approver().to_string(),
            fields,
        })
    }

    pub fn to_json(&self) -> String {
        let fields = self
            .fields
            .iter()
            .map(|field| {
                let source_approval = field
                    .source_approval_id
                    .as_deref()
                    .map(|value| format!("\"{}\"", json_escape(value)))
                    .unwrap_or_else(|| "null".to_string());
                let source_spec = field
                    .source_spec_identity
                    .as_deref()
                    .map(|value| format!("\"{}\"", json_escape(value)))
                    .unwrap_or_else(|| "null".to_string());
                format!(
                    concat!(
                        "{{",
                        "\"key\":\"{}\",",
                        "\"value\":\"{}\",",
                        "\"origin\":\"{}\",",
                        "\"source_approval_id\":{},",
                        "\"source_spec_identity\":{}",
                        "}}"
                    ),
                    json_escape(&field.key),
                    json_escape(&field.value),
                    json_escape(&field.origin),
                    source_approval,
                    source_spec,
                )
            })
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.runtime-subject-projection.v1\",",
                "\"role\":\"semantic_projection_only\",",
                "\"resolution\":{{",
                    "\"intent_id\":\"{}\",",
                    "\"workflow_revision\":{},",
                    "\"approval_id\":\"{}\"",
                "}},",
                "\"subject\":{{",
                    "\"kind\":\"{}\",",
                    "\"identity\":\"{}\"",
                "}},",
                "\"effective_spec\":{{",
                    "\"schema\":\"{}\",",
                    "\"identity\":\"{}\",",
                    "\"fields\":[{}]",
                "}},",
                "\"authorization\":{{",
                    "\"schema\":\"{}\",",
                    "\"policy_id\":\"{}\",",
                    "\"policy_revision\":{},",
                    "\"is_revision\":{},",
                    "\"approver\":\"{}\"",
                "}},",
                "\"planner_policy\":\"opaque_fields_agent_planner_required\"",
                "}}"
            ),
            json_escape(&self.intent_id),
            self.workflow_revision,
            json_escape(&self.approval_id),
            json_escape(&self.subject_kind),
            json_escape(&self.subject_identity),
            json_escape(&self.effective_spec_schema),
            json_escape(&self.effective_spec_identity),
            fields,
            json_escape(&self.authorization_schema),
            json_escape(&self.authorization_policy_id),
            self.authorization_policy_revision,
            self.authorization_is_revision,
            json_escape(&self.approver),
        )
    }
}

pub fn bind_runtime_subject_projection(
    workspace: &mut AgentWorkspace,
    projection: &RuntimeSubjectProjection,
) -> Result<bool, String> {
    workspace.bind_runtime_subject_binding(WorkspaceRuntimeSubjectBinding {
        intent_id: projection.intent_id.clone(),
        workflow_revision: projection.workflow_revision,
        approval_id: projection.approval_id.clone(),
        subject_kind: projection.subject_kind.clone(),
        subject_identity: projection.subject_identity.clone(),
        authorization_policy_id: projection.authorization_policy_id.clone(),
        authorization_policy_revision: projection.authorization_policy_revision,
        authorization_is_revision: projection.authorization_is_revision,
    })
}

pub(crate) fn runtime_subject_binding_json(workspace: &AgentWorkspace) -> String {
    match workspace.runtime_subject_binding() {
        Some(binding) => format!(
            concat!(
                "{{",
                "\"status\":\"bound\",",
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
            json_escape(&binding.intent_id),
            binding.workflow_revision,
            json_escape(&binding.approval_id),
            json_escape(&binding.subject_kind),
            json_escape(&binding.subject_identity),
            json_escape(&binding.authorization_policy_id),
            binding.authorization_policy_revision,
            binding.authorization_is_revision,
        ),
        None => "{\"status\":\"unbound\"}".to_string(),
    }
}

#[wasm_bindgen(js_name = resolutionRuntimeBridgeCapabilities)]
pub fn resolution_runtime_bridge_capabilities() -> String {
    BRIDGE_CONTRACT_V1.to_string()
}

#[wasm_bindgen(js_name = workspaceBindRuntimeSubject)]
#[allow(clippy::too_many_arguments)]
pub fn workspace_bind_runtime_subject(
    workspace: &mut AgentWorkspace,
    intent_id: String,
    workflow_revision: u64,
    approval_id: String,
    subject_kind: String,
    subject_identity: String,
    authorization_policy_id: String,
    authorization_policy_revision: u64,
    authorization_is_revision: bool,
) -> Result<bool, String> {
    validate_nonempty_bounded(&intent_id, MAX_ID_BYTES, "workspaceBindRuntimeSubject.intent_id")?;
    validate_nonempty_bounded(
        &approval_id,
        MAX_ID_BYTES,
        "workspaceBindRuntimeSubject.approval_id",
    )?;
    validate_nonempty_bounded(
        &subject_kind,
        MAX_SUBJECT_KIND_BYTES,
        "workspaceBindRuntimeSubject.subject_kind",
    )?;
    validate_nonempty_bounded(
        &subject_identity,
        MAX_SUBJECT_IDENTITY_BYTES,
        "workspaceBindRuntimeSubject.subject_identity",
    )?;
    validate_nonempty_bounded(
        &authorization_policy_id,
        MAX_ID_BYTES,
        "workspaceBindRuntimeSubject.authorization_policy_id",
    )?;

    workspace.bind_runtime_subject_binding(WorkspaceRuntimeSubjectBinding {
        intent_id,
        workflow_revision,
        approval_id,
        subject_kind,
        subject_identity,
        authorization_policy_id,
        authorization_policy_revision,
        authorization_is_revision,
    })
}

#[wasm_bindgen(js_name = workspaceRuntimeSubject)]
pub fn workspace_runtime_subject(workspace: &AgentWorkspace) -> String {
    runtime_subject_binding_json(workspace)
}

#[cfg(test)]
mod tests {
    use super::{
        bind_runtime_subject_projection, workspace_runtime_subject, RuntimeSubjectProjection,
    };
    use crate::authorization::AuthorizationPolicy;
    use crate::effective_spec::{ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration};
    use crate::resolution_subject::SubjectBoundReviewSession;
    use crate::workspace::AgentWorkspace;

    fn approved_fixture() -> (
        ApprovedEffectiveSpec,
        AuthorizationPolicy,
        crate::authorization::AuthorizationSnapshot,
    ) {
        let spec = EffectiveSpec::root(vec![
            SpecDeclaration::new("objective", "feature_transform").unwrap(),
            SpecDeclaration::new("runtime.hint", "agent_plans_graph").unwrap(),
        ])
        .unwrap();
        let subject = spec.approval_subject().unwrap();
        let mut review = SubjectBoundReviewSession::new("intent-bridge", subject).unwrap();
        review.submit("owner").unwrap();
        let approval = review.approve("owner").unwrap();
        let approved = ApprovedEffectiveSpec::bind_root(spec, approval.clone()).unwrap();
        let policy = AuthorizationPolicy::new("runtime-policy", 7, "owner", vec![]).unwrap();
        let authorization = policy.authorize(&approval).unwrap();
        (approved, policy, authorization)
    }

    #[test]
    fn authorized_effective_spec_projects_without_planning_graph() {
        let (approved, policy, authorization) = approved_fixture();
        let projection =
            RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();

        assert_eq!(projection.intent_id, "intent-bridge");
        assert_eq!(projection.subject_identity, approved.spec.identity);
        assert_eq!(projection.effective_spec_identity, approved.spec.identity);
        assert_eq!(projection.authorization_policy_id, "runtime-policy");
        assert_eq!(projection.fields.len(), 2);

        let json = projection.to_json();
        assert!(json.contains("\"planner_policy\":\"opaque_fields_agent_planner_required\""));
        assert!(!json.contains("AgentLayerSpec"));
    }

    #[test]
    fn stale_policy_authorization_is_rejected() {
        let (approved, _policy, authorization) = approved_fixture();
        let newer_policy = AuthorizationPolicy::new("runtime-policy", 8, "owner", vec![]).unwrap();

        let error = RuntimeSubjectProjection::from_authorized(
            &approved,
            &newer_policy,
            &authorization,
        )
        .unwrap_err();
        assert!(error.contains("stale") || error.contains("policy"));
    }

    #[test]
    fn binding_projection_mutates_only_workspace_control_metadata() {
        let (approved, policy, authorization) = approved_fixture();
        let projection =
            RuntimeSubjectProjection::from_authorized(&approved, &policy, &authorization).unwrap();
        let mut workspace = AgentWorkspace::new(3).unwrap();

        assert!(bind_runtime_subject_projection(&mut workspace, &projection).unwrap());
        let bound = workspace_runtime_subject(&workspace);

        assert!(bound.contains("\"status\":\"bound\""));
        assert!(bound.contains(&approved.spec.identity));
        assert!(workspace.snapshot().contains("\"runtime_subject\":{"));
    }
}
