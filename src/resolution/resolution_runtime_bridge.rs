pub(crate) use crate::facade::resolution::runtime_subject_binding_json;
pub use crate::facade::resolution::{
    bind_runtime_subject_projection, resolution_runtime_bridge_capabilities,
    workspace_bind_runtime_subject, workspace_runtime_program_binding, workspace_runtime_subject,
    RuntimeProjectedField, RuntimeSubjectProjection,
};

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

        let error =
            RuntimeSubjectProjection::from_authorized(&approved, &newer_policy, &authorization)
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
