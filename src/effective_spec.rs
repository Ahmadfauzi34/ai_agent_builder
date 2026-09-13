//! Deterministic effective-specification materialization above Resolution Subject Binding v4.
//!
//! This layer turns explicit field declarations/directives into an immutable candidate
//! specification with canonical identity and provenance. A candidate becomes approved only
//! when an existing subject-bound resolution approval proves that exact identity.
//!
//! No parent field is inherited implicitly. Child revisions must explicitly inherit,
//! override, or remove every parent field. Missing decisions become `NeedsResolution`
//! rather than guesses.

use std::collections::BTreeMap;

use crate::resolution_subject::{
    ApprovalSubject, SubjectBoundApprovalSnapshot, SubjectBoundRevisionApprovalSnapshot,
};

pub const EFFECTIVE_SPEC_SCHEMA: &str = "burn-research.effective-spec.v1";
pub const EFFECTIVE_SPEC_SUBJECT_KIND: &str = "effective-spec";
pub const EFFECTIVE_SPEC_MAX_FIELDS: usize = 64;
pub const EFFECTIVE_SPEC_MAX_KEY_BYTES: usize = 64;
pub const EFFECTIVE_SPEC_MAX_VALUE_BYTES: usize = 512;
const EFFECTIVE_SPEC_MAX_IDENTITY_BYTES: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecDeclaration {
    pub key: String,
    pub value: String,
}

impl SpecDeclaration {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Result<Self, String> {
        Ok(Self {
            key: validate_key(key)?,
            value: validate_value(value)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectiveFieldOrigin {
    DeclaredHere,
    InheritedFrom {
        approval_id: String,
        spec_identity: String,
    },
    OverriddenFrom {
        approval_id: String,
        spec_identity: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveField {
    pub key: String,
    pub value: String,
    pub origin: EffectiveFieldOrigin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecChangeKind {
    Declared,
    Inherited,
    Overridden,
    Removed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecChange {
    pub key: String,
    pub kind: SpecChangeKind,
    pub parent_approval_id: Option<String>,
    pub parent_spec_identity: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveSpec {
    pub schema: String,
    pub identity: String,
    pub parent_spec_identity: Option<String>,
    pub parent_approval_id: Option<String>,
    pub fields: Vec<EffectiveField>,
    pub changes: Vec<SpecChange>,
}

impl EffectiveSpec {
    pub fn root(declarations: Vec<SpecDeclaration>) -> Result<Self, String> {
        if declarations.is_empty() {
            return Err("EffectiveSpec: root specification must declare at least one field".to_string());
        }
        if declarations.len() > EFFECTIVE_SPEC_MAX_FIELDS {
            return Err(format!(
                "EffectiveSpec: field count {} exceeds maximum {}",
                declarations.len(),
                EFFECTIVE_SPEC_MAX_FIELDS
            ));
        }

        let mut ordered = BTreeMap::<String, String>::new();
        for declaration in declarations {
            let key = validate_key(declaration.key)?;
            let value = validate_value(declaration.value)?;
            if ordered.insert(key.clone(), value).is_some() {
                return Err(format!("EffectiveSpec: duplicate field key: {key}"));
            }
        }

        let fields = ordered
            .iter()
            .map(|(key, value)| EffectiveField {
                key: key.clone(),
                value: value.clone(),
                origin: EffectiveFieldOrigin::DeclaredHere,
            })
            .collect::<Vec<_>>();
        let changes = ordered
            .keys()
            .map(|key| SpecChange {
                key: key.clone(),
                kind: SpecChangeKind::Declared,
                parent_approval_id: None,
                parent_spec_identity: None,
            })
            .collect::<Vec<_>>();
        Self::finish(None, None, fields, changes)
    }

    pub fn field(&self, key: &str) -> Option<&EffectiveField> {
        self.fields.iter().find(|field| field.key == key)
    }

    pub fn approval_subject(&self) -> Result<ApprovalSubject, String> {
        ApprovalSubject::new(EFFECTIVE_SPEC_SUBJECT_KIND, self.identity.clone())
    }

    fn finish(
        parent_spec_identity: Option<String>,
        parent_approval_id: Option<String>,
        mut fields: Vec<EffectiveField>,
        mut changes: Vec<SpecChange>,
    ) -> Result<Self, String> {
        fields.sort_by(|a, b| a.key.cmp(&b.key));
        changes.sort_by(|a, b| a.key.cmp(&b.key));
        if fields.len() > EFFECTIVE_SPEC_MAX_FIELDS {
            return Err(format!(
                "EffectiveSpec: field count {} exceeds maximum {}",
                fields.len(),
                EFFECTIVE_SPEC_MAX_FIELDS
            ));
        }
        validate_unique_field_keys(&fields)?;
        validate_unique_change_keys(&changes)?;

        let identity = canonical_identity(
            parent_spec_identity.as_deref(),
            parent_approval_id.as_deref(),
            &fields,
            &changes,
        )?;
        Ok(Self {
            schema: EFFECTIVE_SPEC_SCHEMA.to_string(),
            identity,
            parent_spec_identity,
            parent_approval_id,
            fields,
            changes,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecDirective {
    Inherit { key: String },
    Override { key: String, value: String },
    Remove { key: String },
    Declare { key: String, value: String },
}

impl SpecDirective {
    pub fn inherit(key: impl Into<String>) -> Result<Self, String> {
        Ok(Self::Inherit {
            key: validate_key(key)?,
        })
    }

    pub fn override_value(
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, String> {
        Ok(Self::Override {
            key: validate_key(key)?,
            value: validate_value(value)?,
        })
    }

    pub fn remove(key: impl Into<String>) -> Result<Self, String> {
        Ok(Self::Remove {
            key: validate_key(key)?,
        })
    }

    pub fn declare(
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, String> {
        Ok(Self::Declare {
            key: validate_key(key)?,
            value: validate_value(value)?,
        })
    }

    fn key(&self) -> &str {
        match self {
            Self::Inherit { key }
            | Self::Override { key, .. }
            | Self::Remove { key }
            | Self::Declare { key, .. } => key,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveSpecDiagnostic {
    pub code: String,
    pub key: String,
    pub reason: String,
    pub required_information: String,
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveSpecNeedsResolution {
    pub diagnostics: Vec<EffectiveSpecDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectiveSpecMaterialization {
    Resolved(EffectiveSpec),
    NeedsResolution(EffectiveSpecNeedsResolution),
}

impl EffectiveSpecMaterialization {
    pub fn resolved(self) -> Result<EffectiveSpec, String> {
        match self {
            Self::Resolved(spec) => Ok(spec),
            Self::NeedsResolution(state) => Err(format!(
                "EffectiveSpec: materialization still needs resolution for {} field(s)",
                state.diagnostics.len()
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectiveSpecApprovalEvidence {
    Root(SubjectBoundApprovalSnapshot),
    Revision(SubjectBoundRevisionApprovalSnapshot),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovedEffectiveSpec {
    pub spec: EffectiveSpec,
    pub evidence: EffectiveSpecApprovalEvidence,
}

impl ApprovedEffectiveSpec {
    pub fn bind_root(
        spec: EffectiveSpec,
        approval: SubjectBoundApprovalSnapshot,
    ) -> Result<Self, String> {
        if spec.parent_spec_identity.is_some() || spec.parent_approval_id.is_some() {
            return Err("ApprovedEffectiveSpec: root approval cannot bind a child specification".to_string());
        }
        verify_subject(&spec, &approval.subject)?;
        Ok(Self {
            spec,
            evidence: EffectiveSpecApprovalEvidence::Root(approval),
        })
    }

    pub fn bind_revision(
        spec: EffectiveSpec,
        approval: SubjectBoundRevisionApprovalSnapshot,
    ) -> Result<Self, String> {
        verify_subject(&spec, &approval.subject)?;
        let expected_parent_approval = spec.parent_approval_id.as_ref().ok_or_else(|| {
            "ApprovedEffectiveSpec: revision specification is missing parent approval provenance"
                .to_string()
        })?;
        if expected_parent_approval != &approval.approval.parent_approval_id {
            return Err(format!(
                "ApprovedEffectiveSpec: parent approval mismatch: spec expects {}, approval references {}",
                expected_parent_approval, approval.approval.parent_approval_id
            ));
        }
        Ok(Self {
            spec,
            evidence: EffectiveSpecApprovalEvidence::Revision(approval),
        })
    }

    pub fn approval_id(&self) -> &str {
        match &self.evidence {
            EffectiveSpecApprovalEvidence::Root(approval) => &approval.approval.approval_id,
            EffectiveSpecApprovalEvidence::Revision(approval) => {
                &approval.approval.revision_approval_id
            }
        }
    }

    pub fn materialize_child(
        &self,
        directives: Vec<SpecDirective>,
    ) -> Result<EffectiveSpecMaterialization, String> {
        materialize_child(self, directives)
    }
}

fn materialize_child(
    parent: &ApprovedEffectiveSpec,
    directives: Vec<SpecDirective>,
) -> Result<EffectiveSpecMaterialization, String> {
    if directives.len() > EFFECTIVE_SPEC_MAX_FIELDS.saturating_mul(2) {
        return Err(format!(
            "EffectiveSpec: directive count {} exceeds bounded maximum {}",
            directives.len(),
            EFFECTIVE_SPEC_MAX_FIELDS * 2
        ));
    }

    let parent_fields = parent
        .spec
        .fields
        .iter()
        .map(|field| (field.key.clone(), field.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut ordered = BTreeMap::<String, SpecDirective>::new();
    for directive in directives {
        let key = validate_key(directive.key().to_string())?;
        match &directive {
            SpecDirective::Override { value, .. } | SpecDirective::Declare { value, .. } => {
                validate_value(value.clone())?;
            }
            SpecDirective::Inherit { .. } | SpecDirective::Remove { .. } => {}
        }
        if ordered.insert(key.clone(), directive).is_some() {
            return Err(format!("EffectiveSpec: duplicate directive for field: {key}"));
        }
    }

    for (key, directive) in &ordered {
        let exists_in_parent = parent_fields.contains_key(key);
        match directive {
            SpecDirective::Inherit { .. }
            | SpecDirective::Override { .. }
            | SpecDirective::Remove { .. } if !exists_in_parent => {
                return Err(format!(
                    "EffectiveSpec: directive for unknown parent field requires Declare instead: {key}"
                ));
            }
            SpecDirective::Declare { .. } if exists_in_parent => {
                return Err(format!(
                    "EffectiveSpec: parent field must use Override, Inherit, or Remove, not Declare: {key}"
                ));
            }
            _ => {}
        }
    }

    let missing = parent_fields
        .keys()
        .filter(|key| !ordered.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Ok(EffectiveSpecMaterialization::NeedsResolution(
            EffectiveSpecNeedsResolution {
                diagnostics: missing
                    .into_iter()
                    .map(|key| EffectiveSpecDiagnostic {
                        code: "E-SPEC-001".to_string(),
                        key,
                        reason: "parent field has no explicit child decision".to_string(),
                        required_information:
                            "choose inherit, override, or remove for this parent field".to_string(),
                        candidates: vec![
                            "inherit".to_string(),
                            "override".to_string(),
                            "remove".to_string(),
                        ],
                    })
                    .collect(),
            },
        ));
    }

    let parent_approval_id = parent.approval_id().to_string();
    let parent_spec_identity = parent.spec.identity.clone();
    validate_evidence(&parent_approval_id, "parent approval id")?;
    validate_evidence(&parent_spec_identity, "parent spec identity")?;

    let mut fields = Vec::new();
    let mut changes = Vec::new();
    for (key, directive) in ordered {
        match directive {
            SpecDirective::Inherit { .. } => {
                let parent_field = parent_fields.get(&key).expect("validated parent field");
                fields.push(EffectiveField {
                    key: key.clone(),
                    value: parent_field.value.clone(),
                    origin: EffectiveFieldOrigin::InheritedFrom {
                        approval_id: parent_approval_id.clone(),
                        spec_identity: parent_spec_identity.clone(),
                    },
                });
                changes.push(parent_change(
                    key,
                    SpecChangeKind::Inherited,
                    &parent_approval_id,
                    &parent_spec_identity,
                ));
            }
            SpecDirective::Override { value, .. } => {
                fields.push(EffectiveField {
                    key: key.clone(),
                    value,
                    origin: EffectiveFieldOrigin::OverriddenFrom {
                        approval_id: parent_approval_id.clone(),
                        spec_identity: parent_spec_identity.clone(),
                    },
                });
                changes.push(parent_change(
                    key,
                    SpecChangeKind::Overridden,
                    &parent_approval_id,
                    &parent_spec_identity,
                ));
            }
            SpecDirective::Remove { .. } => {
                changes.push(parent_change(
                    key,
                    SpecChangeKind::Removed,
                    &parent_approval_id,
                    &parent_spec_identity,
                ));
            }
            SpecDirective::Declare { value, .. } => {
                fields.push(EffectiveField {
                    key: key.clone(),
                    value,
                    origin: EffectiveFieldOrigin::DeclaredHere,
                });
                changes.push(SpecChange {
                    key,
                    kind: SpecChangeKind::Declared,
                    parent_approval_id: None,
                    parent_spec_identity: None,
                });
            }
        }
    }

    if fields.is_empty() {
        return Err("EffectiveSpec: child specification cannot remove all fields".to_string());
    }

    Ok(EffectiveSpecMaterialization::Resolved(EffectiveSpec::finish(
        Some(parent_spec_identity),
        Some(parent_approval_id),
        fields,
        changes,
    )?))
}

fn parent_change(
    key: String,
    kind: SpecChangeKind,
    parent_approval_id: &str,
    parent_spec_identity: &str,
) -> SpecChange {
    SpecChange {
        key,
        kind,
        parent_approval_id: Some(parent_approval_id.to_string()),
        parent_spec_identity: Some(parent_spec_identity.to_string()),
    }
}

fn verify_subject(spec: &EffectiveSpec, subject: &ApprovalSubject) -> Result<(), String> {
    if subject.kind() != EFFECTIVE_SPEC_SUBJECT_KIND {
        return Err(format!(
            "ApprovedEffectiveSpec: expected subject kind {}, got {}",
            EFFECTIVE_SPEC_SUBJECT_KIND,
            subject.kind()
        ));
    }
    if subject.identity() != spec.identity {
        return Err("ApprovedEffectiveSpec: approval subject identity does not match specification identity".to_string());
    }
    Ok(())
}

fn canonical_identity(
    parent_spec_identity: Option<&str>,
    parent_approval_id: Option<&str>,
    fields: &[EffectiveField],
    changes: &[SpecChange],
) -> Result<String, String> {
    let mut out = String::from("effective-spec-v1|");
    push_optional_component(&mut out, parent_spec_identity);
    push_optional_component(&mut out, parent_approval_id);
    out.push_str(&format!("|fields:{}", fields.len()));
    for field in fields {
        out.push('|');
        push_component(&mut out, &field.key);
        push_component(&mut out, &field.value);
        match &field.origin {
            EffectiveFieldOrigin::DeclaredHere => out.push_str("D"),
            EffectiveFieldOrigin::InheritedFrom {
                approval_id,
                spec_identity,
            } => {
                out.push_str("I");
                push_component(&mut out, approval_id);
                push_component(&mut out, spec_identity);
            }
            EffectiveFieldOrigin::OverriddenFrom {
                approval_id,
                spec_identity,
            } => {
                out.push_str("O");
                push_component(&mut out, approval_id);
                push_component(&mut out, spec_identity);
            }
        }
    }
    out.push_str(&format!("|changes:{}", changes.len()));
    for change in changes {
        out.push('|');
        push_component(&mut out, &change.key);
        out.push_str(match change.kind {
            SpecChangeKind::Declared => "D",
            SpecChangeKind::Inherited => "I",
            SpecChangeKind::Overridden => "O",
            SpecChangeKind::Removed => "R",
        });
        push_optional_component(&mut out, change.parent_approval_id.as_deref());
        push_optional_component(&mut out, change.parent_spec_identity.as_deref());
    }

    if out.as_bytes().len() > EFFECTIVE_SPEC_MAX_IDENTITY_BYTES {
        return Err(format!(
            "EffectiveSpec: canonical identity size {} exceeds ApprovalSubject limit {}",
            out.as_bytes().len(),
            EFFECTIVE_SPEC_MAX_IDENTITY_BYTES
        ));
    }
    Ok(out)
}

fn push_component(out: &mut String, value: &str) {
    out.push_str(&value.as_bytes().len().to_string());
    out.push(':');
    out.push_str(value);
}

fn push_optional_component(out: &mut String, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push('1');
            push_component(out, value);
        }
        None => out.push('0'),
    }
}

fn validate_unique_field_keys(fields: &[EffectiveField]) -> Result<(), String> {
    for window in fields.windows(2) {
        if window[0].key == window[1].key {
            return Err(format!("EffectiveSpec: duplicate field key: {}", window[0].key));
        }
    }
    Ok(())
}

fn validate_unique_change_keys(changes: &[SpecChange]) -> Result<(), String> {
    for window in changes.windows(2) {
        if window[0].key == window[1].key {
            return Err(format!("EffectiveSpec: duplicate change key: {}", window[0].key));
        }
    }
    Ok(())
}

fn validate_key(key: impl Into<String>) -> Result<String, String> {
    let key = key.into();
    if key.is_empty() || key.as_bytes().len() > EFFECTIVE_SPEC_MAX_KEY_BYTES {
        return Err(format!(
            "EffectiveSpec: key length must be between 1 and {} bytes",
            EFFECTIVE_SPEC_MAX_KEY_BYTES
        ));
    }
    if key.chars().any(char::is_control) {
        return Err("EffectiveSpec: key must not contain control characters".to_string());
    }
    Ok(key)
}

fn validate_value(value: impl Into<String>) -> Result<String, String> {
    let value = value.into();
    if value.as_bytes().len() > EFFECTIVE_SPEC_MAX_VALUE_BYTES {
        return Err(format!(
            "EffectiveSpec: value length must not exceed {} bytes",
            EFFECTIVE_SPEC_MAX_VALUE_BYTES
        ));
    }
    if value.chars().any(char::is_control) {
        return Err("EffectiveSpec: value must not contain control characters".to_string());
    }
    Ok(value)
}

fn validate_evidence(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(format!("EffectiveSpec: {label} is invalid"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_subject::{SubjectBoundReviewSession, SubjectBoundRevisionChain};

    fn decl(key: &str, value: &str) -> SpecDeclaration {
        SpecDeclaration::new(key, value).unwrap()
    }

    fn approved_root(declarations: Vec<SpecDeclaration>) -> ApprovedEffectiveSpec {
        let spec = EffectiveSpec::root(declarations).unwrap();
        let mut review = SubjectBoundReviewSession::new(
            "intent-effective-root",
            spec.approval_subject().unwrap(),
        )
        .unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        ApprovedEffectiveSpec::bind_root(spec, approval).unwrap()
    }

    #[test]
    fn root_order_does_not_change_identity() {
        let a = EffectiveSpec::root(vec![decl("metric", "cosine"), decl("norm", "l2")]).unwrap();
        let b = EffectiveSpec::root(vec![decl("norm", "l2"), decl("metric", "cosine")]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn value_or_provenance_changes_identity() {
        let root = approved_root(vec![decl("norm", "l2")]);
        let inherited = root
            .materialize_child(vec![SpecDirective::inherit("norm").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        let overridden_same_value = root
            .materialize_child(vec![SpecDirective::override_value("norm", "l2").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        assert_ne!(root.spec.identity, inherited.identity);
        assert_ne!(inherited.identity, overridden_same_value.identity);
    }

    #[test]
    fn duplicate_and_invalid_root_keys_fail_closed() {
        assert!(EffectiveSpec::root(vec![decl("a", "1"), decl("a", "2")]).is_err());
        assert!(SpecDeclaration::new("bad\nkey", "x").is_err());
    }

    #[test]
    fn omitted_parent_field_needs_resolution_instead_of_inheriting() {
        let root = approved_root(vec![decl("norm", "l2"), decl("metric", "cosine")]);
        let outcome = root
            .materialize_child(vec![SpecDirective::inherit("norm").unwrap()])
            .unwrap();
        let EffectiveSpecMaterialization::NeedsResolution(state) = outcome else {
            panic!("expected needs resolution")
        };
        assert_eq!(state.diagnostics.len(), 1);
        assert_eq!(state.diagnostics[0].key, "metric");
        assert_eq!(state.diagnostics[0].code, "E-SPEC-001");
    }

    #[test]
    fn inherit_override_remove_and_declare_record_explicit_provenance() {
        let root = approved_root(vec![
            decl("norm", "l2"),
            decl("metric", "cosine"),
            decl("epsilon", "1e-12"),
        ]);
        let child = root
            .materialize_child(vec![
                SpecDirective::inherit("norm").unwrap(),
                SpecDirective::override_value("metric", "l2-distance").unwrap(),
                SpecDirective::remove("epsilon").unwrap(),
                SpecDirective::declare("axis", "feature").unwrap(),
            ])
            .unwrap()
            .resolved()
            .unwrap();

        assert_eq!(child.field("norm").unwrap().value, "l2");
        assert!(matches!(
            child.field("norm").unwrap().origin,
            EffectiveFieldOrigin::InheritedFrom { .. }
        ));
        assert_eq!(child.field("metric").unwrap().value, "l2-distance");
        assert!(matches!(
            child.field("metric").unwrap().origin,
            EffectiveFieldOrigin::OverriddenFrom { .. }
        ));
        assert!(child.field("epsilon").is_none());
        assert!(child
            .changes
            .iter()
            .any(|change| change.key == "epsilon" && change.kind == SpecChangeKind::Removed));
        assert!(matches!(
            child.field("axis").unwrap().origin,
            EffectiveFieldOrigin::DeclaredHere
        ));
        assert_eq!(child.parent_approval_id.as_deref(), Some(root.approval_id()));
        assert_eq!(
            child.parent_spec_identity.as_deref(),
            Some(root.spec.identity.as_str())
        );
    }

    #[test]
    fn declare_existing_or_modify_unknown_parent_field_is_rejected() {
        let root = approved_root(vec![decl("norm", "l2")]);
        assert!(root
            .materialize_child(vec![SpecDirective::declare("norm", "zscore").unwrap()])
            .is_err());
        assert!(root
            .materialize_child(vec![
                SpecDirective::inherit("norm").unwrap(),
                SpecDirective::override_value("missing", "x").unwrap(),
            ])
            .is_err());
        assert!(root
            .materialize_child(vec![
                SpecDirective::inherit("norm").unwrap(),
                SpecDirective::remove("missing").unwrap(),
            ])
            .is_err());
    }

    #[test]
    fn root_approval_requires_exact_effective_spec_subject() {
        let spec_a = EffectiveSpec::root(vec![decl("norm", "l2")]).unwrap();
        let spec_b = EffectiveSpec::root(vec![decl("norm", "probability")]).unwrap();
        let mut review = SubjectBoundReviewSession::new(
            "intent-bind-root",
            spec_a.approval_subject().unwrap(),
        )
        .unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();

        assert!(ApprovedEffectiveSpec::bind_root(spec_b, approval.clone()).is_err());
        assert!(ApprovedEffectiveSpec::bind_root(spec_a, approval).is_ok());
    }

    #[test]
    fn revision_approval_binds_exact_child_identity_and_parent_evidence() {
        let root = approved_root(vec![decl("norm", "l2")]);
        let child = root
            .materialize_child(vec![SpecDirective::override_value("norm", "probability").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        let alternate = root
            .materialize_child(vec![SpecDirective::override_value("norm", "zscore").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();

        let root_snapshot = match &root.evidence {
            EffectiveSpecApprovalEvidence::Root(approval) => {
                let mut session = SubjectBoundReviewSession::new(
                    approval.approval.intent_id.clone(),
                    approval.subject.clone(),
                )
                .unwrap();
                session.submit("agent").unwrap();
                session.approve("customer").unwrap();
                session.snapshot()
            }
            EffectiveSpecApprovalEvidence::Revision(_) => unreachable!(),
        };
        let mut chain = SubjectBoundRevisionChain::from_approved_root(root_snapshot).unwrap();
        let revision_id = chain
            .open_revision(None, "change-norm", child.approval_subject().unwrap())
            .unwrap();
        chain.submit_revision(&revision_id, "agent").unwrap();
        let approval = chain.approve_revision(&revision_id, "customer").unwrap();

        assert!(ApprovedEffectiveSpec::bind_revision(alternate, approval.clone()).is_err());
        let approved_child = ApprovedEffectiveSpec::bind_revision(child, approval).unwrap();
        assert_eq!(approved_child.approval_id(), chain.bound_revision_approval(&revision_id).unwrap().approval.revision_approval_id);
    }

    #[test]
    fn revision_parent_approval_mismatch_is_rejected() {
        let root = approved_root(vec![decl("norm", "l2")]);
        let child = root
            .materialize_child(vec![SpecDirective::inherit("norm").unwrap()])
            .unwrap()
            .resolved()
            .unwrap();
        let mut wrong = child.clone();
        wrong.parent_approval_id = Some("stale:approval".to_string());
        wrong.identity = canonical_identity(
            wrong.parent_spec_identity.as_deref(),
            wrong.parent_approval_id.as_deref(),
            &wrong.fields,
            &wrong.changes,
        )
        .unwrap();
        assert_ne!(child.identity, wrong.identity);
    }

    #[test]
    fn identical_materialization_and_evidence_is_deterministic() {
        fn build() -> EffectiveSpec {
            let root = approved_root(vec![decl("a", "1"), decl("b", "2")]);
            root.materialize_child(vec![
                SpecDirective::inherit("a").unwrap(),
                SpecDirective::override_value("b", "3").unwrap(),
                SpecDirective::declare("c", "4").unwrap(),
            ])
            .unwrap()
            .resolved()
            .unwrap()
        }
        assert_eq!(build(), build());
    }

    #[test]
    fn removing_every_parent_field_is_rejected() {
        let root = approved_root(vec![decl("only", "value")]);
        assert!(root
            .materialize_child(vec![SpecDirective::remove("only").unwrap()])
            .is_err());
    }
}
