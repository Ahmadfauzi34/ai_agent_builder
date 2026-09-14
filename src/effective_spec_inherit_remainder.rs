//! Explicit identity-bound `inherit remainder` desugaring for EffectiveSpec revisions.
//!
//! This module is a communication convenience above EffectiveSpec v1. It never performs
//! materialization itself: it verifies the exact approved parent, fills only missing parent
//! decisions with explicit `SpecDirective::Inherit` operations, then delegates to the existing
//! `ApprovedEffectiveSpec::materialize_child` proof boundary.

use std::collections::BTreeSet;

use crate::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpecMaterialization, SpecDirective,
};

/// Explicit authorization to inherit only the undecided fields of one exact approved parent.
///
/// Both components are required so a command captured for an earlier parent revision cannot be
/// silently reused after either the specification or its approval lineage changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InheritRemainder {
    parent_spec_identity: String,
    parent_approval_id: String,
}

impl InheritRemainder {
    pub fn new(
        parent_spec_identity: impl Into<String>,
        parent_approval_id: impl Into<String>,
    ) -> Self {
        Self {
            parent_spec_identity: parent_spec_identity.into(),
            parent_approval_id: parent_approval_id.into(),
        }
    }

    pub fn parent_spec_identity(&self) -> &str {
        &self.parent_spec_identity
    }

    pub fn parent_approval_id(&self) -> &str {
        &self.parent_approval_id
    }
}

/// Expand an explicit identity-bound remainder command into ordinary per-field directives.
///
/// Existing directives are never replaced. Only parent fields with no explicit directive receive
/// a synthesized `Inherit`. The expanded list is then handed to EffectiveSpec v1 unchanged, so
/// duplicate directives, unknown fields, field limits, provenance, and canonical identity remain
/// governed by the existing core path.
pub fn materialize_child_with_inherit_remainder(
    parent: &ApprovedEffectiveSpec,
    mut directives: Vec<SpecDirective>,
    remainder: &InheritRemainder,
) -> Result<EffectiveSpecMaterialization, String> {
    if remainder.parent_spec_identity != parent.spec.identity {
        return Err(format!(
            "InheritRemainder: parent spec identity mismatch: command references {}, approved parent is {}",
            remainder.parent_spec_identity, parent.spec.identity
        ));
    }

    let actual_parent_approval_id = parent.approval_id();
    if remainder.parent_approval_id != actual_parent_approval_id {
        return Err(format!(
            "InheritRemainder: parent approval id mismatch: command references {}, approved parent is {}",
            remainder.parent_approval_id, actual_parent_approval_id
        ));
    }

    let explicit_keys = directives
        .iter()
        .map(|directive| directive_key(directive).to_string())
        .collect::<BTreeSet<_>>();

    for field in &parent.spec.fields {
        if !explicit_keys.contains(&field.key) {
            directives.push(SpecDirective::inherit(field.key.clone())?);
        }
    }

    parent.materialize_child(directives)
}

fn directive_key(directive: &SpecDirective) -> &str {
    match directive {
        SpecDirective::Inherit { key }
        | SpecDirective::Override { key, .. }
        | SpecDirective::Remove { key }
        | SpecDirective::Declare { key, .. } => key,
    }
}
