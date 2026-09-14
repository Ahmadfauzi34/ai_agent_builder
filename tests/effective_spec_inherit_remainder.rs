use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, EffectiveSpecMaterialization, SpecDeclaration,
    SpecDirective, EFFECTIVE_SPEC_MAX_FIELDS,
};
use burn_research::effective_spec_inherit_remainder::{
    materialize_child_with_inherit_remainder, InheritRemainder,
};
use burn_research::resolution_subject::SubjectBoundReviewSession;

fn approved_root(spec: EffectiveSpec) -> ApprovedEffectiveSpec {
    let mut review = SubjectBoundReviewSession::new(
        "intent-inherit-remainder",
        spec.approval_subject().unwrap(),
    )
    .unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve("customer").unwrap();
    ApprovedEffectiveSpec::bind_root(spec, approval).unwrap()
}

fn command_for(parent: &ApprovedEffectiveSpec) -> InheritRemainder {
    InheritRemainder::new(parent.spec.identity.clone(), parent.approval_id().to_string())
}

fn three_field_parent() -> ApprovedEffectiveSpec {
    approved_root(
        EffectiveSpec::root(vec![
            SpecDeclaration::new("epsilon", "0.001").unwrap(),
            SpecDeclaration::new("legacy", "enabled").unwrap(),
            SpecDeclaration::new("mode", "stable").unwrap(),
        ])
        .unwrap(),
    )
}

#[test]
fn inherit_remainder_is_exactly_equivalent_to_fully_explicit_directives() {
    let parent = three_field_parent();
    let command = command_for(&parent);

    let bulk = materialize_child_with_inherit_remainder(
        &parent,
        vec![
            SpecDirective::override_value("mode", "fast").unwrap(),
            SpecDirective::remove("legacy").unwrap(),
            SpecDirective::declare("child-only", "enabled").unwrap(),
        ],
        &command,
    )
    .unwrap()
    .resolved()
    .unwrap();

    let explicit = parent
        .materialize_child(vec![
            SpecDirective::inherit("epsilon").unwrap(),
            SpecDirective::remove("legacy").unwrap(),
            SpecDirective::override_value("mode", "fast").unwrap(),
            SpecDirective::declare("child-only", "enabled").unwrap(),
        ])
        .unwrap()
        .resolved()
        .unwrap();

    assert_eq!(bulk, explicit);
}

#[test]
fn inherit_remainder_requires_an_explicit_command_and_can_cover_all_undecided_parent_fields() {
    let parent = three_field_parent();

    let without_command = parent.materialize_child(Vec::new()).unwrap();
    let diagnostics = match without_command {
        EffectiveSpecMaterialization::NeedsResolution(state) => state.diagnostics,
        EffectiveSpecMaterialization::Resolved(_) => {
            panic!("parent fields inherited without an explicit remainder command")
        }
    };
    assert_eq!(diagnostics.len(), 3);

    let command = command_for(&parent);
    let bulk = materialize_child_with_inherit_remainder(&parent, Vec::new(), &command)
        .unwrap()
        .resolved()
        .unwrap();
    let explicit = parent
        .materialize_child(vec![
            SpecDirective::inherit("mode").unwrap(),
            SpecDirective::inherit("legacy").unwrap(),
            SpecDirective::inherit("epsilon").unwrap(),
        ])
        .unwrap()
        .resolved()
        .unwrap();

    assert_eq!(bulk, explicit);
}

#[test]
fn explicit_directives_win_and_directive_order_remains_canonical() {
    let parent = three_field_parent();
    let command = command_for(&parent);

    let forward = materialize_child_with_inherit_remainder(
        &parent,
        vec![
            SpecDirective::override_value("mode", "fast").unwrap(),
            SpecDirective::remove("legacy").unwrap(),
        ],
        &command,
    )
    .unwrap()
    .resolved()
    .unwrap();
    let reverse = materialize_child_with_inherit_remainder(
        &parent,
        vec![
            SpecDirective::remove("legacy").unwrap(),
            SpecDirective::override_value("mode", "fast").unwrap(),
        ],
        &command,
    )
    .unwrap()
    .resolved()
    .unwrap();

    assert_eq!(forward, reverse);
    assert_eq!(forward.field("mode").unwrap().value, "fast");
    assert!(forward.field("legacy").is_none());
    assert_eq!(forward.field("epsilon").unwrap().value, "0.001");
}

#[test]
fn inherit_remainder_fails_closed_on_stale_parent_identity_or_approval() {
    let parent = three_field_parent();

    let stale_identity = InheritRemainder::new(
        "effective-spec-v1|stale-parent",
        parent.approval_id().to_string(),
    );
    let identity_err =
        materialize_child_with_inherit_remainder(&parent, Vec::new(), &stale_identity)
            .unwrap_err();
    assert!(identity_err.contains("parent spec identity mismatch"));

    let stale_approval = InheritRemainder::new(parent.spec.identity.clone(), "stale-approval");
    let approval_err =
        materialize_child_with_inherit_remainder(&parent, Vec::new(), &stale_approval)
            .unwrap_err();
    assert!(approval_err.contains("parent approval id mismatch"));
}

#[test]
fn desugaring_does_not_hide_existing_duplicate_or_unknown_directive_failures() {
    let parent = three_field_parent();
    let command = command_for(&parent);

    let duplicate_err = materialize_child_with_inherit_remainder(
        &parent,
        vec![
            SpecDirective::override_value("mode", "fast").unwrap(),
            SpecDirective::override_value("mode", "faster").unwrap(),
        ],
        &command,
    )
    .unwrap_err();
    assert!(duplicate_err.contains("duplicate directive for field: mode"));

    let unknown_err = materialize_child_with_inherit_remainder(
        &parent,
        vec![SpecDirective::override_value("unknown", "value").unwrap()],
        &command,
    )
    .unwrap_err();
    assert!(unknown_err.contains("unknown parent field requires Declare instead: unknown"));
}

#[test]
fn effective_spec_field_bound_is_still_enforced_after_remainder_expansion() {
    let declarations = (0..EFFECTIVE_SPEC_MAX_FIELDS)
        .map(|index| SpecDeclaration::new(format!("field-{index:02}"), "value").unwrap())
        .collect::<Vec<_>>();
    let parent = approved_root(EffectiveSpec::root(declarations).unwrap());
    let command = command_for(&parent);

    let err = materialize_child_with_inherit_remainder(
        &parent,
        vec![SpecDirective::declare("extra", "value").unwrap()],
        &command,
    )
    .unwrap_err();
    assert!(err.contains("field count 65 exceeds maximum 64"));
}
