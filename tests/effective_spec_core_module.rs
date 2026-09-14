use burn_research::effective_spec::{
    ApprovedEffectiveSpec, EffectiveSpec, EffectiveSpecMaterialization, SpecDeclaration,
    SpecDirective, EFFECTIVE_SPEC_MAX_FIELDS, EFFECTIVE_SPEC_SCHEMA,
};
use burn_research::resolution_subject::SubjectBoundReviewSession;

fn approved_root(spec: EffectiveSpec) -> ApprovedEffectiveSpec {
    let mut review =
        SubjectBoundReviewSession::new("intent-effective-spec-core", spec.approval_subject().unwrap())
            .unwrap();
    review.submit("agent").unwrap();
    let approval = review.approve("customer").unwrap();
    ApprovedEffectiveSpec::bind_root(spec, approval).unwrap()
}

#[test]
fn effective_spec_v1_is_available_through_core_crate_module_without_semantic_drift() {
    let forward = EffectiveSpec::root(vec![
        SpecDeclaration::new("mode", "stable").unwrap(),
        SpecDeclaration::new("epsilon", "0.001").unwrap(),
    ])
    .unwrap();
    let reverse = EffectiveSpec::root(vec![
        SpecDeclaration::new("epsilon", "0.001").unwrap(),
        SpecDeclaration::new("mode", "stable").unwrap(),
    ])
    .unwrap();

    assert_eq!(forward.schema, EFFECTIVE_SPEC_SCHEMA);
    assert_eq!(forward.identity, reverse.identity);
    assert_eq!(forward.fields, reverse.fields);
    assert_eq!(EFFECTIVE_SPEC_MAX_FIELDS, 64);

    let parent = approved_root(forward);

    let sparse = parent
        .materialize_child(vec![SpecDirective::override_value("mode", "fast").unwrap()])
        .unwrap();
    let diagnostics = match sparse {
        EffectiveSpecMaterialization::NeedsResolution(state) => state.diagnostics,
        EffectiveSpecMaterialization::Resolved(_) => {
            panic!("omitted parent field unexpectedly inherited implicitly")
        }
    };
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, "E-SPEC-001");
    assert_eq!(diagnostics[0].key, "epsilon");

    let forward_child = parent
        .materialize_child(vec![
            SpecDirective::override_value("mode", "fast").unwrap(),
            SpecDirective::inherit("epsilon").unwrap(),
        ])
        .unwrap()
        .resolved()
        .unwrap();
    let reverse_child = parent
        .materialize_child(vec![
            SpecDirective::inherit("epsilon").unwrap(),
            SpecDirective::override_value("mode", "fast").unwrap(),
        ])
        .unwrap()
        .resolved()
        .unwrap();

    assert_eq!(forward_child.identity, reverse_child.identity);
    assert_eq!(forward_child.fields, reverse_child.fields);
    assert_eq!(forward_child.changes, reverse_child.changes);
}

#[test]
fn effective_spec_core_module_has_no_math_program_execution_authority() {
    let source = include_str!("../src/effective_spec.rs");
    for forbidden in [
        "use crate::math",
        "crate::math::",
        "MathProgramBuilder",
        "MathProgramV4Builder",
        ".run1(",
        ".run2(",
    ] {
        assert!(
            !source.contains(forbidden),
            "EffectiveSpec core contains forbidden execution coupling: {forbidden}"
        );
    }
}
