#[test]
fn authorization_policy_has_no_math_program_execution_coupling() {
    let source = include_str!("../src/authorization.rs");
    let forbidden = [
        "use crate::math",
        "crate::math::",
        "MathProgramBuilder",
        "MathProgramV4Builder",
        ".run1(",
        ".run2(",
    ];

    for needle in forbidden {
        assert!(
            !source.contains(needle),
            "authorization layer contains forbidden Math Program execution coupling: {needle}"
        );
    }
}
