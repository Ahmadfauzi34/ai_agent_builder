use burn_research::agent::AgentGraphBuilder;
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::{
    input_port_consumer_capabilities, input_port_consumer_compatibility, InputPortConsumerSpec,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;

#[test]
fn consumer_compatibility_is_role_aware_and_read_only() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let builder = AgentGraphBuilder::new(3).unwrap();
    let registry = LayerRegistry::new();

    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "fnv1a64:abcd".into(),
    )
    .unwrap();

    let before_workspace = workspace.snapshot();
    let before_steps = builder.num_steps();
    let before_params = registry.total_params();

    let feature_consumer = InputPortConsumerSpec::new(
        "feature-extractor".into(),
        "[\"observation\",\"feature\"]".into(),
        false,
        true,
        10,
    )
    .unwrap();
    let feature_result: serde_json::Value = serde_json::from_str(
        &input_port_consumer_compatibility(&workspace, &feature_consumer),
    )
    .unwrap();

    assert_eq!(feature_result["status"], "compatible");
    assert_eq!(feature_result["compatible"], true);
    assert_eq!(feature_result["execution_authorized"], false);
    assert_eq!(feature_result["decision_authority"], "agent");

    let reward_consumer = InputPortConsumerSpec::new(
        "reward-updater".into(),
        "[\"reward\"]".into(),
        false,
        false,
        0,
    )
    .unwrap();
    let reward_result: serde_json::Value = serde_json::from_str(
        &input_port_consumer_compatibility(&workspace, &reward_consumer),
    )
    .unwrap();

    assert_eq!(reward_result["status"], "incompatible");
    assert_eq!(reward_result["predicates"]["role_match"], false);
    assert_eq!(reward_result["reasons"][0], "role_not_accepted");

    assert_eq!(workspace.snapshot(), before_workspace);
    assert_eq!(builder.num_steps(), before_steps);
    assert_eq!(registry.total_params(), before_params);
}

#[test]
fn provenance_requirements_are_operational_without_becoming_execution_authority() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    workspace_bind_input_port_metadata(
        &mut workspace,
        "state".into(),
        "agent-state".into(),
        3,
        String::new(),
    )
    .unwrap();

    let consumer = InputPortConsumerSpec::new(
        "state-transition".into(),
        "[\"state\"]".into(),
        false,
        true,
        4,
    )
    .unwrap();

    let result: serde_json::Value =
        serde_json::from_str(&input_port_consumer_compatibility(&workspace, &consumer)).unwrap();

    assert_eq!(result["status"], "incompatible");
    assert_eq!(result["predicates"]["role_match"], true);
    assert_eq!(result["predicates"]["fingerprint_ok"], false);
    assert_eq!(result["predicates"]["revision_ok"], false);
    assert_eq!(result["execution_authorized"], false);
}

#[test]
fn unbound_port_is_unknown_and_extension_policy_remains_explicit() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let consumer = InputPortConsumerSpec::new(
        "extension-aware".into(),
        "[\"observation\"]".into(),
        true,
        false,
        0,
    )
    .unwrap();

    let unknown: serde_json::Value =
        serde_json::from_str(&input_port_consumer_compatibility(&workspace, &consumer)).unwrap();
    assert_eq!(unknown["status"], "unknown");
    assert!(unknown["compatible"].is_null());

    workspace_bind_input_port_metadata(
        &mut workspace,
        "x-market-regime".into(),
        "regime-model".into(),
        1,
        String::new(),
    )
    .unwrap();

    let compatible: serde_json::Value =
        serde_json::from_str(&input_port_consumer_compatibility(&workspace, &consumer)).unwrap();
    assert_eq!(compatible["status"], "compatible");
    assert_eq!(compatible["predicates"]["role_match"], true);
}

#[test]
fn capability_contract_keeps_selection_and_execution_outside_the_surface() {
    let caps: serde_json::Value =
        serde_json::from_str(&input_port_consumer_capabilities()).unwrap();

    assert_eq!(caps["scope"]["enforcement"], "advisory_preflight_only");
    assert_eq!(caps["compatibility"]["execution_authorized"], false);
    assert_eq!(caps["compatibility"]["decision_authority"], "agent");
}
