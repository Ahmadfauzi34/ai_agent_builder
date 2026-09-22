use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::input_port_consumer::InputPortConsumerSpec;
use burn_research::input_port_edge_binding::{
    bind_input_port_consumer_edge, semantic_graph_identity,
};
use burn_research::registry::LayerRegistry;
use burn_research::semantic_lifecycle::{
    bind_semantic_lifecycle_transition, semantic_lifecycle_capabilities,
    semantic_lifecycle_identity, semantic_lifecycle_trace,
    semantic_lifecycle_transition, SemanticLifecycleTransitionSpec,
};
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{workspace_init_binary, workspace_init_unary};
use burn_research::WasmTensor;

fn bind_observation_port(workspace: &mut AgentWorkspace) {
    workspace_bind_input_port_metadata(
        workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "fnv1a64:external".into(),
    )
    .unwrap();
}

fn observation_consumer(id: &str) -> InputPortConsumerSpec {
    InputPortConsumerSpec::new(
        id.into(),
        "[\"observation\"]".into(),
        false,
        true,
        1,
    )
    .unwrap()
}

#[test]
fn lifecycle_chain_changes_only_lifecycle_identity() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    bind_observation_port(&mut workspace);

    let id0 = workspace
        .reserve_layer_id(&registry, "relu-feature".into())
        .unwrap();
    let layer0 = AgentLayerSpec::relu(id0);
    let feature_slot = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &layer0,
        0,
        "relu-feature".into(),
    )
    .unwrap();

    let id1 = workspace
        .reserve_layer_id(&registry, "relu-candidate".into())
        .unwrap();
    let layer1 = AgentLayerSpec::relu(id1);
    let candidate_slot = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &layer1,
        feature_slot,
        "relu-candidate".into(),
    )
    .unwrap();

    let consumer = observation_consumer("feature-extractor");
    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
    let compiled_before = builder
        .compile_with_output(&registry, candidate_slot)
        .unwrap();
    let program_before = compiled_before.program_identity();
    let output_before = compiled_before.run(&registry, &input).unwrap().to_array();
    let semantic_graph_before = semantic_graph_identity(&builder);
    let lifecycle_before = semantic_lifecycle_identity(&builder);
    let workspace_before = workspace.snapshot();
    let params_before = registry.total_params();

    let observe_to_feature = SemanticLifecycleTransitionSpec::new(
        "observe-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &mut builder,
        &observe_to_feature,
        0,
    )
    .unwrap());

    let feature_to_candidate = SemanticLifecycleTransitionSpec::new(
        "feature-to-candidate".into(),
        "[\"feature\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(
        &mut builder,
        &feature_to_candidate,
        1,
    )
    .unwrap());

    let compiled_after = builder
        .compile_with_output(&registry, candidate_slot)
        .unwrap();
    let program_after = compiled_after.program_identity();
    let output_after = compiled_after.run(&registry, &input).unwrap().to_array();
    let semantic_graph_after = semantic_graph_identity(&builder);
    let lifecycle_after = semantic_lifecycle_identity(&builder);

    assert_eq!(program_before, program_after);
    assert_eq!(output_before, output_after);
    assert_eq!(semantic_graph_before, semantic_graph_after);
    assert_ne!(lifecycle_before, lifecycle_after);
    assert_eq!(workspace.snapshot(), workspace_before);
    assert_eq!(registry.total_params(), params_before);

    let trace: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_trace(&builder)).unwrap();
    assert_eq!(trace["identity"]["transition_count"], 2);
    assert_eq!(trace["transitions"][0]["transition_id"], "observe-to-feature");
    assert_eq!(trace["transitions"][0]["inputs"][0]["role"], "observation");
    assert_eq!(
        trace["transitions"][0]["inputs"][0]["origin_kind"],
        "external_input_binding"
    );
    assert_eq!(trace["transitions"][0]["output"]["role"], "feature");
    assert_eq!(trace["transitions"][1]["inputs"][0]["role"], "feature");
    assert_eq!(
        trace["transitions"][1]["inputs"][0]["origin_kind"],
        "step_output_transition"
    );
    assert_eq!(trace["transitions"][1]["inputs"][0]["origin_step_index"], 0);
    assert_eq!(trace["transitions"][1]["output"]["role"], "candidate");
}

#[test]
fn binary_transition_preserves_position_order_and_mixed_provenance() {
    let mut workspace = AgentWorkspace::new(5).unwrap();
    let mut builder = AgentGraphBuilder::new(5).unwrap();
    let mut registry = LayerRegistry::new();
    bind_observation_port(&mut workspace);

    let unary_id = workspace
        .reserve_layer_id(&registry, "relu-feature".into())
        .unwrap();
    let unary = AgentLayerSpec::relu(unary_id);
    let feature_slot = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &unary,
        0,
        "relu-feature".into(),
    )
    .unwrap();

    let binary_id = workspace
        .reserve_layer_id(&registry, "add-candidate".into())
        .unwrap();
    let binary = AgentLayerSpec::add(binary_id);
    workspace_init_binary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &binary,
        feature_slot,
        0,
        "add-candidate".into(),
    )
    .unwrap();

    let step0_consumer = observation_consumer("feature-extractor");
    bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &step0_consumer,
        0,
    )
    .unwrap();

    let step1_consumer = observation_consumer("candidate-join");
    bind_input_port_consumer_edge(
        &workspace,
        &mut builder,
        &step1_consumer,
        1,
    )
    .unwrap();

    let first = SemanticLifecycleTransitionSpec::new(
        "observation-to-feature".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    bind_semantic_lifecycle_transition(&mut builder, &first, 0).unwrap();

    let second = SemanticLifecycleTransitionSpec::new(
        "feature-plus-observation-to-candidate".into(),
        "[\"feature\",\"observation\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    bind_semantic_lifecycle_transition(&mut builder, &second, 1).unwrap();

    let trace: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_trace(&builder)).unwrap();
    let inputs = trace["transitions"][1]["inputs"].as_array().unwrap();

    assert_eq!(inputs[0]["position"], "left");
    assert_eq!(inputs[0]["role"], "feature");
    assert_eq!(inputs[0]["origin_kind"], "step_output_transition");
    assert_eq!(inputs[0]["origin_step_index"], 0);

    assert_eq!(inputs[1]["position"], "right");
    assert_eq!(inputs[1]["role"], "observation");
    assert_eq!(inputs[1]["origin_kind"], "external_input_binding");
    assert!(inputs[1]["origin_step_index"].is_null());
}

#[test]
fn lifecycle_binding_fails_closed_on_missing_or_mismatched_upstream_semantics() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    bind_observation_port(&mut workspace);

    let id0 = workspace
        .reserve_layer_id(&registry, "relu-a".into())
        .unwrap();
    let first = AgentLayerSpec::relu(id0);
    let slot1 = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &first,
        0,
        "relu-a".into(),
    )
    .unwrap();

    let id1 = workspace
        .reserve_layer_id(&registry, "relu-b".into())
        .unwrap();
    let second = AgentLayerSpec::relu(id1);
    workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &second,
        slot1,
        "relu-b".into(),
    )
    .unwrap();

    let downstream = SemanticLifecycleTransitionSpec::new(
        "downstream".into(),
        "[\"feature\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    let missing =
        bind_semantic_lifecycle_transition(&mut builder, &downstream, 1).unwrap_err();
    assert!(missing.contains("upstream step has no semantic lifecycle transition"));

    let consumer = observation_consumer("entry");
    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    let upstream = SemanticLifecycleTransitionSpec::new(
        "upstream".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    bind_semantic_lifecycle_transition(&mut builder, &upstream, 0).unwrap();

    let mismatch = SemanticLifecycleTransitionSpec::new(
        "wrong-downstream".into(),
        "[\"state\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    let error =
        bind_semantic_lifecycle_transition(&mut builder, &mismatch, 1).unwrap_err();
    assert!(error.contains("resolved role feature does not match expected role state"));

    let status: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_transition(&builder, 1).unwrap()).unwrap();
    assert_eq!(status["status"], "unbound");
}

#[test]
fn lifecycle_binding_is_idempotent_and_conflicting_rebind_is_rejected() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut builder = AgentGraphBuilder::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    bind_observation_port(&mut workspace);

    let layer_id = workspace
        .reserve_layer_id(&registry, "relu".into())
        .unwrap();
    let layer = AgentLayerSpec::relu(layer_id);
    workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &layer,
        0,
        "relu".into(),
    )
    .unwrap();

    let consumer = observation_consumer("entry");
    bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();

    let feature = SemanticLifecycleTransitionSpec::new(
        "entry-transition".into(),
        "[\"observation\"]".into(),
        "feature".into(),
    )
    .unwrap();
    assert!(bind_semantic_lifecycle_transition(&mut builder, &feature, 0).unwrap());
    assert!(!bind_semantic_lifecycle_transition(&mut builder, &feature, 0).unwrap());

    let before = semantic_lifecycle_transition(&builder, 0).unwrap();
    let conflicting = SemanticLifecycleTransitionSpec::new(
        "entry-transition".into(),
        "[\"observation\"]".into(),
        "candidate".into(),
    )
    .unwrap();
    let error =
        bind_semantic_lifecycle_transition(&mut builder, &conflicting, 0).unwrap_err();
    assert!(error.contains("different immutable semantic lifecycle transition"));
    assert_eq!(semantic_lifecycle_transition(&builder, 0).unwrap(), before);
}

#[test]
fn capability_keeps_lifecycle_outside_execution_and_semantic_graph_identity() {
    let caps: serde_json::Value =
        serde_json::from_str(&semantic_lifecycle_capabilities()).unwrap();

    assert_eq!(caps["scope"]["state_owner"], "AgentGraphBuilder");
    assert_eq!(caps["scope"]["execution_plan_effect"], "none");
    assert_eq!(caps["scope"]["numerical_effect"], "none");
    assert_eq!(caps["scope"]["semantic_graph_identity_effect"], "none");
    assert_eq!(caps["scope"]["decision_authority"], "agent");
}
