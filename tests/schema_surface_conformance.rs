use std::collections::BTreeSet;

use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::contracts::agent_contract_schema;
use burn_research::protocol::LAYER_ACTIVATION;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{
    workspace_capabilities, workspace_compile, workspace_init_unary, workspace_wire_unary,
};
use serde_json::Value;

fn string_set(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .expect("expected JSON array")
        .iter()
        .map(|item| item.as_str().expect("expected string").to_string())
        .collect()
}

#[test]
fn workspace_discovery_matches_schema_surface_inventory_exactly() {
    let capabilities: Value = serde_json::from_str(&workspace_capabilities()).unwrap();
    let schema: Value = serde_json::from_str(&agent_contract_schema()).unwrap();
    let workspace = &schema["surface_inventory"]["workspace"];

    assert_eq!(workspace["constructor"], "AgentWorkspace");

    let classes = ["contracted_direct", "noncanonical_mutators", "read_only"];
    let mut inventoried = BTreeSet::new();
    let mut inventoried_count = 0usize;
    for class in classes {
        let entries = workspace[class].as_array().expect("surface class must be an array");
        inventoried_count += entries.len();
        for entry in entries {
            assert!(
                inventoried.insert(entry.as_str().unwrap().to_string()),
                "workspace surface appears in more than one inventory class: {entry}"
            );
        }
    }

    assert_eq!(inventoried.len(), inventoried_count);
    assert_eq!(inventoried, string_set(&capabilities["workspace_methods"]));
    assert_eq!(
        string_set(&schema["surface_inventory"]["free_operations"]),
        string_set(&capabilities["ops"])
    );
    assert_eq!(
        string_set(&schema["surface_inventory"]["escape_hatches"]),
        string_set(&capabilities["escape_hatches"])
    );
}

#[test]
fn every_operation_guard_references_the_declared_guard_registry() {
    let schema: Value = serde_json::from_str(&agent_contract_schema()).unwrap();
    let guard_registry = schema["implementation_guards"]
        .as_object()
        .expect("implementation_guards registry must exist");

    for (operation, contract) in schema["operations"].as_object().unwrap() {
        for field in ["implementation_guards", "post_reservation_guards"] {
            let Some(guards) = contract.get(field) else {
                continue;
            };
            for guard in guards.as_array().expect("guard list must be an array") {
                let guard_id = guard["guard"].as_str().expect("guard id must be a string");
                assert!(
                    guard_registry.contains_key(guard_id),
                    "operation {operation} references undeclared guard {guard_id}"
                );
            }
        }
    }
}

#[test]
fn canonical_operations_expose_builder_range_guards_at_the_real_boundaries() {
    let schema: Value = serde_json::from_str(&agent_contract_schema()).unwrap();
    let operations = schema["operations"].as_object().unwrap();

    let expected_argument_guards = [
        ("workspaceInitUnary", vec!["input_slot"]),
        ("workspaceInitBinary", vec!["left_slot", "right_slot"]),
        ("workspaceWireUnary", vec!["input_slot"]),
        ("workspaceWireBinary", vec!["left_slot", "right_slot"]),
        ("workspaceCompile", vec!["output_slot"]),
    ];

    for (operation, expected_slots) in expected_argument_guards {
        let guards = operations[operation]["implementation_guards"]
            .as_array()
            .expect("canonical operation must expose implementation guards");
        let actual = guards
            .iter()
            .map(|guard| {
                assert_eq!(guard["guard"], "builder.slot_in_range");
                guard["bind"]["slot"].as_str().unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_slots, "wrong builder-range bindings for {operation}");
    }

    for operation in [
        "workspaceInitUnary",
        "workspaceInitBinary",
        "workspaceWireUnary",
        "workspaceWireBinary",
    ] {
        let guards = operations[operation]["post_reservation_guards"]
            .as_array()
            .expect("auto-output operation must expose post-reservation guard");
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0]["guard"], "builder.slot_in_range");
        assert_eq!(guards[0]["bind"]["slot"], "output_slot");
        assert_eq!(guards[0]["source"], "auto_reserved_output");
    }

    assert_eq!(
        operations["workspaceInitUnary"]["post_reservation_guards"][0]["failure"],
        "workspace_checkpoint_restore"
    );
    assert_eq!(
        operations["workspaceWireUnary"]["post_reservation_guards"][0]["failure"],
        "release_output_slot"
    );
}

#[test]
fn builder_input_range_guard_fails_before_init_or_metadata_mutation() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(1).unwrap();
    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let input_slot = workspace.reserve_slot("readable-input".into()).unwrap();
    assert_eq!(input_slot, 1);

    let before = workspace.snapshot();
    let err = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        input_slot,
        "relu".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside builder num_slots"));
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(builder.num_steps(), 0);
    assert!(!registry.layer_exists(LAYER_ACTIVATION, layer_id));
}

#[test]
fn builder_input_range_guard_keeps_manual_wire_retriable() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut too_small_builder = AgentGraphBuilder::new(1).unwrap();
    let spec = AgentLayerSpec::relu(41);
    registry.init_agent_layer(&spec).unwrap();
    let input_slot = workspace.reserve_slot("readable-input".into()).unwrap();

    let before = workspace.snapshot();
    let err = workspace_wire_unary(
        &mut workspace,
        &mut too_small_builder,
        &registry,
        &spec,
        input_slot,
        "manual-relu".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside builder num_slots"));
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(workspace.get("_layers".into(), "41".into()), "null");
    assert_eq!(too_small_builder.num_steps(), 0);

    let mut compatible_builder = AgentGraphBuilder::new(3).unwrap();
    let output = workspace_wire_unary(
        &mut workspace,
        &mut compatible_builder,
        &registry,
        &spec,
        input_slot,
        "manual-relu".into(),
    )
    .unwrap();
    assert_eq!(output, 2);
    assert_eq!(compatible_builder.num_steps(), 1);
}

#[test]
fn auto_reserved_output_guard_rolls_back_without_restricting_the_api() {
    let mut workspace = AgentWorkspace::new(3).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(1).unwrap();
    let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(layer_id);
    let before = workspace.snapshot();

    let err = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap_err();

    assert!(err.contains("outside builder num_slots"));
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(builder.num_steps(), 0);
    assert!(!registry.layer_exists(LAYER_ACTIVATION, layer_id));
}

#[test]
fn workspace_compile_exposes_builder_range_error_without_state_change() {
    let builder = AgentGraphBuilder::new(1).unwrap();
    let registry = LayerRegistry::new();
    let err = match workspace_compile(&builder, &registry, 1) {
        Ok(_) => panic!("workspaceCompile unexpectedly accepted an out-of-range builder slot"),
        Err(err) => err,
    };
    assert!(err.contains("outside builder num_slots"));
    assert_eq!(builder.num_steps(), 0);
}
