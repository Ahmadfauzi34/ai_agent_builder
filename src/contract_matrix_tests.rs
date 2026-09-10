use serde_json::Value;

use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::contracts::agent_contract_schema;
use crate::protocol::{LAYER_ACTIVATION, LAYER_BINARY};
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace_ops::{
    workspace_compile, workspace_init_binary, workspace_init_unary, workspace_wire_binary,
    workspace_wire_unary,
};

#[derive(Clone, Debug)]
struct MatrixCase {
    operation: String,
    predicate: String,
    slot_binding: Option<String>,
}

fn contract_schema() -> Value {
    serde_json::from_str(&agent_contract_schema()).expect("canonical agent contract schema must parse")
}

fn generated_negative_matrix(schema: &Value) -> Vec<MatrixCase> {
    let operations = schema["operations"]
        .as_object()
        .expect("schema.operations must be an object");
    let mut cases = Vec::new();
    for (operation, contract) in operations {
        let preconditions = contract["preconditions"]
            .as_array()
            .expect("operation.preconditions must be an array");
        for precondition in preconditions {
            let predicate = precondition["predicate"]
                .as_str()
                .expect("precondition.predicate must be a string")
                .to_string();
            let slot_binding = precondition
                .get("bind")
                .and_then(|bind| bind.get("slot"))
                .and_then(Value::as_str)
                .map(str::to_string);
            cases.push(MatrixCase {
                operation: operation.clone(),
                predicate,
                slot_binding,
            });
        }
    }
    cases
}

fn generated_operations(schema: &Value) -> Vec<String> {
    schema["operations"]
        .as_object()
        .expect("schema.operations must be an object")
        .keys()
        .cloned()
        .collect()
}

fn assert_workspace_and_builder_unchanged(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    before_workspace: &str,
    before_steps: u32,
) {
    assert_eq!(workspace.snapshot(), before_workspace);
    assert_eq!(builder.num_steps(), before_steps);
}

fn run_success_case(operation: &str) {
    match operation {
        "reserveSlot" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            assert_eq!(workspace.reserve_slot("matrix".into()).unwrap(), 1);
        }
        "releaseSlot" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let slot = workspace.reserve_slot("matrix".into()).unwrap();
            workspace.release_slot(slot).unwrap();
            assert!(workspace.get("_slots".into(), slot.to_string()).contains("\"state\":\"free\""));
        }
        "workspaceInitUnary" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            let out = workspace_init_unary(
                &mut workspace,
                &mut builder,
                &mut registry,
                &spec,
                0,
                "relu".into(),
            )
            .unwrap();
            assert_eq!(out, 1);
            assert_eq!(builder.num_steps(), 1);
            assert!(registry.layer_exists(LAYER_ACTIVATION, id));
        }
        "workspaceInitBinary" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            let out = workspace_init_binary(
                &mut workspace,
                &mut builder,
                &mut registry,
                &spec,
                0,
                0,
                "add".into(),
            )
            .unwrap();
            assert_eq!(out, 1);
            assert_eq!(builder.num_steps(), 1);
            assert!(registry.layer_exists(LAYER_BINARY, id));
        }
        "workspaceWireUnary" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::relu(41);
            registry.init_agent_layer(&spec).unwrap();
            let out = workspace_wire_unary(
                &mut workspace,
                &mut builder,
                &registry,
                &spec,
                0,
                "relu".into(),
            )
            .unwrap();
            assert_eq!(out, 1);
            assert_eq!(builder.num_steps(), 1);
        }
        "workspaceWireBinary" => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::add(42);
            registry.init_agent_layer(&spec).unwrap();
            let out = workspace_wire_binary(
                &mut workspace,
                &mut builder,
                &registry,
                &spec,
                0,
                0,
                "add".into(),
            )
            .unwrap();
            assert_eq!(out, 1);
            assert_eq!(builder.num_steps(), 1);
        }
        "workspaceCompile" => {
            let mut registry = LayerRegistry::new();
            let spec = AgentLayerSpec::relu(43);
            registry.init_agent_layer(&spec).unwrap();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            builder.add_unary(&spec, 0, 1).unwrap();
            let graph = workspace_compile(&builder, &registry, 1).unwrap();
            assert_eq!(graph.output_slot(), 1);
        }
        other => panic!("contract matrix has no success handler for operation {other}"),
    }
}

fn run_negative_case(case: &MatrixCase) {
    match (case.operation.as_str(), case.predicate.as_str()) {
        ("reserveSlot", "owner.non_empty") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let before = workspace.snapshot();
            assert!(workspace.reserve_slot(String::new()).is_err());
            assert_eq!(workspace.snapshot(), before);
        }
        ("reserveSlot", "slot.free_exists") => {
            let mut workspace = AgentWorkspace::new(2).unwrap();
            workspace.reserve_slot("blocker".into()).unwrap();
            let before = workspace.snapshot();
            assert!(workspace.reserve_slot("overflow".into()).is_err());
            assert_eq!(workspace.snapshot(), before);
        }
        ("releaseSlot", "slot.in_workspace_range") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let before = workspace.snapshot();
            assert!(workspace.release_slot(3).is_err());
            assert_eq!(workspace.snapshot(), before);
        }
        ("releaseSlot", "slot.state_reserved") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let before = workspace.snapshot();
            assert!(workspace.release_slot(1).is_err());
            assert_eq!(workspace.snapshot(), before);
        }
        ("workspaceInitUnary", "spec.unary") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_BINARY, id));
        }
        ("workspaceInitUnary", "slot.readable") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 1, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_ACTIVATION, id));
        }
        ("workspaceInitUnary", "registry.layer_absent") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(registry.layer_exists(LAYER_ACTIVATION, id));
        }
        ("workspaceInitUnary", "workspace.layer_reserved") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::relu(77);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_ACTIVATION, 77));
        }
        ("workspaceInitUnary", "label.within_limit") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 0, "x".repeat(4001)).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_ACTIVATION, id));
        }
        ("workspaceInitUnary", "slot.free_exists") => {
            let mut workspace = AgentWorkspace::new(2).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(2).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            workspace.reserve_slot("blocker".into()).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_unary(&mut workspace, &mut builder, &mut registry, &spec, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_ACTIVATION, id));
        }
        ("workspaceInitBinary", "spec.binary") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
            let spec = AgentLayerSpec::relu(id);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, 0, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_ACTIVATION, id));
        }
        ("workspaceInitBinary", "slot.readable") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            let (left, right) = match case.slot_binding.as_deref() {
                Some("left_slot") => (1, 0),
                Some("right_slot") => (0, 1),
                other => panic!("unexpected slot binding for workspaceInitBinary: {other:?}"),
            };
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, left, right, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_BINARY, id));
        }
        ("workspaceInitBinary", "registry.layer_absent") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, 0, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(registry.layer_exists(LAYER_BINARY, id));
        }
        ("workspaceInitBinary", "workspace.layer_reserved") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::add(78);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, 0, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_BINARY, 78));
        }
        ("workspaceInitBinary", "label.within_limit") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, 0, 0, "x".repeat(4001)).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_BINARY, id));
        }
        ("workspaceInitBinary", "slot.free_exists") => {
            let mut workspace = AgentWorkspace::new(2).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(2).unwrap();
            let id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
            let spec = AgentLayerSpec::add(id);
            workspace.reserve_slot("blocker".into()).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_init_binary(&mut workspace, &mut builder, &mut registry, &spec, 0, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert!(!registry.layer_exists(LAYER_BINARY, id));
        }
        ("workspaceWireUnary", "spec.unary") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::add(81);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_unary(&mut workspace, &mut builder, &registry, &spec, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireUnary", "slot.readable") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::relu(82);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_unary(&mut workspace, &mut builder, &registry, &spec, 1, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireUnary", "registry.identity_exact") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let live = AgentLayerSpec::relu(83);
            let supplied = AgentLayerSpec::sigmoid(83);
            registry.init_agent_layer(&live).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_unary(&mut workspace, &mut builder, &registry, &supplied, 0, "wrong".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireUnary", "label.within_limit") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::relu(84);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_unary(&mut workspace, &mut builder, &registry, &spec, 0, "x".repeat(4001)).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireUnary", "slot.free_exists") => {
            let mut workspace = AgentWorkspace::new(2).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(2).unwrap();
            let spec = AgentLayerSpec::relu(85);
            registry.init_agent_layer(&spec).unwrap();
            workspace.reserve_slot("blocker".into()).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_unary(&mut workspace, &mut builder, &registry, &spec, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert_eq!(workspace.get("_layers".into(), "85".into()), "null");
        }
        ("workspaceWireBinary", "spec.binary") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::relu(86);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_binary(&mut workspace, &mut builder, &registry, &spec, 0, 0, "relu".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireBinary", "slot.readable") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::add(87);
            registry.init_agent_layer(&spec).unwrap();
            let (left, right) = match case.slot_binding.as_deref() {
                Some("left_slot") => (1, 0),
                Some("right_slot") => (0, 1),
                other => panic!("unexpected slot binding for workspaceWireBinary: {other:?}"),
            };
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_binary(&mut workspace, &mut builder, &registry, &spec, left, right, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireBinary", "registry.identity_exact") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let live = AgentLayerSpec::add(88);
            let supplied = AgentLayerSpec::sub(88);
            registry.init_agent_layer(&live).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_binary(&mut workspace, &mut builder, &registry, &supplied, 0, 0, "wrong".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireBinary", "label.within_limit") => {
            let mut workspace = AgentWorkspace::new(3).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            let spec = AgentLayerSpec::add(89);
            registry.init_agent_layer(&spec).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_binary(&mut workspace, &mut builder, &registry, &spec, 0, 0, "x".repeat(4001)).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
        }
        ("workspaceWireBinary", "slot.free_exists") => {
            let mut workspace = AgentWorkspace::new(2).unwrap();
            let mut registry = LayerRegistry::new();
            let mut builder = AgentGraphBuilder::new(2).unwrap();
            let spec = AgentLayerSpec::add(90);
            registry.init_agent_layer(&spec).unwrap();
            workspace.reserve_slot("blocker".into()).unwrap();
            let before = workspace.snapshot();
            let steps = builder.num_steps();
            assert!(workspace_wire_binary(&mut workspace, &mut builder, &registry, &spec, 0, 0, "add".into()).is_err());
            assert_workspace_and_builder_unchanged(&workspace, &builder, &before, steps);
            assert_eq!(workspace.get("_layers".into(), "90".into()), "null");
        }
        ("workspaceCompile", "builder.output_written") => {
            let mut registry = LayerRegistry::new();
            let spec = AgentLayerSpec::relu(91);
            registry.init_agent_layer(&spec).unwrap();
            let mut builder = AgentGraphBuilder::new(3).unwrap();
            builder.add_unary(&spec, 0, 1).unwrap();
            builder.set_output(1).unwrap();
            let before_steps = builder.num_steps();
            assert!(workspace_compile(&builder, &registry, 2).is_err());
            assert_eq!(builder.num_steps(), before_steps);
            assert_eq!(builder.compile(&registry).unwrap().output_slot(), 1);
        }
        other => panic!("unhandled generated contract matrix cell: {other:?} / binding {:?}", case.slot_binding),
    }
}

#[test]
fn schema_driven_contract_matrix_executes_positive_and_negative_cells() {
    let schema = contract_schema();
    let operations = generated_operations(&schema);
    assert!(!operations.is_empty());
    for operation in &operations {
        run_success_case(operation);
    }

    let cases = generated_negative_matrix(&schema);
    assert!(cases.len() >= 20, "expected a non-trivial generated matrix, got {}", cases.len());
    for case in &cases {
        run_negative_case(case);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShadowSlot {
    Input,
    Free,
    Reserved,
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn assert_shadow_matches(workspace: &AgentWorkspace, shadow: &[ShadowSlot]) {
    for (index, expected) in shadow.iter().enumerate() {
        let row = workspace.get("_slots".into(), index.to_string());
        let marker = match expected {
            ShadowSlot::Input => "\"state\":\"input\"",
            ShadowSlot::Free => "\"state\":\"free\"",
            ShadowSlot::Reserved => "\"state\":\"reserved\"",
        };
        assert!(row.contains(marker), "slot {index} expected {expected:?}, row={row}");
    }
}

#[test]
fn seeded_slot_lifecycle_fuzzer_matches_schema_state_machine() {
    let schema = contract_schema();
    let transitions = schema["state_domains"]["slot"]["transitions"]
        .as_array()
        .expect("slot transitions must be declared");
    assert!(transitions.iter().any(|t| {
        t["operation"] == "reserveSlot" && t["from"] == "free" && t["to"] == "reserved"
    }));
    assert!(transitions.iter().any(|t| {
        t["operation"] == "releaseSlot" && t["from"] == "reserved" && t["to"] == "free"
    }));

    for seed in [1_u64, 0xC0FFEE, 0x5EED1234, u64::MAX - 1] {
        let mut rng = seed;
        let mut workspace = AgentWorkspace::new(8).unwrap();
        let mut shadow = vec![ShadowSlot::Free; 8];
        shadow[0] = ShadowSlot::Input;

        for step in 0..512 {
            if next_u64(&mut rng) & 1 == 0 {
                let before = workspace.snapshot();
                let expected_slot = shadow.iter().position(|state| *state == ShadowSlot::Free);
                let result = workspace.reserve_slot(format!("seed-{seed}-step-{step}"));
                match expected_slot {
                    Some(index) => {
                        assert_eq!(result.unwrap(), index as u8);
                        shadow[index] = ShadowSlot::Reserved;
                    }
                    None => {
                        assert!(result.is_err());
                        assert_eq!(workspace.snapshot(), before);
                    }
                }
            } else {
                let slot = (next_u64(&mut rng) % 10) as u8;
                let before = workspace.snapshot();
                let should_succeed = slot > 0
                    && (slot as usize) < shadow.len()
                    && shadow[slot as usize] == ShadowSlot::Reserved;
                let result = workspace.release_slot(slot);
                if should_succeed {
                    result.unwrap();
                    shadow[slot as usize] = ShadowSlot::Free;
                } else {
                    assert!(result.is_err());
                    assert_eq!(workspace.snapshot(), before);
                }
            }
            assert_shadow_matches(&workspace, &shadow);
        }
    }
}
