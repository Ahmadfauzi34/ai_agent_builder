use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::contracts::agent_contract_schema;
use burn_research::protocol::LAYER_NORM;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{workspace_init_unary, workspace_wire_unary};
use serde_json::Value;

#[derive(Debug)]
struct SemanticCase {
    operation: String,
    predicate: String,
    slot_binding: Option<String>,
}

fn contract_schema() -> Value {
    serde_json::from_str(&agent_contract_schema())
        .expect("canonical agent contract schema must parse")
}

fn generated_semantic_matrix(schema: &Value) -> Vec<SemanticCase> {
    let declared_predicates = schema["predicates"]
        .as_object()
        .expect("schema.predicates must be an object");
    let operations = schema["operations"]
        .as_object()
        .expect("schema.operations must be an object");

    let mut cases = Vec::new();
    for (operation, contract) in operations {
        let Some(preconditions) = contract.get("semantic_preconditions") else {
            continue;
        };
        let preconditions = preconditions
            .as_array()
            .expect("operation.semantic_preconditions must be an array");
        for precondition in preconditions {
            let predicate = precondition["predicate"]
                .as_str()
                .expect("semantic precondition predicate must be a string")
                .to_string();
            assert!(
                declared_predicates.contains_key(&predicate),
                "semantic precondition references undeclared predicate {predicate}"
            );
            let slot_binding = precondition
                .get("bind")
                .and_then(|bind| bind.get("slot"))
                .and_then(Value::as_str)
                .map(str::to_string);
            cases.push(SemanticCase {
                operation: operation.clone(),
                predicate,
                slot_binding,
            });
        }
    }
    cases
}

fn seed_linear_output(
    workspace: &mut AgentWorkspace,
    registry: &mut LayerRegistry,
    builder: &mut AgentGraphBuilder,
) -> u8 {
    let linear_id = workspace
        .reserve_layer_id(registry, "matrix-linear".into())
        .unwrap();
    let linear = AgentLayerSpec::linear(linear_id, 4, 4, true).unwrap();
    workspace_init_unary(
        workspace,
        builder,
        registry,
        &linear,
        0,
        "matrix-linear".into(),
    )
    .unwrap()
}

fn run_layout_negative(operation: &str, slot_binding: Option<&str>) {
    assert_eq!(
        slot_binding,
        Some("input_slot"),
        "layout semantic predicate must bind the canonical unary input slot"
    );

    let mut workspace = AgentWorkspace::new(4).unwrap();
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new(4).unwrap();
    let linear_out = seed_linear_output(&mut workspace, &mut registry, &mut builder);

    match operation {
        "workspaceInitUnary" => {
            let norm_id = workspace
                .reserve_layer_id(&registry, "matrix-layernorm".into())
                .unwrap();
            let layer_norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
            let before_workspace = workspace.snapshot();
            let before_steps = builder.num_steps();

            let err = workspace_init_unary(
                &mut workspace,
                &mut builder,
                &mut registry,
                &layer_norm,
                linear_out,
                "matrix-layernorm".into(),
            )
            .expect_err("known incompatible layout must be rejected");

            assert!(err.contains("layout"), "unexpected error: {err}");
            assert_eq!(workspace.snapshot(), before_workspace);
            assert_eq!(builder.num_steps(), before_steps);
            assert!(!registry.layer_exists(LAYER_NORM, norm_id));
        }
        "workspaceWireUnary" => {
            let norm_id = 9001;
            let layer_norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
            registry.init_agent_layer(&layer_norm).unwrap();
            let before_workspace = workspace.snapshot();
            let before_steps = builder.num_steps();
            let before_fingerprint = registry
                .layer_init_fingerprint(LAYER_NORM, norm_id)
                .unwrap();

            let err = workspace_wire_unary(
                &mut workspace,
                &mut builder,
                &registry,
                &layer_norm,
                linear_out,
                "matrix-manual-layernorm".into(),
            )
            .expect_err("known incompatible layout must be rejected");

            assert!(err.contains("layout"), "unexpected error: {err}");
            assert_eq!(workspace.snapshot(), before_workspace);
            assert_eq!(builder.num_steps(), before_steps);
            assert_eq!(workspace.get("_layers".into(), norm_id.to_string()), "null");
            assert_eq!(
                registry.layer_init_fingerprint(LAYER_NORM, norm_id).unwrap(),
                before_fingerprint
            );
        }
        other => panic!("semantic matrix has no layout executor for operation {other}"),
    }
}

fn run_semantic_negative_case(case: &SemanticCase) {
    match case.predicate.as_str() {
        "layout.edge_not_known_incompatible" => {
            run_layout_negative(&case.operation, case.slot_binding.as_deref())
        }
        other => panic!(
            "unhandled semantic contract matrix cell: operation={} predicate={other}",
            case.operation
        ),
    }
}

#[test]
fn schema_driven_semantic_contract_matrix_executes_negative_cells_fail_closed() {
    let schema = contract_schema();
    let cases = generated_semantic_matrix(&schema);
    assert!(
        !cases.is_empty(),
        "expected at least one semantic precondition in the canonical schema"
    );
    assert!(
        cases.iter().any(|case| case.operation == "workspaceInitUnary"
            && case.predicate == "layout.edge_not_known_incompatible")
    );
    assert!(
        cases.iter().any(|case| case.operation == "workspaceWireUnary"
            && case.predicate == "layout.edge_not_known_incompatible")
    );

    for case in &cases {
        run_semantic_negative_case(case);
    }
}
