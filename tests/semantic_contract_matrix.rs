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
}

fn semantic_cases() -> Vec<SemanticCase> {
    let schema: Value = serde_json::from_str(&agent_contract_schema())
        .expect("canonical agent contract schema must parse");
    let operations = schema["operations"]
        .as_object()
        .expect("schema.operations must be an object");

    let mut cases = Vec::new();
    for (operation, contract) in operations {
        let Some(semantic) = contract.get("semantic_preconditions") else {
            continue;
        };
        for precondition in semantic
            .as_array()
            .expect("semantic_preconditions must be an array")
        {
            cases.push(SemanticCase {
                operation: operation.clone(),
                predicate: precondition["predicate"]
                    .as_str()
                    .expect("semantic predicate must be a string")
                    .to_string(),
            });
        }
    }
    cases
}

fn canonical_linear_producer(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &mut LayerRegistry,
) -> u8 {
    let linear_id = workspace
        .reserve_layer_id(registry, "semantic-linear".into())
        .unwrap();
    let linear = AgentLayerSpec::linear(linear_id, 2, 4, true).unwrap();
    workspace_init_unary(
        workspace,
        builder,
        registry,
        &linear,
        0,
        "semantic-linear".into(),
    )
    .unwrap()
}

fn run_layout_negative(operation: &str) {
    let mut workspace = AgentWorkspace::new(5).unwrap();
    let mut builder = AgentGraphBuilder::new(5).unwrap();
    let mut registry = LayerRegistry::new();
    let producer_slot = canonical_linear_producer(&mut workspace, &mut builder, &mut registry);

    match operation {
        "workspaceInitUnary" => {
            let norm_id = workspace
                .reserve_layer_id(&registry, "semantic-layernorm".into())
                .unwrap();
            let norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
            let before_workspace = workspace.snapshot();
            let before_steps = builder.num_steps();

            let err = workspace_init_unary(
                &mut workspace,
                &mut builder,
                &mut registry,
                &norm,
                producer_slot,
                "semantic-layernorm".into(),
            )
            .unwrap_err();
            assert!(err.contains("layout"), "unexpected error: {err}");
            assert_eq!(workspace.snapshot(), before_workspace);
            assert_eq!(builder.num_steps(), before_steps);
            assert!(!registry.layer_exists(LAYER_NORM, norm_id));

            // Unknown external-input layout must remain deferred and the object reusable.
            let retry = workspace_init_unary(
                &mut workspace,
                &mut builder,
                &mut registry,
                &norm,
                0,
                "semantic-layernorm-retry".into(),
            );
            assert!(retry.is_ok());
        }
        "workspaceWireUnary" => {
            let norm_id = 9001;
            let norm = AgentLayerSpec::layer_norm(norm_id, 4, None).unwrap();
            registry.init_agent_layer(&norm).unwrap();
            let before_workspace = workspace.snapshot();
            let before_steps = builder.num_steps();

            let err = workspace_wire_unary(
                &mut workspace,
                &mut builder,
                &registry,
                &norm,
                producer_slot,
                "semantic-layernorm-wire".into(),
            )
            .unwrap_err();
            assert!(err.contains("layout"), "unexpected error: {err}");
            assert_eq!(workspace.snapshot(), before_workspace);
            assert_eq!(builder.num_steps(), before_steps);
            assert!(registry.layer_exists(LAYER_NORM, norm_id));

            // Reconcile from unknown external-input layout after failure.
            let retry = workspace_wire_unary(
                &mut workspace,
                &mut builder,
                &registry,
                &norm,
                0,
                "semantic-layernorm-wire-retry".into(),
            );
            assert!(retry.is_ok());
        }
        other => panic!("unhandled semantic operation {other}"),
    }
}

#[test]
fn schema_driven_semantic_contract_matrix_is_fail_closed() {
    let cases = semantic_cases();
    assert_eq!(
        cases.len(),
        2,
        "expected exactly the canonical unary init/wire semantic layout cells"
    );

    for case in cases {
        match case.predicate.as_str() {
            "layout.edge_not_known_incompatible" => run_layout_negative(&case.operation),
            other => panic!(
                "unhandled semantic contract predicate {other} for operation {}",
                case.operation
            ),
        }
    }
}
