use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::contracts::agent_contract_schema;
use burn_research::protocol::LAYER_NORM;
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{workspace_init_unary, workspace_wire_unary};
use serde_json::Value;

fn schema() -> Value {
    serde_json::from_str(&agent_contract_schema()).unwrap()
}

#[test]
fn semantic_layout_contract_matrix_matches_runtime() {
    let schema = schema();
    let predicate = &schema["predicates"]["layout.edge_not_known_incompatible"];
    assert_eq!(predicate["policy"]["known_incompatible"], "reject");
    assert_eq!(predicate["policy"]["unknown"], "allow_and_defer_to_runtime");

    let mut cells = Vec::new();
    for (operation, contract) in schema["operations"].as_object().unwrap() {
        if let Some(items) = contract.get("semantic_preconditions") {
            for item in items.as_array().unwrap() {
                cells.push((operation.as_str(), item["predicate"].as_str().unwrap()));
            }
        }
    }
    assert_eq!(cells.len(), 2);

    for (operation, predicate) in cells {
        assert_eq!(predicate, "layout.edge_not_known_incompatible");

        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let linear_id = workspace.reserve_layer_id(&registry, "linear".into()).unwrap();
        let linear = AgentLayerSpec::linear(linear_id, 4, 4, true).unwrap();
        let input = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &linear,
            0,
            "linear".into(),
        )
        .unwrap();

        match operation {
            "workspaceInitUnary" => {
                let id = workspace.reserve_layer_id(&registry, "layernorm".into()).unwrap();
                let norm = AgentLayerSpec::layer_norm(id, 4, None).unwrap();
                let before = workspace.snapshot();
                let steps = builder.num_steps();
                assert!(workspace_init_unary(
                    &mut workspace,
                    &mut builder,
                    &mut registry,
                    &norm,
                    input,
                    "layernorm".into(),
                )
                .is_err());
                assert_eq!(workspace.snapshot(), before);
                assert_eq!(builder.num_steps(), steps);
                assert!(!registry.layer_exists(LAYER_NORM, id));
            }
            "workspaceWireUnary" => {
                let norm = AgentLayerSpec::layer_norm(99, 4, None).unwrap();
                registry.init_agent_layer(&norm).unwrap();
                let before = workspace.snapshot();
                let steps = builder.num_steps();
                assert!(workspace_wire_unary(
                    &mut workspace,
                    &mut builder,
                    &registry,
                    &norm,
                    input,
                    "layernorm".into(),
                )
                .is_err());
                assert_eq!(workspace.snapshot(), before);
                assert_eq!(builder.num_steps(), steps);
                assert_eq!(workspace.get("_layers".into(), "99".into()), "null");
            }
            other => panic!("unhandled semantic contract operation {other}"),
        }
    }
}
