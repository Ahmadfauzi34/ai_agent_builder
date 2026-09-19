use std::collections::HashSet;

use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::program_bundle::{export_program_bundle, import_program_bundle};
use burn_research::registry::LayerRegistry;

const PLAN_HEADER_BYTES: usize = 8;
const PLAN_STEP_BYTES: usize = 9;
const PLAN_OUTPUT_BYTES: usize = 1;
const BUNDLE_MAGIC: &[u8; 8] = b"BRPGBNDL";
const BUNDLE_HEADER_BYTES: usize = 28;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn unique_plan_layer_keys(plan: &[u8]) -> Vec<(u8, u32)> {
    assert!(plan.len() >= PLAN_HEADER_BYTES + PLAN_OUTPUT_BYTES);
    let num_steps = read_u32(plan, 0) as usize;
    let expected_len = PLAN_HEADER_BYTES + num_steps * PLAN_STEP_BYTES + PLAN_OUTPUT_BYTES;
    assert_eq!(plan.len(), expected_len);

    let mut seen = HashSet::new();
    let mut keys = Vec::new();
    for index in 0..num_steps {
        let offset = PLAN_HEADER_BYTES + index * PLAN_STEP_BYTES;
        let layer_type = plan[offset + 1];
        let layer_id = read_u32(plan, offset + 2);
        if seen.insert((layer_type, layer_id)) {
            keys.push((layer_type, layer_id));
        }
    }
    keys
}

fn bundle_layer_keys(bundle: &[u8]) -> Vec<(u8, u32)> {
    assert!(bundle.len() >= BUNDLE_HEADER_BYTES);
    assert_eq!(&bundle[..8], BUNDLE_MAGIC);
    assert_eq!(read_u32(bundle, 8), 1);

    let plan_len = read_u32(bundle, 16) as usize;
    let identity_len = read_u32(bundle, 20) as usize;
    let layer_count = read_u32(bundle, 24) as usize;

    let mut cursor = BUNDLE_HEADER_BYTES + plan_len + identity_len;
    let mut keys = Vec::with_capacity(layer_count);

    for _ in 0..layer_count {
        assert!(cursor + 16 <= bundle.len());
        let layer_type = bundle[cursor];
        let reserved = bundle[cursor + 3];
        assert_eq!(reserved, 0);
        let layer_id = read_u32(bundle, cursor + 4);
        let init_len = read_u32(bundle, cursor + 8) as usize;
        let state_len = read_u32(bundle, cursor + 12) as usize;
        cursor += 16;

        let end = cursor
            .checked_add(init_len)
            .and_then(|value| value.checked_add(state_len))
            .expect("bundle record length overflow in audit parser");
        assert!(end <= bundle.len());
        cursor = end;
        keys.push((layer_type, layer_id));
    }

    assert_eq!(cursor, bundle.len());
    keys
}

#[test]
fn public_readers_agree_on_graph_plan_first_use_order() {
    let mut registry = LayerRegistry::new();
    let first = AgentLayerSpec::linear(201, 2, 2, true).unwrap();
    let relu = AgentLayerSpec::relu(202);
    let second = AgentLayerSpec::linear(203, 2, 1, true).unwrap();

    registry.init_agent_layer(&first).unwrap();
    registry.init_agent_layer(&relu).unwrap();
    registry.init_agent_layer(&second).unwrap();

    let mut builder = AgentGraphBuilder::new(5).unwrap();
    builder.add_unary(&first, 0, 1).unwrap();
    builder.add_unary(&relu, 1, 2).unwrap();
    builder.add_unary(&first, 2, 3).unwrap();
    builder.add_unary(&second, 3, 4).unwrap();
    builder.set_output(4).unwrap();

    let graph = builder.compile(&registry).unwrap();
    let plan = graph.program_plan();
    let identity = graph.program_identity();

    let unique_keys = unique_plan_layer_keys(&plan);
    assert_eq!(
        unique_keys,
        vec![
            (first.layer_type(), first.layer_id()),
            (relu.layer_type(), relu.layer_id()),
            (second.layer_type(), second.layer_id()),
        ],
    );

    let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
    let owner_keys = binding
        .owners()
        .iter()
        .map(|owner| (owner.layer_type(), owner.layer_id()))
        .collect::<Vec<_>>();
    assert_eq!(
        owner_keys,
        vec![
            (first.layer_type(), first.layer_id()),
            (second.layer_type(), second.layer_id()),
        ],
    );

    let projected_trainable_order = unique_keys
        .iter()
        .copied()
        .filter(|key| owner_keys.contains(key))
        .collect::<Vec<_>>();
    assert_eq!(projected_trainable_order, owner_keys);

    let bundle_a = export_program_bundle(&graph, &registry, false).unwrap();
    let bundle_b = export_program_bundle(&graph, &registry, false).unwrap();
    assert_eq!(bundle_a, bundle_b);
    assert_eq!(bundle_layer_keys(&bundle_a), unique_keys);

    let mut imported_registry = LayerRegistry::new();
    let imported_graph = import_program_bundle(&mut imported_registry, &bundle_a).unwrap();
    assert_eq!(imported_graph.program_plan(), plan);
    assert_eq!(imported_graph.program_identity(), identity);

    let imported_binding = GraphParameterBinding::build(&imported_graph, &imported_registry).unwrap();
    let imported_owner_keys = imported_binding
        .owners()
        .iter()
        .map(|owner| (owner.layer_type(), owner.layer_id()))
        .collect::<Vec<_>>();
    assert_eq!(imported_owner_keys, owner_keys);
}

#[test]
fn malformed_graph_plan_error_category_remains_stable() {
    let mut registry = LayerRegistry::new();
    let linear = AgentLayerSpec::linear(301, 2, 2, true).unwrap();
    registry.init_agent_layer(&linear).unwrap();

    let mut builder = AgentGraphBuilder::new(2).unwrap();
    builder.add_unary(&linear, 0, 1).unwrap();
    builder.set_output(1).unwrap();
    let graph = builder.compile(&registry).unwrap();
    let plan = graph.program_plan();

    let mut trailing = plan.clone();
    trailing.push(0xFF);
    let trailing_err = match registry.compile_graph(&trailing) {
        Ok(_) => panic!("expected trailing graph plan bytes to fail"),
        Err(err) => err,
    };
    assert!(
        trailing_err.contains("malformed plan length"),
        "unexpected trailing-byte error: {trailing_err}"
    );

    let mut truncated = plan.clone();
    truncated.pop();
    let truncated_err = match registry.compile_graph(&truncated) {
        Ok(_) => panic!("expected truncated graph plan to fail"),
        Err(err) => err,
    };
    assert!(
        truncated_err.contains("malformed plan length"),
        "unexpected truncated-plan error: {truncated_err}"
    );
}
