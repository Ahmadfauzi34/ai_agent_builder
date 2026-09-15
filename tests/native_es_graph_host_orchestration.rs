use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::es::optimizer::EsOptimizer;
use burn_research::graph::CompiledGraph;
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::program_bundle::{export_program_bundle, import_program_bundle};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;

#[derive(Clone, Copy)]
struct Row {
    input: [f32; 2],
    target: f32,
}

fn dataset() -> Vec<Row> {
    let points = [
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, -0.5],
        [-0.5, 0.5],
        [2.0, -1.0],
        [-2.0, 1.0],
        [1.5, 0.25],
        [-1.5, -0.25],
        [0.25, 1.5],
    ];

    points
        .into_iter()
        .map(|[x0, x1]| Row {
            input: [x0, x1],
            target: 1.5 * x0 - 0.75 * x1 + 0.25,
        })
        .collect()
}

fn build_graph(layer_id: u32) -> (LayerRegistry, CompiledGraph, GraphParameterBinding) {
    let mut registry = LayerRegistry::new();
    let linear = AgentLayerSpec::linear(layer_id, 2, 1, true).expect("linear spec");
    registry
        .init_agent_layer(&linear)
        .expect("initialize linear layer");

    let mut builder = AgentGraphBuilder::new(2).expect("graph builder");
    builder
        .add_unary(&linear, 0, 1)
        .expect("add linear graph step");
    builder.set_output(1).expect("set graph output");
    let graph = builder.compile(&registry).expect("compile graph");
    let binding = GraphParameterBinding::build(&graph, &registry).expect("parameter binding");

    assert_eq!(binding.total_len(), 3, "2->1 Linear+bias must expose 3 coordinates");
    assert_eq!(binding.owners().len(), 1, "regression graph should have one trainable owner");

    (registry, graph, binding)
}

fn run_scalar(graph: &CompiledGraph, registry: &LayerRegistry, input: [f32; 2]) -> f32 {
    let input = WasmTensor::new(&input, &[1, 2, 1, 1]);
    let output = graph.run(registry, &input).expect("graph run");
    let values = output.to_array();
    assert_eq!(values.len(), 1, "regression graph must return one scalar");
    assert!(values[0].is_finite(), "graph output must be finite");
    values[0]
}

fn mse(graph: &CompiledGraph, registry: &LayerRegistry, rows: &[Row]) -> f64 {
    let squared = rows.iter().fold(0.0f64, |acc, row| {
        let prediction = run_scalar(graph, registry, row.input) as f64;
        let error = prediction - row.target as f64;
        acc + error * error
    });
    squared / rows.len() as f64
}

#[test]
fn native_rust_host_composes_es_graph_objective_and_checkpoint_without_controller() {
    let rows = dataset();
    let (mut registry, graph, binding) = build_graph(18_001);

    // Canonical graph binding is the only source of optimizer dimension/order.
    let zero = vec![0.0f32; binding.total_len()];
    binding
        .apply_flat(&graph, &mut registry, &zero)
        .expect("apply deterministic baseline");

    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();
    let layout = binding.layout_json();
    let initial_loss = mse(&graph, &registry, &rows);

    let mut optimizer = EsOptimizer::strict(
        binding.total_len() as u32,
        0,
        1777,
        Some(32),
        Some(0.2),
        Some(0.05),
    )
    .expect("strict ES optimizer");

    for _ in 0..60 {
        let candidates = optimizer.ask();
        let batch_size = optimizer.batch_size() as usize;
        assert_eq!(batch_size, 32, "strict ES population must stay fixed");
        assert_eq!(
            candidates.len(),
            batch_size * binding.total_len(),
            "ask output must use canonical graph dimension"
        );

        let mut fitness = Vec::with_capacity(batch_size);
        for candidate in candidates.chunks_exact(binding.total_len()) {
            assert!(
                candidate.iter().all(|value| value.is_finite()),
                "ES candidate must remain finite"
            );
            binding
                .apply_flat(&graph, &mut registry, candidate)
                .expect("canonical candidate apply");
            let loss = mse(&graph, &registry, &rows);
            assert!(loss.is_finite(), "host objective must remain finite");
            fitness.push(-(loss as f32));
        }

        optimizer.tell(&fitness).expect("tell finite fitness batch");
    }

    let best = optimizer.best();
    assert_eq!(best.len(), binding.total_len(), "best candidate dimension mismatch");
    assert!(best.iter().all(|value| value.is_finite()), "best candidate must be finite");

    binding
        .apply_flat(&graph, &mut registry, &best)
        .expect("apply best candidate");
    let final_loss = mse(&graph, &registry, &rows);

    assert!(
        final_loss < initial_loss * 0.1,
        "native host training must materially improve regression loss: {initial_loss} -> {final_loss}"
    );
    assert!(
        final_loss < 0.1,
        "native host final regression loss is unexpectedly high: {final_loss}"
    );

    // Mutable learned state must not change structural or binding identity/layout.
    assert_eq!(graph.program_identity(), program_identity);
    let rebuilt = GraphParameterBinding::build(&graph, &registry).expect("rebuild binding");
    assert_eq!(rebuilt.identity_json(), binding_identity);
    assert_eq!(rebuilt.layout_json(), layout);
    assert_eq!(
        binding
            .read_flat(&graph, &registry)
            .expect("read learned parameters"),
        best,
        "best candidate must read back exactly in canonical order"
    );

    // Existing stateful ProgramBundle is the checkpoint/replay artifact.
    let bundle = export_program_bundle(&graph, &registry, true).expect("export learned checkpoint");
    assert!(!bundle.is_empty(), "stateful checkpoint bundle must not be empty");

    let mut imported_registry = LayerRegistry::new();
    let imported_graph = import_program_bundle(&mut imported_registry, &bundle)
        .expect("import learned checkpoint into fresh registry");
    let imported_binding = GraphParameterBinding::build(&imported_graph, &imported_registry)
        .expect("rebuild imported binding");

    assert_eq!(imported_graph.program_identity(), program_identity);
    assert_eq!(imported_binding.identity_json(), binding_identity);
    assert_eq!(imported_binding.layout_json(), layout);
    assert_eq!(
        imported_binding
            .read_flat(&imported_graph, &imported_registry)
            .expect("read imported learned parameters"),
        best,
        "checkpoint replay must restore the exact canonical best vector"
    );

    let imported_loss = mse(&imported_graph, &imported_registry, &rows);
    assert!(
        (imported_loss - final_loss).abs() <= 1e-8,
        "native checkpoint replay loss mismatch: {final_loss} vs {imported_loss}"
    );

    for row in &rows {
        let trained = run_scalar(&graph, &registry, row.input);
        let replayed = run_scalar(&imported_graph, &imported_registry, row.input);
        assert!(
            (trained - replayed).abs() <= 1e-7,
            "native checkpoint replay output mismatch: {trained} vs {replayed}"
        );
    }
}
