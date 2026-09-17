#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "native-graph-run-internal-decomposition-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

RUST_CONSUMER = r'''
use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::protocol::{LAYER_ACTIVATION, LAYER_LINEAR};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;
use serde_json::json;
use std::hint::black_box;
use std::time::{Duration, Instant};

const POLICY_IN: u32 = 6;
const POLICY_OUT: u32 = 2;
const REPS: usize = 700;
const WARMUP: usize = 70;
const FEATURES: [f32; 6] = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0];

#[derive(Clone, Copy)]
struct StepRef {
    layer_type: u8,
    layer_id: u32,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    hidden_width: u32,
    linear_layers: usize,
}

fn summary(samples: &[Duration]) -> serde_json::Value {
    assert!(!samples.is_empty());
    let mut values = samples.iter().map(|v| v.as_secs_f64() * 1000.0).collect::<Vec<_>>();
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite timing"));
    let mid = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    };
    let p90_index = (((values.len() as f64) * 0.90).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    json!({
        "median_ms": median,
        "min_ms": values[0],
        "p90_ms": values[p90_index],
        "max_ms": values[values.len() - 1],
    })
}

fn deterministic_candidate(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = ((index * 37 + 11) % 101) as i32 - 50;
            bucket as f32 * 0.0005
        })
        .collect()
}

fn run_direct(
    registry: &LayerRegistry,
    input: &WasmTensor,
    steps: &[StepRef],
) -> Result<WasmTensor, String> {
    let mut current = input.clone();
    for step in steps {
        current = registry.forward_layer(step.layer_id, step.layer_type, &current)?;
    }
    Ok(current)
}

fn run_slot_backed(
    registry: &LayerRegistry,
    input: &WasmTensor,
    steps: &[StepRef],
) -> Result<WasmTensor, String> {
    let mut slots: Vec<Option<WasmTensor>> = vec![None; steps.len() + 1];
    slots[0] = Some(input.clone());
    for (index, step) in steps.iter().enumerate() {
        let out = {
            let inp = slots[index]
                .as_ref()
                .ok_or_else(|| format!("manual slot chain: empty slot {index}"))?;
            registry.forward_layer(step.layer_id, step.layer_type, inp)?
        };
        slots[index + 1] = Some(out);
    }
    slots[steps.len()]
        .take()
        .ok_or_else(|| "manual slot chain: missing output".to_string())
}

fn build_case(case_index: usize, case: Case) -> serde_json::Value {
    assert!(case.linear_layers >= 1);
    let graph_steps = case.linear_layers + case.linear_layers.saturating_sub(1);
    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new((graph_steps + 1) as u32).expect("graph builder");
    let mut current_slot = 0u8;
    let mut next_layer_id = 230_000u32 + case_index as u32 * 10_000;
    let mut steps = Vec::with_capacity(graph_steps);

    for linear_index in 0..case.linear_layers {
        let in_dim = if linear_index == 0 { POLICY_IN } else { case.hidden_width };
        let is_final = linear_index + 1 == case.linear_layers;
        let out_dim = if is_final { POLICY_OUT } else { case.hidden_width };

        let linear_id = next_layer_id;
        next_layer_id += 1;
        let linear = AgentLayerSpec::linear(linear_id, in_dim, out_dim, true).expect("linear spec");
        registry.init_agent_layer(&linear).expect("init linear");
        let out_slot = current_slot + 1;
        builder.add_unary(&linear, current_slot, out_slot).expect("add linear");
        steps.push(StepRef { layer_type: LAYER_LINEAR, layer_id: linear_id });
        current_slot = out_slot;

        if !is_final {
            let activation_id = next_layer_id;
            next_layer_id += 1;
            let activation = AgentLayerSpec::relu(activation_id);
            registry.init_agent_layer(&activation).expect("init relu");
            let activation_slot = current_slot + 1;
            builder.add_unary(&activation, current_slot, activation_slot).expect("add relu");
            steps.push(StepRef { layer_type: LAYER_ACTIVATION, layer_id: activation_id });
            current_slot = activation_slot;
        }
    }

    builder.set_output(current_slot).expect("set output");
    let graph = builder.compile(&registry).expect("compile graph");
    assert_eq!(graph.step_count() as usize, steps.len());

    let binding = GraphParameterBinding::build(&graph, &registry).expect("binding");
    let candidate = deterministic_candidate(binding.total_len());
    binding.apply_flat(&graph, &mut registry, &candidate).expect("apply candidate");
    assert_eq!(binding.read_flat(&graph, &registry).expect("read candidate"), candidate);

    let input = WasmTensor::new(&FEATURES, &[1, POLICY_IN as usize, 1, 1]);
    let graph_reference = graph.run(&registry, &input).expect("graph reference").to_array();
    let direct_reference = run_direct(&registry, &input, &steps).expect("direct reference").to_array();
    let slot_reference = run_slot_backed(&registry, &input, &steps).expect("slot reference").to_array();
    assert_eq!(graph_reference, direct_reference, "direct output mismatch");
    assert_eq!(graph_reference, slot_reference, "slot-backed output mismatch");
    assert!(graph_reference.iter().all(|value| value.is_finite()));

    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();

    for _ in 0..WARMUP {
        graph.validate_registry_binding(&registry).expect("warmup validation");
        let _ = run_direct(&registry, &input, &steps).expect("warmup direct");
        let _ = run_slot_backed(&registry, &input, &steps).expect("warmup slot");
        let _ = graph.run(&registry, &input).expect("warmup graph");
    }

    let mut validation_samples = Vec::with_capacity(REPS);
    let mut slot_setup_samples = Vec::with_capacity(REPS);
    let mut direct_samples = Vec::with_capacity(REPS);
    let mut slot_chain_samples = Vec::with_capacity(REPS);
    let mut graph_samples = Vec::with_capacity(REPS);

    for _ in 0..REPS {
        let start = Instant::now();
        graph.validate_registry_binding(black_box(&registry)).expect("validation");
        validation_samples.push(start.elapsed());

        let start = Instant::now();
        let mut slots: Vec<Option<WasmTensor>> = vec![None; steps.len() + 1];
        slots[0] = Some(black_box(&input).clone());
        let elapsed = start.elapsed();
        black_box(&slots);
        slot_setup_samples.push(elapsed);
        drop(slots);

        let start = Instant::now();
        let output = run_direct(black_box(&registry), black_box(&input), black_box(&steps))
            .expect("timed direct");
        direct_samples.push(start.elapsed());
        black_box(&output);
        drop(output);

        let start = Instant::now();
        let output = run_slot_backed(black_box(&registry), black_box(&input), black_box(&steps))
            .expect("timed slot chain");
        slot_chain_samples.push(start.elapsed());
        black_box(&output);
        drop(output);

        let start = Instant::now();
        let output = graph.run(black_box(&registry), black_box(&input)).expect("timed graph");
        graph_samples.push(start.elapsed());
        black_box(&output);
        drop(output);
    }

    let final_graph = graph.run(&registry, &input).expect("final graph").to_array();
    let final_direct = run_direct(&registry, &input, &steps).expect("final direct").to_array();
    let final_slot = run_slot_backed(&registry, &input, &steps).expect("final slot").to_array();
    assert_eq!(graph_reference, final_graph);
    assert_eq!(graph_reference, final_direct);
    assert_eq!(graph_reference, final_slot);
    assert_eq!(graph.program_identity(), program_identity);
    assert_eq!(binding.identity_json(), binding_identity);
    assert_eq!(binding.read_flat(&graph, &registry).expect("final state"), candidate);

    let validation = summary(&validation_samples);
    let slot_setup = summary(&slot_setup_samples);
    let direct = summary(&direct_samples);
    let slot_chain = summary(&slot_chain_samples);
    let graph_run = summary(&graph_samples);

    let graph_median = graph_run["median_ms"].as_f64().expect("graph median");
    let validation_median = validation["median_ms"].as_f64().expect("validation median");
    let slot_setup_median = slot_setup["median_ms"].as_f64().expect("slot setup median");
    let direct_median = direct["median_ms"].as_f64().expect("direct median");
    let slot_chain_median = slot_chain["median_ms"].as_f64().expect("slot median");

    json!({
        "name": case.name,
        "config": {
            "input_dim": POLICY_IN,
            "output_dim": POLICY_OUT,
            "hidden_width": case.hidden_width,
            "linear_layers": case.linear_layers,
            "graph_steps": steps.len(),
            "parameter_count": binding.total_len(),
        },
        "program_identity": program_identity,
        "binding_identity": binding_identity,
        "reference_output": graph_reference,
        "timing": {
            "validate_registry_binding": validation,
            "slot_vector_plus_input_clone": slot_setup,
            "manual_direct_forward_chain": direct,
            "manual_slot_backed_forward_chain": slot_chain,
            "compiled_graph_run": graph_run,
        },
        "derived": {
            "validation_share_of_graph_run": validation_median / graph_median,
            "slot_setup_share_of_graph_run": slot_setup_median / graph_median,
            "direct_chain_share_of_graph_run": direct_median / graph_median,
            "slot_chain_share_of_graph_run": slot_chain_median / graph_median,
            "slot_chain_vs_direct_ratio": slot_chain_median / direct_median,
            "graph_run_vs_slot_chain_ratio": graph_median / slot_chain_median,
        },
        "proofs": {
            "outputs_exact_across_paths": true,
            "candidate_exact_readback": true,
            "trainable_state_stable": true,
            "program_identity_stable": true,
            "binding_identity_stable": true,
            "finite_output": true,
        }
    })
}

fn main() {
    let cases = [
        Case { name: "deep_l8_w32_relu", hidden_width: 32, linear_layers: 8 },
        Case { name: "wide_l3_w128_relu", hidden_width: 128, linear_layers: 3 },
    ];
    let results = cases.iter().copied().enumerate().map(|(i, case)| build_case(i, case)).collect::<Vec<_>>();
    println!("{}", json!({
        "verdict": "PASS",
        "schema": "burn-research.native-graph-run-internal-decomposition.v1",
        "consumer": "external-packaged-rust-release",
        "workload_lineage": "native-graph-runtime-scaling",
        "repetitions_per_case": REPS,
        "warmup_per_case": WARMUP,
        "input": FEATURES,
        "cases": results,
        "interpretation_boundary": {
            "timing_is_descriptive_not_sla": true,
            "no_runtime_optimization_selected_by_script": true,
            "public_surface_only": true
        }
    }));
}
'''


def run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, cwd=cwd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(args)}")
    return completed


def last_json(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("decomposition consumer produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"consumer did not end with JSON: {lines[-1]!r}") from exc
    if payload.get("verdict") != "PASS":
        raise SystemExit(f"unexpected verdict: {payload}")
    return payload


def main() -> None:
    run(["cargo", "package", "--allow-dirty", "--no-verify"], cwd=REPO)
    crate_path = REPO / "target" / "package" / f"{PACKAGE_NAME}-{PACKAGE_VERSION}.crate"
    if not crate_path.is_file():
        raise SystemExit(f"missing packaged crate: {crate_path}")

    with tempfile.TemporaryDirectory(prefix="burn-research-native-run-internal-") as tmp:
        root = Path(tmp)
        extract_root = root / "package"
        extract_root.mkdir()
        with tarfile.open(crate_path, "r:gz") as archive:
            archive.extractall(extract_root)
        package_dir = extract_root / f"{PACKAGE_NAME}-{PACKAGE_VERSION}"
        if not (package_dir / "Cargo.toml").is_file():
            raise SystemExit("packaged crate missing Cargo.toml")

        consumer = root / "consumer"
        (consumer / "src").mkdir(parents=True)
        package_path = package_dir.as_posix().replace('"', '\\"')
        (consumer / "Cargo.toml").write_text(
            f'''[package]\nname = "burn-research-native-run-internal-decomposition"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\nserde_json = "1.0"\n''',
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(RUST_CONSUMER, encoding="utf-8")

        run(["cargo", "generate-lockfile", "--manifest-path", str(consumer / "Cargo.toml")], cwd=consumer)
        completed = run(
            ["cargo", "run", "--release", "--quiet", "--locked", "--manifest-path", str(consumer / "Cargo.toml")],
            cwd=consumer,
        )
        payload = last_json(completed.stdout)

    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
