#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "native-graph-runtime-scaling-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

RUST_CONSUMER = r'''
use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;
use serde_json::json;
use std::hint::black_box;
use std::time::{Duration, Instant};

const POLICY_IN: u32 = 6;
const POLICY_OUT: u32 = 2;
const REPS: usize = 600;
const WARMUP: usize = 60;
const FEATURES: [f32; 6] = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0];

#[derive(Clone, Copy)]
enum Activation {
    None,
    Relu,
    Tanh,
    Gelu,
}

impl Activation {
    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Relu => "relu",
            Self::Tanh => "tanh",
            Self::Gelu => "gelu",
        }
    }

    fn spec(self, layer_id: u32) -> Option<AgentLayerSpec> {
        match self {
            Self::None => None,
            Self::Relu => Some(AgentLayerSpec::relu(layer_id)),
            Self::Tanh => Some(AgentLayerSpec::tanh(layer_id)),
            Self::Gelu => Some(AgentLayerSpec::gelu(layer_id)),
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    axis: &'static str,
    hidden_width: u32,
    linear_layers: usize,
    activation: Activation,
}

fn summary(samples: &[Duration]) -> serde_json::Value {
    assert!(!samples.is_empty(), "empty timing sample set");
    let mut values = samples
        .iter()
        .map(|value| value.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
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

fn build_case(case_index: usize, case: Case) -> serde_json::Value {
    assert!(case.linear_layers >= 1);
    let activation_steps = if matches!(case.activation, Activation::None) {
        0
    } else {
        case.linear_layers.saturating_sub(1)
    };
    let graph_steps = case.linear_layers + activation_steps;
    assert!(graph_steps + 1 <= u8::MAX as usize);

    let mut registry = LayerRegistry::new();
    let mut builder = AgentGraphBuilder::new((graph_steps + 1) as u32).expect("graph builder");
    let mut current_slot = 0u8;
    let mut next_layer_id = 217_000u32 + (case_index as u32) * 1_000;

    for linear_index in 0..case.linear_layers {
        let in_dim = if linear_index == 0 {
            POLICY_IN
        } else {
            case.hidden_width
        };
        let is_final = linear_index + 1 == case.linear_layers;
        let out_dim = if is_final {
            POLICY_OUT
        } else {
            case.hidden_width
        };

        let linear = AgentLayerSpec::linear(next_layer_id, in_dim, out_dim, true)
            .expect("linear spec");
        next_layer_id += 1;
        registry
            .init_agent_layer(&linear)
            .expect("initialize linear");
        let linear_output_slot = current_slot + 1;
        builder
            .add_unary(&linear, current_slot, linear_output_slot)
            .expect("add linear");
        current_slot = linear_output_slot;

        if !is_final {
            if let Some(activation) = case.activation.spec(next_layer_id) {
                next_layer_id += 1;
                registry
                    .init_agent_layer(&activation)
                    .expect("initialize activation");
                let activation_output_slot = current_slot + 1;
                builder
                    .add_unary(&activation, current_slot, activation_output_slot)
                    .expect("add activation");
                current_slot = activation_output_slot;
            }
        }
    }

    builder
        .set_output(current_slot)
        .expect("set graph output");
    let graph = builder.compile(&registry).expect("compile graph");
    let binding = GraphParameterBinding::build(&graph, &registry).expect("build binding");
    let candidate = deterministic_candidate(binding.total_len());
    binding
        .apply_flat(&graph, &mut registry, &candidate)
        .expect("apply deterministic candidate");
    assert_eq!(
        binding
            .read_flat(&graph, &registry)
            .expect("read deterministic candidate"),
        candidate,
        "candidate readback must be exact"
    );

    let input = WasmTensor::new(&FEATURES, &[1, POLICY_IN as usize, 1, 1]);
    let reference = graph.run(&registry, &input).expect("reference graph run");
    let reference_values = reference.to_array();
    assert_eq!(reference_values.len(), POLICY_OUT as usize);
    assert!(reference_values.iter().all(|value| value.is_finite()));

    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();

    for _ in 0..WARMUP {
        let output = graph
            .run(black_box(&registry), black_box(&input))
            .expect("warmup graph run");
        black_box(&output);
    }

    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let output = graph
            .run(black_box(&registry), black_box(&input))
            .expect("timed graph run");
        let elapsed = start.elapsed();
        black_box(&output);
        samples.push(elapsed);
        drop(output);
    }

    let final_values = graph
        .run(&registry, &input)
        .expect("final graph run")
        .to_array();
    assert_eq!(final_values, reference_values, "output changed after timing");
    assert_eq!(graph.program_identity(), program_identity);
    assert_eq!(binding.identity_json(), binding_identity);
    assert_eq!(
        binding
            .read_flat(&graph, &registry)
            .expect("final parameter readback"),
        candidate,
        "trainable state changed during graph execution"
    );

    let timing = summary(&samples);
    let median_ms = timing["median_ms"].as_f64().expect("median");
    let parameter_count = binding.total_len();
    let median_us_per_step = median_ms * 1000.0 / graph_steps as f64;
    let median_ns_per_parameter = if parameter_count == 0 {
        0.0
    } else {
        median_ms * 1_000_000.0 / parameter_count as f64
    };

    json!({
        "name": case.name,
        "axis": case.axis,
        "config": {
            "input_dim": POLICY_IN,
            "output_dim": POLICY_OUT,
            "hidden_width": case.hidden_width,
            "linear_layers": case.linear_layers,
            "activation": case.activation.label(),
            "graph_steps": graph_steps,
            "parameter_count": parameter_count,
        },
        "program_identity": program_identity,
        "binding_identity": binding_identity,
        "reference_output": reference_values,
        "timing": timing,
        "normalized": {
            "median_us_per_graph_step": median_us_per_step,
            "median_ns_per_trainable_parameter": median_ns_per_parameter,
        },
        "proofs": {
            "finite_output": true,
            "deterministic_output": true,
            "candidate_exact_readback": true,
            "trainable_state_stable": true,
            "program_identity_stable": true,
            "binding_identity_stable": true,
        }
    })
}

fn main() {
    let cases = [
        // Depth axis: fixed hidden width, increasing sequential Linear depth.
        Case { name: "depth_l1_w32_relu", axis: "depth", hidden_width: 32, linear_layers: 1, activation: Activation::Relu },
        Case { name: "depth_l2_w32_relu", axis: "depth", hidden_width: 32, linear_layers: 2, activation: Activation::Relu },
        Case { name: "depth_l4_w32_relu", axis: "depth", hidden_width: 32, linear_layers: 4, activation: Activation::Relu },
        Case { name: "depth_l8_w32_relu", axis: "depth", hidden_width: 32, linear_layers: 8, activation: Activation::Relu },
        // Width axis: fixed depth/composition, increasing hidden width.
        Case { name: "width_l3_w8_relu", axis: "width", hidden_width: 8, linear_layers: 3, activation: Activation::Relu },
        Case { name: "width_l3_w16_relu", axis: "width", hidden_width: 16, linear_layers: 3, activation: Activation::Relu },
        Case { name: "width_l3_w32_relu", axis: "width", hidden_width: 32, linear_layers: 3, activation: Activation::Relu },
        Case { name: "width_l3_w64_relu", axis: "width", hidden_width: 64, linear_layers: 3, activation: Activation::Relu },
        Case { name: "width_l3_w128_relu", axis: "width", hidden_width: 128, linear_layers: 3, activation: Activation::Relu },
        // Composition axis: same dimensions/depth for activation comparisons.
        Case { name: "composition_l3_w32_linear_only", axis: "composition", hidden_width: 32, linear_layers: 3, activation: Activation::None },
        Case { name: "composition_l3_w32_relu", axis: "composition", hidden_width: 32, linear_layers: 3, activation: Activation::Relu },
        Case { name: "composition_l3_w32_tanh", axis: "composition", hidden_width: 32, linear_layers: 3, activation: Activation::Tanh },
        Case { name: "composition_l3_w32_gelu", axis: "composition", hidden_width: 32, linear_layers: 3, activation: Activation::Gelu },
    ];

    let results = cases
        .iter()
        .copied()
        .enumerate()
        .map(|(index, case)| build_case(index, case))
        .collect::<Vec<_>>();

    println!("{}", json!({
        "verdict": "PASS",
        "schema": "burn-research.native-graph-runtime-scaling.v1",
        "consumer": "external-packaged-rust-release",
        "workload_lineage": "python-agent-control-policy-6-to-2",
        "repetitions_per_case": REPS,
        "warmup_per_case": WARMUP,
        "input": FEATURES,
        "cases": results,
        "proofs": {
            "public_package_surface_only": true,
            "all_outputs_finite_and_deterministic": true,
            "all_candidate_readbacks_exact": true,
            "all_program_identities_stable": true,
            "all_binding_identities_stable": true,
        }
    }));
}
'''


def run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(args)}"
        )
    return completed


def last_json(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("runtime scaling consumer produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"consumer did not end with JSON: {lines[-1]!r}") from exc
    if payload.get("verdict") != "PASS":
        raise SystemExit(f"unexpected research verdict: {payload}")
    return payload


def case(payload: dict, name: str) -> dict:
    for entry in payload["cases"]:
        if entry["name"] == name:
            return entry
    raise SystemExit(f"missing case {name}")


def median_ms(entry: dict) -> float:
    return float(entry["timing"]["median_ms"])


def main() -> None:
    run(["cargo", "package", "--allow-dirty", "--no-verify"], cwd=REPO)
    crate_path = REPO / "target" / "package" / f"{PACKAGE_NAME}-{PACKAGE_VERSION}.crate"
    if not crate_path.is_file():
        raise SystemExit(f"missing packaged crate: {crate_path}")

    with tempfile.TemporaryDirectory(prefix="burn-research-native-runtime-scaling-") as tmp:
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
            f'''[package]\nname = "burn-research-native-runtime-scaling"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\nserde_json = "1.0"\n''',
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(RUST_CONSUMER, encoding="utf-8")

        run(["cargo", "generate-lockfile", "--manifest-path", str(consumer / "Cargo.toml")], cwd=consumer)
        completed = run(
            [
                "cargo",
                "run",
                "--release",
                "--quiet",
                "--locked",
                "--manifest-path",
                str(consumer / "Cargo.toml"),
            ],
            cwd=consumer,
        )
        payload = last_json(completed.stdout)

    depth_ratio = median_ms(case(payload, "depth_l8_w32_relu")) / median_ms(
        case(payload, "depth_l1_w32_relu")
    )
    width_ratio = median_ms(case(payload, "width_l3_w128_relu")) / median_ms(
        case(payload, "width_l3_w8_relu")
    )
    activation_entries = [
        case(payload, "composition_l3_w32_relu"),
        case(payload, "composition_l3_w32_tanh"),
        case(payload, "composition_l3_w32_gelu"),
    ]
    activation_medians = [median_ms(entry) for entry in activation_entries]
    activation_spread_ratio = max(activation_medians) / min(activation_medians)
    linear_only_ratio = median_ms(case(payload, "composition_l3_w32_relu")) / median_ms(
        case(payload, "composition_l3_w32_linear_only")
    )

    payload["derived"] = {
        "depth_l8_vs_l1_ratio": depth_ratio,
        "width_w128_vs_w8_ratio": width_ratio,
        "activation_max_vs_min_ratio": activation_spread_ratio,
        "relu_vs_linear_only_ratio": linear_only_ratio,
    }
    payload["interpretation_boundary"] = {
        "timing_is_descriptive_not_sla": True,
        "no_runtime_optimization_selected_by_script": True,
        "decision_requires_review_of_depth_width_composition_patterns": True,
    }

    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
