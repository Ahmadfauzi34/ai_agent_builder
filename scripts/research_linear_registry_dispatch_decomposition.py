#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "linear-registry-dispatch-decomposition-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

RUST_CONSUMER = r'''
use burn_research::agent::AgentLayerSpec;
use burn_research::layers::linear::WasmLinear;
use burn_research::protocol::LAYER_LINEAR;
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;
use serde_json::json;
use std::hint::black_box;
use std::time::{Duration, Instant};

const REPS: usize = 1200;
const WARMUP: usize = 120;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    in_dim: usize,
    out_dim: usize,
}

fn summary(samples: &[Duration]) -> serde_json::Value {
    assert!(!samples.is_empty());
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

fn deterministic_weights(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = ((index * 41 + 17) % 127) as i32 - 63;
            bucket as f32 * 0.0004
        })
        .collect()
}

fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = ((index * 29 + 5) % 73) as i32 - 36;
            bucket as f32 * 0.01
        })
        .collect()
}

fn run_case(case_index: usize, case: Case) -> serde_json::Value {
    let layer_id = 240_000u32 + case_index as u32;
    let weight_len = case.in_dim * case.out_dim + case.out_dim;
    let weights = deterministic_weights(weight_len);
    let input_values = deterministic_input(case.in_dim);
    let input = WasmTensor::new(&input_values, &[1, case.in_dim, 1, 1]);

    let mut direct = WasmLinear::new(case.in_dim, case.out_dim, true);
    direct
        .set_weights_flat(&weights)
        .expect("set direct Linear weights");
    assert_eq!(
        direct.get_weights_flat().expect("read direct Linear weights"),
        weights,
        "direct weights must read back exactly"
    );

    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::linear(
        layer_id,
        case.in_dim as u32,
        case.out_dim as u32,
        true,
    )
    .expect("registry Linear spec");
    registry
        .init_agent_layer(&spec)
        .expect("initialize registry Linear");
    registry
        .set_weights_flat(layer_id, LAYER_LINEAR, &weights)
        .expect("set registry Linear weights");
    assert_eq!(
        registry
            .get_weights_flat(layer_id, LAYER_LINEAR)
            .expect("read registry Linear weights"),
        weights,
        "registry weights must read back exactly"
    );
    assert!(registry.layer_exists(LAYER_LINEAR, layer_id));

    let direct_reference = direct.forward(&input).to_array();
    let registry_reference = registry
        .forward_layer(layer_id, LAYER_LINEAR, &input)
        .expect("registry reference forward")
        .to_array();
    assert_eq!(direct_reference, registry_reference, "reference output mismatch");
    assert_eq!(direct_reference.len(), case.out_dim);
    assert!(direct_reference.iter().all(|value| value.is_finite()));

    for _ in 0..WARMUP {
        let direct_output = direct.forward(black_box(&input));
        black_box(&direct_output);
        let registry_output = registry
            .forward_layer(layer_id, LAYER_LINEAR, black_box(&input))
            .expect("registry warmup forward");
        black_box(&registry_output);
    }

    let mut direct_samples = Vec::with_capacity(REPS);
    let mut registry_samples = Vec::with_capacity(REPS);
    let mut lookup_samples = Vec::with_capacity(REPS);

    // Alternate measurement order so runner drift does not always favor one path.
    for index in 0..REPS {
        if index % 2 == 0 {
            let start = Instant::now();
            let output = direct.forward(black_box(&input));
            direct_samples.push(start.elapsed());
            black_box(&output);
            drop(output);

            let start = Instant::now();
            let output = registry
                .forward_layer(layer_id, LAYER_LINEAR, black_box(&input))
                .expect("timed registry forward");
            registry_samples.push(start.elapsed());
            black_box(&output);
            drop(output);
        } else {
            let start = Instant::now();
            let output = registry
                .forward_layer(layer_id, LAYER_LINEAR, black_box(&input))
                .expect("timed registry forward");
            registry_samples.push(start.elapsed());
            black_box(&output);
            drop(output);

            let start = Instant::now();
            let output = direct.forward(black_box(&input));
            direct_samples.push(start.elapsed());
            black_box(&output);
            drop(output);
        }

        let start = Instant::now();
        let exists = registry.layer_exists(LAYER_LINEAR, black_box(layer_id));
        lookup_samples.push(start.elapsed());
        assert!(exists);
    }

    let final_direct = direct.forward(&input).to_array();
    let final_registry = registry
        .forward_layer(layer_id, LAYER_LINEAR, &input)
        .expect("final registry forward")
        .to_array();
    assert_eq!(final_direct, direct_reference);
    assert_eq!(final_registry, direct_reference);
    assert_eq!(direct.get_weights_flat().expect("final direct weights"), weights);
    assert_eq!(
        registry
            .get_weights_flat(layer_id, LAYER_LINEAR)
            .expect("final registry weights"),
        weights
    );

    let direct_timing = summary(&direct_samples);
    let registry_timing = summary(&registry_samples);
    let lookup_timing = summary(&lookup_samples);
    let direct_median = direct_timing["median_ms"].as_f64().expect("direct median");
    let registry_median = registry_timing["median_ms"].as_f64().expect("registry median");
    let lookup_median = lookup_timing["median_ms"].as_f64().expect("lookup median");
    let excess = (registry_median - direct_median).max(0.0);

    json!({
        "name": case.name,
        "config": {
            "in_dim": case.in_dim,
            "out_dim": case.out_dim,
            "bias": true,
            "parameter_count": weight_len,
            "layer_id": layer_id,
        },
        "reference_output": direct_reference,
        "timing": {
            "standalone_wasm_linear_forward": direct_timing,
            "registry_forward_layer_linear": registry_timing,
            "registry_layer_exists_control": lookup_timing,
        },
        "derived": {
            "registry_vs_direct_ratio": registry_median / direct_median,
            "registry_excess_ms": excess,
            "registry_excess_share": excess / registry_median,
            "lookup_share_of_registry_forward": lookup_median / registry_median,
        },
        "proofs": {
            "same_flat_weights_exact": true,
            "outputs_exact": true,
            "outputs_finite": true,
            "direct_output_deterministic": true,
            "registry_output_deterministic": true,
            "weights_stable_after_timing": true,
            "registry_layer_exists": true,
        }
    })
}

fn main() {
    let cases = [
        Case { name: "linear_32_to_32", in_dim: 32, out_dim: 32 },
        Case { name: "linear_128_to_128", in_dim: 128, out_dim: 128 },
    ];
    let results = cases
        .iter()
        .copied()
        .enumerate()
        .map(|(index, case)| run_case(index, case))
        .collect::<Vec<_>>();

    println!("{}", json!({
        "verdict": "PASS",
        "schema": "burn-research.linear-registry-dispatch-decomposition.v1",
        "consumer": "external-packaged-rust-release",
        "repetitions_per_case": REPS,
        "warmup_per_case": WARMUP,
        "cases": results,
        "interpretation_boundary": {
            "timing_is_descriptive_not_sla": true,
            "no_runtime_optimization_selected_by_script": true,
            "public_package_surface_only": true,
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
        raise SystemExit("Linear dispatch decomposition consumer produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"consumer did not end with JSON: {lines[-1]!r}") from exc
    if payload.get("verdict") != "PASS":
        raise SystemExit(f"unexpected research verdict: {payload}")
    return payload


def main() -> None:
    run(["cargo", "package", "--allow-dirty", "--no-verify"], cwd=REPO)
    crate_path = REPO / "target" / "package" / f"{PACKAGE_NAME}-{PACKAGE_VERSION}.crate"
    if not crate_path.is_file():
        raise SystemExit(f"missing packaged crate: {crate_path}")

    with tempfile.TemporaryDirectory(prefix="burn-research-linear-dispatch-") as tmp:
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
            f'''[package]\nname = "burn-research-linear-registry-dispatch-research"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\nserde_json = "1.0"\n''',
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
