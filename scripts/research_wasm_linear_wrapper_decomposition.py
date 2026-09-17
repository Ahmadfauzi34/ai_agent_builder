#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "wasm-linear-wrapper-decomposition-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

RUST_CONSUMER = r'''
use burn_research::layers::linear::WasmLinear;
use burn_research::WasmTensor;
use serde_json::json;
use std::hint::black_box;
use std::time::{Duration, Instant};

const REPS: usize = 1000;
const WARMUP: usize = 100;

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

fn deterministic_values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = (((index + 1) * 37 + salt * 17) % 113) as i32 - 56;
            bucket as f32 * scale
        })
        .collect()
}

fn time_clone(input: &WasmTensor, samples: &mut Vec<Duration>) {
    let start = Instant::now();
    let cloned = black_box(input).clone();
    let elapsed = start.elapsed();
    black_box(&cloned);
    samples.push(elapsed);
    drop(cloned);
}

fn time_weight_dims(layer: &WasmLinear, samples: &mut Vec<Duration>) {
    let start = Instant::now();
    let dims = black_box(layer).weight_dims();
    let elapsed = start.elapsed();
    black_box(&dims);
    samples.push(elapsed);
}

fn time_forward(layer: &WasmLinear, input: &WasmTensor, samples: &mut Vec<Duration>) {
    let start = Instant::now();
    let output = black_box(layer).forward(black_box(input));
    let elapsed = start.elapsed();
    black_box(&output);
    samples.push(elapsed);
    drop(output);
}

fn run_case(dim: usize, salt: usize) -> serde_json::Value {
    let mut layer = WasmLinear::new(dim, dim, true);
    let weight_len = dim * dim + dim;
    let weights = deterministic_values(weight_len, salt, 0.0005);
    layer
        .set_weights_flat(&weights)
        .expect("apply deterministic flat weights");
    assert_eq!(
        layer.get_weights_flat().expect("initial flat weight readback"),
        weights,
        "initial flat weights must read back exactly"
    );
    assert_eq!(layer.weight_dims(), vec![dim, dim]);

    let input_values = deterministic_values(dim, salt + 31, 0.01);
    let input = WasmTensor::new(&input_values, &[1, dim, 1, 1]);
    assert_eq!(input.shape(), vec![1, dim, 1, 1]);

    let reference = layer.forward(&input).to_array();
    assert_eq!(reference.len(), dim);
    assert!(reference.iter().all(|value| value.is_finite()));

    for _ in 0..WARMUP {
        let clone = black_box(&input).clone();
        black_box(&clone);
        let dims = black_box(&layer).weight_dims();
        black_box(&dims);
        let output = black_box(&layer).forward(black_box(&input));
        black_box(&output);
    }

    let mut clone_samples = Vec::with_capacity(REPS);
    let mut dims_samples = Vec::with_capacity(REPS);
    let mut forward_samples = Vec::with_capacity(REPS);

    for repetition in 0..REPS {
        match repetition % 3 {
            0 => {
                time_clone(&input, &mut clone_samples);
                time_weight_dims(&layer, &mut dims_samples);
                time_forward(&layer, &input, &mut forward_samples);
            }
            1 => {
                time_weight_dims(&layer, &mut dims_samples);
                time_forward(&layer, &input, &mut forward_samples);
                time_clone(&input, &mut clone_samples);
            }
            _ => {
                time_forward(&layer, &input, &mut forward_samples);
                time_clone(&input, &mut clone_samples);
                time_weight_dims(&layer, &mut dims_samples);
            }
        }
    }

    let final_output = layer.forward(&input).to_array();
    assert_eq!(final_output, reference, "output changed after timing");
    assert_eq!(
        layer.get_weights_flat().expect("final flat weight readback"),
        weights,
        "weights changed during timing"
    );
    assert_eq!(layer.weight_dims(), vec![dim, dim]);
    assert_eq!(input.shape(), vec![1, dim, 1, 1]);

    let clone_timing = summary(&clone_samples);
    let dims_timing = summary(&dims_samples);
    let forward_timing = summary(&forward_samples);
    let clone_median = clone_timing["median_ms"].as_f64().expect("clone median");
    let dims_median = dims_timing["median_ms"].as_f64().expect("dims median");
    let forward_median = forward_timing["median_ms"].as_f64().expect("forward median");

    json!({
        "name": format!("linear_{dim}_to_{dim}"),
        "config": {
            "input_dim": dim,
            "output_dim": dim,
            "bias": true,
            "weight_count": weight_len,
        },
        "reference_output": reference,
        "timing": {
            "input_clone": clone_timing,
            "weight_dims": dims_timing,
            "forward": forward_timing,
        },
        "ratios": {
            "input_clone_share_of_forward": clone_median / forward_median,
            "weight_dims_share_of_forward": dims_median / forward_median,
            "controls_sum_share_of_forward": (clone_median + dims_median) / forward_median,
        },
        "proofs": {
            "finite_output": true,
            "deterministic_output": true,
            "weights_exact_and_stable": true,
            "input_shape_stable": true,
            "weight_dims_stable": true,
        }
    })
}

fn main() {
    let cases = vec![run_case(32, 223), run_case(128, 447)];
    println!("{}", json!({
        "verdict": "PASS",
        "schema": "burn-research.wasm-linear-wrapper-decomposition.v1",
        "consumer": "external-packaged-rust-release",
        "repetitions_per_case": REPS,
        "warmup_per_case": WARMUP,
        "cases": cases,
        "proofs": {
            "public_package_surface_only": true,
            "all_outputs_finite_and_deterministic": true,
            "all_weights_exact_and_stable": true,
        },
        "note": "Timing buckets are descriptive controls and are not assumed additive or used as CI performance thresholds."
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
        raise SystemExit("WasmLinear wrapper decomposition consumer produced no output")
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

    with tempfile.TemporaryDirectory(prefix="burn-research-wasm-linear-wrapper-") as tmp:
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
            f'''[package]\nname = "burn-research-wasm-linear-wrapper-decomposition"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\nserde_json = "1.0"\n''',
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(RUST_CONSUMER, encoding="utf-8")

        run(
            ["cargo", "generate-lockfile", "--manifest-path", str(consumer / "Cargo.toml")],
            cwd=consumer,
        )
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

    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
