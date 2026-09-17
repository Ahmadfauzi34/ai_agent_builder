#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "wasm-linear-residual-decomposition-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

RUST_CONSUMER = r'''
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_research::layers::linear::WasmLinear;
use burn_research::{WasmBackend, WasmTensor};
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
            let bucket = (((index + 1) * 41 + salt * 19) % 127) as i32 - 63;
            bucket as f32 * scale
        })
        .collect()
}

fn tensor4(values: &[f32], dim: usize) -> Tensor<WasmBackend, 4> {
    let device: <WasmBackend as Backend>::Device = Default::default();
    Tensor::from_data(TensorData::new(values.to_vec(), [1, dim, 1, 1]), &device)
}

fn tensor2(values: &[f32], dim: usize) -> Tensor<WasmBackend, 2> {
    let device: <WasmBackend as Backend>::Device = Default::default();
    Tensor::from_data(TensorData::new(values.to_vec(), [1, dim]), &device)
}

fn time_reshape_in(base: &Tensor<WasmBackend, 4>, dim: usize, samples: &mut Vec<Duration>) {
    let owned = base.clone();
    let start = Instant::now();
    let reshaped = black_box(owned).reshape([1, dim]);
    let elapsed = start.elapsed();
    black_box(&reshaped);
    samples.push(elapsed);
}

fn time_kernel(
    layer: &Linear<WasmBackend>,
    base: &Tensor<WasmBackend, 2>,
    samples: &mut Vec<Duration>,
) {
    let owned = base.clone();
    let start = Instant::now();
    let output = black_box(layer).forward(black_box(owned));
    let elapsed = start.elapsed();
    black_box(&output);
    samples.push(elapsed);
}

fn time_reshape_out(
    base: &Tensor<WasmBackend, 2>,
    dim: usize,
    samples: &mut Vec<Duration>,
) {
    let owned = base.clone();
    let start = Instant::now();
    let reshaped = black_box(owned).reshape([1, dim, 1, 1]);
    let elapsed = start.elapsed();
    black_box(&reshaped);
    samples.push(elapsed);
}

fn time_pipeline(
    layer: &Linear<WasmBackend>,
    base: &Tensor<WasmBackend, 4>,
    dim: usize,
    samples: &mut Vec<Duration>,
) {
    let owned = base.clone();
    let start = Instant::now();
    let x2 = black_box(owned).reshape([1, dim]);
    let y2 = black_box(layer).forward(black_box(x2));
    let y4 = black_box(y2).reshape([1, dim, 1, 1]);
    let elapsed = start.elapsed();
    black_box(&y4);
    samples.push(elapsed);
}

fn time_wrapper(layer: &WasmLinear, input: &WasmTensor, samples: &mut Vec<Duration>) {
    let start = Instant::now();
    let output = black_box(layer).forward(black_box(input));
    let elapsed = start.elapsed();
    black_box(&output);
    samples.push(elapsed);
}

fn run_case(dim: usize, salt: usize) -> serde_json::Value {
    let input_values = deterministic_values(dim, salt + 11, 0.01);
    let wrapper_input = WasmTensor::new(&input_values, &[1, dim, 1, 1]);
    let burn_input4 = tensor4(&input_values, dim);
    let burn_input2 = tensor2(&input_values, dim);

    let device: <WasmBackend as Backend>::Device = Default::default();
    let burn_linear: Linear<WasmBackend> = LinearConfig::new(dim, dim)
        .with_bias(true)
        .init(&device);
    let burn_output2 = burn_linear.forward(burn_input2.clone());
    let burn_output_reference = burn_output2.clone().into_data();
    let burn_output_slice = burn_output_reference
        .as_slice::<f32>()
        .expect("Burn Linear output must be f32");
    assert_eq!(burn_output_slice.len(), dim);
    assert!(burn_output_slice.iter().all(|value| value.is_finite()));

    let mut wrapper = WasmLinear::new(dim, dim, true);
    let weights = deterministic_values(dim * dim + dim, salt + 29, 0.0005);
    wrapper
        .set_weights_flat(&weights)
        .expect("apply deterministic wrapper weights");
    assert_eq!(wrapper.weight_dims(), vec![dim, dim]);
    assert_eq!(wrapper_input.shape(), vec![1, dim, 1, 1]);
    let wrapper_reference = wrapper.forward(&wrapper_input).to_array();
    assert_eq!(wrapper_reference.len(), dim);
    assert!(wrapper_reference.iter().all(|value| value.is_finite()));

    for _ in 0..WARMUP {
        let r_in = burn_input4.clone().reshape([1, dim]);
        black_box(&r_in);
        let k = burn_linear.forward(burn_input2.clone());
        black_box(&k);
        let r_out = burn_output2.clone().reshape([1, dim, 1, 1]);
        black_box(&r_out);
        let p_in = burn_input4.clone().reshape([1, dim]);
        let p_mid = burn_linear.forward(p_in);
        let p_out = p_mid.reshape([1, dim, 1, 1]);
        black_box(&p_out);
        let w = wrapper.forward(&wrapper_input);
        black_box(&w);
    }

    let mut reshape_in_samples = Vec::with_capacity(REPS);
    let mut kernel_samples = Vec::with_capacity(REPS);
    let mut reshape_out_samples = Vec::with_capacity(REPS);
    let mut pipeline_samples = Vec::with_capacity(REPS);
    let mut wrapper_samples = Vec::with_capacity(REPS);

    for repetition in 0..REPS {
        match repetition % 5 {
            0 => {
                time_reshape_in(&burn_input4, dim, &mut reshape_in_samples);
                time_kernel(&burn_linear, &burn_input2, &mut kernel_samples);
                time_reshape_out(&burn_output2, dim, &mut reshape_out_samples);
                time_pipeline(&burn_linear, &burn_input4, dim, &mut pipeline_samples);
                time_wrapper(&wrapper, &wrapper_input, &mut wrapper_samples);
            }
            1 => {
                time_kernel(&burn_linear, &burn_input2, &mut kernel_samples);
                time_reshape_out(&burn_output2, dim, &mut reshape_out_samples);
                time_pipeline(&burn_linear, &burn_input4, dim, &mut pipeline_samples);
                time_wrapper(&wrapper, &wrapper_input, &mut wrapper_samples);
                time_reshape_in(&burn_input4, dim, &mut reshape_in_samples);
            }
            2 => {
                time_reshape_out(&burn_output2, dim, &mut reshape_out_samples);
                time_pipeline(&burn_linear, &burn_input4, dim, &mut pipeline_samples);
                time_wrapper(&wrapper, &wrapper_input, &mut wrapper_samples);
                time_reshape_in(&burn_input4, dim, &mut reshape_in_samples);
                time_kernel(&burn_linear, &burn_input2, &mut kernel_samples);
            }
            3 => {
                time_pipeline(&burn_linear, &burn_input4, dim, &mut pipeline_samples);
                time_wrapper(&wrapper, &wrapper_input, &mut wrapper_samples);
                time_reshape_in(&burn_input4, dim, &mut reshape_in_samples);
                time_kernel(&burn_linear, &burn_input2, &mut kernel_samples);
                time_reshape_out(&burn_output2, dim, &mut reshape_out_samples);
            }
            _ => {
                time_wrapper(&wrapper, &wrapper_input, &mut wrapper_samples);
                time_reshape_in(&burn_input4, dim, &mut reshape_in_samples);
                time_kernel(&burn_linear, &burn_input2, &mut kernel_samples);
                time_reshape_out(&burn_output2, dim, &mut reshape_out_samples);
                time_pipeline(&burn_linear, &burn_input4, dim, &mut pipeline_samples);
            }
        }
    }

    let burn_output_after = burn_linear.forward(burn_input2.clone()).into_data();
    assert_eq!(
        burn_output_after.as_slice::<f32>().expect("Burn output after timing"),
        burn_output_slice,
        "Burn Linear output changed after timing"
    );
    let wrapper_after = wrapper.forward(&wrapper_input).to_array();
    assert_eq!(wrapper_after, wrapper_reference, "wrapper output changed after timing");
    assert_eq!(wrapper.weight_dims(), vec![dim, dim]);
    assert_eq!(
        wrapper.get_weights_flat().expect("wrapper weights after timing"),
        weights,
        "wrapper weights changed during timing"
    );

    let reshape_in = summary(&reshape_in_samples);
    let kernel = summary(&kernel_samples);
    let reshape_out = summary(&reshape_out_samples);
    let pipeline = summary(&pipeline_samples);
    let wrapper_timing = summary(&wrapper_samples);

    let reshape_in_median = reshape_in["median_ms"].as_f64().expect("reshape in median");
    let kernel_median = kernel["median_ms"].as_f64().expect("kernel median");
    let reshape_out_median = reshape_out["median_ms"].as_f64().expect("reshape out median");
    let pipeline_median = pipeline["median_ms"].as_f64().expect("pipeline median");
    let wrapper_median = wrapper_timing["median_ms"].as_f64().expect("wrapper median");

    json!({
        "name": format!("linear_{dim}_to_{dim}"),
        "config": {"dim": dim, "bias": true, "batch": 1},
        "timing": {
            "reshape_4d_to_2d": reshape_in,
            "burn_linear_forward": kernel,
            "reshape_2d_to_4d": reshape_out,
            "analogue_pipeline": pipeline,
            "wasm_linear_forward": wrapper_timing,
        },
        "ratios": {
            "reshape_in_share_of_pipeline": reshape_in_median / pipeline_median,
            "burn_kernel_share_of_pipeline": kernel_median / pipeline_median,
            "reshape_out_share_of_pipeline": reshape_out_median / pipeline_median,
            "reshape_sum_share_of_pipeline": (reshape_in_median + reshape_out_median) / pipeline_median,
            "analogue_pipeline_vs_wrapper": pipeline_median / wrapper_median,
        },
        "proofs": {
            "burn_output_finite_and_stable": true,
            "wrapper_output_finite_and_stable": true,
            "wrapper_weights_exact_and_stable": true,
            "wrapper_dimensions_stable": true,
        }
    })
}

fn main() {
    let cases = vec![run_case(32, 227), run_case(128, 491)];
    println!("{}", json!({
        "verdict": "PASS",
        "schema": "burn-research.wasm-linear-residual-decomposition.v1",
        "consumer": "external-packaged-rust-release",
        "repetitions_per_case": REPS,
        "warmup_per_case": WARMUP,
        "cases": cases,
        "proofs": {
            "public_package_surface_only": true,
            "all_outputs_finite_and_stable": true,
            "wrapper_state_exact_and_stable": true,
        },
        "note": "Burn Linear timing is an equivalent-dimension analogue, not private instrumentation of the wrapper's exact internal module. Timing is descriptive and not a CI SLA."
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
        raise SystemExit("WasmLinear residual decomposition consumer produced no output")
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

    with tempfile.TemporaryDirectory(prefix="burn-research-wasm-linear-residual-") as tmp:
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
            f'''[package]\nname = "burn-research-wasm-linear-residual-decomposition"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\nburn = "0.20.0"\nserde_json = "1.0"\n''',
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
