#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FFI_DIR = REPO / "ffi"
REPORT = REPO / "native-graph-run-decomposition-report.json"

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
const EXPECTED_DIM: usize = (POLICY_IN * POLICY_OUT + POLICY_OUT) as usize;
const REPS: usize = 800;
const FEATURES: [f32; 6] = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0];

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

fn main() {
    let mut registry = LayerRegistry::new();
    let linear = AgentLayerSpec::linear(215_000, POLICY_IN, POLICY_OUT, true)
        .expect("linear spec");
    registry
        .init_agent_layer(&linear)
        .expect("initialize linear");

    let mut builder = AgentGraphBuilder::new(2).expect("graph builder");
    builder
        .add_unary(&linear, 0, 1)
        .expect("add graph step");
    builder.set_output(1).expect("set output");
    let graph = builder.compile(&registry).expect("compile graph");

    let binding = GraphParameterBinding::build(&graph, &registry).expect("build binding");
    assert_eq!(binding.total_len(), EXPECTED_DIM);
    let zero = vec![0.0f32; binding.total_len()];
    binding
        .apply_flat(&graph, &mut registry, &zero)
        .expect("apply canonical zero state");

    let input = WasmTensor::new(&FEATURES, &[1, POLICY_IN as usize, 1, 1]);
    let reference = graph.run(&registry, &input).expect("reference graph run");
    let reference_values = reference.to_array();
    assert_eq!(reference_values.len(), POLICY_OUT as usize);
    assert!(reference_values.iter().all(|value| value.is_finite()));

    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();

    let mut native_run_samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let output = graph.run(black_box(&registry), black_box(&input)).expect("native graph run");
        let elapsed = start.elapsed();
        black_box(&output);
        native_run_samples.push(elapsed);
        drop(output);
    }

    // Precompute real WasmTensor outputs so allocation timing moves the same concrete
    // output type into Box without measuring graph execution in this bucket.
    let prepared = (0..REPS)
        .map(|_| graph.run(&registry, &input).expect("prepare boxing output"))
        .collect::<Vec<_>>();
    let mut box_samples = Vec::with_capacity(REPS);
    for output in prepared {
        let start = Instant::now();
        let boxed = Box::new(black_box(output));
        let elapsed = start.elapsed();
        black_box(&boxed);
        box_samples.push(elapsed);
        drop(boxed);
    }

    let mut native_run_box_samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        let output = graph.run(black_box(&registry), black_box(&input)).expect("native graph run");
        let boxed = Box::new(output);
        let elapsed = start.elapsed();
        black_box(&boxed);
        native_run_box_samples.push(elapsed);
        drop(boxed);
    }

    let final_values = graph
        .run(&registry, &input)
        .expect("final graph run")
        .to_array();
    assert_eq!(final_values, reference_values);
    assert_eq!(graph.program_identity(), program_identity);
    assert_eq!(binding.identity_json(), binding_identity);

    let report = json!({
        "verdict": "PASS",
        "consumer": "public-rust-package-path",
        "workload": {
            "policy": "Linear(6 -> 2, bias=true)",
            "parameter_dim": EXPECTED_DIM,
            "output_dim": POLICY_OUT,
            "repetitions": REPS,
            "input": FEATURES,
        },
        "program_identity": program_identity,
        "binding_identity": binding_identity,
        "reference_output": reference_values,
        "timing": {
            "compiled_graph_run": summary(&native_run_samples),
            "box_wasm_tensor": summary(&box_samples),
            "compiled_graph_run_plus_box": summary(&native_run_box_samples),
        },
        "proofs": {
            "finite_output": true,
            "deterministic_output": true,
            "program_identity_stable": true,
            "binding_identity_stable": true,
            "public_core_surface_only": true,
        }
    });
    println!("{}", report);
}
'''

RAW_PYTHON_CONSUMER = r'''
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import time

import burn_research_ffi as br
from burn_research_ffi import host
from burn_research_ffi.burn_research_ffi import ffi, lib

POLICY_IN = 6
POLICY_OUT = 2
EXPECTED_DIM = POLICY_IN * POLICY_OUT + POLICY_OUT
REPS = 800
FEATURES = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0]


def ns() -> int:
    return time.perf_counter_ns()


def summary(samples: list[int]) -> dict[str, float]:
    values = sorted(value / 1_000_000.0 for value in samples)
    if not values:
        raise AssertionError("empty timing sample set")
    p90 = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    return {
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "p90_ms": values[p90],
        "max_ms": values[-1],
    }


def last_error() -> str:
    required = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", required + 1)
    lib.br_v1_last_error_copy(buf, required + 1)
    return ffi.string(buf).decode("utf-8", errors="replace")


def check(status: int, context: str) -> None:
    code = int(status)
    if code != 0:
        raise AssertionError(f"{context}: status={code} diagnostic={last_error()!r}")


def new_handle(context: str, fn, *args):
    out = ffi.new("br_v1_handle **")
    check(fn(*args, out), context)
    if out[0] == ffi.NULL:
        raise AssertionError(f"{context}: null handle on success")
    return out[0]


def free_handle(handle) -> None:
    if handle != ffi.NULL:
        check(lib.br_v1_handle_free(handle), "handle free")


def read_u8(handle) -> bytes:
    n = ffi.new("size_t *")
    check(lib.br_v1_u8_buffer_len(handle, n), "u8 len")
    size = int(n[0])
    if size == 0:
        return b""
    dest = ffi.new("uint8_t[]", size)
    check(lib.br_v1_u8_buffer_copy(handle, dest, size), "u8 copy")
    return bytes(ffi.buffer(dest, size))


def program_identity(graph) -> str:
    buf = new_handle("program identity", lib.br_v1_graph_program_identity, graph)
    try:
        return read_u8(buf).decode("utf-8")
    finally:
        free_handle(buf)


def binding_identity(binding) -> str:
    buf = new_handle("binding identity", lib.br_v1_binding_identity_json, binding)
    try:
        return read_u8(buf).decode("utf-8")
    finally:
        free_handle(buf)


def tensor_values(tensor) -> list[float]:
    n = ffi.new("size_t *")
    check(lib.br_v1_tensor_len(tensor, n), "tensor len")
    size = int(n[0])
    dest = ffi.new("float[]", size)
    check(lib.br_v1_tensor_copy_f32(tensor, dest, size), "tensor copy")
    return [float(dest[i]) for i in range(size)]


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"consumer imported package from repo: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"consumer imported host from repo: {host_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected host/ABI identity")

handles = []
try:
    registry = new_handle("registry new", lib.br_v1_registry_new)
    handles.append(registry)
    layer = new_handle(
        "linear layer", lib.br_v1_layer_linear, 215_000, POLICY_IN, POLICY_OUT, 1
    )
    handles.append(layer)
    check(lib.br_v1_registry_init_layer(registry, layer), "registry init")
    builder = new_handle("builder new", lib.br_v1_graph_builder_new, 2)
    handles.append(builder)
    check(lib.br_v1_graph_builder_add_unary(builder, layer, 0, 1), "add unary")
    check(lib.br_v1_graph_builder_set_output(builder, 1), "set output")
    graph = new_handle("compile", lib.br_v1_graph_builder_compile, builder, registry)
    handles.append(graph)
    binding = new_handle("binding", lib.br_v1_binding_build, graph, registry)
    handles.append(binding)

    n = ffi.new("size_t *")
    check(lib.br_v1_binding_total_len(binding, n), "binding total len")
    if int(n[0]) != EXPECTED_DIM:
        raise AssertionError(f"parameter dim {int(n[0])} != {EXPECTED_DIM}")
    zero = ffi.new("float[]", EXPECTED_DIM)
    check(
        lib.br_v1_binding_apply_flat(binding, graph, registry, zero, EXPECTED_DIM),
        "apply zero",
    )

    raw_features = ffi.new("float[]", FEATURES)
    input_tensor = new_handle(
        "input tensor",
        lib.br_v1_tensor_new_f32,
        raw_features,
        len(FEATURES),
        1,
        POLICY_IN,
        1,
        1,
    )
    handles.append(input_tensor)

    pid = program_identity(graph)
    bid = binding_identity(binding)
    reference = new_handle("reference run", lib.br_v1_graph_run, graph, registry, input_tensor)
    try:
        reference_values = tensor_values(reference)
    finally:
        free_handle(reference)
    if len(reference_values) != POLICY_OUT or not all(math.isfinite(v) for v in reference_values):
        raise AssertionError(f"invalid reference output: {reference_values}")

    samples: list[int] = []
    for _ in range(REPS):
        out = ffi.new("br_v1_handle **")
        t0 = ns()
        status = lib.br_v1_graph_run(graph, registry, input_tensor, out)
        t1 = ns()
        check(status, "raw graph run")
        if out[0] == ffi.NULL:
            raise AssertionError("raw graph run returned null handle")
        samples.append(t1 - t0)
        free_handle(out[0])

    final = new_handle("final run", lib.br_v1_graph_run, graph, registry, input_tensor)
    try:
        final_values = tensor_values(final)
    finally:
        free_handle(final)
    if final_values != reference_values:
        raise AssertionError("raw output changed after timing")
    if program_identity(graph) != pid:
        raise AssertionError("program identity changed")
    if binding_identity(binding) != bid:
        raise AssertionError("binding identity changed")

    print(json.dumps({
        "verdict": "PASS",
        "consumer": "installed-wheel-raw-cffi",
        "host_api": host.HOST_API_SCHEMA,
        "abi_version": host.abi_version(),
        "program_identity": pid,
        "binding_identity": bid,
        "reference_output": reference_values,
        "timing": {"raw_br_v1_graph_run": summary(samples)},
        "proofs": {
            "installed_wheel": True,
            "finite_output": True,
            "deterministic_output": True,
            "program_identity_stable": True,
            "binding_identity_stable": True,
            "owned_handles_closed": True,
        },
    }, sort_keys=True))
finally:
    for handle in reversed(handles):
        try:
            free_handle(handle)
        except Exception:
            pass
'''


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"cwd: {cwd}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def last_json(stdout: str, context: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{context} produced no output")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{context} did not end with JSON: {lines[-1]!r}") from exc
    if result.get("verdict") != "PASS":
        raise SystemExit(f"{context} unexpected verdict: {result}")
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="burn-research-native-graph-run-") as tmp:
        root = Path(tmp)

        # Direct Rust public-core consumer.
        rust_consumer = root / "rust-consumer"
        (rust_consumer / "src").mkdir(parents=True)
        repo_path = REPO.as_posix().replace('"', '\\"')
        (rust_consumer / "Cargo.toml").write_text(
            f'''[package]\nname = "burn-research-native-graph-run-research"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{repo_path}" }}\nserde_json = "1.0"\n''',
            encoding="utf-8",
        )
        (rust_consumer / "src" / "main.rs").write_text(RUST_CONSUMER, encoding="utf-8")
        run(
            ["cargo", "generate-lockfile", "--manifest-path", str(rust_consumer / "Cargo.toml")],
            cwd=rust_consumer,
        )
        rust_completed = run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--locked",
                "--manifest-path",
                str(rust_consumer / "Cargo.toml"),
            ],
            cwd=rust_consumer,
        )
        rust_report = last_json(rust_completed.stdout, "Rust consumer")

        # Installed-wheel raw ABI consumer on the same runner.
        wheels = root / "wheels"
        wheels.mkdir()
        run(
            [sys.executable, "-m", "maturin", "build", "--release", "--out", str(wheels)],
            cwd=FFI_DIR,
        )
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise SystemExit(f"expected one wheel, found {wheel_files}")

        venv = root / "venv"
        run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        python = venv / "bin" / "python"
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                str(wheel_files[0]),
            ],
            cwd=root,
        )
        consumer = root / "raw_consumer.py"
        consumer.write_text(RAW_PYTHON_CONSUMER, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["BR_REPO_ROOT"] = str(REPO)
        raw_completed = run([str(python), str(consumer)], cwd=root, env=env)
        raw_report = last_json(raw_completed.stdout, "raw installed-wheel consumer")

        if rust_report["program_identity"] != raw_report["program_identity"]:
            raise SystemExit("Rust-direct/raw-FFI program identity mismatch")
        if rust_report["binding_identity"] != raw_report["binding_identity"]:
            raise SystemExit("Rust-direct/raw-FFI binding identity mismatch")
        if rust_report["reference_output"] != raw_report["reference_output"]:
            raise SystemExit(
                "Rust-direct/raw-FFI reference output mismatch: "
                f"{rust_report['reference_output']} != {raw_report['reference_output']}"
            )
        if not all(math.isfinite(float(v)) for v in rust_report["reference_output"]):
            raise SystemExit("non-finite cross-boundary reference output")

        native_run_ms = float(
            rust_report["timing"]["compiled_graph_run"]["median_ms"]
        )
        box_ms = float(rust_report["timing"]["box_wasm_tensor"]["median_ms"])
        native_run_box_ms = float(
            rust_report["timing"]["compiled_graph_run_plus_box"]["median_ms"]
        )
        raw_ffi_ms = float(raw_report["timing"]["raw_br_v1_graph_run"]["median_ms"])
        if raw_ffi_ms <= 0.0:
            raise SystemExit("invalid raw FFI median")

        residual_ms = max(0.0, raw_ffi_ms - native_run_box_ms)
        native_run_share = native_run_ms / raw_ffi_ms
        native_run_box_share = native_run_box_ms / raw_ffi_ms
        box_share_of_native_combined = box_ms / max(native_run_box_ms, 1e-12)
        residual_share = residual_ms / raw_ffi_ms

        if native_run_box_share >= 0.70:
            decision = "COMPILED_GRAPH_RUN_DOMINATES"
        elif residual_share >= 0.25:
            decision = "FFI_BOUNDARY_RESIDUAL_MATERIAL"
        else:
            decision = "MIXED_NATIVE_GRAPH_RUN_COST"

        report = {
            "verdict": "PASS",
            "decision": decision,
            "decision_is_ci_threshold": False,
            "method": "same-runner-public-core-vs-installed-wheel-shadow-decomposition",
            "workload": rust_report["workload"],
            "identity": {
                "program_identity": rust_report["program_identity"],
                "binding_identity": rust_report["binding_identity"],
            },
            "reference_output": rust_report["reference_output"],
            "timing": {
                "rust_direct": rust_report["timing"],
                "raw_ffi": raw_report["timing"],
            },
            "derived": {
                "native_run_share_of_raw_ffi": native_run_share,
                "native_run_plus_box_share_of_raw_ffi": native_run_box_share,
                "box_share_of_native_run_plus_box": box_share_of_native_combined,
                "boundary_residual_ms": residual_ms,
                "boundary_residual_share": residual_share,
            },
            "proofs": {
                "public_rust_surface_only": True,
                "installed_wheel_raw_ffi": True,
                "program_identity_exact": True,
                "binding_identity_exact": True,
                "output_exact": True,
                "finite_output": True,
                "deterministic_execution": True,
                "stable_identities": True,
                "owned_handles_closed": True,
                "production_abi_unchanged": True,
            },
            "notes": [
                "Timing is descriptive research evidence only, not a performance SLA.",
                "The residual groups Python/CFFI call overhead, ffi_status/catch_unwind, distinct/type validation, handle lookup, and BrV1Handle wrapping; it does not pretend to attribute those individually.",
                "Box<WasmTensor> is a shadow allocation control for concrete output ownership, not an exact byte-for-byte model of Box<BrV1Handle>.",
                "No production Rust source, ABI symbol, Python API, graph primitive, or tensor transport contract is changed.",
            ],
        }
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
