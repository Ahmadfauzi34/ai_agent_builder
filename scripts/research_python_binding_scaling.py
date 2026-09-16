#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FFI_DIR = REPO / "ffi"
REPORT = REPO / "python-binding-scaling-report.json"

CONSUMER = r'''
from __future__ import annotations

import json
import math
import os
import statistics
import struct
import time
from pathlib import Path

import burn_research_ffi
from burn_research_ffi import (
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    ProgramBundle,
    Registry,
    Tensor,
)

BATCH = 8
BUILD_REPS = 7
READ_REPS = 7
EVAL_REPS = 15

CASES = [
    {"name": "single-w8", "family": "parameter_length", "width": 8, "depth": 1},
    {"name": "single-w32", "family": "parameter_length", "width": 32, "depth": 1},
    {"name": "single-w64", "family": "parameter_length", "width": 64, "depth": 1},
    {"name": "single-w128", "family": "parameter_length", "width": 128, "depth": 1},
    {"name": "owners-d4-w64", "family": "owner_count", "width": 64, "depth": 4},
    {"name": "owners-d8-w64", "family": "owner_count", "width": 64, "depth": 8},
    {"name": "owners-d16-w64", "family": "owner_count", "width": 64, "depth": 16},
]


def ns() -> int:
    return time.perf_counter_ns()


def ms(value_ns: int) -> float:
    return value_ns / 1_000_000.0


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def summary(samples_ns: list[int]) -> dict[str, float]:
    values = sorted(ms(sample) for sample in samples_ns)
    if not values:
        raise AssertionError("timing sample set is empty")
    p90_index = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    return {
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "p90_ms": values[p90_index],
        "max_ms": values[-1],
    }


def expected_params(width: int, depth: int) -> int:
    return depth * (width * width + width)


def make_candidate(length: int) -> list[float]:
    # Small deterministic f32 values keep deep linear chains finite while still
    # forcing every canonical parameter coordinate through the apply path.
    return [f32((((i * 37) % 101) - 50) * 0.0005) for i in range(length)]


def make_input(width: int) -> list[float]:
    values: list[float] = []
    for b in range(BATCH):
        for j in range(width):
            values.append(f32(0.35 * math.sin((b + 1) * (j + 2) * 0.071)))
    return values


def close_all(items: list[object]) -> None:
    while items:
        item = items.pop()
        close = getattr(item, "close", None)
        if close is not None:
            close()


def build_case(width: int, depth: int):
    owned: list[object] = []
    registry = Registry(); owned.append(registry)
    builder = GraphBuilder(depth + 1); owned.append(builder)
    layers: list[LinearLayerSpec] = []
    for i in range(depth):
        layer = LinearLayerSpec(70_000 + depth * 100 + i, width, width, bias=True)
        owned.append(layer)
        layers.append(layer)
        registry.init_layer(layer)
        builder.add_unary(layer, i, i + 1)
    builder.set_output(depth)
    graph = builder.compile(registry); owned.append(graph)
    return registry, builder, graph, layers, owned


def evaluate_case(case: dict[str, object]) -> dict[str, object]:
    width = int(case["width"])
    depth = int(case["depth"])
    expected = expected_params(width, depth)
    registry, builder, graph, layers, owned = build_case(width, depth)
    try:
        program_identity = graph.program_identity()

        # Build once for semantics and identity.
        binding = GraphParameterBinding.build(graph, registry); owned.append(binding)
        if binding.total_len != expected:
            raise AssertionError(
                f"{case['name']}: total_len={binding.total_len}, expected={expected}"
            )
        binding_identity = binding.identity()
        initial = binding.read_flat(graph, registry)
        if len(initial) != expected or not all(math.isfinite(v) for v in initial):
            raise AssertionError(f"{case['name']}: invalid initial flat vector")

        candidate = make_candidate(expected)
        binding.apply_flat(graph, registry, candidate)
        roundtrip = binding.read_flat(graph, registry)
        if roundtrip != candidate:
            raise AssertionError(f"{case['name']}: candidate did not read back exactly")
        if graph.program_identity() != program_identity:
            raise AssertionError(f"{case['name']}: program identity changed after apply")
        if binding.identity() != binding_identity:
            raise AssertionError(f"{case['name']}: binding identity changed after apply")

        input_tensor = Tensor.from_f32(make_input(width), (BATCH, width, 1, 1)); owned.append(input_tensor)

        # Warm-up core paths before collecting timing evidence.
        binding.apply_flat(graph, registry, candidate)
        warm = graph.run(registry, input_tensor)
        warm_values = warm.to_f32()
        warm.close()
        if len(warm_values) != BATCH * width or not all(math.isfinite(v) for v in warm_values):
            raise AssertionError(f"{case['name']}: invalid warm-up output")

        build_samples: list[int] = []
        for _ in range(BUILD_REPS):
            t0 = ns()
            temp = GraphParameterBinding.build(graph, registry)
            t1 = ns()
            if temp.total_len != expected or temp.identity() != binding_identity:
                temp.close()
                raise AssertionError(f"{case['name']}: rebuilt binding mismatch")
            temp.close()
            build_samples.append(t1 - t0)

        read_samples: list[int] = []
        for _ in range(READ_REPS):
            t0 = ns()
            values = binding.read_flat(graph, registry)
            t1 = ns()
            if values != candidate:
                raise AssertionError(f"{case['name']}: read_flat changed state")
            read_samples.append(t1 - t0)

        apply_samples: list[int] = []
        run_samples: list[int] = []
        copy_samples: list[int] = []
        combined_samples: list[int] = []
        reference_output: list[float] | None = None
        for _ in range(EVAL_REPS):
            t0 = ns()
            binding.apply_flat(graph, registry, candidate)
            t1 = ns()
            output = graph.run(registry, input_tensor)
            t2 = ns()
            values = output.to_f32()
            t3 = ns()
            output.close()
            if len(values) != BATCH * width or not all(math.isfinite(v) for v in values):
                raise AssertionError(f"{case['name']}: non-finite or malformed output")
            if reference_output is None:
                reference_output = values
            elif values != reference_output:
                raise AssertionError(f"{case['name']}: repeated apply/run is not deterministic")
            apply_samples.append(t1 - t0)
            run_samples.append(t2 - t1)
            copy_samples.append(t3 - t2)
            combined_samples.append(t3 - t0)

        if binding.read_flat(graph, registry) != candidate:
            raise AssertionError(f"{case['name']}: final state drifted")
        if graph.program_identity() != program_identity or binding.identity() != binding_identity:
            raise AssertionError(f"{case['name']}: identity drifted during timing loop")

        apply_summary = summary(apply_samples)
        run_summary = summary(run_samples)
        copy_summary = summary(copy_samples)
        combined_summary = summary(combined_samples)
        apply_share = apply_summary["median_ms"] / max(combined_summary["median_ms"], 1e-12)

        return {
            "name": case["name"],
            "family": case["family"],
            "width": width,
            "depth": depth,
            "expected_owners": depth,
            "parameter_count": expected,
            "program_identity": program_identity,
            "binding_identity": binding_identity,
            "timing": {
                "binding_build": summary(build_samples),
                "read_flat": summary(read_samples),
                "apply_flat": apply_summary,
                "graph_run": run_summary,
                "output_copy": copy_summary,
                "apply_run_copy": combined_summary,
                "apply_share_of_combined_median": apply_share,
                "apply_ns_per_parameter_median": (
                    apply_summary["median_ms"] * 1_000_000.0 / expected
                ),
            },
            "reference_output": reference_output,
            "candidate": candidate,
        }
    finally:
        close_all(owned)


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(burn_research_ffi.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from checkout: {module_path}")

results: list[dict[str, object]] = []
for case in CASES:
    results.append(evaluate_case(case))

largest = max(results, key=lambda item: int(item["parameter_count"]))
width = int(largest["width"])
depth = int(largest["depth"])
registry, builder, graph, layers, owned = build_case(width, depth)
try:
    binding = GraphParameterBinding.build(graph, registry); owned.append(binding)
    candidate = list(largest["candidate"])
    binding.apply_flat(graph, registry, candidate)
    input_tensor = Tensor.from_f32(make_input(width), (BATCH, width, 1, 1)); owned.append(input_tensor)
    before_tensor = graph.run(registry, input_tensor)
    before = before_tensor.to_f32(); before_tensor.close()
    program_identity = graph.program_identity()
    binding_identity = binding.identity()

    bundle = ProgramBundle.export(graph, registry, include_state=True)
    imported_registry = Registry(); owned.append(imported_registry)
    imported_graph = ProgramBundle.import_graph(imported_registry, bundle); owned.append(imported_graph)
    imported_binding = GraphParameterBinding.build(imported_graph, imported_registry); owned.append(imported_binding)
    after_tensor = imported_graph.run(imported_registry, input_tensor)
    after = after_tensor.to_f32(); after_tensor.close()

    if imported_graph.program_identity() != program_identity:
        raise AssertionError("largest-case ProgramBundle program identity mismatch")
    if imported_binding.identity() != binding_identity:
        raise AssertionError("largest-case ProgramBundle binding identity mismatch")
    if imported_binding.read_flat(imported_graph, imported_registry) != candidate:
        raise AssertionError("largest-case ProgramBundle flat state mismatch")
    if after != before:
        raise AssertionError("largest-case ProgramBundle output replay mismatch")
finally:
    close_all(owned)

# Remove bulky vectors from the durable report after semantic proof.
for item in results:
    item.pop("candidate", None)
    item.pop("reference_output", None)

report = {
    "verdict": "PASS",
    "consumer": "installed-wheel-typed-facade",
    "workload": "python-graph-parameter-binding-scaling-v1",
    "module_path": str(module_path),
    "samples": {
        "binding_build_repetitions": BUILD_REPS,
        "read_repetitions": READ_REPS,
        "candidate_evaluation_repetitions": EVAL_REPS,
        "batch": BATCH,
    },
    "cases": results,
    "largest_checkpoint": {
        "case": largest["name"],
        "parameter_count": largest["parameter_count"],
        "expected_owners": largest["expected_owners"],
        "bundle_bytes": len(bundle),
        "replay_exact": True,
    },
    "decision_boundary": {
        "timings_are_ci_thresholds": False,
        "binding_cache_added": False,
        "abi_changed": False,
        "core_semantics_changed": False,
    },
}
print(json.dumps(report, sort_keys=True))
'''


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="burn-research-binding-scaling-") as tmp:
        root = Path(tmp)
        wheels = root / "wheels"
        wheels.mkdir()

        run(
            [sys.executable, "-m", "maturin", "build", "--out", str(wheels)],
            cwd=FFI_DIR,
        )
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise SystemExit(f"expected exactly one wheel, found: {wheel_files}")
        wheel = wheel_files[0]

        venv = root / "venv"
        run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        venv_python = venv / "bin" / "python"
        if not venv_python.is_file():
            raise SystemExit(f"missing venv interpreter: {venv_python}")

        run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                str(wheel),
            ],
            cwd=root,
        )

        consumer = root / "consumer.py"
        consumer.write_text(CONSUMER, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["BR_REPO_ROOT"] = str(REPO.resolve())
        completed = run([str(venv_python), str(consumer)], cwd=root, env=env)
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("binding-scaling consumer produced no output")

        report = json.loads(lines[-1])
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
