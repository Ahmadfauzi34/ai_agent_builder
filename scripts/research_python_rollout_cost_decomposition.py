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
REPORT = REPO / "python-rollout-cost-decomposition-report.json"

CONSUMER = r'''
from __future__ import annotations

from array import array
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import time

import burn_research_ffi as br
from burn_research_ffi import host

POLICY_IN = 6
POLICY_OUT = 2
EXPECTED_DIM = POLICY_IN * POLICY_OUT + POLICY_OUT
HORIZON = 16
MICRO_REPS = 320
ROLLOUT_REPS = 20

SCENARIOS = [
    (1.20, -0.80, 0.00, 0.00, 0.00, 0.00),
    (-1.00, 0.90, 0.20, -0.10, 0.30, -0.20),
    (0.70, 1.10, -0.10, 0.15, -0.50, 0.40),
    (-1.30, -0.60, 0.10, 0.05, 0.40, 0.30),
]
STEPS_PER_ROLLOUT = len(SCENARIOS) * HORIZON


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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_policy(stack: ExitStack):
    registry = stack.enter_context(host.Registry())
    layer = stack.enter_context(
        host.LinearLayerSpec(211_000, POLICY_IN, POLICY_OUT, bias=True)
    )
    registry.init_layer(layer)
    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(layer, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    if binding.total_len != EXPECTED_DIM:
        raise AssertionError(
            f"policy parameter dim {binding.total_len} != expected {EXPECTED_DIM}"
        )
    zero = array("f", [0.0]) * binding.total_len
    binding.apply_flat(graph, registry, zero)
    return registry, graph, binding


def policy_action(graph, registry, features: list[float]) -> tuple[float, float]:
    with host.Tensor.vector(features) as inp:
        with graph.run(registry, inp) as out:
            values = out.to_f32()
    if len(values) != POLICY_OUT or not all(math.isfinite(v) for v in values):
        raise AssertionError(f"invalid policy output: {values}")
    return clamp(values[0], -1.5, 1.5), clamp(values[1], -1.5, 1.5)


def rollout_graph(graph, registry) -> float:
    total_cost = 0.0
    for scenario_index, (x0, y0, vx0, vy0, tx, ty) in enumerate(SCENARIOS):
        x, y, vx, vy = x0, y0, vx0, vy0
        prev_ax = 0.0
        prev_ay = 0.0
        for step in range(HORIZON):
            ex = x - tx
            ey = y - ty
            ax, ay = policy_action(
                graph, registry, [ex, vx, ey, vy, prev_ax, prev_ay]
            )
            expert_ax = clamp(-0.80 * ex - 0.40 * vx, -1.5, 1.5)
            expert_ay = clamp(-0.80 * ey - 0.40 * vy, -1.5, 1.5)
            total_cost += (
                ex * ex
                + ey * ey
                + 0.15 * (vx * vx + vy * vy)
                + 0.02 * (ax * ax + ay * ay)
                + 0.25
                * ((ax - expert_ax) ** 2 + (ay - expert_ay) ** 2)
            )
            disturbance_x = 0.012 * math.sin(
                (scenario_index + 1) * (step + 1) * 0.61
            )
            disturbance_y = 0.010 * math.cos(
                (scenario_index + 2) * (step + 1) * 0.47
            )
            vx = 0.84 * vx + 0.24 * ax + disturbance_x
            vy = 0.84 * vy + 0.24 * ay + disturbance_y
            x += vx
            y += vy
            prev_ax, prev_ay = ax, ay
        total_cost += 0.75 * ((x - tx) ** 2 + (y - ty) ** 2)
    reward = -(total_cost / STEPS_PER_ROLLOUT)
    if not math.isfinite(reward):
        raise AssertionError("non-finite graph rollout reward")
    return reward


def rollout_python_only() -> float:
    total_cost = 0.0
    for scenario_index, (x0, y0, vx0, vy0, tx, ty) in enumerate(SCENARIOS):
        x, y, vx, vy = x0, y0, vx0, vy0
        prev_ax = 0.0
        prev_ay = 0.0
        for step in range(HORIZON):
            ex = x - tx
            ey = y - ty
            ax = clamp(-0.80 * ex - 0.40 * vx, -1.5, 1.5)
            ay = clamp(-0.80 * ey - 0.40 * vy, -1.5, 1.5)
            total_cost += (
                ex * ex
                + ey * ey
                + 0.15 * (vx * vx + vy * vy)
                + 0.02 * (ax * ax + ay * ay)
            )
            disturbance_x = 0.012 * math.sin(
                (scenario_index + 1) * (step + 1) * 0.61
            )
            disturbance_y = 0.010 * math.cos(
                (scenario_index + 2) * (step + 1) * 0.47
            )
            vx = 0.84 * vx + 0.24 * ax + disturbance_x
            vy = 0.84 * vy + 0.24 * ay + disturbance_y
            x += vx
            y += vy
            prev_ax, prev_ay = ax, ay
        total_cost += 0.75 * ((x - tx) ** 2 + (y - ty) ** 2)
    reward = -(total_cost / STEPS_PER_ROLLOUT)
    if not math.isfinite(reward):
        raise AssertionError("non-finite Python-only rollout reward")
    return reward


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from repo: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"research imported host from repo: {host_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected host/ABI identity")

with ExitStack() as stack:
    registry, graph, binding = build_policy(stack)
    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    features = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0]

    baseline_reward = rollout_graph(graph, registry)
    python_only_reward = rollout_python_only()

    tensor_samples: list[int] = []
    for _ in range(MICRO_REPS):
        t0 = ns()
        with host.Tensor.vector(features) as inp:
            if inp.length != POLICY_IN:
                raise AssertionError("unexpected input tensor length")
        t1 = ns()
        tensor_samples.append(t1 - t0)

    graph_samples: list[int] = []
    with host.Tensor.vector(features) as shared_input:
        for _ in range(MICRO_REPS):
            t0 = ns()
            with graph.run(registry, shared_input) as out:
                values = out.to_f32()
            t1 = ns()
            if len(values) != POLICY_OUT or not all(math.isfinite(v) for v in values):
                raise AssertionError("invalid graph microbenchmark output")
            graph_samples.append(t1 - t0)

    action_samples: list[int] = []
    for _ in range(MICRO_REPS):
        t0 = ns()
        ax, ay = policy_action(graph, registry, features)
        t1 = ns()
        if not math.isfinite(ax) or not math.isfinite(ay):
            raise AssertionError("non-finite policy action")
        action_samples.append(t1 - t0)

    env_samples: list[int] = []
    graph_rollout_samples: list[int] = []
    for _ in range(ROLLOUT_REPS):
        t0 = ns()
        reward = rollout_python_only()
        t1 = ns()
        if reward != python_only_reward:
            raise AssertionError("Python-only rollout is not deterministic")
        env_samples.append(t1 - t0)

        r0 = ns()
        graph_reward = rollout_graph(graph, registry)
        r1 = ns()
        if graph_reward != baseline_reward:
            raise AssertionError("graph rollout is not deterministic")
        graph_rollout_samples.append(r1 - r0)

    final_reward = rollout_graph(graph, registry)
    if final_reward != baseline_reward:
        raise AssertionError("graph rollout reward changed after timing loops")
    if graph.program_identity() != program_identity:
        raise AssertionError("program identity changed during timing loops")
    if binding.identity() != binding_identity:
        raise AssertionError("binding identity changed during timing loops")

summaries = {
    "python_environment_rollout": summary(env_samples),
    "tensor_vector_construct_close": summary(tensor_samples),
    "graph_run_output_copy": summary(graph_samples),
    "full_policy_action": summary(action_samples),
    "full_graph_rollout": summary(graph_rollout_samples),
}
full_rollout_ms = summaries["full_graph_rollout"]["median_ms"]
full_action_ms = summaries["full_policy_action"]["median_ms"]
env_rollout_ms = summaries["python_environment_rollout"]["median_ms"]
tensor_ms = summaries["tensor_vector_construct_close"]["median_ms"]
graph_ms = summaries["graph_run_output_copy"]["median_ms"]

env_ratio = env_rollout_ms / max(full_rollout_ms, 1e-12)
tensor_action_ratio = tensor_ms / max(full_action_ms, 1e-12)
graph_action_ratio = graph_ms / max(full_action_ms, 1e-12)

if env_ratio >= 0.55:
    decision = "PYTHON_ENVIRONMENT_DOMINATES"
elif graph_action_ratio >= 0.55:
    decision = "GRAPH_RUN_OUTPUT_BOUNDARY_DOMINATES"
elif tensor_action_ratio >= 0.55:
    decision = "TENSOR_INPUT_MARSHALLING_DOMINATES"
else:
    decision = "MIXED_ROLLOUT_COST"

report = {
    "verdict": "PASS",
    "decision": decision,
    "decision_is_ci_threshold": False,
    "host_api": host.HOST_API_SCHEMA,
    "abi_version": host.abi_version(),
    "workload": {
        "policy": "Linear(6 -> 2, bias=true)",
        "parameter_dim": EXPECTED_DIM,
        "scenarios": len(SCENARIOS),
        "horizon": HORIZON,
        "steps_per_rollout": STEPS_PER_ROLLOUT,
        "micro_repetitions": MICRO_REPS,
        "rollout_repetitions": ROLLOUT_REPS,
    },
    "reward": {
        "graph_zero_policy": baseline_reward,
        "python_expert_policy": python_only_reward,
        "graph_recheck": final_reward,
    },
    "timing": summaries,
    "ratios": {
        "python_environment_vs_full_rollout": env_ratio,
        "tensor_construct_vs_full_policy_action": tensor_action_ratio,
        "graph_run_output_vs_full_policy_action": graph_action_ratio,
    },
    "proofs": {
        "installed_wheel": True,
        "finite_outputs": True,
        "deterministic_graph_rollout": True,
        "deterministic_python_rollout": True,
        "program_identity_stable": True,
        "binding_identity_stable": True,
        "owned_handles_closed": True,
    },
    "notes": [
        "Classification heuristics are descriptive research only, not performance SLAs.",
        "graph_run_output_copy includes native graph execution, FFI crossing, output handle creation, and output copy; this test does not pretend to separate those without native instrumentation.",
        "No ABI, facade, graph, tensor, optimizer, or Math primitive change is introduced.",
    ],
}
print(json.dumps(report, sort_keys=True))
'''


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None):
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
    with tempfile.TemporaryDirectory(prefix="burn-research-rollout-decomp-") as tmp:
        root = Path(tmp)
        wheels = root / "wheels"
        wheels.mkdir()
        run(
            [sys.executable, "-m", "maturin", "build", "--out", str(wheels)],
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

        consumer = root / "consumer.py"
        consumer.write_text(CONSUMER, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["BR_REPO_ROOT"] = str(REPO)
        completed = run([str(python), str(consumer)], cwd=root, env=env)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("consumer produced no report")
        report = json.loads(lines[-1])
        if report.get("verdict") != "PASS":
            raise SystemExit(f"unexpected report verdict: {report}")
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
