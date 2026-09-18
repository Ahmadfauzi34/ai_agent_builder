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
REPORT = REPO / "python-nonlinear-es-depth-768-report.json"

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

SEED = 20_260_917
SIGMA = 0.15
LEARNING_RATE = 0.06
HORIZON = 16
POLICY_IN = 6
POLICY_HIDDEN = 8
POLICY_OUT = 2
EXPECTED_DIM = (POLICY_IN * POLICY_HIDDEN + POLICY_HIDDEN) + (
    POLICY_HIDDEN * POLICY_OUT + POLICY_OUT
)

BUDGETS = [
    ("C", 8, 48),
    ("D", 16, 48),
    ("E", 8, 96),
]

SCENARIOS = [
    (1.20, -0.80, 0.00, 0.00, 0.00, 0.00),
    (-1.00, 0.90, 0.20, -0.10, 0.30, -0.20),
    (0.70, 1.10, -0.10, 0.15, -0.50, 0.40),
    (-1.30, -0.60, 0.10, 0.05, 0.40, 0.30),
]


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
    linear_in = stack.enter_context(
        host.LinearLayerSpec(236_000, POLICY_IN, POLICY_HIDDEN, bias=True)
    )
    relu = stack.enter_context(host.ReluLayerSpec(236_001))
    linear_out = stack.enter_context(
        host.LinearLayerSpec(236_002, POLICY_HIDDEN, POLICY_OUT, bias=True)
    )
    for layer in (linear_in, relu, linear_out):
        registry.init_layer(layer)

    builder = stack.enter_context(host.GraphBuilder(4))
    builder.add_unary(linear_in, 0, 1)
    builder.add_unary(relu, 1, 2)
    builder.add_unary(linear_out, 2, 3)
    builder.set_output(3)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    if binding.total_len != EXPECTED_DIM:
        raise AssertionError(
            f"policy parameter dim {binding.total_len} != expected {EXPECTED_DIM}"
        )
    return registry, graph, binding


def policy_action(graph, registry, features: list[float]) -> tuple[float, float]:
    with host.Tensor.vector(features) as inp:
        with graph.run(registry, inp) as out:
            values = out.to_f32()
    if len(values) != POLICY_OUT or not all(math.isfinite(v) for v in values):
        raise AssertionError(f"invalid policy output: {values}")
    return clamp(values[0], -1.5, 1.5), clamp(values[1], -1.5, 1.5)


def rollout(graph, registry) -> float:
    total_cost = 0.0
    for scenario_index, (x0, y0, vx0, vy0, tx, ty) in enumerate(SCENARIOS):
        x, y, vx, vy = x0, y0, vx0, vy0
        prev_ax = 0.0
        prev_ay = 0.0
        for step in range(HORIZON):
            ex = x - tx
            ey = y - ty
            features = [ex, vx, ey, vy, prev_ax, prev_ay]
            ax, ay = policy_action(graph, registry, features)

            expert_ax = clamp(-0.80 * ex - 0.40 * vx, -1.5, 1.5)
            expert_ay = clamp(-0.80 * ey - 0.40 * vy, -1.5, 1.5)

            state_cost = ex * ex + ey * ey + 0.15 * (vx * vx + vy * vy)
            action_cost = 0.02 * (ax * ax + ay * ay)
            imitation_cost = 0.25 * (
                (ax - expert_ax) * (ax - expert_ax)
                + (ay - expert_ay) * (ay - expert_ay)
            )
            total_cost += state_cost + action_cost + imitation_cost

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

        terminal_ex = x - tx
        terminal_ey = y - ty
        total_cost += 0.75 * (terminal_ex * terminal_ex + terminal_ey * terminal_ey)

    normalized_cost = total_cost / (len(SCENARIOS) * HORIZON)
    reward = -normalized_cost
    if not math.isfinite(reward):
        raise AssertionError(f"non-finite rollout reward: {reward}")
    return reward


def train_case(
    *,
    label: str,
    population: int,
    generations: int,
    graph,
    registry,
    binding,
    zero_policy_reward: float,
    program_identity: str,
    binding_identity: str,
):
    zero = array("f", [0.0]) * binding.total_len
    binding.apply_flat(graph, registry, zero)
    reset_reward = rollout(graph, registry)
    if abs(reset_reward - zero_policy_reward) > 1e-12:
        raise AssertionError(
            f"case {label} zero-policy reset mismatch: {reset_reward} vs {zero_policy_reward}"
        )

    ask_samples: list[int] = []
    apply_samples: list[int] = []
    rollout_samples: list[int] = []
    tell_samples: list[int] = []
    champion_rewards: list[float] = []
    generation_best_rewards: list[float] = []
    report_flags: list[list[str]] = []
    observed_batch = None

    training_start = ns()
    with host.EsOptimizer.strict(
        binding.total_len,
        strategy=0,
        seed=SEED,
        population=population,
        sigma=SIGMA,
        learning_rate=LEARNING_RATE,
    ) as optimizer:
        if optimizer.batch_size != 0:
            raise AssertionError(f"case {label}: new optimizer batch_size must be zero")

        for generation in range(generations):
            t0 = ns()
            candidates = optimizer.ask_f32()
            t1 = ns()
            ask_samples.append(t1 - t0)

            if not isinstance(candidates, array) or candidates.typecode != "f":
                raise AssertionError(f"case {label}: ask_f32 did not return array('f')")
            batch = optimizer.batch_size
            if batch <= 0 or len(candidates) != batch * binding.total_len:
                raise AssertionError(f"case {label}: ask_f32 cardinality mismatch")
            if observed_batch is None:
                observed_batch = batch
            elif batch != observed_batch:
                raise AssertionError(
                    f"case {label}: optimizer batch_size changed across generations"
                )

            flat_view = memoryview(candidates)
            if (
                flat_view.format != "f"
                or flat_view.itemsize != 4
                or flat_view.ndim != 1
                or not flat_view.c_contiguous
            ):
                raise AssertionError(
                    f"case {label}: ask_f32 storage is not contiguous native f32"
                )

            fitness: list[float] = []
            for index in range(batch):
                start = index * binding.total_len
                candidate = flat_view[start : start + binding.total_len]
                if len(candidate) != binding.total_len or not candidate.c_contiguous:
                    raise AssertionError(
                        f"case {label}: candidate window contract failed"
                    )

                a0 = ns()
                binding.apply_flat(graph, registry, candidate)
                a1 = ns()
                apply_samples.append(a1 - a0)

                r0 = ns()
                reward = rollout(graph, registry)
                r1 = ns()
                rollout_samples.append(r1 - r0)
                fitness.append(reward)

            if not all(math.isfinite(value) for value in fitness):
                raise AssertionError(f"case {label}: non-finite generation fitness")
            generation_best_rewards.append(max(fitness))

            q0 = ns()
            generation_report = optimizer.tell(fitness)
            q1 = ns()
            tell_samples.append(q1 - q0)
            if int(generation_report["gen"]) != generation + 1:
                raise AssertionError(
                    f"case {label}: optimizer generation counter mismatch"
                )
            if int(generation_report["evals"]) != batch:
                raise AssertionError(
                    f"case {label}: optimizer report eval count mismatch"
                )
            if optimizer.batch_size != batch:
                raise AssertionError(
                    f"case {label}: batch cardinality changed after tell"
                )
            report_flags.append(list(generation_report.get("flags", [])))

            champion = optimizer.best()
            if len(champion) != binding.total_len or not all(
                math.isfinite(value) for value in champion
            ):
                raise AssertionError(f"case {label}: invalid optimizer champion")
            binding.apply_flat(graph, registry, array("f", champion))
            champion_reward = rollout(graph, registry)
            champion_rewards.append(champion_reward)

            if graph.program_identity() != program_identity:
                raise AssertionError(
                    f"case {label}: program identity changed during training"
                )
            if binding.identity() != binding_identity:
                raise AssertionError(
                    f"case {label}: binding identity changed during training"
                )

        if observed_batch != population:
            raise AssertionError(
                f"case {label}: observed batch {observed_batch} != requested {population}"
            )

        final_params = optimizer.best()

    training_end = ns()
    if len(final_params) != binding.total_len:
        raise AssertionError(f"case {label}: final parameter length mismatch")

    binding.apply_flat(graph, registry, array("f", final_params))
    final_reward = rollout(graph, registry)
    if abs(final_reward - champion_rewards[-1]) > 1e-9:
        raise AssertionError(f"case {label}: final reward is not reproducible")

    initial_champion_reward = champion_rewards[0]
    baseline_cost = -zero_policy_reward
    initial_cost = -initial_champion_reward
    final_cost = -final_reward
    baseline_gain = baseline_cost - final_cost
    training_gain = initial_cost - final_cost
    baseline_gain_ratio = baseline_gain / max(abs(baseline_cost), 1e-12)
    training_gain_ratio = training_gain / max(abs(initial_cost), 1e-12)

    expected_evaluations = population * generations
    actual_evaluations = observed_batch * generations
    if actual_evaluations != expected_evaluations:
        raise AssertionError(
            f"case {label}: evaluation count {actual_evaluations} != "
            f"expected {expected_evaluations}"
        )

    return {
        "label": label,
        "population": population,
        "generations": generations,
        "candidate_evaluations": actual_evaluations,
        "zero_policy_reward": zero_policy_reward,
        "initial_champion_reward": initial_champion_reward,
        "final_champion_reward": final_reward,
        "baseline_gain_ratio": baseline_gain_ratio,
        "training_gain_ratio": training_gain_ratio,
        "generation_best": generation_best_rewards,
        "champion_history": champion_rewards,
        "optimizer_flags": report_flags,
        "timing": {
            "ask_f32": summary(ask_samples),
            "apply_candidate": summary(apply_samples),
            "rollout_candidate": summary(rollout_samples),
            "tell": summary(tell_samples),
            "training_wall_ms": (training_end - training_start) / 1_000_000.0,
            "timing_is_ci_threshold": False,
        },
        "final_params": [float(value) for value in final_params],
        "proofs": {
            "finite_fitness": True,
            "batch_cardinality_stable": True,
            "candidate_evaluation_count_exact": True,
            "program_identity_stable": True,
            "binding_identity_stable": True,
            "final_reward_reproducible_within_1e-9": True,
        },
    }


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from repo: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"research imported host from repo: {host_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected host/ABI identity")
if br.ReluLayerSpec is not host.ReluLayerSpec:
    raise AssertionError("typed ReLU host export mismatch")

with ExitStack() as stack:
    registry, graph, binding = build_policy(stack)
    program_identity = graph.program_identity()
    binding_identity = binding.identity()

    zero = array("f", [0.0]) * binding.total_len
    binding.apply_flat(graph, registry, zero)
    zero_policy_reward = rollout(graph, registry)

    cases = []
    for label, population, generations in BUDGETS:
        cases.append(
            train_case(
                label=label,
                population=population,
                generations=generations,
                graph=graph,
                registry=registry,
                binding=binding,
                zero_policy_reward=zero_policy_reward,
                program_identity=program_identity,
                binding_identity=binding_identity,
            )
        )

    selected = max(cases, key=lambda case: case["final_champion_reward"])
    selected_params = array("f", selected["final_params"])
    binding.apply_flat(graph, registry, selected_params)
    selected_reward = rollout(graph, registry)
    if abs(selected_reward - selected["final_champion_reward"]) > 1e-9:
        raise AssertionError("selected case reward mismatch before checkpoint")

    final_state = binding.read_flat(graph, registry)
    bundle = host.ProgramBundle.export(graph, registry, include_state=True)
    if not bundle:
        raise AssertionError("empty stateful ProgramBundle")

    replay_registry = stack.enter_context(host.Registry())
    replay_graph = stack.enter_context(
        host.ProgramBundle.import_graph(replay_registry, bundle)
    )
    replay_binding = stack.enter_context(
        host.GraphParameterBinding.build(replay_graph, replay_registry)
    )
    if replay_graph.program_identity() != program_identity:
        raise AssertionError("replay program identity mismatch")
    if replay_binding.identity() != binding_identity:
        raise AssertionError("replay binding identity mismatch")
    if replay_binding.total_len != EXPECTED_DIM:
        raise AssertionError("replay binding length mismatch")
    if replay_binding.read_flat(replay_graph, replay_registry) != final_state:
        raise AssertionError("replay parameter state mismatch")

    replay_reward = rollout(replay_graph, replay_registry)
    if abs(replay_reward - selected_reward) > 1e-9:
        raise AssertionError(
            f"replay reward mismatch: {replay_reward} vs {selected_reward}"
        )

for case in cases:
    case.pop("final_params")

case_by_label = {case["label"]: case for case in cases}
c_reward = case_by_label["C"]["final_champion_reward"]
d_reward = case_by_label["D"]["final_champion_reward"]
e_reward = case_by_label["E"]["final_champion_reward"]

equal_768_preference = (
    "DEPTH"
    if e_reward > d_reward
    else "BREADTH"
    if d_reward > e_reward
    else "TIE"
)
depth_gain_48_to_96 = e_reward - c_reward
equal_768_depth_minus_breadth = e_reward - d_reward

if e_reward > d_reward and e_reward > c_reward:
    decision = "DEPTH_REMAINS_STRONGER_AT_EQUAL_768_EVALUATIONS"
elif d_reward > e_reward:
    decision = "BREADTH_BECOMES_STRONGER_AT_EQUAL_768_EVALUATIONS"
else:
    decision = "DEPTH_SATURATION_OR_TIE_AT_768_EVALUATIONS"

report = {
    "verdict": "PASS",
    "decision": decision,
    "decision_is_ci_threshold": False,
    "host_api": host.HOST_API_SCHEMA,
    "abi_version": host.abi_version(),
    "workload": {
        "kind": "deterministic_2d_control_rollout",
        "policy": "Linear(6 -> 8, bias=true) -> ReLU -> Linear(8 -> 2, bias=true)",
        "parameter_dim": EXPECTED_DIM,
        "scenarios": len(SCENARIOS),
        "horizon": HORIZON,
        "seed": SEED,
        "sigma": SIGMA,
        "learning_rate": LEARNING_RATE,
    },
    "budget_matrix": [
        {
            "label": label,
            "population": population,
            "generations": generations,
            "candidate_evaluations": population * generations,
        }
        for label, population, generations in BUDGETS
    ],
    "cases": cases,
    "comparison": {
        "depth_baseline_case": "C",
        "breadth_768_case": "D",
        "depth_768_case": "E",
        "C_final_reward": c_reward,
        "D_final_reward": d_reward,
        "E_final_reward": e_reward,
        "E_minus_C_reward": depth_gain_48_to_96,
        "E_minus_D_reward": equal_768_depth_minus_breadth,
        "equal_768_preference": equal_768_preference,
        "selected_best_case": selected["label"],
        "selected_final_reward": selected["final_champion_reward"],
    },
    "replay": {
        "selected_case": selected["label"],
        "reward_before_export": selected_reward,
        "reward_after_import": replay_reward,
        "checkpoint_bytes": len(bundle),
    },
    "proofs": {
        "installed_wheel": True,
        "typed_relu_surface": True,
        "canonical_parameter_dim_74": True,
        "independent_optimizer_runs_same_seed": True,
        "same_zero_policy_baseline_each_case": True,
        "ask_f32_native_f32": True,
        "candidate_windows_contiguous": True,
        "finite_fitness_all_cases": True,
        "batch_cardinality_stable_all_cases": True,
        "program_identity_stable_all_cases": True,
        "binding_identity_stable_all_cases": True,
        "stateful_checkpoint_replay_selected_case": True,
        "replay_parameter_state_exact": True,
        "replay_reward_exact_within_1e-9": True,
    },
    "notes": [
        "Only population and generation count vary across cases C-D-E.",
        "D and E both use 768 candidate evaluations, directly comparing breadth versus depth at equal total search budget.",
        "C and E hold population at 8 and double generation depth from 48 to 96.",
        "Training-quality decisions are research interpretation only and are not CI pass/fail thresholds.",
        "No optimizer algorithm, hyperparameter, ABI symbol, activation surface, graph semantic, or support claim changes in this slice.",
    ],
}
print(json.dumps(report, sort_keys=True))
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


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="burn-research-nonlinear-es-depth-768-") as tmp:
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
        env["BR_REPO_ROOT"] = str(REPO.resolve())
        completed = run([str(python), str(consumer)], cwd=root, env=env)
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("nonlinear ES depth-768 consumer produced no output")
        payload = json.loads(lines[-1])
        if payload.get("verdict") != "PASS":
            raise SystemExit(f"unexpected nonlinear ES depth-768 verdict: {payload}")

    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
