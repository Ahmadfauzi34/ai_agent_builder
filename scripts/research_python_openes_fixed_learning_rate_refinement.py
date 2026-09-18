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
REPORT = REPO / "python-openes-fixed-learning-rate-refinement-report.json"

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
POPULATION = 8
GENERATIONS = 192
HORIZON = 16
POLICY_IN = 6
POLICY_HIDDEN = 8
POLICY_OUT = 2
EXPECTED_DIM = (POLICY_IN * POLICY_HIDDEN + POLICY_HIDDEN) + (
    POLICY_HIDDEN * POLICY_OUT + POLICY_OUT
)

CASES = [
    ("LR_008", 0.08, 0.08),
    ("LR_010", 0.08, 0.10),
    ("LR_012", 0.08, 0.12),
]
MILESTONES = (48, 96, 144, 192)
INTERVALS = ((1, 48), (49, 96), (97, 144), (145, 192))
EXPECTED_BASE_MILESTONE_REWARDS = {
    48: -0.48940773015802336,
    96: -0.3302926615428852,
    144: -0.3082627693620379,
    192: -0.3046468133328745,
}
MILESTONE_TOLERANCE = 5e-6

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
        host.LinearLayerSpec(237_000, POLICY_IN, POLICY_HIDDEN, bias=True)
    )
    relu = stack.enter_context(host.ReluLayerSpec(237_001))
    linear_out = stack.enter_context(
        host.LinearLayerSpec(237_002, POLICY_HIDDEN, POLICY_OUT, bias=True)
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


def plateau_metrics(
    champion_history: list[float],
    report_flags: list[list[str]],
) -> dict[str, object]:
    if not champion_history:
        raise AssertionError("empty champion history")

    improvement_generations: list[int] = []
    longest_no_improvement = 0
    current_no_improvement = 0
    for index in range(1, len(champion_history)):
        if champion_history[index] > champion_history[index - 1]:
            generation = index + 1
            improvement_generations.append(generation)
            current_no_improvement = 0
        else:
            current_no_improvement += 1
            longest_no_improvement = max(
                longest_no_improvement,
                current_no_improvement,
            )

    no_improvement_flag_count = sum(
        1
        for flags in report_flags
        if "NO_IMPROVEMENT" in flags
    )
    return {
        "champion_improvement_count": len(improvement_generations),
        "improvement_generations": improvement_generations,
        "first_improvement_generation": (
            improvement_generations[0] if improvement_generations else None
        ),
        "last_improvement_generation": (
            improvement_generations[-1] if improvement_generations else None
        ),
        "longest_consecutive_no_improvement_generations": longest_no_improvement,
        "no_improvement_flag_count": no_improvement_flag_count,
    }


def train_case(
    stack: ExitStack,
    label: str,
    sigma: float,
    learning_rate: float,
    graph,
    registry,
    binding,
    program_identity: str,
    binding_identity: str,
    zero_policy_reward: float,
) -> tuple[dict[str, object], list[float]]:
    zero = array("f", [0.0]) * binding.total_len
    binding.apply_flat(graph, registry, zero)
    zero_recheck = rollout(graph, registry)
    if abs(zero_recheck - zero_policy_reward) > 1e-9:
        raise AssertionError(
            f"case {label}: zero-policy reset mismatch {zero_recheck} vs {zero_policy_reward}"
        )

    optimizer = stack.enter_context(
        host.EsOptimizer.strict(
            binding.total_len,
            strategy=0,
            seed=SEED,
            population=POPULATION,
            sigma=sigma,
            learning_rate=learning_rate,
        )
    )
    if optimizer.batch_size != 0:
        raise AssertionError(f"case {label}: new optimizer batch_size must be zero")

    ask_samples: list[int] = []
    apply_samples: list[int] = []
    rollout_samples: list[int] = []
    tell_samples: list[int] = []
    champion_rewards: list[float] = []
    generation_best_rewards: list[float] = []
    report_flags: list[list[str]] = []
    telemetry_history: list[dict[str, float | int]] = []
    evaluation_count = 0
    observed_batch = None

    wall_start = ns()
    for generation in range(GENERATIONS):
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
            raise AssertionError(f"case {label}: batch_size changed across generations")

        flat_view = memoryview(candidates)
        if (
            flat_view.format != "f"
            or flat_view.itemsize != 4
            or flat_view.ndim != 1
            or not flat_view.c_contiguous
        ):
            raise AssertionError(f"case {label}: ask_f32 storage contract failed")

        fitness: list[float] = []
        for index in range(batch):
            start = index * binding.total_len
            candidate = flat_view[start : start + binding.total_len]
            if len(candidate) != binding.total_len or not candidate.c_contiguous:
                raise AssertionError(f"case {label}: candidate window contract failed")

            a0 = ns()
            binding.apply_flat(graph, registry, candidate)
            a1 = ns()
            apply_samples.append(a1 - a0)

            r0 = ns()
            reward = rollout(graph, registry)
            r1 = ns()
            rollout_samples.append(r1 - r0)
            fitness.append(reward)
            evaluation_count += 1

        if not all(math.isfinite(value) for value in fitness):
            raise AssertionError(f"case {label}: non-finite generation fitness")
        generation_best_rewards.append(max(fitness))

        q0 = ns()
        report = optimizer.tell(fitness)
        q1 = ns()
        tell_samples.append(q1 - q0)
        if int(report["gen"]) != generation + 1:
            raise AssertionError(f"case {label}: generation counter mismatch")
        if int(report["evals"]) != batch:
            raise AssertionError(f"case {label}: report eval count mismatch")
        if optimizer.batch_size != batch:
            raise AssertionError(f"case {label}: batch_size changed after tell")

        required_numeric_fields = (
            "best",
            "worst",
            "mean",
            "std",
            "diversity",
            "sigma",
            "lr",
            "mean_norm",
            "best_norm",
        )
        for field in required_numeric_fields:
            value = float(report[field])
            if not math.isfinite(value):
                raise AssertionError(
                    f"case {label}: telemetry field {field} is non-finite: {value}"
                )
        if int(report["dim"]) != binding.total_len:
            raise AssertionError(f"case {label}: telemetry dim mismatch")
        if int(report["pop"]) != batch:
            raise AssertionError(f"case {label}: telemetry population mismatch")
        if abs(float(report["sigma"]) - sigma) > 1e-6:
            raise AssertionError(f"case {label}: telemetry sigma drift")
        if abs(float(report["lr"]) - learning_rate) > 1e-6:
            raise AssertionError(f"case {label}: telemetry learning-rate drift")
        if abs(float(report["best"]) - generation_best_rewards[-1]) > 1e-6:
            raise AssertionError(f"case {label}: report generation-best mismatch")
        if batch % 2 != 0:
            raise AssertionError(f"case {label}: antithetic batch must be even")

        pair_abs_diffs = [
            abs(fitness[index] - fitness[index + 1])
            for index in range(0, batch, 2)
        ]
        fitness_scale = max(float(report["std"]), 1e-8)
        normalized_pair_abs_diffs = [
            value / fitness_scale for value in pair_abs_diffs
        ]

        report_flags.append(list(report.get("flags", [])))
        telemetry_history.append(
            {
                "generation": int(report["gen"]),
                "fitness_std": float(report["std"]),
                "diversity": float(report["diversity"]),
                "population_mean_fitness": float(report["mean"]),
                "generation_best_fitness": float(report["best"]),
                "stagnation": int(report["stagnation"]),
                "mean_norm": float(report["mean_norm"]),
                "best_norm": float(report["best_norm"]),
                "normalized_antithetic_contrast_median": statistics.median(
                    normalized_pair_abs_diffs
                ),
            }
        )

        champion = optimizer.best()
        if len(champion) != binding.total_len or not all(
            math.isfinite(value) for value in champion
        ):
            raise AssertionError(f"case {label}: invalid optimizer champion")
        binding.apply_flat(graph, registry, array("f", champion))
        champion_reward = rollout(graph, registry)
        champion_rewards.append(champion_reward)

        if graph.program_identity() != program_identity:
            raise AssertionError(f"case {label}: program identity changed")
        if binding.identity() != binding_identity:
            raise AssertionError(f"case {label}: binding identity changed")

    wall_end = ns()

    if observed_batch != POPULATION:
        raise AssertionError(
            f"case {label}: observed batch {observed_batch} != requested {POPULATION}"
        )
    expected_evaluations = POPULATION * GENERATIONS
    if evaluation_count != expected_evaluations:
        raise AssertionError(
            f"case {label}: evaluations {evaluation_count} != expected {expected_evaluations}"
        )

    best_params = optimizer.best()
    if len(best_params) != binding.total_len:
        raise AssertionError(f"case {label}: best parameter length mismatch")
    binding.apply_flat(graph, registry, array("f", best_params))
    final_reward_recheck = rollout(graph, registry)
    final_champion_reward = champion_rewards[-1]
    if abs(final_reward_recheck - final_champion_reward) > 1e-9:
        raise AssertionError(f"case {label}: final reward is not reproducible")

    initial_champion_reward = champion_rewards[0]
    baseline_cost = -zero_policy_reward
    initial_cost = -initial_champion_reward
    final_cost = -final_champion_reward
    baseline_gain_ratio = (
        (baseline_cost - final_cost) / max(abs(baseline_cost), 1e-12)
    )
    training_gain_ratio = (
        (initial_cost - final_cost) / max(abs(initial_cost), 1e-12)
    )

    stagnation = plateau_metrics(champion_rewards, report_flags)
    improvement_generations = list(stagnation["improvement_generations"])

    milestone_rewards = {
        str(generation): champion_rewards[generation - 1]
        for generation in MILESTONES
    }
    if label == "LR_010":
        for generation, expected in EXPECTED_BASE_MILESTONE_REWARDS.items():
            actual = champion_rewards[generation - 1]
            if abs(actual - expected) > MILESTONE_TOLERANCE:
                raise AssertionError(
                    f"case {label}: milestone {generation} reward drift "
                    f"{actual} vs expected {expected}"
                )

    interval_reports: list[dict[str, object]] = []
    for interval_start, interval_end in INTERVALS:
        telemetry_slice = telemetry_history[interval_start - 1 : interval_end]
        if len(telemetry_slice) != interval_end - interval_start + 1:
            raise AssertionError(f"case {label}: incomplete telemetry interval")
        interval_improvements = [
            generation
            for generation in improvement_generations
            if interval_start <= generation <= interval_end
        ]
        mean_norm_values = [
            float(item["mean_norm"]) for item in telemetry_slice
        ]
        interval_reports.append(
            {
                "start_generation": interval_start,
                "end_generation": interval_end,
                "champion_reward_at_end": champion_rewards[interval_end - 1],
                "champion_improvement_count": len(interval_improvements),
                "fitness_std_median": statistics.median(
                    float(item["fitness_std"]) for item in telemetry_slice
                ),
                "diversity_median": statistics.median(
                    float(item["diversity"]) for item in telemetry_slice
                ),
                "normalized_antithetic_contrast_median": statistics.median(
                    float(item["normalized_antithetic_contrast_median"])
                    for item in telemetry_slice
                ),
                "max_stagnation": max(
                    int(item["stagnation"]) for item in telemetry_slice
                ),
                "mean_norm_path_length": sum(
                    abs(current - previous)
                    for previous, current in zip(
                        mean_norm_values, mean_norm_values[1:]
                    )
                ),
            }
        )

    return (
        {
            "label": label,
            "sigma": sigma,
            "learning_rate": learning_rate,
            "population": POPULATION,
            "generations": GENERATIONS,
            "candidate_evaluations": evaluation_count,
            "zero_policy_reward": zero_policy_reward,
            "initial_champion_reward": initial_champion_reward,
            "final_champion_reward": final_champion_reward,
            "baseline_gain_ratio": baseline_gain_ratio,
            "training_gain_ratio": training_gain_ratio,
            "generation_best": generation_best_rewards,
            "champion_history": champion_rewards,
            "optimizer_flags": report_flags,
            "stagnation": stagnation,
            "milestone_champion_rewards": milestone_rewards,
            "intervals": interval_reports,
            "timing": {
                "ask_f32": summary(ask_samples),
                "apply_candidate": summary(apply_samples),
                "rollout_candidate": summary(rollout_samples),
                "tell": summary(tell_samples),
                "training_wall_ms": (wall_end - wall_start) / 1_000_000.0,
                "timing_is_ci_threshold": False,
            },
        },
        [float(value) for value in best_params],
    )


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

    case_reports: list[dict[str, object]] = []
    case_params: dict[str, list[float]] = {}
    for label, sigma, learning_rate in CASES:
        case_report, best_params = train_case(
            stack,
            label,
            sigma,
            learning_rate,
            graph,
            registry,
            binding,
            program_identity,
            binding_identity,
            zero_policy_reward,
        )
        case_reports.append(case_report)
        case_params[label] = best_params

    selected = max(
        case_reports,
        key=lambda item: float(item["final_champion_reward"]),
    )
    selected_label = str(selected["label"])
    selected_params = case_params[selected_label]
    binding.apply_flat(graph, registry, array("f", selected_params))
    selected_reward = rollout(graph, registry)
    if abs(selected_reward - float(selected["final_champion_reward"])) > 1e-9:
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

case_by_label = {str(item["label"]): item for item in case_reports}
lr008_reward = float(case_by_label["LR_008"]["final_champion_reward"])
lr010_reward = float(case_by_label["LR_010"]["final_champion_reward"])
lr012_reward = float(case_by_label["LR_012"]["final_champion_reward"])

final_leader = max(
    ("LR_008", "LR_010", "LR_012"),
    key=lambda label: float(case_by_label[label]["final_champion_reward"]),
)
early_leader = max(
    ("LR_008", "LR_010", "LR_012"),
    key=lambda label: float(
        case_by_label[label]["milestone_champion_rewards"]["48"]
    ),
)

if final_leader == "LR_008":
    sensitivity_pattern = "LR_008_LEADS_AT_192"
elif final_leader == "LR_012":
    sensitivity_pattern = "LR_012_LEADS_AT_192"
else:
    sensitivity_pattern = "LR_010_LEADS_AT_192"

depth_tradeoff = (
    "LEARNING_RATE_LEADER_CHANGES_WITH_DEPTH"
    if early_leader != final_leader
    else "LEARNING_RATE_LEADER_STABLE_WITH_DEPTH"
)

report = {
    "verdict": "PASS",
    "schema": "burn-research.python-openes-fixed-learning-rate-refinement.v1",
    "decision_is_ci_threshold": False,
    "host_api": host.HOST_API_SCHEMA,
    "abi_version": host.abi_version(),
    "workload": {
        "kind": "deterministic_2d_control_rollout",
        "policy": "Linear(6 -> 8, bias=true) -> ReLU -> Linear(8 -> 2, bias=true)",
        "parameter_dim": EXPECTED_DIM,
        "population": POPULATION,
        "generations": GENERATIONS,
        "candidate_evaluations_per_case": POPULATION * GENERATIONS,
        "sigma": 0.08,
        "learning_rates": [0.08, 0.10, 0.12],
        "scenarios": len(SCENARIOS),
        "horizon": HORIZON,
        "seed": SEED,
    },
    "cases": case_reports,
    "comparison": {
        "continuity_case": "LR_010",
        "learning_rate_cases": ["LR_008", "LR_010", "LR_012"],
        "lr_008_final_reward": lr008_reward,
        "lr_010_final_reward": lr010_reward,
        "lr_012_final_reward": lr012_reward,
        "early_leader_generation_48": early_leader,
        "final_leader_generation_192": final_leader,
        "sensitivity_pattern": sensitivity_pattern,
        "depth_tradeoff": depth_tradeoff,
        "selected_replay_case": selected_label,
        "selected_final_reward": selected_reward,
        "replay_reward": replay_reward,
    },
    "proofs": {
        "installed_wheel": True,
        "typed_relu_surface": True,
        "canonical_parameter_dim_74": True,
        "one_variable_at_a_time_matrix": True,
        "sigma_fixed_at_0_08": True,
        "lr_010_milestone_continuity": True,
        "independent_optimizers": True,
        "zero_state_reset_each_case": True,
        "equal_candidate_evaluation_count_all_cases": True,
        "ask_f32_native_f32": True,
        "candidate_windows_contiguous": True,
        "batch_cardinality_stable": True,
        "finite_fitness_all_cases": True,
        "program_identity_stable_all_cases": True,
        "binding_identity_stable_all_cases": True,
        "selected_stateful_checkpoint_replay": True,
        "replay_parameter_state_exact": True,
        "replay_reward_exact_within_1e-9": True,
    },
    "checkpoint_bytes": len(bundle),
    "notes": [
        "All cases use population 8, sigma 0.08, 192 generations, and exactly 1536 candidate evaluations.",
        "Only learning rate changes across LR_008, LR_010, and LR_012.",
        "LR_010 milestones reproduce the validated #255 fixed-rate winner trajectory.",
        "Search-state interval telemetry is descriptive evidence and does not alter candidates or tell().",
        "Case selection is only evidence plumbing for replay and does not promote production optimizer settings.",
        "Timing and reward comparisons are research evidence only, not CI performance thresholds.",
        "No optimizer algorithm, adaptive schedule, ABI, graph/controller, activation surface, or support claim changes in this research.",
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
    with tempfile.TemporaryDirectory(prefix="burn-research-openes-fixed-lr-refinement-") as tmp:
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
            raise SystemExit("fixed OpenES learning-rate refinement consumer produced no output")
        payload = json.loads(lines[-1])
        if payload.get("verdict") != "PASS":
            raise SystemExit(
                f"unexpected fixed OpenES learning-rate refinement verdict: {payload}"
            )

    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
