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
REPORT = REPO / "python-nonlinear-es-depth-saturation-report.json"

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
SIGMA = 0.08
LEARNING_RATE = 0.06
HORIZON = 16
POLICY_IN = 6
POLICY_HIDDEN = 8
POLICY_OUT = 2
EXPECTED_DIM = (POLICY_IN * POLICY_HIDDEN + POLICY_HIDDEN) + (
    POLICY_HIDDEN * POLICY_OUT + POLICY_OUT
)

POPULATION = 8
GENERATIONS = 192
MILESTONES = (48, 96, 144, 192)
INTERVALS = ((1, 48), (49, 96), (97, 144), (145, 192))

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
        host.LinearLayerSpec(233_000, POLICY_IN, POLICY_HIDDEN, bias=True)
    )
    relu = stack.enter_context(host.ReluLayerSpec(233_001))
    linear_out = stack.enter_context(
        host.LinearLayerSpec(233_002, POLICY_HIDDEN, POLICY_OUT, bias=True)
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
    stack: ExitStack,
    label: str,
    population: int,
    generations: int,
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
            population=population,
            sigma=SIGMA,
            learning_rate=LEARNING_RATE,
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
    observed_batch = None
    evaluation_count = 0

    wall_start = ns()
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
        report_flags.append(list(report.get("flags", [])))

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

    if observed_batch != population:
        raise AssertionError(
            f"case {label}: observed batch {observed_batch} != requested {population}"
        )
    expected_evaluations = population * generations
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

    improvement_generations: list[int] = []
    longest_plateau = 0
    current_plateau = 0
    previous_champion = None
    for generation_index, reward in enumerate(champion_rewards, start=1):
        if previous_champion is None or reward > previous_champion + 1e-12:
            improvement_generations.append(generation_index)
            previous_champion = reward
            current_plateau = 0
        else:
            current_plateau += 1
            longest_plateau = max(longest_plateau, current_plateau)

    no_improvement_count = sum(
        1
        for flags in report_flags
        for flag in flags
        if flag == "NO_IMPROVEMENT"
    )

    improvement_set = set(improvement_generations)
    interval_reports: list[dict[str, object]] = []
    for interval_start, interval_end in INTERVALS:
        interval_improvements = [
            generation
            for generation in improvement_generations
            if interval_start <= generation <= interval_end
        ]
        interval_no_improvement = sum(
            1
            for flags in report_flags[interval_start - 1 : interval_end]
            for flag in flags
            if flag == "NO_IMPROVEMENT"
        )
        interval_longest_plateau = 0
        interval_current_plateau = 0
        for generation in range(interval_start, interval_end + 1):
            if generation in improvement_set:
                interval_current_plateau = 0
            else:
                interval_current_plateau += 1
                interval_longest_plateau = max(
                    interval_longest_plateau,
                    interval_current_plateau,
                )
        interval_reports.append(
            {
                "start_generation": interval_start,
                "end_generation": interval_end,
                "candidate_evaluations_at_end": interval_end * population,
                "champion_reward_at_end": champion_rewards[interval_end - 1],
                "champion_improvement_count": len(interval_improvements),
                "champion_improvement_generations": interval_improvements,
                "longest_champion_plateau_generations": interval_longest_plateau,
                "no_improvement_flag_count": interval_no_improvement,
            }
        )

    milestone_rewards = {
        str(generation): champion_rewards[generation - 1]
        for generation in MILESTONES
    }

    return (
        {
            "label": label,
            "population": population,
            "generations": generations,
            "candidate_evaluations": evaluation_count,
            "zero_policy_reward": zero_policy_reward,
            "initial_champion_reward": initial_champion_reward,
            "final_champion_reward": final_champion_reward,
            "baseline_gain_ratio": baseline_gain_ratio,
            "training_gain_ratio": training_gain_ratio,
            "generation_best": generation_best_rewards,
            "champion_history": champion_rewards,
            "optimizer_flags": report_flags,
            "champion_improvement_generations": improvement_generations,
            "champion_improvement_count": len(improvement_generations),
            "longest_champion_plateau_generations": longest_plateau,
            "first_champion_improvement_generation": (
                improvement_generations[0] if improvement_generations else None
            ),
            "last_champion_improvement_generation": (
                improvement_generations[-1] if improvement_generations else None
            ),
            "no_improvement_flag_count": no_improvement_count,
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

    trajectory_report, best_params = train_case(
        stack,
        "DEPTH_192",
        POPULATION,
        GENERATIONS,
        graph,
        registry,
        binding,
        program_identity,
        binding_identity,
        zero_policy_reward,
    )

    binding.apply_flat(graph, registry, array("f", best_params))
    final_reward = rollout(graph, registry)
    if abs(final_reward - float(trajectory_report["final_champion_reward"])) > 1e-9:
        raise AssertionError("final reward mismatch before checkpoint")

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
    if abs(replay_reward - final_reward) > 1e-9:
        raise AssertionError(
            f"replay reward mismatch: {replay_reward} vs {final_reward}"
        )

intervals = trajectory_report["intervals"]
final_interval = intervals[-1]
final_interval_improvements = int(final_interval["champion_improvement_count"])
if final_interval_improvements > 0:
    saturation_decision = "DEPTH_SATURATION_NOT_OBSERVED_AT_192"
else:
    saturation_decision = "DEPTH_SATURATION_SIGNAL_AT_192"

interval_improvement_counts = [
    int(interval["champion_improvement_count"])
    for interval in intervals
]
if (
    final_interval_improvements > 0
    and final_interval_improvements < max(interval_improvement_counts[:-1])
):
    progress_shape = "DEPTH_RETURNS_DIMINISHING_BUT_ACTIVE"
elif final_interval_improvements > 0:
    progress_shape = "DEPTH_PROGRESS_REMAINS_ACTIVE"
else:
    progress_shape = "FINAL_INTERVAL_NO_CHAMPION_IMPROVEMENT"

report = {
    "verdict": "PASS",
    "schema": "burn-research.python-nonlinear-es-depth-saturation.v1",
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
    "trajectory": trajectory_report,
    "comparison": {
        "milestones": list(MILESTONES),
        "total_generations": GENERATIONS,
        "total_candidate_evaluations": GENERATIONS * POPULATION,
        "saturation_decision": saturation_decision,
        "progress_shape": progress_shape,
        "interval_improvement_counts": interval_improvement_counts,
        "final_reward": final_reward,
        "replay_reward": replay_reward,
    },
    "proofs": {
        "installed_wheel": True,
        "typed_relu_surface": True,
        "canonical_parameter_dim_74": True,
        "single_continuous_optimizer": True,
        "zero_state_initialized_once": True,
        "ask_f32_native_f32": True,
        "candidate_windows_contiguous": True,
        "batch_cardinality_stable": True,
        "finite_fitness_full_trajectory": True,
        "program_identity_stable_trajectory": True,
        "binding_identity_stable_trajectory": True,
        "candidate_evaluation_count_exact": True,
        "final_stateful_checkpoint_replay": True,
        "replay_parameter_state_exact": True,
        "replay_reward_exact_within_1e-9": True,
    },
    "checkpoint_bytes": len(bundle),
    "notes": [
        "One continuous OpenES trajectory runs for 192 generations with population=8, sigma=0.08, and learning_rate=0.06.",
        "Milestones at generations 48, 96, 144, and 192 observe continuation of the same optimizer state rather than independent restarts.",
        "The 48/96 milestones provide continuity with the preceding corrected-sigma depth evidence.",
        "The final 192-generation champion alone is used for stateful ProgramBundle replay.",
        "Timing and training-quality comparisons are evidence only and are not CI performance thresholds.",
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
    with tempfile.TemporaryDirectory(prefix="burn-research-nonlinear-es-depth-saturation-") as tmp:
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
            raise SystemExit("nonlinear ES depth-saturation consumer produced no output")
        payload = json.loads(lines[-1])
        if payload.get("verdict") != "PASS":
            raise SystemExit(f"unexpected nonlinear ES depth-saturation verdict: {payload}")

    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
