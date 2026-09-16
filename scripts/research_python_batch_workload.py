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
REPORT = REPO / "python-batch-workload-report.json"

CONSUMER = r'''
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import burn_research_ffi
from burn_research_ffi import (
    EsOptimizer,
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    ProgramBundle,
    Registry,
    Tensor,
)

FEATURES = 8
TRAIN_ROWS = 96
HOLDOUT_ROWS = 32
POPULATION = 16
GENERATIONS = 36
TARGET_WEIGHTS = [0.40, -0.30, 0.20, 0.10, -0.16, 0.12, 0.07, -0.05]
TARGET_BIAS = 0.08


def now_ns() -> int:
    return time.perf_counter_ns()


def ms(delta_ns: int) -> float:
    return delta_ns / 1_000_000.0


def make_rows(count: int, offset: int) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for i in range(count):
        k = i + offset + 1
        row = [
            (math.sin(k * (j + 1) * 0.173) + 0.5 * math.cos((k + 2) * (j + 3) * 0.117)) / 1.5
            for j in range(FEATURES)
        ]
        target = sum(w * x for w, x in zip(TARGET_WEIGHTS, row)) + TARGET_BIAS
        rows.append(row)
        targets.append(target)
    return rows, targets


def flatten(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def mse(outputs: list[float], targets: list[float]) -> float:
    if len(outputs) != len(targets):
        raise AssertionError(f"output/target length mismatch: {len(outputs)} != {len(targets)}")
    value = sum((out - target) ** 2 for out, target in zip(outputs, targets)) / len(targets)
    if not math.isfinite(value):
        raise AssertionError(f"non-finite MSE: {value}")
    return value


def run_batch(graph, registry, tensor: Tensor) -> tuple[list[float], float, float]:
    t0 = now_ns()
    output = graph.run(registry, tensor)
    t1 = now_ns()
    values = output.to_f32()
    t2 = now_ns()
    output.close()
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("batch output contains non-finite values")
    return values, ms(t1 - t0), ms(t2 - t1)


def run_row_profile(graph, registry, rows: list[list[float]]) -> tuple[list[float], dict[str, float]]:
    outputs: list[float] = []
    create_ns = 0
    run_ns = 0
    copy_ns = 0
    for row in rows:
        t0 = now_ns()
        tensor = Tensor.vector(row)
        t1 = now_ns()
        output = graph.run(registry, tensor)
        t2 = now_ns()
        values = output.to_f32()
        t3 = now_ns()
        output.close()
        tensor.close()
        if len(values) != 1 or not math.isfinite(values[0]):
            raise AssertionError(f"bad row output: {values}")
        outputs.append(values[0])
        create_ns += t1 - t0
        run_ns += t2 - t1
        copy_ns += t3 - t2
    return outputs, {
        "tensor_create_ms": ms(create_ns),
        "graph_run_ms": ms(run_ns),
        "output_copy_ms": ms(copy_ns),
        "total_ms": ms(create_ns + run_ns + copy_ns),
    }


def run_batch_profile(graph, registry, rows: list[list[float]]) -> tuple[list[float], dict[str, float]]:
    t0 = now_ns()
    tensor = Tensor.from_f32(flatten(rows), (len(rows), FEATURES, 1, 1))
    t1 = now_ns()
    output = graph.run(registry, tensor)
    t2 = now_ns()
    values = output.to_f32()
    t3 = now_ns()
    output.close()
    tensor.close()
    if len(values) != len(rows):
        raise AssertionError(f"bad batch output length: {len(values)} != {len(rows)}")
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("batch output contains non-finite values")
    return values, {
        "tensor_create_ms": ms(t1 - t0),
        "graph_run_ms": ms(t2 - t1),
        "output_copy_ms": ms(t3 - t2),
        "total_ms": ms(t3 - t0),
    }


def median_profile(samples: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
    }


def close_all(items: list[object]) -> None:
    while items:
        item = items.pop()
        close = getattr(item, "close", None)
        if close is not None:
            close()


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(burn_research_ffi.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from checkout: {module_path}")

owned: list[object] = []
try:
    train_rows, train_targets = make_rows(TRAIN_ROWS, 0)
    holdout_rows, holdout_targets = make_rows(HOLDOUT_ROWS, 10_000)

    registry = Registry(); owned.append(registry)
    layer = LinearLayerSpec(61_001, FEATURES, 1, bias=True); owned.append(layer)
    registry.init_layer(layer)
    builder = GraphBuilder(2); owned.append(builder)
    builder.add_unary(layer, 0, 1).set_output(1)
    graph = builder.compile(registry); owned.append(graph)
    binding = GraphParameterBinding.build(graph, registry); owned.append(binding)

    if binding.total_len != FEATURES + 1:
        raise AssertionError(f"unexpected parameter dimension: {binding.total_len}")

    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    layout = binding.layout()

    zero = [0.0] * binding.total_len
    binding.apply_flat(graph, registry, zero)
    if binding.read_flat(graph, registry) != zero:
        raise AssertionError("zero candidate did not read back exactly")

    # Equivalence and crossing/copy profile. Timing is evidence only.
    row_profiles: list[dict[str, float]] = []
    batch_profiles: list[dict[str, float]] = []
    reference_row: list[float] | None = None
    reference_batch: list[float] | None = None
    for _ in range(3):
        row_values, row_profile = run_row_profile(graph, registry, holdout_rows)
        batch_values, batch_profile = run_batch_profile(graph, registry, holdout_rows)
        row_profiles.append(row_profile)
        batch_profiles.append(batch_profile)
        reference_row = row_values
        reference_batch = batch_values

    assert reference_row is not None and reference_batch is not None
    max_row_batch_abs_diff = max(
        abs(a - b) for a, b in zip(reference_row, reference_batch)
    )
    if max_row_batch_abs_diff > 1e-6:
        raise AssertionError(
            f"row/batch semantic mismatch: max_abs_diff={max_row_batch_abs_diff}"
        )

    row_profile = median_profile(row_profiles)
    batch_profile = median_profile(batch_profiles)
    row_vs_batch_ratio = row_profile["total_ms"] / max(batch_profile["total_ms"], 1e-9)

    # Persistent batch tensors model a host that reuses a dataset across candidates.
    t0 = now_ns()
    train_tensor = Tensor.from_f32(
        flatten(train_rows), (TRAIN_ROWS, FEATURES, 1, 1)
    ); owned.append(train_tensor)
    t1 = now_ns()
    holdout_tensor = Tensor.from_f32(
        flatten(holdout_rows), (HOLDOUT_ROWS, FEATURES, 1, 1)
    ); owned.append(holdout_tensor)
    t2 = now_ns()
    persistent_tensor_create_ms = ms(t2 - t0)

    baseline_outputs, _, _ = run_batch(graph, registry, holdout_tensor)
    baseline_holdout_loss = mse(baseline_outputs, holdout_targets)

    optimizer = EsOptimizer.strict(
        binding.total_len,
        strategy=0,
        seed=20_260_916,
        population=POPULATION,
        sigma=0.22,
        learning_rate=0.06,
    ); owned.append(optimizer)

    ask_ns = 0
    apply_ns = 0
    run_ns = 0
    copy_ns = 0
    objective_ns = 0
    tell_ns = 0
    best_observed_loss = float("inf")
    first_generation_best_loss: float | None = None

    for generation in range(GENERATIONS):
        ta0 = now_ns()
        candidates = optimizer.ask()
        ta1 = now_ns()
        ask_ns += ta1 - ta0

        expected = optimizer.batch_size * binding.total_len
        if len(candidates) != expected:
            raise AssertionError(f"ask cardinality mismatch: {len(candidates)} != {expected}")
        if not all(math.isfinite(value) for value in candidates):
            raise AssertionError("ES emitted non-finite candidate")

        fitness: list[float] = []
        generation_losses: list[float] = []
        for i in range(optimizer.batch_size):
            start = i * binding.total_len
            candidate = candidates[start : start + binding.total_len]

            tp0 = now_ns()
            binding.apply_flat(graph, registry, candidate)
            tp1 = now_ns()
            apply_ns += tp1 - tp0

            outputs, run_ms, copy_ms = run_batch(graph, registry, train_tensor)
            run_ns += int(run_ms * 1_000_000)
            copy_ns += int(copy_ms * 1_000_000)

            to0 = now_ns()
            loss = mse(outputs, train_targets)
            to1 = now_ns()
            objective_ns += to1 - to0
            generation_losses.append(loss)
            fitness.append(-loss)
            best_observed_loss = min(best_observed_loss, loss)

        if generation == 0:
            first_generation_best_loss = min(generation_losses)

        tt0 = now_ns()
        optimizer.tell(fitness)
        tt1 = now_ns()
        tell_ns += tt1 - tt0

    if first_generation_best_loss is None:
        raise AssertionError("missing first-generation loss")

    best = optimizer.best()
    if len(best) != binding.total_len or not all(math.isfinite(value) for value in best):
        raise AssertionError("invalid optimizer best vector")
    binding.apply_flat(graph, registry, best)
    learned = binding.read_flat(graph, registry)
    if learned != best:
        raise AssertionError("best candidate did not read back exactly")

    final_train_outputs, _, _ = run_batch(graph, registry, train_tensor)
    final_train_loss = mse(final_train_outputs, train_targets)
    final_holdout_outputs, _, _ = run_batch(graph, registry, holdout_tensor)
    final_holdout_loss = mse(final_holdout_outputs, holdout_targets)

    if final_holdout_loss >= baseline_holdout_loss * 0.90:
        raise AssertionError(
            "training did not materially improve held-out loss: "
            f"baseline={baseline_holdout_loss} final={final_holdout_loss}"
        )
    if graph.program_identity() != program_identity:
        raise AssertionError("program identity changed during training")
    if binding.identity() != binding_identity:
        raise AssertionError("binding identity changed during training")

    tc0 = now_ns()
    bundle = ProgramBundle.export(graph, registry, include_state=True)
    tc1 = now_ns()
    imported_registry = Registry(); owned.append(imported_registry)
    imported_graph = ProgramBundle.import_graph(imported_registry, bundle); owned.append(imported_graph)
    imported_binding = GraphParameterBinding.build(imported_graph, imported_registry); owned.append(imported_binding)
    tc2 = now_ns()

    if imported_graph.program_identity() != program_identity:
        raise AssertionError("checkpoint program identity mismatch")
    if imported_binding.identity() != binding_identity:
        raise AssertionError("checkpoint binding identity mismatch")
    if imported_binding.layout() != layout:
        raise AssertionError("checkpoint binding layout mismatch")
    if imported_binding.read_flat(imported_graph, imported_registry) != learned:
        raise AssertionError("checkpoint learned state mismatch")

    tr0 = now_ns()
    replay_outputs, replay_run_ms, replay_copy_ms = run_batch(
        imported_graph, imported_registry, holdout_tensor
    )
    replay_loss = mse(replay_outputs, holdout_targets)
    tr1 = now_ns()
    replay_max_abs_diff = max(
        abs(a - b) for a, b in zip(replay_outputs, final_holdout_outputs)
    )
    if replay_max_abs_diff > 1e-7 or abs(replay_loss - final_holdout_loss) > 1e-9:
        raise AssertionError(
            "checkpoint replay mismatch: "
            f"max_abs_diff={replay_max_abs_diff} loss_delta={abs(replay_loss-final_holdout_loss)}"
        )

    candidate_evaluations = GENERATIONS * POPULATION
    report = {
        "verdict": "PASS",
        "consumer": "installed-wheel-typed-facade",
        "workload": "python-batch-vs-row-linear-regression-v1",
        "module_path": str(module_path),
        "shape": {
            "features": FEATURES,
            "train_rows": TRAIN_ROWS,
            "holdout_rows": HOLDOUT_ROWS,
            "parameter_dim": binding.total_len,
        },
        "optimizer": {
            "population": POPULATION,
            "generations": GENERATIONS,
            "candidate_evaluations": candidate_evaluations,
            "first_generation_best_loss": first_generation_best_loss,
            "best_observed_train_loss": best_observed_loss,
        },
        "semantic": {
            "max_row_batch_abs_diff": max_row_batch_abs_diff,
            "baseline_holdout_loss": baseline_holdout_loss,
            "final_train_loss": final_train_loss,
            "final_holdout_loss": final_holdout_loss,
            "heldout_improvement_ratio": baseline_holdout_loss / final_holdout_loss,
            "checkpoint_bytes": len(bundle),
            "replay_max_abs_diff": replay_max_abs_diff,
            "replay_loss": replay_loss,
        },
        "profile_ms": {
            "row_median": row_profile,
            "batch_median": batch_profile,
            "row_vs_batch_total_ratio": row_vs_batch_ratio,
            "persistent_train_plus_holdout_tensor_create": persistent_tensor_create_ms,
            "training_total": {
                "ask": ms(ask_ns),
                "candidate_apply": ms(apply_ns),
                "graph_run": ms(run_ns),
                "output_copy": ms(copy_ns),
                "python_objective": ms(objective_ns),
                "tell": ms(tell_ns),
            },
            "training_per_candidate": {
                "candidate_apply": ms(apply_ns) / candidate_evaluations,
                "graph_run": ms(run_ns) / candidate_evaluations,
                "output_copy": ms(copy_ns) / candidate_evaluations,
                "python_objective": ms(objective_ns) / candidate_evaluations,
            },
            "checkpoint": {
                "export": ms(tc1 - tc0),
                "import_and_binding_rebuild": ms(tc2 - tc1),
                "replay_total": ms(tr1 - tr0),
                "replay_graph_run": replay_run_ms,
                "replay_output_copy": replay_copy_ms,
            },
        },
        "decision_boundary": {
            "timings_are_ci_thresholds": False,
            "abi_widened": False,
            "numpy_required": False,
            "zero_copy_claimed": False,
        },
    }
    print(json.dumps(report, sort_keys=True))
finally:
    close_all(owned)
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
    with tempfile.TemporaryDirectory(prefix="burn-research-python-workload-") as tmp:
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
            raise SystemExit("research consumer produced no output")

        report = json.loads(lines[-1])
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
