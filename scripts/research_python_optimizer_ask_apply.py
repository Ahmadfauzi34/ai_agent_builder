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
REPORT = REPO / "python-optimizer-ask-apply-report.json"

CONSUMER = r'''
from __future__ import annotations

from array import array
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import burn_research_ffi as br
from burn_research_ffi import host

TRANSPORT_REPS = 7
CONTROL_REPS = 11
OBJECTIVE_REPS = 5
POPULATION = 4
LARGE_WIDTH = 64
LARGE_DEPTH = 16
EXPECTED_LARGE_DIM = 66_560
SEED = 20_260_916


def ns():
    return time.perf_counter_ns()


def summary(samples):
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


def digest_f32(values):
    return hashlib.sha256(array("f", values).tobytes()).hexdigest()


def build_chain(stack, width, depth, layer_base):
    registry = stack.enter_context(host.Registry())
    builder = stack.enter_context(host.GraphBuilder(depth + 1))
    for index in range(depth):
        layer = stack.enter_context(
            host.LinearLayerSpec(layer_base + index, width, width, bias=True)
        )
        registry.init_layer(layer)
        builder.add_unary(layer, index, index + 1)
    builder.set_output(depth)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    return registry, graph, binding


def new_optimizer(dim):
    return host.EsOptimizer.strict(
        dim,
        strategy=0,
        seed=SEED,
        population=POPULATION,
        sigma=0.2,
        learning_rate=0.05,
    )


def transport_variant(name, graph, registry, binding):
    dim = binding.total_len
    ask_ns, slice_ns, convert_ns, view_ns, apply_ns = [], [], [], [], []
    post_ns, total_ns, digests = [], [], []

    with new_optimizer(dim) as optimizer:
        if optimizer.batch_size != POPULATION:
            raise AssertionError("optimizer batch size mismatch")
        for rep in range(TRANSPORT_REPS):
            whole_start = ns()
            t0 = ns()
            batch = optimizer.ask()
            t1 = ns()
            ask_ns.append(t1 - t0)
            if len(batch) != POPULATION * dim:
                raise AssertionError(f"{name}: ask cardinality mismatch")
            digests.append(digest_f32(batch))

            post_start = ns()
            expected = None
            validation_buffer = None
            validation_view = None

            if name == "list_slice":
                for index in range(POPULATION):
                    start = index * dim
                    s0 = ns()
                    candidate = batch[start : start + dim]
                    s1 = ns()
                    slice_ns.append(s1 - s0)
                    a0 = ns()
                    binding.apply_flat(graph, registry, candidate)
                    a1 = ns()
                    apply_ns.append(a1 - a0)
                    if rep == 0 and index == POPULATION - 1:
                        expected = candidate

            elif name == "per_candidate_array":
                for index in range(POPULATION):
                    start = index * dim
                    s0 = ns()
                    candidate_list = batch[start : start + dim]
                    s1 = ns()
                    slice_ns.append(s1 - s0)
                    c0 = ns()
                    candidate = array("f", candidate_list)
                    c1 = ns()
                    convert_ns.append(c1 - c0)
                    a0 = ns()
                    binding.apply_flat(graph, registry, candidate)
                    a1 = ns()
                    apply_ns.append(a1 - a0)
                    if rep == 0 and index == POPULATION - 1:
                        expected = candidate_list
                        validation_buffer = candidate

            elif name == "whole_batch_buffer_view":
                c0 = ns()
                batch_buffer = array("f", batch)
                c1 = ns()
                convert_ns.append(c1 - c0)
                whole_view = memoryview(batch_buffer)
                if (
                    whole_view.ndim != 1
                    or not whole_view.c_contiguous
                    or whole_view.format != "f"
                    or whole_view.itemsize != 4
                ):
                    raise AssertionError("whole-batch buffer is not native contiguous f32")
                for index in range(POPULATION):
                    start = index * dim
                    v0 = ns()
                    candidate = whole_view[start : start + dim]
                    v1 = ns()
                    view_ns.append(v1 - v0)
                    if (
                        candidate.ndim != 1
                        or not candidate.c_contiguous
                        or candidate.format != "f"
                        or candidate.itemsize != 4
                        or len(candidate) != dim
                    ):
                        raise AssertionError("candidate view lost native f32 contiguity")
                    a0 = ns()
                    binding.apply_flat(graph, registry, candidate)
                    a1 = ns()
                    apply_ns.append(a1 - a0)
                    if rep == 0 and index == POPULATION - 1:
                        validation_view = candidate
            else:
                raise AssertionError(f"unknown transport variant {name}")

            post_end = ns()
            post_ns.append(post_end - post_start)
            total_ns.append(post_end - whole_start)

            # Semantic checks are deliberately outside the aggregate transport timer.
            if rep == 0:
                if validation_buffer is not None and list(validation_buffer) != expected:
                    raise AssertionError("per-candidate f32 conversion changed values")
                if name == "whole_batch_buffer_view":
                    if list(batch_buffer) != batch:
                        raise AssertionError("whole-batch f32 conversion changed values")
                    if validation_view is None:
                        raise AssertionError("missing final candidate view")
                    expected = list(validation_view)
                if expected is None or binding.read_flat(graph, registry) != expected:
                    raise AssertionError(f"{name}: final candidate state mismatch")

            optimizer.tell([0.0] * POPULATION)

    return {
        "variant": name,
        "ask": summary(ask_ns),
        "list_slice": summary(slice_ns) if slice_ns else None,
        "f32_conversion": summary(convert_ns) if convert_ns else None,
        "memoryview_window": summary(view_ns) if view_ns else None,
        "apply": summary(apply_ns),
        "post_ask_transport": summary(post_ns),
        "ask_plus_transport": summary(total_ns),
        "generation_digests": digests,
    }


def apply_control(graph, registry, binding):
    dim = binding.total_len
    candidate_buffer = array(
        "f", [((((index * 37) % 101) - 50) * 0.0005) for index in range(dim)]
    )
    candidate_list = list(candidate_buffer)
    list_ns, buffer_ns = [], []
    binding.apply_flat(graph, registry, candidate_list)
    binding.apply_flat(graph, registry, candidate_buffer)
    for _ in range(CONTROL_REPS):
        t0 = ns()
        binding.apply_flat(graph, registry, candidate_list)
        t1 = ns()
        binding.apply_flat(graph, registry, candidate_buffer)
        t2 = ns()
        list_ns.append(t1 - t0)
        buffer_ns.append(t2 - t1)
    if binding.read_flat(graph, registry) != candidate_list:
        raise AssertionError("apply control state mismatch")
    list_result = summary(list_ns)
    buffer_result = summary(buffer_ns)
    return {
        "list_fallback": list_result,
        "f32_buffer": buffer_result,
        "list_over_buffer_ratio": list_result["median_ms"]
        / max(buffer_result["median_ms"], 1e-12),
    }


def prove_atomic_rejection(graph, registry, binding):
    before = binding.read_flat(graph, registry)
    poisoned = array("f", before)
    poisoned[len(poisoned) // 2] = math.nan
    try:
        binding.apply_flat(graph, registry, poisoned)
    except host.BurnResearchError as exc:
        if exc.status != host.Status.CORE_ERROR:
            raise AssertionError(f"unexpected non-finite status: {exc.status}")
    else:
        raise AssertionError("non-finite f32 candidate unexpectedly succeeded")
    if binding.read_flat(graph, registry) != before:
        raise AssertionError("non-finite rejection mutated state")
    return True


def make_rows(width):
    rows = []
    for row in range(8):
        values = [
            0.35 * math.sin((row + 1) * (column + 2) * 0.071)
            for column in range(width)
        ]
        target = sum((column + 1) * value for column, value in enumerate(values)) / width
        rows.append((values, target))
    return rows


def build_objective_graph(stack, layer_id):
    registry = stack.enter_context(host.Registry())
    layer = stack.enter_context(host.LinearLayerSpec(layer_id, 8, 1, bias=True))
    registry.init_layer(layer)
    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(layer, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    if binding.total_len != 9:
        raise AssertionError("objective binding dimension mismatch")
    return registry, graph, binding


def run_scalar(graph, registry, values):
    with host.Tensor.vector(values) as input_tensor:
        with graph.run(registry, input_tensor) as output:
            result = output.to_f32()
            if len(result) != 1 or not math.isfinite(result[0]):
                raise AssertionError(f"invalid objective output: {result}")
            return result[0]


def objective_variant(name):
    ask_ns, slice_ns, convert_ns, view_ns = [], [], [], []
    apply_ns, objective_ns, tell_ns, total_ns = [], [], [], []
    digests, fitness_history = [], []

    with ExitStack() as stack:
        registry, graph, binding = build_objective_graph(stack, 97_001)
        optimizer = stack.enter_context(new_optimizer(binding.total_len))
        rows = make_rows(8)
        program_identity = graph.program_identity()
        binding_identity = binding.identity()

        for rep in range(OBJECTIVE_REPS):
            generation_start = ns()
            t0 = ns()
            batch = optimizer.ask()
            t1 = ns()
            ask_ns.append(t1 - t0)
            digests.append(digest_f32(batch))

            batch_buffer = None
            batch_view = None
            validation_pair = None
            if name == "whole_batch_buffer_view":
                c0 = ns()
                batch_buffer = array("f", batch)
                c1 = ns()
                convert_ns.append(c1 - c0)
                batch_view = memoryview(batch_buffer)
                if (
                    batch_view.ndim != 1
                    or not batch_view.c_contiguous
                    or batch_view.format != "f"
                    or batch_view.itemsize != 4
                ):
                    raise AssertionError("objective whole-batch buffer layout mismatch")

            fitness = []
            for index in range(POPULATION):
                start = index * binding.total_len
                if name == "list_slice":
                    s0 = ns()
                    candidate = batch[start : start + binding.total_len]
                    s1 = ns()
                    slice_ns.append(s1 - s0)
                elif name == "per_candidate_array":
                    s0 = ns()
                    candidate_list = batch[start : start + binding.total_len]
                    s1 = ns()
                    slice_ns.append(s1 - s0)
                    c0 = ns()
                    candidate = array("f", candidate_list)
                    c1 = ns()
                    convert_ns.append(c1 - c0)
                    if rep == 0 and index == POPULATION - 1:
                        validation_pair = (candidate, candidate_list)
                elif name == "whole_batch_buffer_view":
                    v0 = ns()
                    candidate = batch_view[start : start + binding.total_len]
                    v1 = ns()
                    view_ns.append(v1 - v0)
                    if (
                        candidate.ndim != 1
                        or not candidate.c_contiguous
                        or candidate.format != "f"
                        or candidate.itemsize != 4
                        or len(candidate) != binding.total_len
                    ):
                        raise AssertionError("objective candidate view layout mismatch")
                else:
                    raise AssertionError(f"unknown objective variant {name}")

                a0 = ns()
                binding.apply_flat(graph, registry, candidate)
                a1 = ns()
                apply_ns.append(a1 - a0)

                o0 = ns()
                squared = 0.0
                for values, target in rows:
                    error = run_scalar(graph, registry, values) - target
                    squared += error * error
                fitness.append(-(squared / len(rows)))
                o1 = ns()
                objective_ns.append(o1 - o0)

            t2 = ns()
            optimizer.tell(fitness)
            t3 = ns()
            tell_ns.append(t3 - t2)
            total_ns.append(t3 - generation_start)
            fitness_history.append(fitness)

            # Validation happens after the timed full generation.
            if rep == 0:
                if validation_pair is not None:
                    candidate_buffer, candidate_list = validation_pair
                    if list(candidate_buffer) != candidate_list:
                        raise AssertionError("objective per-candidate conversion changed values")
                if name == "whole_batch_buffer_view" and list(batch_buffer) != batch:
                    raise AssertionError("objective whole-batch conversion changed values")
            if graph.program_identity() != program_identity:
                raise AssertionError("objective program identity changed")
            if binding.identity() != binding_identity:
                raise AssertionError("objective binding identity changed")

        bundle = host.ProgramBundle.export(graph, registry, include_state=True)
        replay_registry = stack.enter_context(host.Registry())
        replay_graph = stack.enter_context(host.ProgramBundle.import_graph(replay_registry, bundle))
        replay_binding = stack.enter_context(
            host.GraphParameterBinding.build(replay_graph, replay_registry)
        )
        if replay_graph.program_identity() != program_identity:
            raise AssertionError("objective replay program identity mismatch")
        if replay_binding.identity() != binding_identity:
            raise AssertionError("objective replay binding identity mismatch")
        if replay_binding.read_flat(replay_graph, replay_registry) != binding.read_flat(graph, registry):
            raise AssertionError("objective replay state mismatch")

    return {
        "variant": name,
        "ask": summary(ask_ns),
        "list_slice": summary(slice_ns) if slice_ns else None,
        "f32_conversion": summary(convert_ns) if convert_ns else None,
        "memoryview_window": summary(view_ns) if view_ns else None,
        "apply": summary(apply_ns),
        "objective": summary(objective_ns),
        "tell": summary(tell_ns),
        "full_generation": summary(total_ns),
        "generation_digests": digests,
        "fitness_history": fitness_history,
        "checkpoint_replay": True,
    }


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from repository: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"research imported host from repository: {host_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected Python host / ABI identity")

with ExitStack() as stack:
    registry, graph, binding = build_chain(stack, LARGE_WIDTH, LARGE_DEPTH, 98_000)
    if binding.total_len != EXPECTED_LARGE_DIM:
        raise AssertionError(
            f"large binding dim {binding.total_len} != {EXPECTED_LARGE_DIM}"
        )
    program_identity = graph.program_identity()
    binding_identity = binding.identity()

    transport = {
        name: transport_variant(name, graph, registry, binding)
        for name in ("list_slice", "per_candidate_array", "whole_batch_buffer_view")
    }
    reference = transport["list_slice"]["generation_digests"]
    for name, result in transport.items():
        if result["generation_digests"] != reference:
            raise AssertionError(f"{name}: optimizer candidate sequence changed")
    if graph.program_identity() != program_identity:
        raise AssertionError("large program identity changed")
    if binding.identity() != binding_identity:
        raise AssertionError("large binding identity changed")

    control = apply_control(graph, registry, binding)
    atomic_rejection = prove_atomic_rejection(graph, registry, binding)

objective = {
    name: objective_variant(name)
    for name in ("list_slice", "per_candidate_array", "whole_batch_buffer_view")
}
reference_digests = objective["list_slice"]["generation_digests"]
reference_fitness = objective["list_slice"]["fitness_history"]
for name, result in objective.items():
    if result["generation_digests"] != reference_digests:
        raise AssertionError(f"{name}: objective optimizer sequence changed")
    if result["fitness_history"] != reference_fitness:
        raise AssertionError(f"{name}: objective fitness changed")

baseline_post = transport["list_slice"]["post_ask_transport"]["median_ms"]
whole_post = transport["whole_batch_buffer_view"]["post_ask_transport"]["median_ms"]
per_array_post = transport["per_candidate_array"]["post_ask_transport"]["median_ms"]
whole_speedup = baseline_post / max(whole_post, 1e-12)
per_array_speedup = baseline_post / max(per_array_post, 1e-12)
slice_median = transport["list_slice"]["list_slice"]["median_ms"]
apply_median = transport["list_slice"]["apply"]["median_ms"]
slice_share = slice_median / max(slice_median + apply_median, 1e-12)

# Interpretation only. The threshold does not gate CI success.
decision = (
    "PYTHON_BUFFER_RETURN_WORTH_PROTOTYPING"
    if whole_speedup >= 1.20
    else "KEEP_CURRENT_API"
)

report = {
    "verdict": "PASS",
    "decision": decision,
    "decision_is_ci_threshold": False,
    "host_api": host.HOST_API_SCHEMA,
    "abi_version": host.abi_version(),
    "module_path": str(module_path),
    "host_module_path": str(host_path),
    "large_case": {
        "width": LARGE_WIDTH,
        "depth": LARGE_DEPTH,
        "parameter_dim": EXPECTED_LARGE_DIM,
        "population": POPULATION,
        "repetitions": TRANSPORT_REPS,
        "transport": transport,
        "apply_control": control,
        "derived": {
            "whole_batch_buffer_post_ask_speedup": whole_speedup,
            "per_candidate_array_post_ask_speedup": per_array_speedup,
            "per_candidate_list_slice_share_of_slice_plus_apply": slice_share,
        },
    },
    "objective_case": {
        "input_width": 8,
        "parameter_dim": 9,
        "population": POPULATION,
        "repetitions": OBJECTIVE_REPS,
        "variants": objective,
        "semantic_equivalence": True,
        "checkpoint_replay": True,
    },
    "semantic_proofs": {
        "optimizer_candidate_sequence_equal": True,
        "objective_fitness_equal": True,
        "program_identity_stable": True,
        "binding_identity_stable": True,
        "atomic_nonfinite_rejection": atomic_rejection,
        "whole_batch_memoryview_contiguous_f32": True,
    },
    "notes": [
        "Timing is evidence only and never a CI threshold.",
        "The whole-batch variant includes the cost of converting the existing ask() list to array('f').",
        "A future optimizer buffer-return prototype would need separate semantic and installed-wheel proof before support.",
        "This research does not justify an ABI batch primitive by itself.",
    ],
}
print(json.dumps(report, sort_keys=True))
'''


def run(args, *, cwd, env=None):
    completed = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"cwd: {cwd}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main():
    with tempfile.TemporaryDirectory(prefix="burn-research-optimizer-transport-") as tmp:
        root = Path(tmp)
        wheels = root / "wheels"
        wheels.mkdir()
        run([sys.executable, "-m", "maturin", "build", "--out", str(wheels)], cwd=FFI_DIR)
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise SystemExit(f"expected exactly one wheel, found: {wheel_files}")

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
        env["BR_REPO_ROOT"] = str(REPO.resolve())
        completed = run([str(python), str(consumer)], cwd=root, env=env)
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("optimizer transport research produced no output")
        report = json.loads(lines[-1])
        REPORT.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
