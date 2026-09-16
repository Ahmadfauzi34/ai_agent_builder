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
REPORT = REPO / "python-optimizer-direct-f32-report.json"

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
from burn_research_ffi import host, ffi, lib

REPS = 7
OBJECTIVE_REPS = 5
REQUESTED_POPULATION = 4
SEED = 20_260_916
LARGE_WIDTH = 64
LARGE_DEPTH = 16
EXPECTED_DIM = 66_560


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


def digest_f32_buffer(values):
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def last_error():
    n = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", n + 1)
    lib.br_v1_last_error_copy(buf, n + 1)
    return ffi.string(buf).decode("utf-8", errors="replace")


def check(status, context):
    if int(status) != 0:
        raise AssertionError(f"{context}: status={int(status)} error={last_error()!r}")


def raw_free(handle):
    if handle != ffi.NULL:
        check(lib.br_v1_handle_free(handle), "handle free")


def raw_new_optimizer(dim):
    out = ffi.new("br_v1_handle **")
    check(
        lib.br_v1_es_strict(
            dim, 0, SEED, REQUESTED_POPULATION, 0.2, 1, 0.05, out
        ),
        "es strict",
    )
    if out[0] == ffi.NULL:
        raise AssertionError("es strict returned null handle")
    return out[0]


def raw_batch_size(optimizer):
    out = ffi.new("uint32_t *")
    check(lib.br_v1_es_batch_size(optimizer, out), "es batch size")
    return int(out[0])


def raw_ask_array(optimizer):
    """Call existing ABI v1 ask and copy its f32 handle directly into array('f')."""
    out = ffi.new("br_v1_handle **")
    check(lib.br_v1_es_ask(optimizer, out), "es ask")
    handle = out[0]
    if handle == ffi.NULL:
        raise AssertionError("es ask returned null f32 handle")
    try:
        n = ffi.new("size_t *")
        check(lib.br_v1_f32_buffer_len(handle, n), "f32 buffer len")
        size = int(n[0])
        if size == 0:
            return array("f")
        dest = array("f", [0.0]) * size
        raw = ffi.from_buffer("float[]", dest)
        check(lib.br_v1_f32_buffer_copy(handle, raw, size), "f32 buffer copy")
        return dest
    finally:
        raw_free(handle)


def raw_tell(optimizer, fitness):
    values = array("f", fitness)
    raw = ffi.from_buffer("float[]", values) if values else ffi.NULL
    out = ffi.new("br_v1_handle **")
    check(lib.br_v1_es_tell(optimizer, raw, len(values), out), "es tell")
    if out[0] == ffi.NULL:
        raise AssertionError("es tell returned null report")
    raw_free(out[0])


def new_facade_optimizer(dim):
    return host.EsOptimizer.strict(
        dim,
        strategy=0,
        seed=SEED,
        population=REQUESTED_POPULATION,
        sigma=0.2,
        learning_rate=0.05,
    )


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


def build_objective_graph(stack, layer_id):
    registry = stack.enter_context(host.Registry())
    layer = stack.enter_context(host.LinearLayerSpec(layer_id, 8, 1, bias=True))
    registry.init_layer(layer)
    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(layer, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    if binding.total_len != 9:
        raise AssertionError(f"objective dim {binding.total_len} != 9")
    return registry, graph, binding


def run_transport(kind, graph, registry, binding):
    ask_samples = []
    window_samples = []
    apply_samples = []
    post_samples = []
    total_samples = []
    generation_digests = []
    observed_batch = None

    facade = new_facade_optimizer(binding.total_len) if kind == "facade_list" else None
    raw_optimizer = raw_new_optimizer(binding.total_len) if kind == "direct_f32_array" else None
    try:
        if kind == "facade_list" and facade.batch_size != 0:
            raise AssertionError("new facade optimizer batch_size must be 0")
        if kind == "direct_f32_array" and raw_batch_size(raw_optimizer) != 0:
            raise AssertionError("new raw optimizer batch_size must be 0")

        for _ in range(REPS):
            a0 = ns()
            if kind == "facade_list":
                batch = facade.ask()
                batch_size = facade.batch_size
            else:
                batch = raw_ask_array(raw_optimizer)
                batch_size = raw_batch_size(raw_optimizer)
            a1 = ns()
            ask_elapsed = a1 - a0
            ask_samples.append(ask_elapsed)

            if batch_size <= 0 or len(batch) != batch_size * binding.total_len:
                raise AssertionError(f"{kind}: invalid ask cardinality")
            if observed_batch is None:
                observed_batch = batch_size
            elif batch_size != observed_batch:
                raise AssertionError(f"{kind}: batch_size changed")

            if kind == "facade_list":
                proof_buffer = array("f", batch)
            else:
                proof_buffer = batch
            generation_digests.append(digest_f32_buffer(proof_buffer))

            p0 = ns()
            if kind == "facade_list":
                for index in range(batch_size):
                    start = index * binding.total_len
                    w0 = ns()
                    candidate = batch[start : start + binding.total_len]
                    w1 = ns()
                    window_samples.append(w1 - w0)
                    x0 = ns()
                    binding.apply_flat(graph, registry, candidate)
                    x1 = ns()
                    apply_samples.append(x1 - x0)
            else:
                view = memoryview(batch)
                if view.format != "f" or view.itemsize != 4 or not view.c_contiguous:
                    raise AssertionError("direct batch is not contiguous native f32")
                for index in range(batch_size):
                    start = index * binding.total_len
                    w0 = ns()
                    candidate = view[start : start + binding.total_len]
                    w1 = ns()
                    window_samples.append(w1 - w0)
                    if len(candidate) != binding.total_len or not candidate.c_contiguous:
                        raise AssertionError("direct candidate window contract failed")
                    x0 = ns()
                    binding.apply_flat(graph, registry, candidate)
                    x1 = ns()
                    apply_samples.append(x1 - x0)
            p1 = ns()
            post_elapsed = p1 - p0
            post_samples.append(post_elapsed)
            # Deliberately excludes semantic digest work from timing.
            total_samples.append(ask_elapsed + post_elapsed)

            zeros = [0.0] * batch_size
            if kind == "facade_list":
                facade.tell(zeros)
                if facade.batch_size != batch_size:
                    raise AssertionError("facade last-candidate cardinality changed after tell")
            else:
                raw_tell(raw_optimizer, zeros)
                if raw_batch_size(raw_optimizer) != batch_size:
                    raise AssertionError("raw last-candidate cardinality changed after tell")

        return {
            "kind": kind,
            "batch_size": observed_batch,
            "ask": summary(ask_samples),
            "candidate_window": summary(window_samples),
            "apply": summary(apply_samples),
            "post_ask_transport": summary(post_samples),
            "ask_plus_transport": summary(total_samples),
            "generation_digests": generation_digests,
        }
    finally:
        if facade is not None:
            facade.close()
        if raw_optimizer is not None:
            raw_free(raw_optimizer)


def objective_rows():
    result = []
    for row in range(8):
        x = [0.35 * math.sin((row + 1) * (col + 2) * 0.071) for col in range(8)]
        target = sum((col + 1) * value for col, value in enumerate(x)) / 8
        result.append((x, target))
    return result


ROWS = objective_rows()


def scalar(graph, registry, values):
    with host.Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            data = out.to_f32()
            if len(data) != 1 or not math.isfinite(data[0]):
                raise AssertionError(f"invalid objective output: {data}")
            return data[0]


def objective_variant(kind):
    generation_digests = []
    fitness_history = []
    total_samples = []
    with ExitStack() as stack:
        registry, graph, binding = build_objective_graph(stack, 199_000)
        program_identity = graph.program_identity()
        binding_identity = binding.identity()
        facade = new_facade_optimizer(binding.total_len) if kind == "facade_list" else None
        raw_optimizer = raw_new_optimizer(binding.total_len) if kind == "direct_f32_array" else None
        try:
            for _ in range(OBJECTIVE_REPS):
                t0 = ns()
                if kind == "facade_list":
                    batch = facade.ask()
                    batch_size = facade.batch_size
                else:
                    batch = raw_ask_array(raw_optimizer)
                    batch_size = raw_batch_size(raw_optimizer)
                if len(batch) != batch_size * binding.total_len:
                    raise AssertionError("objective ask cardinality mismatch")

                if kind == "facade_list":
                    source = batch
                else:
                    source = memoryview(batch)

                fitness = []
                for index in range(batch_size):
                    start = index * binding.total_len
                    candidate = source[start : start + binding.total_len]
                    binding.apply_flat(graph, registry, candidate)
                    squared = 0.0
                    for x, target in ROWS:
                        error = scalar(graph, registry, x) - target
                        squared += error * error
                    fitness.append(-(squared / len(ROWS)))

                if kind == "facade_list":
                    facade.tell(fitness)
                else:
                    raw_tell(raw_optimizer, fitness)
                t1 = ns()
                total_samples.append(t1 - t0)

                # Semantic digest is intentionally outside the timed generation region.
                proof_buffer = array("f", batch) if kind == "facade_list" else batch
                generation_digests.append(digest_f32_buffer(proof_buffer))
                fitness_history.append(fitness)

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
                raise AssertionError("replay program identity mismatch")
            if replay_binding.identity() != binding_identity:
                raise AssertionError("replay binding identity mismatch")
            if replay_binding.read_flat(replay_graph, replay_registry) != binding.read_flat(graph, registry):
                raise AssertionError("replay state mismatch")
        finally:
            if facade is not None:
                facade.close()
            if raw_optimizer is not None:
                raw_free(raw_optimizer)

    return {
        "kind": kind,
        "generation_digests": generation_digests,
        "fitness_history": fitness_history,
        "full_generation": summary(total_samples),
        "checkpoint_replay": True,
    }


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from repo: {module_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected host/ABI identity")

with ExitStack() as stack:
    registry, graph, binding = build_chain(stack, LARGE_WIDTH, LARGE_DEPTH, 198_000)
    if binding.total_len != EXPECTED_DIM:
        raise AssertionError(f"large dim {binding.total_len} != {EXPECTED_DIM}")
    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    facade = run_transport("facade_list", graph, registry, binding)
    direct = run_transport("direct_f32_array", graph, registry, binding)
    if facade["generation_digests"] != direct["generation_digests"]:
        raise AssertionError("direct f32 path changed candidate bytes")
    if facade["batch_size"] != direct["batch_size"]:
        raise AssertionError("direct f32 path changed batch cardinality")
    if graph.program_identity() != program_identity or binding.identity() != binding_identity:
        raise AssertionError("large graph/binding identity changed")

objective_facade = objective_variant("facade_list")
objective_direct = objective_variant("direct_f32_array")
if objective_facade["generation_digests"] != objective_direct["generation_digests"]:
    raise AssertionError("objective candidate sequence changed")
if objective_facade["fitness_history"] != objective_direct["fitness_history"]:
    raise AssertionError("objective fitness history changed")

ask_speedup = facade["ask"]["median_ms"] / max(direct["ask"]["median_ms"], 1e-12)
total_speedup = facade["ask_plus_transport"]["median_ms"] / max(
    direct["ask_plus_transport"]["median_ms"], 1e-12
)
decision = (
    "PUBLIC_ASK_F32_WORTH_IMPLEMENTING"
    if total_speedup >= 1.10
    else "KEEP_LIST_API_ONLY"
)

report = {
    "verdict": "PASS",
    "decision": decision,
    "decision_is_ci_threshold": False,
    "host_api": host.HOST_API_SCHEMA,
    "abi_version": host.abi_version(),
    "large_case": {
        "parameter_dim": EXPECTED_DIM,
        "batch_size": facade["batch_size"],
        "facade_list": facade,
        "direct_f32_array": direct,
        "derived": {
            "direct_ask_speedup": ask_speedup,
            "direct_ask_plus_transport_speedup": total_speedup,
        },
    },
    "objective_case": {
        "parameter_dim": 9,
        "facade_list": objective_facade,
        "direct_f32_array": objective_direct,
        "semantic_equivalence": True,
    },
    "semantic_proofs": {
        "candidate_bytes_equal": True,
        "batch_cardinality_equal": True,
        "objective_fitness_equal": True,
        "program_identity_stable": True,
        "binding_identity_stable": True,
        "checkpoint_replay": True,
        "raw_handles_freed": True,
    },
    "notes": [
        "Prototype uses only existing ABI v1 symbols and standard-library array('f').",
        "Existing host.EsOptimizer.ask() remains unchanged.",
        "Semantic digest work is excluded from timing regions.",
        "Timing is evidence only and never a CI threshold.",
        "A positive decision only justifies a later additive public ask_f32 product slice.",
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
    with tempfile.TemporaryDirectory(prefix="burn-research-direct-f32-") as tmp:
        root = Path(tmp)
        wheels = root / "wheels"
        wheels.mkdir()
        run([sys.executable, "-m", "maturin", "build", "--out", str(wheels)], cwd=FFI_DIR)
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise SystemExit(f"expected one wheel, found {wheel_files}")

        venv = root / "venv"
        run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        python = venv / "bin" / "python"
        run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", str(wheel_files[0])],
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
            raise SystemExit("direct-f32 prototype produced no output")
        report = json.loads(lines[-1])
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
