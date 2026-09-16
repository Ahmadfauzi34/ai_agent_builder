#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FFI_DIR = REPO / "ffi"

CONSUMER = r'''
from __future__ import annotations

import json
import math
import os
import statistics
import time
from array import array
from contextlib import ExitStack
from pathlib import Path

import burn_research_ffi as br
from burn_research_ffi import (
    BurnResearchError,
    ClosedHandleError,
    EsOptimizer,
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    ProgramBundle,
    Registry,
    Status,
    Tensor,
)


class BufferOnlyF32(array):
    """f32 buffer whose Python iteration must never be used by the fast path."""

    def __new__(cls, values):
        return array.__new__(cls, "f", values)

    def __iter__(self):
        raise AssertionError("compatible f32 candidate fell back to Sequence iteration")


def run_scalar(graph, registry, values: list[float]) -> float:
    with Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            assert out.length == 1
            result = out.to_f32()
            assert len(result) == 1 and math.isfinite(result[0])
            return result[0]


def build_linear_chain(stack: ExitStack, width: int, depth: int, layer_base: int):
    registry = stack.enter_context(Registry())
    builder = stack.enter_context(GraphBuilder(depth + 1))
    for i in range(depth):
        layer = stack.enter_context(
            LinearLayerSpec(layer_base + i, width, width, bias=True)
        )
        registry.init_layer(layer)
        builder.add_unary(layer, i, i + 1)
    builder.set_output(depth)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(GraphParameterBinding.build(graph, registry))
    return registry, graph, binding


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"wheel consumer imported from repository: {module_path}")

# The ergonomic facade must not hide or replace the low-level ABI objects.
assert br.ffi is not None and br.lib is not None
assert int(br.lib.br_v1_abi_version()) == 1
assert br.abi_version() == 1
assert module_path.with_name("py.typed").is_file()

caps = br.capabilities()
assert caps["schema"] == "burn-research.ffi.v1"
assert caps["host_policy"] == "external"

# Python ownership must be deterministic: double-close is harmless and a
# closed object is rejected locally before another FFI call is attempted.
closed_registry = Registry()
closed_registry.close()
closed_registry.close()
assert closed_registry.closed
with LinearLayerSpec(99_999, 1, 1) as scratch_layer:
    try:
        closed_registry.init_layer(scratch_layer)
    except ClosedHandleError:
        pass
    else:
        raise AssertionError("use-after-close was not rejected locally")

with ExitStack() as stack:
    registry = stack.enter_context(Registry())
    linear = stack.enter_context(LinearLayerSpec(52_001, 2, 1, bias=True))
    registry.init_layer(linear)

    builder = stack.enter_context(GraphBuilder(2))
    builder.add_unary(linear, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(GraphParameterBinding.build(graph, registry))

    dim = binding.total_len
    assert dim == 3
    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    assert binding.layout()

    # Historical generic Sequence compatibility remains unchanged.
    list_candidate = [0.125, -0.25, 0.5]
    binding.apply_flat(graph, registry, list_candidate)
    assert binding.read_flat(graph, registry) == list_candidate

    tuple_candidate = (0.25, 0.5, -0.75)
    binding.apply_flat(graph, registry, tuple_candidate)
    assert binding.read_flat(graph, registry) == list(tuple_candidate)

    # A non-f32 buffer-backed Sequence falls back to the generic Sequence path.
    f64_candidate = array("d", [0.5, -0.25, 0.125])
    binding.apply_flat(graph, registry, f64_candidate)
    assert binding.read_flat(graph, registry) == [0.5, -0.25, 0.125]

    # This candidate cannot be iterated. Success therefore proves the facade
    # borrowed its compatible f32 buffer instead of normalizing a Sequence.
    fast_candidate = BufferOnlyF32([0.75, -0.5, 0.25])
    binding.apply_flat(graph, registry, fast_candidate)
    assert binding.read_flat(graph, registry) == [0.75, -0.5, 0.25]

    # A contiguous native-f32 memoryview follows the same stateless fast path.
    memory_candidate_backing = array("f", [-0.25, 0.375, 0.625])
    memory_candidate = memoryview(memory_candidate_backing)
    binding.apply_flat(graph, registry, memory_candidate)
    assert binding.read_flat(graph, registry) == list(memory_candidate_backing)

    # Once an object presents native f32 storage, malformed layout/length fails
    # locally instead of silently reinterpreting bytes through Sequence fallback.
    wrong_length = array("f", [1.0, 2.0])
    try:
        binding.apply_flat(graph, registry, wrong_length)
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("wrong-length f32 buffer was not rejected locally")

    non_contiguous_backing = array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    non_contiguous = memoryview(non_contiguous_backing)[::2]
    assert len(non_contiguous) == dim and not non_contiguous.c_contiguous
    try:
        binding.apply_flat(graph, registry, non_contiguous)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("non-contiguous f32 buffer was not rejected locally")

    # Stable ABI status -> Python exception mapping remains intact for the new
    # buffer path, and the core finite-only rejection remains atomic.
    before_reject = binding.read_flat(graph, registry)
    poisoned = array("f", [before_reject[0], math.nan, before_reject[2]])
    try:
        binding.apply_flat(graph, registry, poisoned)
    except BurnResearchError as exc:
        assert exc.status == Status.CORE_ERROR
        assert exc.status_code == int(Status.CORE_ERROR)
    else:
        raise AssertionError("non-finite f32 buffer candidate unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_reject
    assert graph.program_identity() == program_identity
    assert binding.identity() == binding_identity

    optimizer = stack.enter_context(
        EsOptimizer.strict(
            dim,
            strategy=0,
            seed=9917,
            population=4,
            sigma=0.2,
            learning_rate=0.05,
        )
    )
    candidates = optimizer.ask()
    batch = optimizer.batch_size
    assert len(candidates) == batch * dim

    rows = [(-1.0, 0.5), (0.0, 0.0), (1.0, -0.5)]
    fitness: list[float] = []
    for i in range(batch):
        candidate = candidates[i * dim : (i + 1) * dim]
        assert all(math.isfinite(value) for value in candidate)
        binding.apply_flat(graph, registry, candidate)

        squared = 0.0
        for x0, x1 in rows:
            target = 1.5 * x0 - 0.75 * x1 + 0.25
            error = run_scalar(graph, registry, [x0, x1]) - target
            squared += error * error
        fitness.append(-(squared / len(rows)))

    report = optimizer.tell(fitness)
    assert int(report["gen"]) == 1

    best = optimizer.best()
    assert len(best) == dim and all(math.isfinite(value) for value in best)
    binding.apply_flat(graph, registry, array("f", best))
    learned = binding.read_flat(graph, registry)
    assert learned == list(array("f", best))
    probe_before = run_scalar(graph, registry, [1.0, 2.0])

    bundle = ProgramBundle.export(graph, registry, include_state=True)
    assert bundle

    imported_registry = stack.enter_context(Registry())
    imported_graph = stack.enter_context(
        ProgramBundle.import_graph(imported_registry, bundle)
    )
    imported_binding = stack.enter_context(
        GraphParameterBinding.build(imported_graph, imported_registry)
    )

    assert imported_graph.program_identity() == program_identity
    assert imported_binding.identity() == binding_identity
    assert imported_binding.read_flat(imported_graph, imported_registry) == learned
    probe_after = run_scalar(imported_graph, imported_registry, [1.0, 2.0])
    assert abs(probe_after - probe_before) <= 1e-7

# Timing evidence only: no CI performance threshold. The large case mirrors the
# research scale where host marshalling was previously material.
PERF_REPS = 9
with ExitStack() as perf_stack:
    _, perf_graph, perf_binding = build_linear_chain(
        perf_stack, width=64, depth=16, layer_base=88_000
    )
    perf_dim = perf_binding.total_len
    assert perf_dim == 66_560
    perf_buffer = array(
        "f",
        [((((i * 37) % 101) - 50) * 0.0005) for i in range(perf_dim)],
    )
    perf_list = list(perf_buffer)

    # Warm both paths before timing.
    perf_binding.apply_flat(perf_graph, perf_stack._exit_callbacks and next(iter([]), None), perf_buffer) if False else None

    # Registry is needed explicitly; build once more with a retained name.

with ExitStack() as perf_stack:
    perf_registry, perf_graph, perf_binding = build_linear_chain(
        perf_stack, width=64, depth=16, layer_base=89_000
    )
    perf_dim = perf_binding.total_len
    assert perf_dim == 66_560
    perf_buffer = array(
        "f",
        [((((i * 37) % 101) - 50) * 0.0005) for i in range(perf_dim)],
    )
    perf_list = list(perf_buffer)
    perf_binding.apply_flat(perf_graph, perf_registry, perf_list)
    perf_binding.apply_flat(perf_graph, perf_registry, perf_buffer)

    list_samples: list[int] = []
    buffer_samples: list[int] = []
    for _ in range(PERF_REPS):
        t0 = time.perf_counter_ns()
        perf_binding.apply_flat(perf_graph, perf_registry, perf_list)
        t1 = time.perf_counter_ns()
        perf_binding.apply_flat(perf_graph, perf_registry, perf_buffer)
        t2 = time.perf_counter_ns()
        list_samples.append(t1 - t0)
        buffer_samples.append(t2 - t1)

    list_median_ms = statistics.median(list_samples) / 1_000_000.0
    buffer_median_ms = statistics.median(buffer_samples) / 1_000_000.0
    performance_ratio = list_median_ms / max(buffer_median_ms, 1e-12)

print(json.dumps({
    "verdict": "PASS",
    "consumer": "installed-wheel-python-facade",
    "abi_version": br.abi_version(),
    "module_path": str(module_path),
    "parameter_dim": dim,
    "checkpoint_bytes": len(bundle),
    "replay_output": probe_after,
    "raw_cffi_available": True,
    "typed_marker": True,
    "f32_buffer_fast_path": True,
    "sequence_fallback": True,
    "performance_evidence": {
        "parameter_dim": perf_dim,
        "repetitions": PERF_REPS,
        "list_fallback_median_ms": list_median_ms,
        "f32_buffer_median_ms": buffer_median_ms,
        "list_over_buffer_ratio": performance_ratio,
        "timing_is_ci_threshold": False,
    },
}, sort_keys=True))
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
    with tempfile.TemporaryDirectory(prefix="burn-research-python-wheel-") as tmp:
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
            raise SystemExit("installed-wheel consumer produced no output")
        print(lines[-1])


if __name__ == "__main__":
    main()
