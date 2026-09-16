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
from array import array
from contextlib import ExitStack
from pathlib import Path

import burn_research_ffi as br
from burn_research_ffi import host


class BufferOnlyF32(array):
    """Writable native-f32 buffer whose Python iteration must never be used."""

    def __new__(cls, values):
        return array.__new__(cls, "f", values)

    def __iter__(self):
        raise AssertionError("compatible f32 candidate fell back to Sequence iteration")


def run_scalar(graph, registry, values: list[float]) -> float:
    with host.Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            assert out.length == 1
            result = out.to_f32()
            assert len(result) == 1 and math.isfinite(result[0])
            return result[0]


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"wheel consumer imported from repository: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"host namespace imported from repository: {host_path}")

# The first-class host namespace is the primary typed surface, while the raw
# ABI remains separately available for diagnostics/escape-hatch consumers.
assert br.host is host
assert br.ffi is not None and br.lib is not None
assert int(br.lib.br_v1_abi_version()) == 1
assert br.abi_version() == 1
assert host.abi_version() == 1
assert host.HOST_API_VERSION == 1
assert host.HOST_API_SCHEMA == "burn-research.python-host.v1"
assert module_path.with_name("py.typed").is_file()

# Root-level facade exports remain compatibility aliases, not a second engine.
assert br.Registry is host.Registry
assert br.GraphBuilder is host.GraphBuilder
assert br.GraphParameterBinding is host.GraphParameterBinding
assert br.EsOptimizer is host.EsOptimizer
assert br.ProgramBundle is host.ProgramBundle

foreign_caps = br.capabilities()
assert foreign_caps["schema"] == "burn-research.ffi.v1"
assert foreign_caps["host_policy"] == "external"
host_caps = host.host_capabilities()
assert host_caps == {
    "schema": "burn-research.python-host.v1",
    "version": 1,
    "abi_version": 1,
    "abi_schema": "burn-research.ffi.v1",
    "orchestration": "host_owned",
    "typed": True,
    "raw_ffi_primary": False,
}

# Python ownership must be deterministic: double-close is harmless and a
# closed object is rejected locally before another FFI call is attempted.
closed_registry = host.Registry()
closed_registry.close()
closed_registry.close()
assert closed_registry.closed
with host.LinearLayerSpec(99_999, 1, 1) as scratch_layer:
    try:
        closed_registry.init_layer(scratch_layer)
    except host.ClosedHandleError:
        pass
    else:
        raise AssertionError("use-after-close was not rejected locally")

with ExitStack() as stack:
    registry = stack.enter_context(host.Registry())
    linear = stack.enter_context(host.LinearLayerSpec(52_001, 2, 1, bias=True))
    registry.init_layer(linear)

    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(linear, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))

    dim = binding.total_len
    assert dim == 3
    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    assert binding.layout()

    # Historical generic Sequence behavior remains intact.
    list_candidate = [0.125, -0.25, 0.5]
    binding.apply_flat(graph, registry, list_candidate)
    assert binding.read_flat(graph, registry) == list_candidate

    tuple_candidate = (0.25, 0.5, -0.75)
    binding.apply_flat(graph, registry, tuple_candidate)
    assert binding.read_flat(graph, registry) == list(tuple_candidate)

    # A non-f32 buffer-backed Sequence must not be reinterpreted as f32. It uses
    # the generic conversion path and therefore remains compatible.
    f64_candidate = array("d", [0.5, -0.25, 0.125])
    binding.apply_flat(graph, registry, f64_candidate)
    assert binding.read_flat(graph, registry) == [0.5, -0.25, 0.125]

    # This object deliberately cannot be iterated. Success proves the public
    # method borrowed its compatible native-f32 buffer instead of normalizing it.
    fast_candidate = BufferOnlyF32([0.75, -0.5, 0.25])
    binding.apply_flat(graph, registry, fast_candidate)
    assert binding.read_flat(graph, registry) == [0.75, -0.5, 0.25]

    # A writable contiguous native-f32 memoryview follows the same stateless path.
    memory_candidate_backing = array("f", [-0.25, 0.375, 0.625])
    memory_candidate = memoryview(memory_candidate_backing)
    binding.apply_flat(graph, registry, memory_candidate)
    assert binding.read_flat(graph, registry) == list(memory_candidate_backing)

    # Non-contiguous native-f32 storage is not borrowed as a raw pointer. Because
    # this memoryview is still a valid Python sequence, compatibility fallback
    # applies it element-by-element without changing public behavior.
    non_contiguous_backing = array("f", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    non_contiguous = memoryview(non_contiguous_backing)[::2]
    assert len(non_contiguous) == dim and not non_contiguous.c_contiguous
    binding.apply_flat(graph, registry, non_contiguous)
    assert binding.read_flat(graph, registry) == [1.0, 3.0, 5.0]

    # Length validation still belongs to the existing ABI/core boundary. The
    # transport fast path forwards the actual buffer length rather than creating
    # a new local ValueError contract, and structural rejection remains atomic.
    before_wrong_length = binding.read_flat(graph, registry)
    wrong_length = array("f", [1.0, 2.0])
    try:
        binding.apply_flat(graph, registry, wrong_length)
    except host.BurnResearchError as exc:
        assert exc.status == host.Status.CORE_ERROR
        assert exc.status_code == int(host.Status.CORE_ERROR)
    else:
        raise AssertionError("wrong-length f32 buffer unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_wrong_length

    # Stable ABI status -> Python exception mapping remains intact on the fast
    # path, and the core finite-only rejection remains atomic.
    before_reject = binding.read_flat(graph, registry)
    poisoned = array("f", [before_reject[0], math.nan, before_reject[2]])
    try:
        binding.apply_flat(graph, registry, poisoned)
    except host.BurnResearchError as exc:
        assert exc.status == host.Status.CORE_ERROR
        assert exc.status_code == int(host.Status.CORE_ERROR)
    else:
        raise AssertionError("non-finite f32 buffer candidate unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_reject
    assert graph.program_identity() == program_identity
    assert binding.identity() == binding_identity

    optimizer = stack.enter_context(
        host.EsOptimizer.strict(
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

    # Python owns the objective and evaluation schedule. Optimizer-generated
    # Python lists continue to exercise the compatibility Sequence path.
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
    best_buffer = array("f", best)
    binding.apply_flat(graph, registry, best_buffer)
    learned = binding.read_flat(graph, registry)
    assert learned == list(best_buffer)
    assert graph.program_identity() == program_identity
    assert binding.identity() == binding_identity
    probe_before = run_scalar(graph, registry, [1.0, 2.0])

    bundle = host.ProgramBundle.export(graph, registry, include_state=True)
    assert bundle

    imported_registry = stack.enter_context(host.Registry())
    imported_graph = stack.enter_context(
        host.ProgramBundle.import_graph(imported_registry, bundle)
    )
    imported_binding = stack.enter_context(
        host.GraphParameterBinding.build(imported_graph, imported_registry)
    )

    assert imported_graph.program_identity() == program_identity
    assert imported_binding.identity() == binding_identity
    assert imported_binding.read_flat(imported_graph, imported_registry) == learned
    probe_after = run_scalar(imported_graph, imported_registry, [1.0, 2.0])
    assert abs(probe_after - probe_before) <= 1e-7

print(json.dumps({
    "verdict": "PASS",
    "consumer": "installed-wheel-python-host-v1",
    "host_api": host.HOST_API_SCHEMA,
    "host_api_version": host.HOST_API_VERSION,
    "abi_version": host.abi_version(),
    "module_path": str(module_path),
    "host_module_path": str(host_path),
    "parameter_dim": dim,
    "checkpoint_bytes": len(bundle),
    "replay_output": probe_after,
    "orchestration": host_caps["orchestration"],
    "raw_cffi_available": True,
    "typed_marker": True,
    "f32_buffer_fast_path_proven": True,
    "f32_buffer_fast_path_stateless": True,
    "sequence_fallback_proven": True,
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
