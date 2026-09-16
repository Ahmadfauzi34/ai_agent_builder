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


def run_scalar(graph, registry, values: list[float]) -> float:
    with Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            assert out.length == 1
            result = out.to_f32()
            assert len(result) == 1 and math.isfinite(result[0])
            return result[0]


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

    # Stable ABI status -> Python exception mapping, while preserving atomicity.
    before_reject = binding.read_flat(graph, registry)
    try:
        binding.apply_flat(graph, registry, [math.nan, 0.0, 0.0])
    except BurnResearchError as exc:
        assert exc.status == Status.CORE_ERROR
        assert exc.status_code == int(Status.CORE_ERROR)
    else:
        raise AssertionError("non-finite candidate unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_reject

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
    binding.apply_flat(graph, registry, best)
    learned = binding.read_flat(graph, registry)
    assert learned == best
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
