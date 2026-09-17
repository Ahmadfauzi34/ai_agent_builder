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
REPORT = REPO / "python-relu-layer-spec-proof-report.json"

CONSUMER = r'''
from __future__ import annotations

from array import array
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path

import burn_research_ffi as br
from burn_research_ffi import host

EXPECTED_DIM = 74


def run_policy(graph, registry, values: list[float]) -> list[float]:
    with host.Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            result = out.to_f32()
    if len(result) != 2 or not all(math.isfinite(value) for value in result):
        raise AssertionError(f"invalid policy output: {result}")
    return result


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"consumer imported package from repository: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"consumer imported host from repository: {host_path}")

assert br.host is host
assert br.abi_version() == 1
assert host.abi_version() == 1
assert host.HOST_API_SCHEMA == "burn-research.python-host.v1"
assert br.ReluLayerSpec is host.ReluLayerSpec
assert br.LinearLayerSpec is host.LinearLayerSpec
assert br.lib.br_v1_layer_relu is not None

with ExitStack() as stack:
    registry = stack.enter_context(host.Registry())
    linear_in = stack.enter_context(host.LinearLayerSpec(229_001, 6, 8, bias=True))
    relu = stack.enter_context(host.ReluLayerSpec(229_002))
    linear_out = stack.enter_context(host.LinearLayerSpec(229_003, 8, 2, bias=True))

    for layer in (linear_in, relu, linear_out):
        registry.init_layer(layer)

    try:
        registry.init_layer(object())
    except TypeError:
        pass
    else:
        raise AssertionError("registry accepted a non-layer-spec object")

    builder = stack.enter_context(host.GraphBuilder(4))
    builder.add_unary(linear_in, 0, 1)
    builder.add_unary(relu, 1, 2)
    builder.add_unary(linear_out, 2, 3)
    builder.set_output(3)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))

    if binding.total_len != EXPECTED_DIM:
        raise AssertionError(
            f"MLP parameter dim {binding.total_len} != expected {EXPECTED_DIM}"
        )

    # The canonical first-seen trainable owners are the two Linear layers only:
    # 6*8 + 8 bias = 56, then 8*2 + 2 bias = 18. ReLU contributes zero state.
    candidate = array(
        "f",
        [1.0] * 48
        + [0.0] * 8
        + [1.0] * 16
        + [0.0] * 2,
    )
    assert len(candidate) == EXPECTED_DIM

    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    binding.apply_flat(graph, registry, candidate)
    if binding.read_flat(graph, registry) != list(candidate):
        raise AssertionError("canonical MLP state did not read back exactly")

    positive = run_policy(graph, registry, [1.0] * 6)
    negative = run_policy(graph, registry, [-1.0] * 6)

    if not all(value > 1.0 for value in positive):
        raise AssertionError(f"expected positive ReLU branch, got {positive}")
    if not all(abs(value) <= 1e-7 for value in negative):
        raise AssertionError(f"expected ReLU-clamped negative branch, got {negative}")

    # With zero biases, a purely Linear chain would be odd: f(-x) == -f(x).
    # The observed zero negative branch and positive positive branch proves the
    # installed consumer graph contains a nonlinear ReLU transition.
    odd_symmetry_residual = max(
        abs(positive[index] + negative[index]) for index in range(2)
    )
    if odd_symmetry_residual <= 1.0:
        raise AssertionError(
            f"nonlinear symmetry proof too small: {odd_symmetry_residual}"
        )

    if graph.program_identity() != program_identity:
        raise AssertionError("program identity changed after apply/run")
    if binding.identity() != binding_identity:
        raise AssertionError("binding identity changed after apply/run")

    state_before = binding.read_flat(graph, registry)
    bundle = host.ProgramBundle.export(graph, registry, include_state=True)
    if not bundle:
        raise AssertionError("stateful ProgramBundle is empty")

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
    if replay_binding.read_flat(replay_graph, replay_registry) != state_before:
        raise AssertionError("replay parameter state mismatch")

    replay_positive = run_policy(replay_graph, replay_registry, [1.0] * 6)
    replay_negative = run_policy(replay_graph, replay_registry, [-1.0] * 6)
    for expected, actual in zip(positive + negative, replay_positive + replay_negative):
        if abs(expected - actual) > 1e-7:
            raise AssertionError(
                f"replay output mismatch: expected {expected}, got {actual}"
            )

report = {
    "verdict": "PASS",
    "schema": "burn-research.python-relu-layer-spec-proof.v1",
    "consumer": "fresh-installed-wheel",
    "abi_version": host.abi_version(),
    "host_api": host.HOST_API_SCHEMA,
    "module_path": str(module_path),
    "host_module_path": str(host_path),
    "graph": "Linear(6->8) -> ReLU -> Linear(8->2)",
    "parameter_dim": EXPECTED_DIM,
    "relu_trainable_parameters": 0,
    "positive_output": positive,
    "negative_output": negative,
    "odd_symmetry_residual": odd_symmetry_residual,
    "checkpoint_bytes": len(bundle),
    "proofs": {
        "raw_relu_symbol_available": True,
        "typed_relu_exported": True,
        "layer_spec_type_boundary": True,
        "canonical_binding_length": True,
        "relu_zero_trainable_state": True,
        "nonlinear_behavior": True,
        "program_identity_stable": True,
        "binding_identity_stable": True,
        "stateful_program_bundle_replay": True,
        "replay_parameters_exact": True,
        "replay_outputs_exact_within_1e-7": True,
    },
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
    with tempfile.TemporaryDirectory(prefix="burn-research-python-relu-") as tmp:
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
            raise SystemExit("ReLU installed-wheel proof produced no output")
        payload = json.loads(lines[-1])
        if payload.get("verdict") != "PASS":
            raise SystemExit(f"unexpected ReLU proof verdict: {payload}")

    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
