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
from pathlib import Path

import burn_research_ffi
from burn_research_ffi import ffi, lib

OK = 0
owned = []


def last_error() -> str:
    required = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", required + 1)
    lib.br_v1_last_error_copy(buf, required + 1)
    return ffi.string(buf).decode("utf-8", errors="replace")


def check(status, context: str) -> None:
    if int(status) != OK:
        raise AssertionError(f"{context}: status={int(status)} error={last_error()!r}")


def new_handle(fn, *args):
    out = ffi.new("br_v1_handle **")
    check(fn(*args, out), "constructor/call")
    if out[0] == ffi.NULL:
        raise AssertionError("ABI returned null handle on success")
    owned.append(out[0])
    return out[0]


def read_u8(handle) -> bytes:
    n = ffi.new("size_t *")
    check(lib.br_v1_u8_buffer_len(handle, n), "u8 len")
    if int(n[0]) == 0:
        return b""
    buf = ffi.new("uint8_t[]", int(n[0]))
    check(lib.br_v1_u8_buffer_copy(handle, buf, n[0]), "u8 copy")
    return bytes(ffi.buffer(buf, int(n[0])))


def read_f32(handle) -> list[float]:
    n = ffi.new("size_t *")
    check(lib.br_v1_f32_buffer_len(handle, n), "f32 len")
    if int(n[0]) == 0:
        return []
    buf = ffi.new("float[]", int(n[0]))
    check(lib.br_v1_f32_buffer_copy(handle, buf, n[0]), "f32 copy")
    return [float(buf[i]) for i in range(int(n[0]))]


def tensor(values: list[float]):
    raw = ffi.new("float[]", values)
    return new_handle(lib.br_v1_tensor_new_f32, raw, len(values), 1, len(values), 1, 1)


def run_scalar(graph, registry, values: list[float]) -> float:
    inp = tensor(values)
    out = new_handle(lib.br_v1_graph_run, graph, registry, inp)
    n = ffi.new("size_t *")
    check(lib.br_v1_tensor_len(out, n), "tensor len")
    assert int(n[0]) == 1
    buf = ffi.new("float[]", 1)
    check(lib.br_v1_tensor_copy_f32(out, buf, 1), "tensor copy")
    value = float(buf[0])
    assert math.isfinite(value)
    return value


try:
    repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
    module_path = Path(burn_research_ffi.__file__).resolve()
    if repo_root == module_path or repo_root in module_path.parents:
        raise AssertionError(f"wheel consumer imported from repository: {module_path}")

    assert int(lib.br_v1_abi_version()) == 1
    caps_h = new_handle(lib.br_v1_capabilities_json)
    caps = json.loads(read_u8(caps_h).decode("utf-8"))
    assert caps["schema"] == "burn-research.ffi.v1"
    assert caps["host_policy"] == "external"

    registry = new_handle(lib.br_v1_registry_new)
    linear = new_handle(lib.br_v1_layer_linear, 52_001, 2, 1, 1)
    check(lib.br_v1_registry_init_layer(registry, linear), "registry init")

    builder = new_handle(lib.br_v1_graph_builder_new, 2)
    check(lib.br_v1_graph_builder_add_unary(builder, linear, 0, 1), "add unary")
    check(lib.br_v1_graph_builder_set_output(builder, 1), "set output")
    graph = new_handle(lib.br_v1_graph_builder_compile, builder, registry)
    binding = new_handle(lib.br_v1_binding_build, graph, registry)

    dim = ffi.new("size_t *")
    check(lib.br_v1_binding_total_len(binding, dim), "binding total len")
    assert int(dim[0]) == 3

    program_id_h = new_handle(lib.br_v1_graph_program_identity, graph)
    binding_id_h = new_handle(lib.br_v1_binding_identity_json, binding)
    program_identity = read_u8(program_id_h)
    binding_identity = read_u8(binding_id_h)

    optimizer = new_handle(lib.br_v1_es_strict, 3, 0, 9917, 4, 0.2, 1, 0.05)
    ask_h = new_handle(lib.br_v1_es_ask, optimizer)
    candidates = read_f32(ask_h)
    batch = ffi.new("uint32_t *")
    check(lib.br_v1_es_batch_size(optimizer, batch), "batch size")
    assert len(candidates) == int(batch[0]) * 3

    rows = [(-1.0, 0.5), (0.0, 0.0), (1.0, -0.5)]
    fitness = []
    for i in range(int(batch[0])):
        candidate = candidates[i * 3 : (i + 1) * 3]
        assert all(math.isfinite(v) for v in candidate)
        raw = ffi.new("float[]", candidate)
        check(lib.br_v1_binding_apply_flat(binding, graph, registry, raw, 3), "candidate apply")

        squared = 0.0
        for x0, x1 in rows:
            target = 1.5 * x0 - 0.75 * x1 + 0.25
            error = run_scalar(graph, registry, [x0, x1]) - target
            squared += error * error
        fitness.append(-(squared / len(rows)))

    raw_fitness = ffi.new("float[]", fitness)
    report_h = new_handle(lib.br_v1_es_tell, optimizer, raw_fitness, len(fitness))
    report = json.loads(read_u8(report_h).decode("utf-8"))
    assert int(report["gen"]) == 1

    best_h = new_handle(lib.br_v1_es_best, optimizer)
    best = read_f32(best_h)
    assert len(best) == 3 and all(math.isfinite(v) for v in best)
    raw_best = ffi.new("float[]", best)
    check(lib.br_v1_binding_apply_flat(binding, graph, registry, raw_best, 3), "apply best")

    learned_h = new_handle(lib.br_v1_binding_read_flat, binding, graph, registry)
    learned = read_f32(learned_h)
    assert learned == best
    probe_before = run_scalar(graph, registry, [1.0, 2.0])

    bundle_h = new_handle(lib.br_v1_program_bundle_export, graph, registry, 1)
    bundle = read_u8(bundle_h)
    assert bundle

    imported_registry = new_handle(lib.br_v1_registry_new)
    raw_bundle = ffi.new("uint8_t[]", bundle)
    imported_graph = new_handle(
        lib.br_v1_program_bundle_import,
        imported_registry,
        raw_bundle,
        len(bundle),
    )
    imported_binding = new_handle(lib.br_v1_binding_build, imported_graph, imported_registry)
    imported_program_id_h = new_handle(lib.br_v1_graph_program_identity, imported_graph)
    imported_binding_id_h = new_handle(lib.br_v1_binding_identity_json, imported_binding)
    imported_flat_h = new_handle(
        lib.br_v1_binding_read_flat,
        imported_binding,
        imported_graph,
        imported_registry,
    )

    assert read_u8(imported_program_id_h) == program_identity
    assert read_u8(imported_binding_id_h) == binding_identity
    assert read_f32(imported_flat_h) == learned
    probe_after = run_scalar(imported_graph, imported_registry, [1.0, 2.0])
    assert abs(probe_after - probe_before) <= 1e-7

    print(json.dumps({
        "verdict": "PASS",
        "consumer": "installed-wheel-cffi",
        "abi_version": 1,
        "module_path": str(module_path),
        "parameter_dim": 3,
        "checkpoint_bytes": len(bundle),
        "replay_output": probe_after,
    }, sort_keys=True))
finally:
    while owned:
        handle = owned.pop()
        status = int(lib.br_v1_handle_free(handle))
        if status != OK:
            raise AssertionError(f"handle cleanup failed: {status} {last_error()!r}")
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
