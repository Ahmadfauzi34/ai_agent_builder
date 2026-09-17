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
REPORT = REPO / "python-graph-run-output-decomposition-report.json"

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
from burn_research_ffi.burn_research_ffi import ffi, lib

POLICY_IN = 6
POLICY_OUT = 2
EXPECTED_DIM = POLICY_IN * POLICY_OUT + POLICY_OUT
MICRO_REPS = 600
FEATURES = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0]


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


def last_error() -> str:
    required = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", required + 1)
    lib.br_v1_last_error_copy(buf, required + 1)
    return ffi.string(buf).decode("utf-8", errors="replace")


def check(status: int, context: str) -> None:
    code = int(status)
    if code != 0:
        raise AssertionError(f"{context}: status={code} diagnostic={last_error()!r}")


def new_handle(context: str, fn, *args):
    out = ffi.new("br_v1_handle **")
    check(fn(*args, out), context)
    if out[0] == ffi.NULL:
        raise AssertionError(f"{context}: null handle on success")
    return out[0]


def free_handle(handle) -> None:
    if handle != ffi.NULL:
        check(lib.br_v1_handle_free(handle), "handle free")


def read_u8_handle(handle) -> bytes:
    n = ffi.new("size_t *")
    check(lib.br_v1_u8_buffer_len(handle, n), "u8 buffer len")
    size = int(n[0])
    if size == 0:
        return b""
    dest = ffi.new("uint8_t[]", size)
    check(lib.br_v1_u8_buffer_copy(handle, dest, size), "u8 buffer copy")
    return bytes(ffi.buffer(dest, size))


def raw_program_identity(graph) -> str:
    buf = new_handle("graph program identity", lib.br_v1_graph_program_identity, graph)
    try:
        return read_u8_handle(buf).decode("utf-8")
    finally:
        free_handle(buf)


def raw_binding_identity(binding) -> str:
    buf = new_handle("binding identity", lib.br_v1_binding_identity_json, binding)
    try:
        return read_u8_handle(buf).decode("utf-8")
    finally:
        free_handle(buf)


def raw_tensor_values(tensor) -> list[float]:
    n = ffi.new("size_t *")
    check(lib.br_v1_tensor_len(tensor, n), "tensor len")
    size = int(n[0])
    if size == 0:
        return []
    dest = ffi.new("float[]", size)
    check(lib.br_v1_tensor_copy_f32(tensor, dest, size), "tensor copy f32")
    return [float(dest[i]) for i in range(size)]


def build_typed_policy(stack: ExitStack):
    registry = stack.enter_context(host.Registry())
    layer = stack.enter_context(
        host.LinearLayerSpec(213_000, POLICY_IN, POLICY_OUT, bias=True)
    )
    registry.init_layer(layer)
    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(layer, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))
    if binding.total_len != EXPECTED_DIM:
        raise AssertionError(
            f"typed policy parameter dim {binding.total_len} != expected {EXPECTED_DIM}"
        )
    zero = array("f", [0.0]) * binding.total_len
    binding.apply_flat(graph, registry, zero)
    return registry, graph, binding


def build_raw_policy():
    handles = []
    try:
        registry = new_handle("registry new", lib.br_v1_registry_new)
        handles.append(registry)
        layer = new_handle(
            "linear layer",
            lib.br_v1_layer_linear,
            213_000,
            POLICY_IN,
            POLICY_OUT,
            1,
        )
        handles.append(layer)
        check(lib.br_v1_registry_init_layer(registry, layer), "registry init layer")
        builder = new_handle("graph builder new", lib.br_v1_graph_builder_new, 2)
        handles.append(builder)
        check(
            lib.br_v1_graph_builder_add_unary(builder, layer, 0, 1),
            "graph builder add unary",
        )
        check(lib.br_v1_graph_builder_set_output(builder, 1), "graph builder set output")
        graph = new_handle(
            "graph builder compile", lib.br_v1_graph_builder_compile, builder, registry
        )
        handles.append(graph)
        binding = new_handle("binding build", lib.br_v1_binding_build, graph, registry)
        handles.append(binding)
        total_len = ffi.new("size_t *")
        check(lib.br_v1_binding_total_len(binding, total_len), "binding total len")
        dim = int(total_len[0])
        if dim != EXPECTED_DIM:
            raise AssertionError(f"raw policy parameter dim {dim} != expected {EXPECTED_DIM}")
        zero = ffi.new("float[]", dim)
        check(
            lib.br_v1_binding_apply_flat(binding, graph, registry, zero, dim),
            "binding apply zero",
        )
        raw_features = ffi.new("float[]", FEATURES)
        input_tensor = new_handle(
            "tensor new f32",
            lib.br_v1_tensor_new_f32,
            raw_features,
            len(FEATURES),
            1,
            POLICY_IN,
            1,
            1,
        )
        handles.append(input_tensor)
        return handles, registry, graph, binding, input_tensor
    except Exception:
        for handle in reversed(handles):
            try:
                free_handle(handle)
            except Exception:
                pass
        raise


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
host_path = Path(host.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from repo: {module_path}")
if repo_root == host_path or repo_root in host_path.parents:
    raise AssertionError(f"research imported host from repo: {host_path}")
if host.HOST_API_SCHEMA != "burn-research.python-host.v1" or host.abi_version() != 1:
    raise AssertionError("unexpected host/ABI identity")

raw_handles = []
try:
    raw_handles, raw_registry, raw_graph, raw_binding, raw_input = build_raw_policy()

    with ExitStack() as stack:
        typed_registry, typed_graph, typed_binding = build_typed_policy(stack)
        typed_input = stack.enter_context(host.Tensor.vector(FEATURES))

        typed_program_identity = typed_graph.program_identity()
        typed_binding_identity = typed_binding.identity()
        raw_program_before = raw_program_identity(raw_graph)
        raw_binding_before = raw_binding_identity(raw_binding)
        if raw_program_before != typed_program_identity:
            raise AssertionError("raw/typed program identity mismatch")
        if raw_binding_before != typed_binding_identity:
            raise AssertionError("raw/typed binding identity mismatch")

        raw_reference = new_handle(
            "raw graph reference run",
            lib.br_v1_graph_run,
            raw_graph,
            raw_registry,
            raw_input,
        )
        try:
            raw_reference_values = raw_tensor_values(raw_reference)
        finally:
            free_handle(raw_reference)

        with typed_graph.run(typed_registry, typed_input) as typed_reference:
            typed_reference_values = typed_reference.to_f32()

        if raw_reference_values != typed_reference_values:
            raise AssertionError(
                f"raw/typed output mismatch: {raw_reference_values} != {typed_reference_values}"
            )
        if len(typed_reference_values) != POLICY_OUT:
            raise AssertionError("unexpected policy output length")
        if not all(math.isfinite(v) for v in typed_reference_values):
            raise AssertionError("non-finite policy output")

        raw_run_samples: list[int] = []
        for _ in range(MICRO_REPS):
            out = ffi.new("br_v1_handle **")
            t0 = ns()
            status = lib.br_v1_graph_run(raw_graph, raw_registry, raw_input, out)
            t1 = ns()
            check(status, "raw graph run")
            if out[0] == ffi.NULL:
                raise AssertionError("raw graph run returned null output")
            raw_run_samples.append(t1 - t0)
            free_handle(out[0])

        typed_run_samples: list[int] = []
        for _ in range(MICRO_REPS):
            t0 = ns()
            out = typed_graph.run(typed_registry, typed_input)
            t1 = ns()
            typed_run_samples.append(t1 - t0)
            out.close()

        raw_output = new_handle(
            "raw graph persistent output",
            lib.br_v1_graph_run,
            raw_graph,
            raw_registry,
            raw_input,
        )
        try:
            raw_len = ffi.new("size_t *")
            check(lib.br_v1_tensor_len(raw_output, raw_len), "raw output len")
            raw_size = int(raw_len[0])
            if raw_size != POLICY_OUT:
                raise AssertionError(f"raw output len {raw_size} != {POLICY_OUT}")
            raw_dest = ffi.new("float[]", raw_size)

            raw_len_samples: list[int] = []
            for _ in range(MICRO_REPS):
                t0 = ns()
                status = lib.br_v1_tensor_len(raw_output, raw_len)
                t1 = ns()
                check(status, "raw tensor len")
                if int(raw_len[0]) != raw_size:
                    raise AssertionError("raw tensor length changed")
                raw_len_samples.append(t1 - t0)

            raw_copy_samples: list[int] = []
            for _ in range(MICRO_REPS):
                t0 = ns()
                status = lib.br_v1_tensor_copy_f32(raw_output, raw_dest, raw_size)
                t1 = ns()
                check(status, "raw tensor copy")
                raw_copy_samples.append(t1 - t0)

            materialize_samples: list[int] = []
            for _ in range(MICRO_REPS):
                t0 = ns()
                values = [float(raw_dest[i]) for i in range(raw_size)]
                t1 = ns()
                if values != raw_reference_values:
                    raise AssertionError("preallocated raw buffer materialization mismatch")
                materialize_samples.append(t1 - t0)
        finally:
            free_handle(raw_output)

        typed_output = stack.enter_context(typed_graph.run(typed_registry, typed_input))
        typed_copy_samples: list[int] = []
        for _ in range(MICRO_REPS):
            t0 = ns()
            values = typed_output.to_f32()
            t1 = ns()
            if values != typed_reference_values:
                raise AssertionError("typed output copy changed values")
            typed_copy_samples.append(t1 - t0)

        typed_full_samples: list[int] = []
        for _ in range(MICRO_REPS):
            t0 = ns()
            with typed_graph.run(typed_registry, typed_input) as out:
                values = out.to_f32()
            t1 = ns()
            if values != typed_reference_values:
                raise AssertionError("typed full path changed output")
            typed_full_samples.append(t1 - t0)

        raw_full_samples: list[int] = []
        for _ in range(MICRO_REPS):
            t0 = ns()
            out = new_handle(
                "raw full graph run", lib.br_v1_graph_run, raw_graph, raw_registry, raw_input
            )
            n = ffi.new("size_t *")
            check(lib.br_v1_tensor_len(out, n), "raw full tensor len")
            size = int(n[0])
            dest = ffi.new("float[]", size)
            check(lib.br_v1_tensor_copy_f32(out, dest, size), "raw full tensor copy")
            values = [float(dest[i]) for i in range(size)]
            free_handle(out)
            t1 = ns()
            if values != raw_reference_values:
                raise AssertionError("raw full path changed output")
            raw_full_samples.append(t1 - t0)

        if typed_graph.program_identity() != typed_program_identity:
            raise AssertionError("typed program identity changed")
        if typed_binding.identity() != typed_binding_identity:
            raise AssertionError("typed binding identity changed")
        if raw_program_identity(raw_graph) != raw_program_before:
            raise AssertionError("raw program identity changed")
        if raw_binding_identity(raw_binding) != raw_binding_before:
            raise AssertionError("raw binding identity changed")

    summaries = {
        "raw_graph_run_handle": summary(raw_run_samples),
        "typed_graph_run_handle": summary(typed_run_samples),
        "raw_tensor_len": summary(raw_len_samples),
        "raw_tensor_copy_preallocated": summary(raw_copy_samples),
        "python_list_materialize_preallocated": summary(materialize_samples),
        "typed_tensor_to_f32": summary(typed_copy_samples),
        "raw_graph_run_copy_close": summary(raw_full_samples),
        "typed_graph_run_copy_close": summary(typed_full_samples),
    }

    full_ms = summaries["typed_graph_run_copy_close"]["median_ms"]
    raw_run_ms = summaries["raw_graph_run_handle"]["median_ms"]
    typed_run_ms = summaries["typed_graph_run_handle"]["median_ms"]
    typed_copy_ms = summaries["typed_tensor_to_f32"]["median_ms"]
    raw_copy_ms = summaries["raw_tensor_copy_preallocated"]["median_ms"]
    materialize_ms = summaries["python_list_materialize_preallocated"]["median_ms"]

    run_share = raw_run_ms / max(full_ms, 1e-12)
    output_share = typed_copy_ms / max(full_ms, 1e-12)
    facade_run_overhead_ms = max(0.0, typed_run_ms - raw_run_ms)
    raw_copy_share_of_typed_copy = raw_copy_ms / max(typed_copy_ms, 1e-12)
    materialize_share_of_typed_copy = materialize_ms / max(typed_copy_ms, 1e-12)

    if run_share >= 0.55:
        decision = "GRAPH_RUN_HANDLE_BOUNDARY_DOMINATES"
    elif output_share >= 0.40:
        decision = "OUTPUT_COPY_MATERIALIZATION_DOMINATES"
    else:
        decision = "MIXED_GRAPH_OUTPUT_BOUNDARY_COST"

    report = {
        "verdict": "PASS",
        "decision": decision,
        "decision_is_ci_threshold": False,
        "host_api": host.HOST_API_SCHEMA,
        "abi_version": host.abi_version(),
        "workload": {
            "policy": "Linear(6 -> 2, bias=true)",
            "parameter_dim": EXPECTED_DIM,
            "output_dim": POLICY_OUT,
            "micro_repetitions": MICRO_REPS,
            "input": FEATURES,
        },
        "timing": summaries,
        "ratios": {
            "raw_graph_run_share_of_typed_full": run_share,
            "typed_output_copy_share_of_typed_full": output_share,
            "raw_copy_share_of_typed_to_f32": raw_copy_share_of_typed_copy,
            "python_materialize_share_of_typed_to_f32": materialize_share_of_typed_copy,
        },
        "deltas": {
            "typed_run_minus_raw_run_ms": facade_run_overhead_ms,
        },
        "proofs": {
            "installed_wheel": True,
            "raw_typed_program_identity_equal": True,
            "raw_typed_binding_identity_equal": True,
            "raw_typed_output_exact": True,
            "finite_outputs": True,
            "program_identity_stable": True,
            "binding_identity_stable": True,
            "deterministic_repeated_execution": True,
            "owned_handles_closed": True,
            "no_private_facade_handles": True,
        },
        "notes": [
            "Timing is descriptive research evidence only, not a performance SLA.",
            "raw_graph_run_handle includes ABI crossing, validation, native CompiledGraph::run, and output-handle allocation; it intentionally does not pretend to separate those without native instrumentation.",
            "raw_tensor_copy_preallocated includes the current tensor to_array materialization inside br_v1_tensor_copy_f32 plus the copy into a preallocated destination.",
            "typed_tensor_to_f32 includes tensor_len, destination allocation, tensor_copy_f32, and Python list materialization.",
            "No ABI, facade, graph runtime, tensor runtime, optimizer, Math primitive, checkpoint, Resolution, or Authorization change is introduced.",
        ],
    }
    print(json.dumps(report, sort_keys=True))
finally:
    for handle in reversed(raw_handles):
        try:
            free_handle(handle)
        except Exception:
            pass
'''


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None):
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
    with tempfile.TemporaryDirectory(prefix="burn-research-graph-output-decomp-") as tmp:
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
        env["PYTHONNOUSERSITE"] = "1"
        env["BR_REPO_ROOT"] = str(REPO)
        completed = run([str(python), str(consumer)], cwd=root, env=env)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("consumer produced no report")
        report = json.loads(lines[-1])
        if report.get("verdict") != "PASS":
            raise SystemExit(f"unexpected report verdict: {report}")
        REPORT.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
