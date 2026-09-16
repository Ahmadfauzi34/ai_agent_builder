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
REPORT = REPO / "python-f32-buffer-fast-path-report.json"

CONSUMER = r'''
from __future__ import annotations

from array import array
import json
import math
import os
import statistics
import struct
import time
from pathlib import Path

import burn_research_ffi
from burn_research_ffi import (
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    Registry,
    Tensor,
    ffi,
    lib,
)

REPS = 17
BATCH = 8
CASES = [
    {"name": "owner1-w64", "width": 64, "depth": 1},
    {"name": "owner4-w64", "width": 64, "depth": 4},
    {"name": "owner16-w64", "width": 64, "depth": 16},
]


def ns() -> int:
    return time.perf_counter_ns()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def summary(samples_ns: list[int]) -> dict[str, float]:
    values = sorted(sample / 1_000_000.0 for sample in samples_ns)
    if not values:
        raise AssertionError("empty timing sample set")
    p90_index = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
    return {
        "median_ms": statistics.median(values),
        "min_ms": values[0],
        "p90_ms": values[p90_index],
        "max_ms": values[-1],
    }


def expected_params(width: int, depth: int) -> int:
    return depth * (width * width + width)


def make_candidate(length: int) -> tuple[float, ...]:
    return tuple(f32((((i * 37) % 101) - 50) * 0.0005) for i in range(length))


def make_input(width: int) -> list[float]:
    values: list[float] = []
    for b in range(BATCH):
        for j in range(width):
            values.append(f32(0.35 * math.sin((b + 1) * (j + 2) * 0.071)))
    return values


def check(status: int, context: str) -> None:
    if int(status) == 0:
        return
    required = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", required + 1)
    lib.br_v1_last_error_copy(buf, required + 1)
    message = ffi.string(buf).decode("utf-8", errors="replace")
    raise AssertionError(f"{context}: status={int(status)} error={message!r}")


def validate_native_f32_buffer(obj: object, expected_len: int) -> memoryview:
    view = memoryview(obj)
    if view.ndim != 1:
        raise ValueError("candidate buffer must be one-dimensional")
    if not view.c_contiguous:
        raise ValueError("candidate buffer must be C-contiguous")
    if view.itemsize != 4:
        raise ValueError("candidate buffer itemsize must be 4 bytes")
    if view.format != "f":
        raise ValueError(f"candidate buffer format must be native float32 'f', got {view.format!r}")
    if len(view) != expected_len:
        raise ValueError(
            f"candidate buffer length mismatch: expected {expected_len}, got {len(view)}"
        )
    return view


def acquire_native_f32_buffer(obj: object, expected_len: int):
    validate_native_f32_buffer(obj, expected_len)
    cdata = ffi.from_buffer("float[]", obj)
    if len(cdata) != expected_len:
        raise AssertionError(
            f"CFFI buffer length mismatch: expected {expected_len}, got {len(cdata)}"
        )
    return cdata


class RawHandles:
    def __init__(self) -> None:
        self.owned: list[object] = []

    def new(self, fn, *args):
        out = ffi.new("br_v1_handle **")
        check(fn(*args, out), getattr(fn, "__name__", "raw constructor"))
        if out[0] == ffi.NULL:
            raise AssertionError("raw ABI constructor returned null on success")
        self.owned.append(out[0])
        return out[0]

    def free(self, handle) -> None:
        if handle == ffi.NULL:
            return
        check(lib.br_v1_handle_free(handle), "br_v1_handle_free")
        for index, candidate in enumerate(self.owned):
            if candidate == handle:
                self.owned.pop(index)
                break

    def close(self) -> None:
        while self.owned:
            handle = self.owned.pop()
            check(lib.br_v1_handle_free(handle), "br_v1_handle_free")

    def read_u8(self, handle) -> bytes:
        out_len = ffi.new("size_t *")
        check(lib.br_v1_u8_buffer_len(handle, out_len), "br_v1_u8_buffer_len")
        length = int(out_len[0])
        if length == 0:
            return b""
        dest = ffi.new("uint8_t[]", length)
        check(lib.br_v1_u8_buffer_copy(handle, dest, length), "br_v1_u8_buffer_copy")
        return bytes(ffi.buffer(dest, length))

    def read_f32_buffer(self, handle) -> list[float]:
        out_len = ffi.new("size_t *")
        check(lib.br_v1_f32_buffer_len(handle, out_len), "br_v1_f32_buffer_len")
        length = int(out_len[0])
        if length == 0:
            return []
        dest = ffi.new("float[]", length)
        check(lib.br_v1_f32_buffer_copy(handle, dest, length), "br_v1_f32_buffer_copy")
        return [float(dest[i]) for i in range(length)]

    def text(self, fn, *args) -> str:
        handle = self.new(fn, *args)
        try:
            return self.read_u8(handle).decode("utf-8")
        finally:
            self.free(handle)

    def read_flat(self, binding, graph, registry) -> list[float]:
        handle = self.new(lib.br_v1_binding_read_flat, binding, graph, registry)
        try:
            return self.read_f32_buffer(handle)
        finally:
            self.free(handle)


def close_facade(items: list[object]) -> None:
    while items:
        item = items.pop()
        close = getattr(item, "close", None)
        if close is not None:
            close()


def build_raw(width: int, depth: int, owner: RawHandles):
    registry = owner.new(lib.br_v1_registry_new)
    builder = owner.new(lib.br_v1_graph_builder_new, depth + 1)
    layers = []
    for index in range(depth):
        layer_id = 93_000 + depth * 100 + index
        layer = owner.new(lib.br_v1_layer_linear, layer_id, width, width, 1)
        layers.append(layer)
        check(lib.br_v1_registry_init_layer(registry, layer), "raw registry init")
        check(
            lib.br_v1_graph_builder_add_unary(builder, layer, index, index + 1),
            "raw graph add unary",
        )
    check(lib.br_v1_graph_builder_set_output(builder, depth), "raw set output")
    graph = owner.new(lib.br_v1_graph_builder_compile, builder, registry)
    binding = owner.new(lib.br_v1_binding_build, graph, registry)
    return registry, builder, graph, binding, layers


def build_facade(width: int, depth: int):
    owned: list[object] = []
    registry = Registry(); owned.append(registry)
    builder = GraphBuilder(depth + 1); owned.append(builder)
    layers: list[LinearLayerSpec] = []
    for index in range(depth):
        layer_id = 93_000 + depth * 100 + index
        layer = LinearLayerSpec(layer_id, width, width, bias=True)
        owned.append(layer)
        layers.append(layer)
        registry.init_layer(layer)
        builder.add_unary(layer, index, index + 1)
    builder.set_output(depth)
    graph = builder.compile(registry); owned.append(graph)
    binding = GraphParameterBinding.build(graph, registry); owned.append(binding)
    return registry, builder, graph, binding, layers, owned


def raw_total_len(binding) -> int:
    out = ffi.new("size_t *")
    check(lib.br_v1_binding_total_len(binding, out), "raw binding total len")
    return int(out[0])


def raw_tensor(owner: RawHandles, values: list[float], shape: tuple[int, int, int, int]):
    raw = ffi.new("float[]", values)
    return owner.new(
        lib.br_v1_tensor_new_f32,
        raw,
        len(values),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
    )


def raw_run(owner: RawHandles, graph, registry, input_tensor) -> list[float]:
    output = owner.new(lib.br_v1_graph_run, graph, registry, input_tensor)
    try:
        length = ffi.new("size_t *")
        check(lib.br_v1_tensor_len(output, length), "raw tensor len")
        n = int(length[0])
        dest = ffi.new("float[]", n)
        check(lib.br_v1_tensor_copy_f32(output, dest, n), "raw tensor copy")
        return [float(dest[i]) for i in range(n)]
    finally:
        owner.free(output)


def prove_rejections() -> dict[str, bool]:
    rejected = {
        "float64_itemsize": False,
        "non_contiguous": False,
        "wrong_format": False,
        "wrong_length": False,
    }
    try:
        validate_native_f32_buffer(array("d", [1.0, 2.0]), 2)
    except ValueError:
        rejected["float64_itemsize"] = True

    base = array("f", [1.0, 2.0, 3.0, 4.0])
    try:
        validate_native_f32_buffer(memoryview(base)[::2], 2)
    except ValueError:
        rejected["non_contiguous"] = True

    try:
        validate_native_f32_buffer(bytearray(8), 2)
    except ValueError:
        rejected["wrong_format"] = True

    try:
        validate_native_f32_buffer(base, 3)
    except ValueError:
        rejected["wrong_length"] = True

    if not all(rejected.values()):
        raise AssertionError(f"buffer rejection proof incomplete: {rejected}")
    return rejected


def evaluate(case: dict[str, int | str]) -> dict[str, object]:
    width = int(case["width"])
    depth = int(case["depth"])
    expected = expected_params(width, depth)
    candidate = make_candidate(expected)
    candidate_list = list(candidate)
    candidate_array = array("f", candidate)
    if candidate_array.itemsize != 4:
        raise AssertionError(f"array('f') itemsize is {candidate_array.itemsize}, expected 4")
    if list(candidate_array) != candidate_list:
        raise AssertionError("array('f') changed already-rounded f32 candidate")
    input_values = make_input(width)

    raw_owner = RawHandles()
    facade_owned: list[object] = []
    try:
        raw_registry, raw_builder, raw_graph, raw_binding, raw_layers = build_raw(
            width, depth, raw_owner
        )
        (
            facade_registry,
            facade_builder,
            facade_graph,
            facade_binding,
            facade_layers,
            facade_owned,
        ) = build_facade(width, depth)

        if raw_total_len(raw_binding) != expected or facade_binding.total_len != expected:
            raise AssertionError(f"{case['name']}: canonical parameter count mismatch")

        raw_program_identity = raw_owner.text(lib.br_v1_graph_program_identity, raw_graph)
        raw_binding_identity = raw_owner.text(lib.br_v1_binding_identity_json, raw_binding)
        if raw_program_identity != facade_graph.program_identity():
            raise AssertionError(f"{case['name']}: program identity mismatch")
        if raw_binding_identity != facade_binding.identity():
            raise AssertionError(f"{case['name']}: binding identity mismatch")

        list_cdata = ffi.new("float[]", candidate_list)
        check(
            lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, list_cdata, expected
            ),
            "baseline raw list candidate apply",
        )
        baseline_state = raw_owner.read_flat(raw_binding, raw_graph, raw_registry)
        if baseline_state != candidate_list:
            raise AssertionError(f"{case['name']}: baseline candidate readback mismatch")

        persistent_buffer = acquire_native_f32_buffer(candidate_array, expected)
        # Keep both Python backing storage and CFFI view strongly referenced for the full loop.
        persistent_lifetime_guard = (candidate_array, persistent_buffer)
        if persistent_lifetime_guard[0] is not candidate_array:
            raise AssertionError("persistent backing buffer lifetime guard failed")
        check(
            lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, persistent_buffer, expected
            ),
            "buffer candidate initial apply",
        )
        if raw_owner.read_flat(raw_binding, raw_graph, raw_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: buffer candidate readback mismatch")

        facade_binding.apply_flat(facade_graph, facade_registry, candidate)
        if facade_binding.read_flat(facade_graph, facade_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: facade candidate readback mismatch")

        raw_input = raw_tensor(raw_owner, input_values, (BATCH, width, 1, 1))
        facade_input = Tensor.from_f32(input_values, (BATCH, width, 1, 1))
        facade_owned.append(facade_input)
        raw_output = raw_run(raw_owner, raw_graph, raw_registry, raw_input)
        facade_output_tensor = facade_graph.run(facade_registry, facade_input)
        facade_output = facade_output_tensor.to_f32()
        facade_output_tensor.close()
        if not all(math.isfinite(value) for value in raw_output + facade_output):
            raise AssertionError(f"{case['name']}: non-finite output")
        if raw_output != facade_output:
            raise AssertionError(f"{case['name']}: buffer/list execution output mismatch")

        if raw_owner.text(lib.br_v1_graph_program_identity, raw_graph) != raw_program_identity:
            raise AssertionError(f"{case['name']}: raw program identity drift")
        if raw_owner.text(lib.br_v1_binding_identity_json, raw_binding) != raw_binding_identity:
            raise AssertionError(f"{case['name']}: raw binding identity drift")

        memoryview_supported = True
        memoryview_error = None
        candidate_memoryview = memoryview(candidate_array)
        try:
            memoryview_cdata = acquire_native_f32_buffer(candidate_memoryview, expected)
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, memoryview_cdata, expected
                ),
                "memoryview candidate apply",
            )
            if raw_owner.read_flat(raw_binding, raw_graph, raw_registry) != candidate_list:
                raise AssertionError(f"{case['name']}: memoryview readback mismatch")
        except (TypeError, ValueError, BufferError) as exc:
            memoryview_supported = False
            memoryview_error = f"{type(exc).__name__}: {exc}"

        # Warm up each measured path.
        _ = [float(value) for value in candidate]
        _ = ffi.new("float[]", candidate_list)
        _ = array("f", candidate)
        _ = validate_native_f32_buffer(candidate_array, expected)
        _ = ffi.from_buffer("float[]", candidate_array)
        for _ in range(3):
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, list_cdata, expected
                ),
                "list preallocated warmup",
            )
            fresh_view = acquire_native_f32_buffer(candidate_array, expected)
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, fresh_view, expected
                ),
                "fresh buffer warmup",
            )
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, persistent_buffer, expected
                ),
                "persistent buffer warmup",
            )
            facade_binding.apply_flat(facade_graph, facade_registry, candidate)

        normalization_samples: list[int] = []
        list_allocation_samples: list[int] = []
        array_construction_samples: list[int] = []
        validation_samples: list[int] = []
        from_buffer_samples: list[int] = []
        list_preallocated_apply_samples: list[int] = []
        fresh_buffer_apply_samples: list[int] = []
        persistent_buffer_apply_samples: list[int] = []
        facade_samples: list[int] = []
        memoryview_fresh_apply_samples: list[int] = []

        for _ in range(REPS):
            t0 = ns()
            normalized = [float(value) for value in candidate]
            t1 = ns()
            if len(normalized) != expected:
                raise AssertionError("normalization length drift")
            normalization_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            allocated = ffi.new("float[]", candidate_list)
            t1 = ns()
            if expected and float(allocated[expected - 1]) != candidate_list[-1]:
                raise AssertionError("list CFFI allocation content mismatch")
            list_allocation_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            constructed = array("f", candidate)
            t1 = ns()
            if constructed.itemsize != 4 or len(constructed) != expected:
                raise AssertionError("array construction mismatch")
            array_construction_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            view = validate_native_f32_buffer(candidate_array, expected)
            t1 = ns()
            if len(view) != expected:
                raise AssertionError("validated view length mismatch")
            validation_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            acquired = ffi.from_buffer("float[]", candidate_array)
            t1 = ns()
            if len(acquired) != expected:
                raise AssertionError("from_buffer length mismatch")
            from_buffer_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            status = lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, list_cdata, expected
            )
            t1 = ns()
            check(status, "preallocated list-CFFI apply")
            list_preallocated_apply_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            fresh_view = acquire_native_f32_buffer(candidate_array, expected)
            status = lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, fresh_view, expected
            )
            t1 = ns()
            check(status, "fresh buffer-view apply")
            fresh_buffer_apply_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            status = lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, persistent_buffer, expected
            )
            t1 = ns()
            check(status, "persistent buffer-view apply")
            persistent_buffer_apply_samples.append(t1 - t0)

        if memoryview_supported:
            for _ in range(REPS):
                t0 = ns()
                fresh_mv = acquire_native_f32_buffer(candidate_memoryview, expected)
                status = lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, fresh_mv, expected
                )
                t1 = ns()
                check(status, "memoryview fresh apply")
                memoryview_fresh_apply_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            facade_binding.apply_flat(facade_graph, facade_registry, candidate)
            t1 = ns()
            facade_samples.append(t1 - t0)

        if raw_owner.read_flat(raw_binding, raw_graph, raw_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: raw final state drift")
        if facade_binding.read_flat(facade_graph, facade_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: facade final state drift")
        if raw_owner.text(lib.br_v1_graph_program_identity, raw_graph) != raw_program_identity:
            raise AssertionError(f"{case['name']}: raw program identity changed")
        if raw_owner.text(lib.br_v1_binding_identity_json, raw_binding) != raw_binding_identity:
            raise AssertionError(f"{case['name']}: raw binding identity changed")

        list_preallocated = summary(list_preallocated_apply_samples)
        fresh_buffer = summary(fresh_buffer_apply_samples)
        persistent = summary(persistent_buffer_apply_samples)
        facade = summary(facade_samples)

        return {
            "name": case["name"],
            "width": width,
            "depth": depth,
            "owner_count": depth,
            "parameter_count": expected,
            "semantic": {
                "array_itemsize": candidate_array.itemsize,
                "candidate_readback_exact": True,
                "raw_facade_output_exact": True,
                "identities_stable": True,
                "backing_array_strong_reference_held": True,
                "memoryview_supported": memoryview_supported,
                "memoryview_error": memoryview_error,
            },
            "timing": {
                "python_list_normalization": summary(normalization_samples),
                "cffi_list_allocation_copy": summary(list_allocation_samples),
                "array_f_construction": summary(array_construction_samples),
                "buffer_validation": summary(validation_samples),
                "from_buffer_acquisition": summary(from_buffer_samples),
                "raw_preallocated_list_cffi_apply": list_preallocated,
                "raw_array_fresh_view_apply": fresh_buffer,
                "raw_array_persistent_view_apply": persistent,
                "raw_memoryview_fresh_apply": (
                    summary(memoryview_fresh_apply_samples)
                    if memoryview_fresh_apply_samples
                    else None
                ),
                "facade_end_to_end_apply": facade,
                "fresh_buffer_over_persistent_ratio": (
                    fresh_buffer["median_ms"] / max(persistent["median_ms"], 1e-12)
                ),
                "facade_over_fresh_buffer_ratio": (
                    facade["median_ms"] / max(fresh_buffer["median_ms"], 1e-12)
                ),
                "facade_over_persistent_buffer_ratio": (
                    facade["median_ms"] / max(persistent["median_ms"], 1e-12)
                ),
                "fresh_buffer_saved_vs_facade_ms": (
                    facade["median_ms"] - fresh_buffer["median_ms"]
                ),
            },
        }
    finally:
        close_facade(facade_owned)
        raw_owner.close()


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(burn_research_ffi.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"research imported package from checkout: {module_path}")
if int(lib.br_v1_abi_version()) != 1:
    raise AssertionError("unexpected ABI version")

rejections = prove_rejections()
results = [evaluate(case) for case in CASES]
report = {
    "verdict": "PASS",
    "consumer": "installed-wheel-public-raw-abi-buffer-research",
    "workload": "python-standard-library-f32-buffer-fast-path-v1",
    "module_path": str(module_path),
    "samples": {"repetitions": REPS, "batch": BATCH},
    "rejection_proof": rejections,
    "cases": results,
    "decision_boundary": {
        "timings_are_ci_thresholds": False,
        "abi_changed": False,
        "facade_changed": False,
        "core_semantics_changed": False,
        "binding_cache_added": False,
        "third_party_array_dependency_added": False,
        "private_facade_handles_used": False,
    },
}
print(json.dumps(report, sort_keys=True))
'''


def run(
    args: list[str], *, cwd: Path, env: dict[str, str] | None = None
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
    with tempfile.TemporaryDirectory(prefix="burn-research-f32-buffer-") as tmp:
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
            raise SystemExit("f32-buffer consumer produced no output")
        report = json.loads(lines[-1])
        REPORT.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
