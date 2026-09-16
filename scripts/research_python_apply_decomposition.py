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
REPORT = REPO / "python-apply-decomposition-report.json"

CONSUMER = r'''
from __future__ import annotations

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
    {"name": "owner1-w128", "width": 128, "depth": 1},
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
        layer_id = 91_000 + depth * 100 + index
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
        layer_id = 91_000 + depth * 100 + index
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


def evaluate(case: dict[str, int | str]) -> dict[str, object]:
    width = int(case["width"])
    depth = int(case["depth"])
    expected = expected_params(width, depth)
    candidate = make_candidate(expected)
    candidate_list = list(candidate)
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

        if raw_total_len(raw_binding) != expected:
            raise AssertionError(f"{case['name']}: raw total_len mismatch")
        if facade_binding.total_len != expected:
            raise AssertionError(f"{case['name']}: facade total_len mismatch")

        raw_program_identity = raw_owner.text(lib.br_v1_graph_program_identity, raw_graph)
        raw_binding_identity = raw_owner.text(lib.br_v1_binding_identity_json, raw_binding)
        facade_program_identity = facade_graph.program_identity()
        facade_binding_identity = facade_binding.identity()
        if raw_program_identity != facade_program_identity:
            raise AssertionError(f"{case['name']}: raw/facade program identity mismatch")
        if raw_binding_identity != facade_binding_identity:
            raise AssertionError(f"{case['name']}: raw/facade binding identity mismatch")

        # Apply the same canonical f32 vector through both public paths.
        preallocated = ffi.new("float[]", candidate_list)
        check(
            lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, preallocated, expected
            ),
            "raw initial candidate apply",
        )
        facade_binding.apply_flat(facade_graph, facade_registry, candidate)
        if raw_owner.read_flat(raw_binding, raw_graph, raw_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: raw candidate readback mismatch")
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
            raise AssertionError(f"{case['name']}: raw/facade output mismatch")

        if raw_owner.text(lib.br_v1_graph_program_identity, raw_graph) != raw_program_identity:
            raise AssertionError(f"{case['name']}: raw program identity drift")
        if raw_owner.text(lib.br_v1_binding_identity_json, raw_binding) != raw_binding_identity:
            raise AssertionError(f"{case['name']}: raw binding identity drift")
        if facade_graph.program_identity() != facade_program_identity:
            raise AssertionError(f"{case['name']}: facade program identity drift")
        if facade_binding.identity() != facade_binding_identity:
            raise AssertionError(f"{case['name']}: facade binding identity drift")

        # Warm up all measured paths independently.
        _ = [float(value) for value in candidate]
        _ = ffi.new("float[]", candidate_list)
        for _ in range(3):
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, preallocated, expected
                ),
                "raw preallocated warmup",
            )
            fresh = ffi.new("float[]", candidate_list)
            check(
                lib.br_v1_binding_apply_flat(
                    raw_binding, raw_graph, raw_registry, fresh, expected
                ),
                "raw allocate warmup",
            )
            facade_binding.apply_flat(facade_graph, facade_registry, candidate)

        normalization_samples: list[int] = []
        cffi_allocation_samples: list[int] = []
        raw_preallocated_samples: list[int] = []
        raw_allocate_apply_samples: list[int] = []
        facade_samples: list[int] = []

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
                raise AssertionError("CFFI allocation content mismatch")
            cffi_allocation_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            status = lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, preallocated, expected
            )
            t1 = ns()
            check(status, "raw preallocated apply")
            raw_preallocated_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            fresh = ffi.new("float[]", candidate_list)
            status = lib.br_v1_binding_apply_flat(
                raw_binding, raw_graph, raw_registry, fresh, expected
            )
            t1 = ns()
            check(status, "raw allocate+apply")
            raw_allocate_apply_samples.append(t1 - t0)

        for _ in range(REPS):
            t0 = ns()
            facade_binding.apply_flat(facade_graph, facade_registry, candidate)
            t1 = ns()
            facade_samples.append(t1 - t0)

        if raw_owner.read_flat(raw_binding, raw_graph, raw_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: raw state drift after timing")
        if facade_binding.read_flat(facade_graph, facade_registry) != candidate_list:
            raise AssertionError(f"{case['name']}: facade state drift after timing")
        if raw_owner.text(lib.br_v1_graph_program_identity, raw_graph) != raw_program_identity:
            raise AssertionError(f"{case['name']}: raw program identity changed during timing")
        if raw_owner.text(lib.br_v1_binding_identity_json, raw_binding) != raw_binding_identity:
            raise AssertionError(f"{case['name']}: raw binding identity changed during timing")
        if facade_graph.program_identity() != facade_program_identity:
            raise AssertionError(f"{case['name']}: facade program identity changed during timing")
        if facade_binding.identity() != facade_binding_identity:
            raise AssertionError(f"{case['name']}: facade binding identity changed during timing")

        normalization = summary(normalization_samples)
        allocation = summary(cffi_allocation_samples)
        raw_preallocated = summary(raw_preallocated_samples)
        raw_allocate_apply = summary(raw_allocate_apply_samples)
        facade = summary(facade_samples)
        raw_median = raw_preallocated["median_ms"]
        facade_median = facade["median_ms"]
        component_sum = (
            normalization["median_ms"]
            + allocation["median_ms"]
            + raw_median
        )

        return {
            "name": case["name"],
            "width": width,
            "depth": depth,
            "owner_count": depth,
            "parameter_count": expected,
            "program_identity": facade_program_identity,
            "binding_identity": facade_binding_identity,
            "semantic": {
                "raw_facade_output_exact": True,
                "candidate_readback_exact": True,
                "identities_stable": True,
                "finite_output": True,
            },
            "timing": {
                "python_normalization": normalization,
                "cffi_allocation_copy": allocation,
                "raw_preallocated_apply": raw_preallocated,
                "raw_allocate_apply": raw_allocate_apply,
                "facade_end_to_end_apply": facade,
                "facade_over_raw_preallocated_ratio": (
                    facade_median / max(raw_median, 1e-12)
                ),
                "raw_preallocated_share_of_facade_median": (
                    raw_median / max(facade_median, 1e-12)
                ),
                "facade_minus_raw_preallocated_ms": facade_median - raw_median,
                "raw_allocate_minus_preallocated_ms": (
                    raw_allocate_apply["median_ms"] - raw_median
                ),
                "component_sum_median_ms": component_sum,
                "facade_minus_component_sum_ms": facade_median - component_sum,
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

results = [evaluate(case) for case in CASES]
report = {
    "verdict": "PASS",
    "consumer": "installed-wheel-public-raw-abi-and-typed-facade",
    "workload": "python-binding-apply-decomposition-v1",
    "module_path": str(module_path),
    "samples": {"repetitions": REPS, "batch": BATCH},
    "cases": results,
    "decision_boundary": {
        "timings_are_ci_thresholds": False,
        "abi_changed": False,
        "facade_changed": False,
        "core_semantics_changed": False,
        "binding_cache_added": False,
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
    with tempfile.TemporaryDirectory(prefix="burn-research-apply-decomposition-") as tmp:
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
            raise SystemExit("apply-decomposition consumer produced no output")
        report = json.loads(lines[-1])
        REPORT.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
