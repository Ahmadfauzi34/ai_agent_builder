#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from cffi import FFI


REPO = Path(__file__).resolve().parents[1]
FFI_MANIFEST = REPO / "ffi" / "Cargo.toml"
LIBRARY = REPO / "ffi" / "target" / "debug" / "libburn_research_ffi.so"

CDEF = r"""
typedef struct br_v1_handle br_v1_handle;
typedef int32_t br_v1_status;

uint32_t br_v1_abi_version(void);
size_t br_v1_last_error_len(void);
size_t br_v1_last_error_copy(char *dest, size_t capacity);
br_v1_status br_v1_handle_free(br_v1_handle *handle);
br_v1_status br_v1_capabilities_json(br_v1_handle **out);

br_v1_status br_v1_registry_new(br_v1_handle **out);
br_v1_status br_v1_layer_linear(uint32_t layer_id, uint32_t in_dim, uint32_t out_dim, uint8_t bias, br_v1_handle **out);
br_v1_status br_v1_registry_init_layer(br_v1_handle *registry, const br_v1_handle *layer_spec);
br_v1_status br_v1_graph_builder_new(uint32_t num_slots, br_v1_handle **out);
br_v1_status br_v1_graph_builder_add_unary(br_v1_handle *builder, const br_v1_handle *layer_spec, uint8_t input_slot, uint8_t output_slot);
br_v1_status br_v1_graph_builder_set_output(br_v1_handle *builder, uint8_t output_slot);
br_v1_status br_v1_graph_builder_compile(const br_v1_handle *builder, const br_v1_handle *registry, br_v1_handle **out_graph);
br_v1_status br_v1_graph_program_identity(const br_v1_handle *graph, br_v1_handle **out_utf8);

br_v1_status br_v1_tensor_new_f32(const float *data, size_t len, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, br_v1_handle **out_tensor);
br_v1_status br_v1_tensor_len(const br_v1_handle *tensor, size_t *out_len);
br_v1_status br_v1_tensor_copy_f32(const br_v1_handle *tensor, float *dest, size_t dest_len);
br_v1_status br_v1_graph_run(const br_v1_handle *graph, const br_v1_handle *registry, const br_v1_handle *input, br_v1_handle **out_tensor);

br_v1_status br_v1_binding_build(const br_v1_handle *graph, const br_v1_handle *registry, br_v1_handle **out_binding);
br_v1_status br_v1_binding_total_len(const br_v1_handle *binding, size_t *out_len);
br_v1_status br_v1_binding_identity_json(const br_v1_handle *binding, br_v1_handle **out_utf8);
br_v1_status br_v1_binding_read_flat(const br_v1_handle *binding, const br_v1_handle *graph, const br_v1_handle *registry, br_v1_handle **out_f32);
br_v1_status br_v1_binding_apply_flat(const br_v1_handle *binding, const br_v1_handle *graph, br_v1_handle *registry, const float *candidate, size_t candidate_len);

br_v1_status br_v1_es_strict(uint32_t dim, uint8_t strategy, uint32_t seed, uint32_t pop, float sigma, uint8_t has_lr, float lr, br_v1_handle **out_optimizer);
br_v1_status br_v1_es_ask(br_v1_handle *optimizer, br_v1_handle **out_f32);
br_v1_status br_v1_es_batch_size(const br_v1_handle *optimizer, uint32_t *out_batch_size);
br_v1_status br_v1_es_set_learning_rate(br_v1_handle *optimizer, float learning_rate);
br_v1_status br_v1_es_tell(br_v1_handle *optimizer, const float *fitness, size_t fitness_len, br_v1_handle **out_report_utf8);
br_v1_status br_v1_es_best(const br_v1_handle *optimizer, br_v1_handle **out_f32);

br_v1_status br_v1_program_bundle_export(const br_v1_handle *graph, const br_v1_handle *registry, uint8_t include_state, br_v1_handle **out_bytes);
br_v1_status br_v1_program_bundle_import(br_v1_handle *registry, const uint8_t *bytes, size_t len, br_v1_handle **out_graph);
br_v1_status br_v1_f32_buffer_len(const br_v1_handle *buffer, size_t *out_len);
br_v1_status br_v1_f32_buffer_copy(const br_v1_handle *buffer, float *dest, size_t dest_len);
br_v1_status br_v1_u8_buffer_len(const br_v1_handle *buffer, size_t *out_len);
br_v1_status br_v1_u8_buffer_copy(const br_v1_handle *buffer, uint8_t *dest, size_t dest_len);
"""

BR_V1_OK = 0
BR_V1_INVALID_HANDLE_TYPE = 2
BR_V1_INVALID_ARGUMENT = 3
BR_V1_CORE_ERROR = 4


def run(args: list[str]) -> None:
    completed = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> None:
    if os.name != "posix":
        raise SystemExit("python FFI semantic proof currently targets the Linux CI runner only")

    run(["cargo", "build", "--manifest-path", str(FFI_MANIFEST)])
    if not LIBRARY.is_file():
        raise SystemExit(f"missing FFI library: {LIBRARY}")

    ffi = FFI()
    ffi.cdef(CDEF)
    lib = ffi.dlopen(str(LIBRARY))
    owned: list[object] = []

    def last_error() -> str:
        required = int(lib.br_v1_last_error_len())
        buf = ffi.new("char[]", required + 1)
        lib.br_v1_last_error_copy(buf, required + 1)
        return ffi.string(buf).decode("utf-8", errors="replace")

    def check(status: int, context: str) -> None:
        if int(status) != BR_V1_OK:
            raise AssertionError(f"{context}: status={int(status)} error={last_error()!r}")

    def expect_status(status: int, expected: int, context: str) -> None:
        if int(status) != expected:
            raise AssertionError(
                f"{context}: expected status={expected}, got={int(status)} error={last_error()!r}"
            )

    def new_handle(fn, *args):
        out = ffi.new("br_v1_handle **")
        check(fn(*args, out), fn.__name__ if hasattr(fn, "__name__") else "new_handle")
        if out[0] == ffi.NULL:
            raise AssertionError("FFI constructor returned a null handle on success")
        owned.append(out[0])
        return out[0]

    def free_handle(handle) -> None:
        if handle == ffi.NULL:
            return
        check(lib.br_v1_handle_free(handle), "br_v1_handle_free")
        try:
            owned.remove(handle)
        except ValueError:
            pass

    def read_u8(handle) -> bytes:
        out_len = ffi.new("size_t *")
        check(lib.br_v1_u8_buffer_len(handle, out_len), "br_v1_u8_buffer_len")
        if out_len[0] == 0:
            return b""
        buf = ffi.new("uint8_t[]", out_len[0])
        check(lib.br_v1_u8_buffer_copy(handle, buf, out_len[0]), "br_v1_u8_buffer_copy")
        return bytes(ffi.buffer(buf, out_len[0]))

    def read_f32(handle) -> list[float]:
        out_len = ffi.new("size_t *")
        check(lib.br_v1_f32_buffer_len(handle, out_len), "br_v1_f32_buffer_len")
        if out_len[0] == 0:
            return []
        buf = ffi.new("float[]", out_len[0])
        check(lib.br_v1_f32_buffer_copy(handle, buf, out_len[0]), "br_v1_f32_buffer_copy")
        return [float(buf[i]) for i in range(out_len[0])]

    def tensor(values: list[float], shape: tuple[int, int, int, int]):
        raw = ffi.new("float[]", values)
        return new_handle(
            lib.br_v1_tensor_new_f32,
            raw,
            len(values),
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

    def run_scalar(graph, registry, values: list[float]) -> float:
        inp = tensor(values, (1, len(values), 1, 1))
        out = new_handle(lib.br_v1_graph_run, graph, registry, inp)
        out_len = ffi.new("size_t *")
        check(lib.br_v1_tensor_len(out, out_len), "br_v1_tensor_len")
        assert int(out_len[0]) == 1
        data = ffi.new("float[]", 1)
        check(lib.br_v1_tensor_copy_f32(out, data, 1), "br_v1_tensor_copy_f32")
        value = float(data[0])
        free_handle(out)
        free_handle(inp)
        return value

    try:
        assert int(lib.br_v1_abi_version()) == 1
        caps_h = new_handle(lib.br_v1_capabilities_json)
        caps = json.loads(read_u8(caps_h).decode("utf-8"))
        assert caps["schema"] == "burn-research.ffi.v1"
        assert caps["host_policy"] == "external"

        registry = new_handle(lib.br_v1_registry_new)
        linear = new_handle(lib.br_v1_layer_linear, 42_101, 2, 1, 1)
        check(lib.br_v1_registry_init_layer(registry, linear), "br_v1_registry_init_layer")
        builder = new_handle(lib.br_v1_graph_builder_new, 2)
        check(lib.br_v1_graph_builder_add_unary(builder, linear, 0, 1), "add unary")
        check(lib.br_v1_graph_builder_set_output(builder, 1), "set output")
        graph = new_handle(lib.br_v1_graph_builder_compile, builder, registry)
        binding = new_handle(lib.br_v1_binding_build, graph, registry)

        # Cross-type misuse is rejected by the ABI instead of interpreting object layout.
        wrong_len = ffi.new("size_t *")
        expect_status(
            lib.br_v1_binding_total_len(registry, wrong_len),
            BR_V1_INVALID_HANDLE_TYPE,
            "wrong handle type",
        )

        total_len = ffi.new("size_t *")
        check(lib.br_v1_binding_total_len(binding, total_len), "binding total len")
        assert int(total_len[0]) == 3

        # Invalid shape is rejected before WasmTensor::new can hit its native panic path.
        bad_out = ffi.new("br_v1_handle **")
        bad_data = ffi.new("float[]", [1.0, 2.0])
        expect_status(
            lib.br_v1_tensor_new_f32(bad_data, 2, 1, 3, 1, 1, bad_out),
            BR_V1_INVALID_ARGUMENT,
            "invalid tensor shape",
        )
        assert bad_out[0] == ffi.NULL

        program_id_h = new_handle(lib.br_v1_graph_program_identity, graph)
        binding_id_h = new_handle(lib.br_v1_binding_identity_json, binding)
        program_identity = read_u8(program_id_h)
        binding_identity = read_u8(binding_id_h)

        initial_h = new_handle(lib.br_v1_binding_read_flat, binding, graph, registry)
        initial = read_f32(initial_h)
        assert len(initial) == 3

        # Non-finite graph candidates retain the core finite-only/atomic boundary.
        poisoned = ffi.new("float[]", [initial[0], float("nan"), initial[2]])
        expect_status(
            lib.br_v1_binding_apply_flat(binding, graph, registry, poisoned, 3),
            BR_V1_CORE_ERROR,
            "non-finite candidate",
        )
        after_reject_h = new_handle(lib.br_v1_binding_read_flat, binding, graph, registry)
        assert read_f32(after_reject_h) == initial

        # In-place OpenES learning-rate control preserves search/RNG state.
        lr_control = new_handle(lib.br_v1_es_strict, 3, 0, 4242, 8, 0.08, 1, 0.10)
        lr_changed = new_handle(lib.br_v1_es_strict, 3, 0, 4242, 8, 0.08, 1, 0.10)

        first_control_h = new_handle(lib.br_v1_es_ask, lr_control)
        first_changed_h = new_handle(lib.br_v1_es_ask, lr_changed)
        assert read_f32(first_control_h) == read_f32(first_changed_h)

        continuity_fitness_1 = ffi.new(
            "float[]", [2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.5, -0.5]
        )
        first_control_report_h = new_handle(
            lib.br_v1_es_tell, lr_control, continuity_fitness_1, 8
        )
        first_changed_report_h = new_handle(
            lib.br_v1_es_tell, lr_changed, continuity_fitness_1, 8
        )
        assert json.loads(read_u8(first_control_report_h).decode("utf-8"))["gen"] == 1
        assert json.loads(read_u8(first_changed_report_h).decode("utf-8"))["gen"] == 1

        expect_status(
            lib.br_v1_es_set_learning_rate(lr_changed, 0.0),
            BR_V1_INVALID_ARGUMENT,
            "invalid learning rate",
        )
        check(
            lib.br_v1_es_set_learning_rate(lr_changed, 0.08),
            "set OpenES learning rate between generations",
        )

        # LR is not part of ask(), so unchanged mean + RNG must produce an
        # exactly identical next batch after an in-place LR mutation.
        second_control_h = new_handle(lib.br_v1_es_ask, lr_control)
        second_changed_h = new_handle(lib.br_v1_es_ask, lr_changed)
        assert read_f32(second_control_h) == read_f32(second_changed_h)

        # A pending batch is immutable with respect to optimizer control.
        expect_status(
            lib.br_v1_es_set_learning_rate(lr_changed, 0.07),
            BR_V1_CORE_ERROR,
            "pending-batch learning-rate mutation",
        )

        continuity_fitness_2 = ffi.new(
            "float[]", [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.25, -0.25]
        )
        second_control_report_h = new_handle(
            lib.br_v1_es_tell, lr_control, continuity_fitness_2, 8
        )
        second_changed_report_h = new_handle(
            lib.br_v1_es_tell, lr_changed, continuity_fitness_2, 8
        )
        second_control_report = json.loads(
            read_u8(second_control_report_h).decode("utf-8")
        )
        second_changed_report = json.loads(
            read_u8(second_changed_report_h).decode("utf-8")
        )
        assert second_control_report["gen"] == second_changed_report["gen"] == 2
        assert abs(float(second_control_report["lr"]) - 0.10) <= 1e-6
        assert abs(float(second_changed_report["lr"]) - 0.08) <= 1e-6

        third_control_h = new_handle(lib.br_v1_es_ask, lr_control)
        third_changed_h = new_handle(lib.br_v1_es_ask, lr_changed)
        assert read_f32(third_control_h) != read_f32(third_changed_h)

        mu_lambda = new_handle(lib.br_v1_es_strict, 3, 1, 4242, 8, 0.08, 0, 0.0)
        expect_status(
            lib.br_v1_es_set_learning_rate(mu_lambda, 0.08),
            BR_V1_CORE_ERROR,
            "mu-lambda learning-rate mutation",
        )

        optimizer = new_handle(lib.br_v1_es_strict, 3, 0, 1777, 8, 0.2, 1, 0.05)
        rows = [
            (-1.0, -1.0),
            (-1.0, 1.0),
            (0.0, 0.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ]

        # Python owns the objective and evaluation schedule; Rust owns candidate semantics.
        for _ in range(5):
            ask_h = new_handle(lib.br_v1_es_ask, optimizer)
            candidates = read_f32(ask_h)
            batch = ffi.new("uint32_t *")
            check(lib.br_v1_es_batch_size(optimizer, batch), "ES batch size")
            assert len(candidates) == int(batch[0]) * 3
            fitness: list[float] = []
            for index in range(int(batch[0])):
                candidate = candidates[index * 3 : (index + 1) * 3]
                assert all(math.isfinite(value) for value in candidate)
                raw_candidate = ffi.new("float[]", candidate)
                check(
                    lib.br_v1_binding_apply_flat(binding, graph, registry, raw_candidate, 3),
                    "candidate apply",
                )
                squared = 0.0
                for x0, x1 in rows:
                    target = 1.5 * x0 - 0.75 * x1 + 0.25
                    error = run_scalar(graph, registry, [x0, x1]) - target
                    squared += error * error
                fitness.append(-(squared / len(rows)))
            raw_fitness = ffi.new("float[]", fitness)
            report_h = new_handle(lib.br_v1_es_tell, optimizer, raw_fitness, len(fitness))
            report = json.loads(read_u8(report_h).decode("utf-8"))
            assert int(report["gen"]) >= 1
            free_handle(report_h)
            free_handle(ask_h)

        best_h = new_handle(lib.br_v1_es_best, optimizer)
        best = read_f32(best_h)
        assert len(best) == 3 and all(math.isfinite(value) for value in best)
        raw_best = ffi.new("float[]", best)
        check(lib.br_v1_binding_apply_flat(binding, graph, registry, raw_best, 3), "apply best")

        probe_before = run_scalar(graph, registry, [1.0, 2.0])
        learned_h = new_handle(lib.br_v1_binding_read_flat, binding, graph, registry)
        learned = read_f32(learned_h)
        assert learned == best

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

        print(
            json.dumps(
                {
                    "verdict": "PASS",
                    "schema": caps["schema"],
                    "consumer": "python-cffi-semantic-proof",
                    "abi_version": 1,
                    "parameter_dim": 3,
                    "checkpoint_bytes": len(bundle),
                    "replay_output": probe_after,
                },
                sort_keys=True,
            )
        )
    finally:
        while owned:
            handle = owned.pop()
            if handle != ffi.NULL:
                status = int(lib.br_v1_handle_free(handle))
                if status != BR_V1_OK:
                    raise AssertionError(f"handle cleanup failed: {status} {last_error()!r}")


if __name__ == "__main__":
    main()
