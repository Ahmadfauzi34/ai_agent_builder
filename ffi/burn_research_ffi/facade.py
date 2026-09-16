from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Sequence

from .burn_research_ffi import ffi, lib


class Status(IntEnum):
    OK = 0
    NULL_POINTER = 1
    INVALID_HANDLE_TYPE = 2
    INVALID_ARGUMENT = 3
    CORE_ERROR = 4
    PANIC = 5
    BUFFER_TOO_SMALL = 6


class BurnResearchError(RuntimeError):
    """Failure returned by the versioned burn-research foreign ABI."""

    def __init__(self, status_code: int, context: str, diagnostic: str) -> None:
        self.status_code = int(status_code)
        try:
            self.status: Status | int = Status(self.status_code)
        except ValueError:
            self.status = self.status_code
        self.context = context
        self.diagnostic = diagnostic
        super().__init__(
            f"{context}: status={self.status_code} diagnostic={diagnostic!r}"
        )


class ClosedHandleError(RuntimeError):
    """Raised locally when Python tries to use a facade object after close()."""


def _last_error() -> str:
    required = int(lib.br_v1_last_error_len())
    buf = ffi.new("char[]", required + 1)
    lib.br_v1_last_error_copy(buf, required + 1)
    return ffi.string(buf).decode("utf-8", errors="replace")


def _check(status: Any, context: str) -> None:
    code = int(status)
    if code != int(Status.OK):
        raise BurnResearchError(code, context, _last_error())


def _new_handle(context: str, fn: Any, *args: Any) -> Any:
    out = ffi.new("br_v1_handle **")
    _check(fn(*args, out), context)
    if out[0] == ffi.NULL:
        raise RuntimeError(f"{context}: ABI returned a null handle on success")
    return out[0]


def _u32(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0 or value > 0xFFFF_FFFF:
        raise ValueError(f"{name} must fit uint32")
    return value


def _u8(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0 or value > 0xFF:
        raise ValueError(f"{name} must fit uint8")
    return value


def _f32_buffer_view(
    candidate: object,
    expected_len: int,
) -> tuple[memoryview, Any] | None:
    """Borrow a compatible native f32 buffer for exactly one ABI call.

    Objects with non-f32 buffers fall back to the historical Sequence path. Once
    an object presents itself as native f32 storage, however, malformed shape,
    contiguity, or length fails locally rather than being silently reinterpreted.
    """

    try:
        view = memoryview(candidate)
    except TypeError:
        return None

    if view.format != "f" or view.itemsize != 4:
        return None
    if view.ndim != 1:
        raise ValueError("candidate f32 buffer must be one-dimensional")
    if not view.c_contiguous:
        raise ValueError("candidate f32 buffer must be C-contiguous")
    if len(view) != expected_len:
        raise ValueError(
            f"candidate f32 buffer length must equal binding total_len "
            f"({expected_len}), got {len(view)}"
        )

    raw = ffi.from_buffer("float[]", view)
    return view, raw


class _OwnedHandle:
    __slots__ = ("_handle", "_closed")

    def __init__(self, handle: Any) -> None:
        if handle == ffi.NULL:
            raise ValueError("owned handle cannot be null")
        self._handle = handle
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _borrow(self) -> Any:
        if self._closed:
            raise ClosedHandleError(
                f"{type(self).__name__} has already been closed"
            )
        return self._handle

    def close(self) -> None:
        if self._closed:
            return
        handle = self._handle
        self._handle = ffi.NULL
        self._closed = True
        _check(lib.br_v1_handle_free(handle), f"close {type(self).__name__}")

    def __enter__(self) -> _OwnedHandle:
        self._borrow()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must never leak exceptions. Deterministic callers should
            # use close() or a context manager and will receive the ABI error there.
            pass


class _Buffer(_OwnedHandle):
    def to_bytes(self) -> bytes:
        handle = self._borrow()
        n = ffi.new("size_t *")
        _check(lib.br_v1_u8_buffer_len(handle, n), "u8 buffer len")
        size = int(n[0])
        if size == 0:
            return b""
        dest = ffi.new("uint8_t[]", size)
        _check(lib.br_v1_u8_buffer_copy(handle, dest, size), "u8 buffer copy")
        return bytes(ffi.buffer(dest, size))

    def to_f32(self) -> list[float]:
        handle = self._borrow()
        n = ffi.new("size_t *")
        _check(lib.br_v1_f32_buffer_len(handle, n), "f32 buffer len")
        size = int(n[0])
        if size == 0:
            return []
        dest = ffi.new("float[]", size)
        _check(lib.br_v1_f32_buffer_copy(handle, dest, size), "f32 buffer copy")
        return [float(dest[i]) for i in range(size)]


def abi_version() -> int:
    return int(lib.br_v1_abi_version())


def capabilities() -> dict[str, Any]:
    with _Buffer(_new_handle("capabilities", lib.br_v1_capabilities_json)) as buf:
        return json.loads(buf.to_bytes().decode("utf-8"))


class LinearLayerSpec(_OwnedHandle):
    def __init__(
        self,
        layer_id: int,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
    ) -> None:
        handle = _new_handle(
            "linear layer spec",
            lib.br_v1_layer_linear,
            _u32(layer_id, "layer_id"),
            _u32(in_dim, "in_dim"),
            _u32(out_dim, "out_dim"),
            1 if bias else 0,
        )
        super().__init__(handle)


class Registry(_OwnedHandle):
    def __init__(self) -> None:
        super().__init__(_new_handle("registry new", lib.br_v1_registry_new))

    def init_layer(self, layer: LinearLayerSpec) -> None:
        if not isinstance(layer, LinearLayerSpec):
            raise TypeError("layer must be LinearLayerSpec")
        _check(
            lib.br_v1_registry_init_layer(self._borrow(), layer._borrow()),
            "registry init layer",
        )


class Tensor(_OwnedHandle):
    @classmethod
    def from_f32(
        cls,
        values: Sequence[float],
        shape: tuple[int, int, int, int],
    ) -> Tensor:
        if len(shape) != 4:
            raise ValueError("shape must contain exactly four dimensions")
        dims = tuple(_u32(dim, f"shape[{i}]") for i, dim in enumerate(shape))
        data = [float(value) for value in values]
        raw = ffi.new("float[]", data)
        return cls(
            _new_handle(
                "tensor new f32",
                lib.br_v1_tensor_new_f32,
                raw,
                len(data),
                dims[0],
                dims[1],
                dims[2],
                dims[3],
            )
        )

    @classmethod
    def vector(cls, values: Sequence[float]) -> Tensor:
        data = [float(value) for value in values]
        return cls.from_f32(data, (1, len(data), 1, 1))

    @property
    def length(self) -> int:
        out = ffi.new("size_t *")
        _check(lib.br_v1_tensor_len(self._borrow(), out), "tensor len")
        return int(out[0])

    def to_f32(self) -> list[float]:
        size = self.length
        if size == 0:
            return []
        dest = ffi.new("float[]", size)
        _check(
            lib.br_v1_tensor_copy_f32(self._borrow(), dest, size),
            "tensor copy f32",
        )
        return [float(dest[i]) for i in range(size)]


class Graph(_OwnedHandle):
    def program_identity(self) -> str:
        with _Buffer(
            _new_handle(
                "graph program identity",
                lib.br_v1_graph_program_identity,
                self._borrow(),
            )
        ) as buf:
            return buf.to_bytes().decode("utf-8")

    def run(self, registry: Registry, input_tensor: Tensor) -> Tensor:
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        if not isinstance(input_tensor, Tensor):
            raise TypeError("input_tensor must be Tensor")
        return Tensor(
            _new_handle(
                "graph run",
                lib.br_v1_graph_run,
                self._borrow(),
                registry._borrow(),
                input_tensor._borrow(),
            )
        )


class GraphBuilder(_OwnedHandle):
    def __init__(self, num_slots: int) -> None:
        super().__init__(
            _new_handle(
                "graph builder new",
                lib.br_v1_graph_builder_new,
                _u32(num_slots, "num_slots"),
            )
        )

    def add_unary(
        self,
        layer: LinearLayerSpec,
        input_slot: int,
        output_slot: int,
    ) -> GraphBuilder:
        if not isinstance(layer, LinearLayerSpec):
            raise TypeError("layer must be LinearLayerSpec")
        _check(
            lib.br_v1_graph_builder_add_unary(
                self._borrow(),
                layer._borrow(),
                _u8(input_slot, "input_slot"),
                _u8(output_slot, "output_slot"),
            ),
            "graph builder add unary",
        )
        return self

    def set_output(self, output_slot: int) -> GraphBuilder:
        _check(
            lib.br_v1_graph_builder_set_output(
                self._borrow(), _u8(output_slot, "output_slot")
            ),
            "graph builder set output",
        )
        return self

    def compile(self, registry: Registry) -> Graph:
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        return Graph(
            _new_handle(
                "graph builder compile",
                lib.br_v1_graph_builder_compile,
                self._borrow(),
                registry._borrow(),
            )
        )


class GraphParameterBinding(_OwnedHandle):
    @classmethod
    def build(cls, graph: Graph, registry: Registry) -> GraphParameterBinding:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be Graph")
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        return cls(
            _new_handle(
                "binding build",
                lib.br_v1_binding_build,
                graph._borrow(),
                registry._borrow(),
            )
        )

    @property
    def total_len(self) -> int:
        out = ffi.new("size_t *")
        _check(lib.br_v1_binding_total_len(self._borrow(), out), "binding total len")
        return int(out[0])

    def layout(self) -> dict[str, Any]:
        with _Buffer(
            _new_handle(
                "binding layout",
                lib.br_v1_binding_layout_json,
                self._borrow(),
            )
        ) as buf:
            return json.loads(buf.to_bytes().decode("utf-8"))

    def identity(self) -> str:
        with _Buffer(
            _new_handle(
                "binding identity",
                lib.br_v1_binding_identity_json,
                self._borrow(),
            )
        ) as buf:
            return buf.to_bytes().decode("utf-8")

    def read_flat(self, graph: Graph, registry: Registry) -> list[float]:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be Graph")
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        with _Buffer(
            _new_handle(
                "binding read flat",
                lib.br_v1_binding_read_flat,
                self._borrow(),
                graph._borrow(),
                registry._borrow(),
            )
        ) as buf:
            return buf.to_f32()

    def apply_flat(
        self,
        graph: Graph,
        registry: Registry,
        candidate: Sequence[float],
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be Graph")
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")

        expected_len = self.total_len
        borrowed = _f32_buffer_view(candidate, expected_len)
        if borrowed is not None:
            # Keep both the memoryview and CFFI cdata alive until the ABI call
            # returns. No pointer/view is stored on the facade object.
            view, raw = borrowed
            _check(
                lib.br_v1_binding_apply_flat(
                    self._borrow(),
                    graph._borrow(),
                    registry._borrow(),
                    raw,
                    len(view),
                ),
                "binding apply flat",
            )
            return

        values = [float(value) for value in candidate]
        raw = ffi.new("float[]", values)
        _check(
            lib.br_v1_binding_apply_flat(
                self._borrow(),
                graph._borrow(),
                registry._borrow(),
                raw,
                len(values),
            ),
            "binding apply flat",
        )


class EsOptimizer(_OwnedHandle):
    @classmethod
    def strict(
        cls,
        dim: int,
        *,
        strategy: int = 0,
        seed: int = 0,
        population: int = 4,
        sigma: float = 0.2,
        learning_rate: float | None = None,
    ) -> EsOptimizer:
        return cls(
            _new_handle(
                "es strict",
                lib.br_v1_es_strict,
                _u32(dim, "dim"),
                _u8(strategy, "strategy"),
                _u32(seed, "seed"),
                _u32(population, "population"),
                float(sigma),
                0 if learning_rate is None else 1,
                0.0 if learning_rate is None else float(learning_rate),
            )
        )

    @property
    def batch_size(self) -> int:
        out = ffi.new("uint32_t *")
        _check(lib.br_v1_es_batch_size(self._borrow(), out), "es batch size")
        return int(out[0])

    def ask(self) -> list[float]:
        with _Buffer(
            _new_handle("es ask", lib.br_v1_es_ask, self._borrow())
        ) as buf:
            return buf.to_f32()

    def tell(self, fitness: Sequence[float]) -> dict[str, Any]:
        values = [float(value) for value in fitness]
        raw = ffi.new("float[]", values)
        with _Buffer(
            _new_handle(
                "es tell",
                lib.br_v1_es_tell,
                self._borrow(),
                raw,
                len(values),
            )
        ) as buf:
            return json.loads(buf.to_bytes().decode("utf-8"))

    def best(self) -> list[float]:
        with _Buffer(
            _new_handle("es best", lib.br_v1_es_best, self._borrow())
        ) as buf:
            return buf.to_f32()


class ProgramBundle:
    """Canonical ProgramBundle bytes exposed through the existing ABI v1."""

    @staticmethod
    def export(graph: Graph, registry: Registry, *, include_state: bool = True) -> bytes:
        if not isinstance(graph, Graph):
            raise TypeError("graph must be Graph")
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        with _Buffer(
            _new_handle(
                "program bundle export",
                lib.br_v1_program_bundle_export,
                graph._borrow(),
                registry._borrow(),
                1 if include_state else 0,
            )
        ) as buf:
            return buf.to_bytes()

    @staticmethod
    def import_graph(registry: Registry, payload: bytes | bytearray | memoryview) -> Graph:
        if not isinstance(registry, Registry):
            raise TypeError("registry must be Registry")
        data = bytes(payload)
        raw = ffi.new("uint8_t[]", data)
        return Graph(
            _new_handle(
                "program bundle import",
                lib.br_v1_program_bundle_import,
                registry._borrow(),
                raw,
                len(data),
            )
        )


__all__ = [
    "BurnResearchError",
    "ClosedHandleError",
    "EsOptimizer",
    "Graph",
    "GraphBuilder",
    "GraphParameterBinding",
    "LinearLayerSpec",
    "ProgramBundle",
    "Registry",
    "Status",
    "Tensor",
    "abi_version",
    "capabilities",
]
