from __future__ import annotations

from typing import Any

from .facade import (
    BurnResearchError,
    ClosedHandleError,
    EsOptimizer,
    Graph,
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    ProgramBundle,
    Registry,
    Status,
    Tensor,
    abi_version,
    capabilities,
)


HOST_API_VERSION = 1
HOST_API_SCHEMA = "burn-research.python-host.v1"


def host_capabilities() -> dict[str, Any]:
    """Describe the Python host surface without changing the native ABI.

    The Python host layer owns ergonomics and orchestration at the package level.
    Execution, trainable-state semantics, optimizer behavior, and checkpoint
    semantics remain delegated to the existing versioned ABI and Rust core.
    """

    foreign = capabilities()
    return {
        "schema": HOST_API_SCHEMA,
        "version": HOST_API_VERSION,
        "abi_version": abi_version(),
        "abi_schema": foreign.get("schema"),
        "orchestration": "host_owned",
        "typed": True,
        "raw_ffi_primary": False,
    }


__all__ = [
    "BurnResearchError",
    "ClosedHandleError",
    "EsOptimizer",
    "Graph",
    "GraphBuilder",
    "GraphParameterBinding",
    "HOST_API_SCHEMA",
    "HOST_API_VERSION",
    "LinearLayerSpec",
    "ProgramBundle",
    "Registry",
    "Status",
    "Tensor",
    "abi_version",
    "host_capabilities",
]
