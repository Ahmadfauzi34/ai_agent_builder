"""Python package for the burn-research language-neutral ABI v1.

The typed facade is the ergonomic default. Raw ``ffi`` and ``lib`` remain
available for low-level consumers and ABI diagnostics.
"""

from .burn_research_ffi import ffi, lib
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
    "ffi",
    "lib",
]
