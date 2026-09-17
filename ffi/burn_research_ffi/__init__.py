"""Python host package for the burn-research reference machine.

``burn_research_ffi.host`` is the primary typed host API. Raw ``ffi`` and
``lib`` remain available for low-level consumers and ABI diagnostics.
The historical root-level facade exports remain for compatibility.
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
    ReluLayerSpec,
    Status,
    Tensor,
    abi_version,
    capabilities,
)
from . import host as host
from .host import HOST_API_SCHEMA, HOST_API_VERSION, host_capabilities

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
    "ReluLayerSpec",
    "Status",
    "Tensor",
    "abi_version",
    "capabilities",
    "ffi",
    "host",
    "host_capabilities",
    "lib",
]
