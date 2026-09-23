#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURFACE_PATH = ROOT / "docs" / "wasm-surface.v1.json"
AGENT_CONTRACT_PATH = ROOT / "docs" / "agent-contracts.v1.json"
LAYOUT_CONTRACT_PATH = ROOT / "docs" / "agent-layout-contracts.v1.json"
DEFAULT_PKG_PATH = ROOT / "pkg"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_class_members(body: str):
    members = set()
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith(("/**", "*", "*/")):
            continue
        if line.startswith("free():") or line.startswith("[Symbol.dispose]"):
            continue
        if line.startswith("private constructor") or line.startswith("constructor("):
            members.add("constructor")
            continue
        method = re.match(r"(?:static\s+)?([A-Za-z_]\w*)\s*\(", line)
        if method:
            members.add(method.group(1))
            continue
        field = re.match(r"([A-Za-z_]\w*)\s*:", line)
        if field:
            members.add(field.group(1))
    return members


def parse_dts(text: str):
    classes = {}
    for match in re.finditer(r"^export class (\w+) \{\n(.*?)^\}", text, re.MULTILINE | re.DOTALL):
        classes[match.group(1)] = parse_class_members(match.group(2))

    functions = set(re.findall(r"^export function ([A-Za-z0-9_]+)\(", text, re.MULTILINE))
    default_match = re.search(r"^export default function ([A-Za-z0-9_]+)", text, re.MULTILINE)
    if not default_match:
        raise AssertionError("generated d.ts has no default wasm initializer export")
    return classes, functions, default_match.group(1)


def require_equal(label, actual, expected):
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"{label} mismatch; missing={missing}; extra={extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkg",
        type=Path,
        default=DEFAULT_PKG_PATH,
        help="wasm-pack output directory (default: repository pkg/)",
    )
    args = parser.parse_args()
    pkg_path = args.pkg.resolve()
    dts_path = pkg_path / "burn_research.d.ts"
    bindings_surface_path = pkg_path / "wasm-surface.bindings.actual.json"

    surface = load_json(SURFACE_PATH)
    agent_contract = load_json(AGENT_CONTRACT_PATH)
    layout_contract = load_json(LAYOUT_CONTRACT_PATH)
    dts_text = dts_path.read_text(encoding="utf-8")
    actual_classes, actual_functions, actual_default = parse_dts(dts_text)

    dts_bytes = dts_path.read_bytes()
    bindings_surface_path.write_text(
        json.dumps(
            {
                "schema": "burn-research.wasm-bindgen-surface.actual.v1",
                "artifact": dts_path.name,
                "artifact_sha256": f"sha256:{hashlib.sha256(dts_bytes).hexdigest()}",
                "default_init": actual_default,
                "functions": sorted(actual_functions),
                "classes": {
                    name: sorted(members)
                    for name, members in sorted(actual_classes.items())
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    expected_classes = {name: set(members) for name, members in surface["classes"].items()}
    require_equal("exported classes", set(actual_classes), set(expected_classes))
    for class_name, expected_members in expected_classes.items():
        require_equal(
            f"{class_name} packaged members",
            actual_classes[class_name],
            expected_members,
        )

    require_equal("exported free functions", actual_functions, set(surface["functions"]))
    if actual_default != surface["default_init"]:
        raise AssertionError(
            f"default initializer mismatch; actual={actual_default}; expected={surface['default_init']}"
        )

    workspace = agent_contract["surface_inventory"]["workspace"]
    workspace_inventory = {"constructor"}
    for group in ("contracted_direct", "noncanonical_mutators", "read_only"):
        for member in workspace[group]:
            if member in workspace_inventory:
                raise AssertionError(f"duplicate AgentWorkspace inventory member: {member}")
            workspace_inventory.add(member)
    require_equal(
        "AgentWorkspace schema vs packaged surface manifest",
        set(surface["classes"]["AgentWorkspace"]),
        workspace_inventory,
    )

    free_operations = set(agent_contract["surface_inventory"]["free_operations"])
    if not free_operations.issubset(actual_functions):
        raise AssertionError(
            f"canonical free operations missing from package: {sorted(free_operations - actual_functions)}"
        )

    required_progressive_classes = {
        "AgentWorkspace",
        "AgentLayerSpec",
        "AgentGraphBuilder",
        "LayerRegistry",
        "PacketHeader",
        "CompiledGraph",
        "WasmTensor",
    }
    if not required_progressive_classes.issubset(actual_classes):
        raise AssertionError(
            "progressive-disclosure classes missing from package: "
            + str(sorted(required_progressive_classes - set(actual_classes)))
        )

    typed_constructors = set(layout_contract["constructors"])
    spec_nonconstructors = {"constructor", "layerId", "layerType", "variant"}
    packaged_typed_constructors = actual_classes["AgentLayerSpec"] - spec_nonconstructors
    require_equal(
        "AgentLayerSpec constructors vs layout contract",
        packaged_typed_constructors,
        typed_constructors,
    )

    print(
        "WASM surface conformance PASS: "
        f"{len(actual_classes)} classes, {len(actual_functions)} free functions, "
        f"{len(actual_classes['AgentWorkspace'])} AgentWorkspace members; "
        f"declaration projection: {bindings_surface_path}"
    )


if __name__ == "__main__":
    main()
