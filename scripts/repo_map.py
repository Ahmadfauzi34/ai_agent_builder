#!/usr/bin/env python3
"""Small deterministic repository mapper for ai_agent_builder.

Default profile (v1):
- human summary
- machine-readable JSON
- Graphviz DOT
- Rust module/dependency edges
- inventory of all non-generated repository files

Opt-in cross-surface profile (v2) retains the v1 data and also maps:
- local Cargo path-dependency edges
- local Python package import edges
- GitHub workflow -> tracked script references

The mapper uses only the Python standard library and never infers support or
policy ownership from file names. docs/host-support.v1.json remains the
authority for host support status.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_V1 = "ai-agent-builder.repo-map.v1"
SCHEMA_V2 = "ai-agent-builder.repo-map.v2"
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "target",
    "pkg",
}
MOD_RE = re.compile(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)\s*;")
USE_RE = re.compile(r"(?m)^\s*(?:pub\s+)?use\s+crate::([^;]+);")
CARGO_SECTION_RE = re.compile(r"^\[([^]]+)\]$")
CARGO_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"\s*$')
CARGO_INLINE_DEP_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*=\s*\{([^}]*)\}\s*$")
CARGO_PATH_RE = re.compile(r'\bpath\s*=\s*"([^"]+)"')
CARGO_TABLE_PATH_RE = re.compile(r'^path\s*=\s*"([^"]+)"\s*$')
WORKFLOW_SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_./-]+\.(?:py|mjs|js)")


def repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        files.append(rel)
    return sorted(files, key=lambda p: p.as_posix())


def area_for(path: Path) -> str:
    return path.parts[0] if len(path.parts) > 1 else "."


def rust_module_for(path: Path) -> str | None:
    if path.suffix != ".rs" or not path.parts or path.parts[0] != "src":
        return None

    parts = list(path.parts[1:])
    if not parts:
        return None

    if parts == ["lib.rs"]:
        return "crate"

    if parts[-1] == "mod.rs":
        parts = parts[:-1]
    else:
        parts[-1] = Path(parts[-1]).stem

    return "crate" + ("::" + "::".join(parts) if parts else "")


def declared_child(source_module: str, child: str, modules: set[str]) -> str | None:
    candidate = f"{source_module}::{child}" if source_module != "crate" else f"crate::{child}"
    if candidate in modules:
        return candidate

    if source_module != "crate":
        parent = source_module.rsplit("::", 1)[0]
        candidate = f"{parent}::{child}"
        if candidate in modules:
            return candidate
    return None


def longest_existing_module(import_expr: str, modules: set[str]) -> str | None:
    # Strip aliases, glob markers, and brace groups; then resolve the longest
    # module prefix that actually exists in the checkout.
    head = import_expr.split("{", 1)[0].strip().rstrip(":")
    head = head.split(" as ", 1)[0].strip()
    head = head.rstrip("::*").rstrip(":")
    if not head:
        return "crate" if "crate" in modules else None

    parts = [p for p in head.split("::") if p]
    while parts:
        candidate = "crate::" + "::".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return "crate" if "crate" in modules else None


def parse_rust_edges(
    root: Path, files: list[Path]
) -> tuple[dict[str, str], list[dict[str, str]]]:
    module_to_file: dict[str, str] = {}
    for rel in files:
        module = rust_module_for(rel)
        if module:
            module_to_file[module] = rel.as_posix()

    modules = set(module_to_file)
    edges: set[tuple[str, str, str]] = set()

    for module, rel_s in sorted(module_to_file.items()):
        rel = Path(rel_s)
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for child in MOD_RE.findall(text):
            target = declared_child(module, child, modules)
            if target and target != module:
                edges.add((module, target, "mod"))

        for expr in USE_RE.findall(text):
            target = longest_existing_module(expr, modules)
            if target and target != module:
                edges.add((module, target, "use"))

    edge_rows = [
        {"from": src, "to": dst, "kind": kind}
        for src, dst, kind in sorted(edges)
    ]
    return dict(sorted(module_to_file.items())), edge_rows


def build_map(root: Path) -> dict:
    """Build the frozen Rust-focused v1 map."""

    files = repo_files(root)
    module_to_file, edges = parse_rust_edges(root, files)

    by_area = Counter(area_for(p) for p in files)
    by_extension = Counter(p.suffix or "<none>" for p in files)

    return {
        "schema": SCHEMA_V1,
        "root": ".",
        "summary": {
            "file_count": len(files),
            "rust_module_count": len(module_to_file),
            "rust_edge_count": len(edges),
        },
        "areas": dict(sorted(by_area.items())),
        "extensions": dict(sorted(by_extension.items())),
        "files": [p.as_posix() for p in files],
        "rust_modules": [
            {"module": module, "file": file}
            for module, file in module_to_file.items()
        ],
        "rust_edges": edges,
    }


def dependency_section(section: str) -> tuple[str, str | None] | None:
    parts = section.split(".")
    if not parts:
        return None
    if parts[-1] in {"dependencies", "dev-dependencies", "build-dependencies"}:
        return parts[-1], None
    if len(parts) >= 2 and parts[-2] in {
        "dependencies",
        "dev-dependencies",
        "build-dependencies",
    }:
        return parts[-2], parts[-1]
    return None


def parse_cargo_manifest(root: Path, rel: Path) -> tuple[dict, list[dict]]:
    text = (root / rel).read_text(encoding="utf-8")
    section = ""
    package_name: str | None = None
    path_dependencies: list[dict] = []

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        section_match = CARGO_SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1).strip()
            continue

        if section == "package" and package_name is None:
            name_match = CARGO_NAME_RE.match(line)
            if name_match:
                package_name = name_match.group(1)
                continue

        dep_section = dependency_section(section)
        if not dep_section:
            continue
        dep_kind, table_dependency = dep_section

        if table_dependency is not None:
            path_match = CARGO_TABLE_PATH_RE.match(line)
            if path_match:
                path_dependencies.append(
                    {
                        "dependency": table_dependency,
                        "dependency_kind": dep_kind,
                        "path": path_match.group(1),
                    }
                )
            continue

        inline_match = CARGO_INLINE_DEP_RE.match(line)
        if not inline_match:
            continue
        path_match = CARGO_PATH_RE.search(inline_match.group(2))
        if path_match:
            path_dependencies.append(
                {
                    "dependency": inline_match.group(1),
                    "dependency_kind": dep_kind,
                    "path": path_match.group(1),
                }
            )

    row = {
        "name": package_name or rel.parent.name or "<root>",
        "manifest": rel.as_posix(),
        "root": rel.parent.as_posix() if rel.parent.as_posix() != "." else ".",
    }
    return row, path_dependencies


def parse_cargo_graph(
    root: Path, files: list[Path]
) -> tuple[list[dict], list[dict]]:
    manifests = sorted(
        (rel for rel in files if rel.name == "Cargo.toml"),
        key=lambda p: p.as_posix(),
    )
    packages: list[dict] = []
    deps_by_manifest: dict[str, list[dict]] = {}

    for rel in manifests:
        try:
            package, deps = parse_cargo_manifest(root, rel)
        except UnicodeDecodeError:
            continue
        packages.append(package)
        deps_by_manifest[rel.as_posix()] = deps

    by_manifest = {row["manifest"]: row for row in packages}
    edges: list[dict] = []
    for source_manifest, deps in sorted(deps_by_manifest.items()):
        source = by_manifest[source_manifest]
        source_dir = Path(source_manifest).parent
        for dep in deps:
            target_manifest_abs = (
                root / source_dir / dep["path"] / "Cargo.toml"
            ).resolve()
            try:
                target_manifest = target_manifest_abs.relative_to(root).as_posix()
            except ValueError:
                continue
            target = by_manifest.get(target_manifest)
            if not target:
                continue
            edges.append(
                {
                    "from": source["name"],
                    "to": target["name"],
                    "kind": "path",
                    "dependency": dep["dependency"],
                    "dependency_kind": dep["dependency_kind"],
                    "from_manifest": source_manifest,
                    "to_manifest": target_manifest,
                }
            )

    packages.sort(key=lambda row: (row["manifest"], row["name"]))
    edges.sort(
        key=lambda row: (
            row["from_manifest"],
            row["to_manifest"],
            row["dependency_kind"],
            row["dependency"],
        )
    )
    return packages, edges


def python_module_for(
    rel: Path, file_set: set[str]
) -> tuple[str, bool] | None:
    if rel.suffix != ".py":
        return None

    parent_parts = list(rel.parent.parts)
    package_parts: list[str] = []
    cursor = len(parent_parts)
    while cursor > 0:
        init_path = Path(*parent_parts[:cursor], "__init__.py").as_posix()
        if init_path not in file_set:
            break
        package_parts.insert(0, parent_parts[cursor - 1])
        cursor -= 1

    if not package_parts:
        return None

    is_package = rel.name == "__init__.py"
    module_parts = package_parts if is_package else package_parts + [rel.stem]
    return ".".join(module_parts), is_package


def resolve_python_module(candidate: str, modules: set[str]) -> str | None:
    parts = [part for part in candidate.split(".") if part]
    while parts:
        value = ".".join(parts)
        if value in modules:
            return value
        parts.pop()
    return None


def relative_python_candidate(
    source_module: str,
    source_is_package: bool,
    level: int,
    imported_module: str | None,
) -> str | None:
    base = source_module.split(".")
    if not source_is_package:
        base = base[:-1]

    ascend = level - 1
    if ascend > len(base):
        return None
    if ascend:
        base = base[: len(base) - ascend]

    if imported_module:
        base += imported_module.split(".")
    return ".".join(base)


def parse_python_graph(
    root: Path, files: list[Path]
) -> tuple[list[dict], list[dict]]:
    file_set = {rel.as_posix() for rel in files}
    modules: dict[str, dict] = {}

    for rel in files:
        mapped = python_module_for(rel, file_set)
        if not mapped:
            continue
        module, is_package = mapped
        modules[module] = {
            "module": module,
            "file": rel.as_posix(),
            "package": is_package,
        }

    module_names = set(modules)
    local_roots = {name.split(".", 1)[0] for name in module_names}
    edges: set[tuple[str, str, str, bool]] = set()

    for module, row in sorted(modules.items()):
        try:
            source = (root / row["file"]).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=row["file"])
        except (UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = relative_python_candidate(
                        module,
                        bool(row["package"]),
                        node.level,
                        node.module,
                    )
                    if not base:
                        continue
                    if node.module is None:
                        candidates.extend(
                            f"{base}.{alias.name}" if base else alias.name
                            for alias in node.names
                            if alias.name != "*"
                        )
                    else:
                        candidates.append(base)
                elif node.module:
                    candidates.append(node.module)

            for candidate in candidates:
                if not candidate:
                    continue
                root_name = candidate.split(".", 1)[0]
                if root_name not in local_roots:
                    continue
                resolved = resolve_python_module(candidate, module_names)
                target = resolved or candidate
                if target != module:
                    edges.add((module, target, "import", resolved is not None))

    module_rows = [modules[name] for name in sorted(modules)]
    edge_rows = [
        {
            "from": src,
            "to": dst,
            "kind": kind,
            "resolved": resolved,
        }
        for src, dst, kind, resolved in sorted(edges)
    ]
    return module_rows, edge_rows


def parse_workflow_script_edges(root: Path, files: list[Path]) -> list[dict]:
    file_set = {rel.as_posix() for rel in files}
    workflows = sorted(
        (
            rel
            for rel in files
            if rel.parts[:2] == (".github", "workflows")
            and rel.suffix in {".yml", ".yaml"}
        ),
        key=lambda p: p.as_posix(),
    )
    edges: set[tuple[str, str, str]] = set()

    for rel in workflows:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for script in WORKFLOW_SCRIPT_RE.findall(text):
            if script in file_set:
                edges.add((rel.as_posix(), script, "workflow_script"))

    return [
        {"from": src, "to": dst, "kind": kind}
        for src, dst, kind in sorted(edges)
    ]


def build_cross_surface_map(root: Path) -> dict:
    base = build_map(root)
    files = [Path(value) for value in base["files"]]

    cargo_packages, cargo_edges = parse_cargo_graph(root, files)
    python_modules, python_edges = parse_python_graph(root, files)
    workflow_script_edges = parse_workflow_script_edges(root, files)

    summary = dict(base["summary"])
    summary.update(
        {
            "cargo_package_count": len(cargo_packages),
            "cargo_edge_count": len(cargo_edges),
            "python_module_count": len(python_modules),
            "python_edge_count": len(python_edges),
            "workflow_script_edge_count": len(workflow_script_edges),
        }
    )

    return {
        **base,
        "schema": SCHEMA_V2,
        "summary": summary,
        "cargo_packages": cargo_packages,
        "cargo_edges": cargo_edges,
        "python_modules": python_modules,
        "python_edges": python_edges,
        "workflow_script_edges": workflow_script_edges,
    }


def render_summary(data: dict) -> str:
    s = data["summary"]
    lines = [
        f"schema: {data['schema']}",
        f"files: {s['file_count']}",
        f"rust modules: {s['rust_module_count']}",
        f"rust edges: {s['rust_edge_count']}",
        "",
        "areas:",
    ]
    for area, count in data["areas"].items():
        lines.append(f"  {area}: {count}")

    lines += ["", "rust modules:"]
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in data["rust_edges"]:
        outgoing[edge["from"]].append(f"{edge['kind']}->{edge['to']}")

    for row in data["rust_modules"]:
        deps = ", ".join(outgoing.get(row["module"], []))
        suffix = f"  [{deps}]" if deps else ""
        lines.append(f"  {row['module']} = {row['file']}{suffix}")

    if data["schema"] == SCHEMA_V2:
        lines += [
            "",
            "cross-surface:",
            f"  cargo packages: {s['cargo_package_count']}",
            f"  cargo path edges: {s['cargo_edge_count']}",
            f"  python modules: {s['python_module_count']}",
            f"  python import edges: {s['python_edge_count']}",
            f"  workflow->script edges: {s['workflow_script_edge_count']}",
            "",
            "cargo path edges:",
        ]
        for edge in data["cargo_edges"]:
            lines.append(
                f"  {edge['from']} -> {edge['to']} "
                f"[{edge['dependency_kind']}:{edge['dependency']}]"
            )

        lines += ["", "python import edges:"]
        for edge in data["python_edges"]:
            state = "resolved" if edge["resolved"] else "unresolved-local"
            lines.append(f"  {edge['from']} -> {edge['to']} [{state}]")

        lines += ["", "workflow -> script edges:"]
        for edge in data["workflow_script_edges"]:
            lines.append(f"  {edge['from']} -> {edge['to']}")

    return "\n".join(lines) + "\n"


def dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_dot(data: dict) -> str:
    label = (
        "ai_agent_builder Rust module map"
        if data["schema"] == SCHEMA_V1
        else "ai_agent_builder cross-surface repository map"
    )
    lines = [
        "digraph repo_map {",
        "  rankdir=LR;",
        f"  graph [label={dot_quote(label)}, labelloc=t];",
        "  node [shape=box];",
    ]
    for row in data["rust_modules"]:
        label_value = f"{row['module']}\\n{row['file']}"
        lines.append(
            f"  {dot_quote(row['module'])} [label={dot_quote(label_value)}];"
        )
    for edge in data["rust_edges"]:
        style = "solid" if edge["kind"] == "mod" else "dashed"
        lines.append(
            f"  {dot_quote(edge['from'])} -> {dot_quote(edge['to'])} "
            f"[label={dot_quote(edge['kind'])}, style={style}];"
        )

    if data["schema"] == SCHEMA_V2:
        for row in data["cargo_packages"]:
            node = f"cargo:{row['manifest']}"
            label_value = f"cargo:{row['name']}\\n{row['manifest']}"
            lines.append(
                f"  {dot_quote(node)} [label={dot_quote(label_value)}, shape=ellipse];"
            )
        cargo_by_name = {
            row["name"]: f"cargo:{row['manifest']}"
            for row in data["cargo_packages"]
        }
        for edge in data["cargo_edges"]:
            lines.append(
                f"  {dot_quote(cargo_by_name[edge['from']])} -> "
                f"{dot_quote(cargo_by_name[edge['to']])} "
                f"[label={dot_quote('cargo:path')}, style=bold];"
            )

        python_nodes = {row["module"] for row in data["python_modules"]}
        for row in data["python_modules"]:
            node = f"python:{row['module']}"
            label_value = f"python:{row['module']}\\n{row['file']}"
            lines.append(
                f"  {dot_quote(node)} [label={dot_quote(label_value)}, shape=note];"
            )
        for edge in data["python_edges"]:
            target = f"python:{edge['to']}"
            if edge["to"] not in python_nodes:
                label_value = f"python:{edge['to']}\\nunresolved local target"
                lines.append(
                    f"  {dot_quote(target)} "
                    f"[label={dot_quote(label_value)}, shape=note, style=dashed];"
                )
            style = "dotted" if edge["resolved"] else "dashed"
            lines.append(
                f"  {dot_quote('python:' + edge['from'])} -> {dot_quote(target)} "
                f"[label={dot_quote('python:import')}, style={style}];"
            )

        workflow_nodes = sorted(
            {edge["from"] for edge in data["workflow_script_edges"]}
        )
        script_nodes = sorted(
            {edge["to"] for edge in data["workflow_script_edges"]}
        )
        for workflow in workflow_nodes:
            node = f"workflow:{workflow}"
            lines.append(
                f"  {dot_quote(node)} "
                f"[label={dot_quote(workflow)}, shape=component];"
            )
        for script in script_nodes:
            node = f"script:{script}"
            lines.append(
                f"  {dot_quote(node)} [label={dot_quote(script)}, shape=folder];"
            )
        for edge in data["workflow_script_edges"]:
            lines.append(
                f"  {dot_quote('workflow:' + edge['from'])} -> "
                f"{dot_quote('script:' + edge['to'])} "
                f"[label={dot_quote('workflow:script')}, style=dashed];"
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map repository files and dependency relationships."
    )
    parser.add_argument(
        "--root", default=".", help="Repository root (default: current directory)."
    )
    parser.add_argument(
        "--profile",
        choices=("rust", "cross-surface"),
        default="rust",
        help="Mapping profile. 'rust' preserves repo-map.v1; cross-surface emits v2.",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "json", "dot"),
        default="summary",
        help="Output format.",
    )
    parser.add_argument("-o", "--output", help="Write output to a file instead of stdout.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "Cargo.toml").exists():
        parser.error(f"{root} does not look like the repository root (Cargo.toml missing)")

    data = (
        build_map(root)
        if args.profile == "rust"
        else build_cross_surface_map(root)
    )
    if args.format == "json":
        output = json.dumps(data, indent=2, sort_keys=True) + "\n"
    elif args.format == "dot":
        output = render_dot(data)
    else:
        output = render_summary(data)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
