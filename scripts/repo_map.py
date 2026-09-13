#!/usr/bin/env python3
"""Small deterministic repository mapper for ai_agent_builder.

Scans the checkout without external dependencies and emits:
- human summary
- machine-readable JSON
- Graphviz DOT

The graph focuses on Rust module/dependency edges while still inventorying
all non-generated repository files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "ai-agent-builder.repo-map.v1"
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
    files = repo_files(root)
    module_to_file, edges = parse_rust_edges(root, files)

    by_area = Counter(area_for(p) for p in files)
    by_extension = Counter(p.suffix or "<none>" for p in files)

    return {
        "schema": SCHEMA,
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

    return "\n".join(lines) + "\n"


def dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_dot(data: dict) -> str:
    lines = [
        "digraph repo_map {",
        "  rankdir=LR;",
        '  graph [label="ai_agent_builder Rust module map", labelloc=t];',
        "  node [shape=box];",
    ]
    for row in data["rust_modules"]:
        label = f"{row['module']}\\n{row['file']}"
        lines.append(f"  {dot_quote(row['module'])} [label={dot_quote(label)}];")
    for edge in data["rust_edges"]:
        style = "solid" if edge["kind"] == "mod" else "dashed"
        lines.append(
            f"  {dot_quote(edge['from'])} -> {dot_quote(edge['to'])} "
            f"[label={dot_quote(edge['kind'])}, style={style}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map repository files and Rust module relationships."
    )
    parser.add_argument(
        "--root", default=".", help="Repository root (default: current directory)."
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

    data = build_map(root)
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
