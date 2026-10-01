#!/usr/bin/env python3
"""Enforce the canonical contract registry (Opsi D).

Checks:
1. Every docs/contracts/*.json is registered in docs/contracts/REGISTRY.md.
2. Every include_str!("...docs/contracts/<name>") in src/ points at an existing
   file and that file is registered.
3. No stale REGISTRY.md entries (a registered contract whose file is gone).
4. Every docs/contracts/*.json parses as JSON.

Exit non-zero with a human-readable error on the first violation class found.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = ROOT / "docs" / "contracts"
REGISTRY = CONTRACTS_DIR / "REGISTRY.md"
SRC_DIR = ROOT / "src"

TABLE_ROW_RE = re.compile(r"^\|\s*`([a-z0-9_.-]+\.v\d+\.json)`\s*\|")
INCLUDE_RE = re.compile(r'include_str!\("(?:\.\./)+docs/contracts/([^"]+)"\)')


def fail(msg: str) -> "NoReturn":
    print(f"contract-registry check FAILED: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not REGISTRY.is_file():
        fail("docs/contracts/REGISTRY.md is missing")

    registered = set()
    for line in REGISTRY.read_text().splitlines():
        m = TABLE_ROW_RE.match(line)
        if m:
            registered.add(m.group(1))

    files = {p.name for p in CONTRACTS_DIR.glob("*.json")}

    unregistered = sorted(files - registered)
    if unregistered:
        fail(f"contracts not registered in REGISTRY.md: {', '.join(unregistered)}")

    stale = sorted(registered - files)
    if stale:
        fail(f"REGISTRY.md entries with no file: {', '.join(stale)}")

    # include_str! references
    missing_files: set[str] = set()
    unregistered_refs: set[str] = set()
    for rs in SRC_DIR.rglob("*.rs"):
        for ref in INCLUDE_RE.findall(rs.read_text()):
            if "*" in ref:
                continue  # doc-comment glob, not a real include_str! target
            name = Path(ref).name
            if not (CONTRACTS_DIR / name).is_file():
                missing_files.add(f"{rs.relative_to(ROOT)} -> {ref}")
            elif name not in registered:
                unregistered_refs.add(f"{rs.relative_to(ROOT)} -> {name}")
    if missing_files:
        fail(f"include_str! to missing contract files: {', '.join(sorted(missing_files))}")
    if unregistered_refs:
        fail(f"include_str! to unregistered contracts: {', '.join(sorted(unregistered_refs))}")

    # JSON validity for every contract
    for name in sorted(files):
        try:
            json.loads((CONTRACTS_DIR / name).read_text())
        except json.JSONDecodeError as e:
            fail(f"{name} is not valid JSON: {e}")

    print(f"contract-registry check OK: {len(files)} contracts, all registered and valid")


if __name__ == "__main__":
    main()
