from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_MAP_PATH = REPO_ROOT / "scripts" / "repo_map.py"

_spec = importlib.util.spec_from_file_location("repo_map_under_test", REPO_MAP_PATH)
assert _spec is not None and _spec.loader is not None
repo_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repo_map)


class RepoMapTests(unittest.TestCase):
    def write(self, root: Path, rel: str, content: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def base_repo(self, root: Path) -> None:
        self.write(
            root,
            "Cargo.toml",
            """[package]
name = "core"
version = "0.1.0"
edition = "2021"
""",
        )
        self.write(
            root,
            "src/lib.rs",
            """mod alpha;
use crate::alpha::Thing;
""",
        )
        self.write(root, "src/alpha.rs", "pub struct Thing;\n")

    def cli_json(self, root: Path, *args: str) -> tuple[str, dict]:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = repo_map.main(
                ["--root", str(root), "--format", "json", *args]
            )
        self.assertEqual(code, 0)
        raw = stdout.getvalue()
        return raw, json.loads(raw)

    def test_default_profile_preserves_v1_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.base_repo(root)

            raw, data = self.cli_json(root)

            self.assertEqual(data["schema"], "ai-agent-builder.repo-map.v1")
            self.assertEqual(
                set(data),
                {
                    "schema",
                    "root",
                    "summary",
                    "areas",
                    "extensions",
                    "files",
                    "rust_modules",
                    "rust_edges",
                },
            )
            self.assertNotIn("cross-surface", raw)
            self.assertEqual(data["summary"]["rust_module_count"], 2)
            self.assertIn(
                {"from": "crate", "to": "crate::alpha", "kind": "mod"},
                data["rust_edges"],
            )
            self.assertIn(
                {"from": "crate", "to": "crate::alpha", "kind": "use"},
                data["rust_edges"],
            )

    def test_cross_surface_profile_maps_evidence_edges_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.base_repo(root)
            self.write(
                root,
                "ffi/Cargo.toml",
                """[package]
name = "bridge"
version = "0.1.0"
edition = "2021"

[dependencies]
core = { path = ".." }
""",
            )
            self.write(
                root,
                "ffi/pkg/__init__.py",
                "from .facade import Thing\n",
            )
            self.write(
                root,
                "ffi/pkg/facade.py",
                "class Thing:\n    pass\n",
            )
            self.write(root, "scripts/check.py", "print('ok')\n")
            self.write(
                root,
                ".github/workflows/research.yml",
                """name: Research
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: python3 scripts/check.py
""",
            )

            raw_a, data = self.cli_json(root, "--profile", "cross-surface")
            raw_b, data_b = self.cli_json(root, "--profile", "cross-surface")

            self.assertEqual(raw_a, raw_b)
            self.assertEqual(data, data_b)
            self.assertEqual(data["schema"], "ai-agent-builder.repo-map.v2")

            self.assertIn(
                {
                    "from": "bridge",
                    "to": "core",
                    "kind": "path",
                    "dependency": "core",
                    "dependency_kind": "dependencies",
                    "from_manifest": "ffi/Cargo.toml",
                    "to_manifest": "Cargo.toml",
                },
                data["cargo_edges"],
            )
            self.assertIn(
                {
                    "from": "pkg",
                    "to": "pkg.facade",
                    "kind": "import",
                    "resolved": True,
                },
                data["python_edges"],
            )
            self.assertIn(
                {
                    "from": ".github/workflows/research.yml",
                    "to": "scripts/check.py",
                    "kind": "workflow_script",
                },
                data["workflow_script_edges"],
            )

            dot = repo_map.render_dot(data)
            self.assertIn("cargo:path", dot)
            self.assertIn("python:import", dot)
            self.assertIn("workflow:script", dot)


if __name__ == "__main__":
    unittest.main()
