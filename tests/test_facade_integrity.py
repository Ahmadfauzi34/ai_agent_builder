#!/usr/bin/env python3
"""Regression tests for Opsi C Fase 1 (WASM facade).

Covers the four confirmed regression cases from the Fase 1 move:
  RC-truncated-splice-corruption : no truncated-splice fragments in sources
  RC-facade-cut-completeness     : every moved item is intact at its destination
  RC-lib-wiring                  : every facade item has root + legacy path re-exports
  RC-pure-move                    : JS export surface identical to pre-move HEAD

Run: python3 -m unittest tests.test_facade_integrity -v
"""

import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
FACADE = SRC / "facade"
# Opsi C Fase 1 moved items off this base commit.
BASE = "3830dc5bc02199f9ced6e2403914b4432025046b"

# Fragments produced by the truncated byte-range splice bug (2026-10-04).
# The mover script cut mid-token, leaving shards like "_ tier Ok(...)",
# ".to_oner", ".to_vecnner", "WelasmTensor", "//Mos(...)", "//gLs(...)".
CORRUPTION_FRAGMENTS = [
    "_ tier Ok(",
    ".to_oner",
    ".to_vecnner",
    "WelasmTensor",
    "//Mos(",
    "//gLs(",
    ".to_vecnner()",
]


def git_show(ref_path: str) -> str:
    return subprocess.run(
        ["git", "show", ref_path],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout


def head_rs_files():
    out = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BASE, "src/"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.splitlines() if f.endswith(".rs")]


def js_export_names(files, read):
    """(kind, js_name) for every #[wasm_bindgen] item.

    kind is 'fn' (top-level function), 'class' (struct), or 'method'
    (ClassName.method). js_name honors js_name = "..." renames.
    """
    exports = set()
    for f in files:
        try:
            src = read(f)
        except Exception:
            continue
        lines = src.split("\n")
        i, n = 0, len(lines)
        in_wasm_impl = False
        impl_struct = None
        while i < n:
            line = lines[i].strip()
            attr = re.match(r'#\[wasm_bindgen(\(js_name\s*=\s*"(\w+)"\))?\]', line)
            if attr and i + 1 < n:
                nxt = lines[i + 1].strip()
                ms = re.match(r"pub struct (\w+)", nxt)
                if ms:
                    exports.add(("class", attr.group(2) or ms.group(1)))
                    i += 2
                    continue
                mi = re.match(r"impl (\w+)", nxt)
                if mi:
                    in_wasm_impl, impl_struct = True, mi.group(1)
                    i += 2
                    continue
            if in_wasm_impl:
                if re.match(r"^}", line):
                    in_wasm_impl, impl_struct = False, None
                else:
                    mj = re.match(
                        r'#\[wasm_bindgen(\(js_name\s*=\s*"(\w+)"\)|\(constructor\)|(.*))?\]',
                        line,
                    )
                    if mj and i + 1 < n:
                        mnxt = lines[i + 1].strip()
                        mm = re.match(r"pub fn (\w+)", mnxt)
                        if mm and "constructor" not in (mj.group(1) or ""):
                            exports.add(
                                ("method", f"{impl_struct}.{mj.group(2) or mm.group(1)}")
                            )
                            i += 2
                            continue
            m = re.match(r'#\[wasm_bindgen(\(js_name\s*=\s*(\w+)\))?\]', line)
            if m and i + 1 < n and lines[i + 1].startswith("pub fn "):
                mf = re.match(r"pub fn (\w+)", lines[i + 1])
                if mf:
                    exports.add(("fn", m.group(2) or mf.group(1)))
            i += 1
    return exports


def facade_functions():
    """{module: [fn names]} for pub fns in src/facade/*.rs (except mod/tensor/wasm_types)."""
    result = {}
    for path in sorted(FACADE.glob("*.rs")):
        mod = path.stem
        if mod in ("mod", "tensor", "wasm_types"):
            continue
        fns = re.findall(r"^pub fn (\w+)", path.read_text(), re.M)
        if fns:
            result[mod] = fns
    return result


def facade_types():
    """Wasm* struct names declared in facade/tensor.rs and facade/wasm_types.rs."""
    names = []
    for mod in ("tensor", "wasm_types"):
        src = (FACADE / f"{mod}.rs").read_text()
        names += re.findall(r"^pub struct (\w+)", src, re.M)
    return names


class TestTruncatedSpliceCorruption(unittest.TestCase):
    """RC-truncated-splice-corruption: no splice shards anywhere in src/."""

    def test_no_corruption_fragments(self):
        hits = []
        for path in SRC.rglob("*.rs"):
            text = path.read_text()
            for frag in CORRUPTION_FRAGMENTS:
                if frag in text:
                    hits.append((str(path.relative_to(REPO)), frag))
        self.assertEqual(hits, [], f"corruption fragments found: {hits}")

    def test_no_suspicious_comment_shards(self):
        # The corruptor left comment-prefixed shards like "//Mos(", "//gLs(".
        hits = []
        for path in (list(SRC.rglob("*.rs"))):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if re.match(r"\s*//[A-Z][a-z]{0,2}\(", line):
                    hits.append(f"{path.relative_to(REPO)}:{lineno}:{line.strip()}")
        self.assertEqual(hits, [], f"suspicious shards: {hits[:10]}")


class TestFacadeCutCompleteness(unittest.TestCase):
    """RC-facade-cut-completeness: every moved item exists whole at its destination."""

    def test_facade_files_nonempty(self):
        empty = [
            p.name for p in FACADE.glob("*.rs")
            if p.name != "mod.rs" and not p.read_text().strip()
        ]
        self.assertEqual(empty, [], f"empty facade files: {empty}")

    def test_no_wasm_bindgen_fn_left_outside_facade(self):
        left = []
        for path in SRC.rglob("*.rs"):
            if path.is_relative_to(FACADE):
                continue
            src = path.read_text()
            for m in re.finditer(
                r"#\[wasm_bindgen[^\n]*\]\n(?:#\[[^\n]*\]\n)*pub fn (\w+)", src
            ):
                # top-level only (column 0)
                idx = src.find("pub fn " + m.group(1), m.start())
                if idx - src.rfind("\n", 0, idx) - 1 == 0:
                    left.append((str(path.relative_to(REPO)), m.group(1)))
        self.assertEqual(left, [], f"unmoved #[wasm_bindgen] fns: {left}")

    def test_every_facade_fn_has_body(self):
        """A moved fn must have a real body, not a truncated stub."""
        bad = []
        for mod, fns in facade_functions().items():
            src = (FACADE / f"{mod}.rs").read_text()
            for fn in fns:
                m = re.search(rf"^pub fn {fn}\b[^{{]*\{{", src, re.M)
                if not m:
                    bad.append(f"{mod}::{fn}")
        self.assertEqual(bad, [], f"fns without bodies: {bad}")


class TestLibWiring(unittest.TestCase):
    """RC-lib-wiring: every facade item reachable via root and legacy paths."""

    def test_root_reexport_for_every_fn(self):
        lib = (SRC / "lib.rs").read_text()
        missing = []
        for mod, fns in facade_functions().items():
            for fn in fns:
                # root re-export: `pub use facade::<mod>::{..., fn, ...};`
                # or single: `pub use facade::<mod>::fn;`
                if not (
                    re.search(rf"pub use facade::{mod}::\{{[^}}]*\b{fn}\b", lib)
                    or re.search(rf"pub use facade::{mod}::{fn};", lib)
                ):
                    missing.append(f"{mod}::{fn}")
        self.assertEqual(missing, [], f"missing root re-exports: {missing}")

    def test_root_reexport_for_every_type(self):
        lib = (SRC / "lib.rs").read_text()
        missing = [t for t in facade_types() if f"pub use facade::" not in lib or t not in lib]
        # precise check: each type appears in a `pub use facade::...::{...}` list
        missing = []
        for t in facade_types():
            if not re.search(rf"pub use facade::\w+::\{{[^}}]*\b{t}\b", lib):
                missing.append(t)
        self.assertEqual(missing, [], f"missing root type re-exports: {missing}")

    def test_legacy_path_reexport_for_every_fn(self):
        """Each facade fn must be re-exported from its original domain module."""
        non_facade = []
        for path in SRC.rglob("*.rs"):
            if not path.is_relative_to(FACADE):
                non_facade.append(path)
        blob = "\n".join(p.read_text() for p in non_facade)
        missing = []
        for mod, fns in facade_functions().items():
            for fn in fns:
                if not (
                    re.search(rf"pub use crate::facade::{mod}::\{{[^}}]*\b{fn}\b", blob)
                    or re.search(rf"pub use crate::facade::{mod}::{fn};", blob)
                ):
                    missing.append(f"{mod}::{fn}")
        self.assertEqual(missing, [], f"missing legacy re-exports: {missing}")


class TestPureMove(unittest.TestCase):
    """RC-pure-move: the JS surface is byte-identical in name set to pre-move HEAD."""

    def test_js_surface_unchanged(self):
        before = js_export_names(
            head_rs_files(), lambda f: git_show(f"{BASE}:{f}")
        )
        after = js_export_names(
            [str(p.relative_to(REPO)) for p in SRC.rglob("*.rs")],
            lambda f: (REPO / f).read_text(),
        )
        self.assertEqual(
            before ^ after, set(),
            f"surface drift: missing={sorted(before - after)[:10]} "
            f"added={sorted(after - before)[:10]}",
        )


# Opsi C Fase 2 moved #[wasm_bindgen] impl blocks for these 13 domain structs
# into src/facade/; the struct definitions stay in their domain modules but
# MUST keep #[wasm_bindgen] (it generates the WasmDescribe/IntoWasmAbi/
# FromWasmAbi trait impls every wasm signature needs — removing it is E0277).
FASE2_STRUCTS = [
    "CompiledGraph",
    "CompiledMultiInputGraph",
    "MultiInputVerificationCases",
    "TracedMultiInputRun",
    "CheckpointBranchSet",
    "GraphMutationTransaction",
    "LayerRegistry",
    "PacketHeader",
    "EsOptimizer",
    "AgentGraphBuilder",
    "SemanticIngressManifestV2",
    "SemanticTransitionSpec",
    "InputPortConsumerSpec",
]


def facade_impl_targets():
    """Type names with #[wasm_bindgen] impl blocks under src/facade/."""
    targets = set()
    for path in FACADE.rglob("*.rs"):
        src = path.read_text()
        for m in re.finditer(
            r"#\[wasm_bindgen[^\n]*\]\n(?:#\[[^\n]*\]\n)*impl (\w+)", src
        ):
            targets.add(m.group(1))
    return targets


def struct_has_wasm_attr(name):
    """True/False whether `pub struct <name>` carries #[wasm_bindgen];
    None if the struct is not defined anywhere in src/."""
    for path in SRC.rglob("*.rs"):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if re.match(rf"pub struct {re.escape(name)}\b", line.strip()):
                j = i - 1
                while j >= 0:
                    s = lines[j].strip()
                    if s.startswith("#["):
                        if re.match(r"#\[wasm_bindgen", s):
                            return True
                    elif s.startswith("///") or s.startswith("//") or s == "":
                        pass
                    else:
                        break
                    j -= 1
                return False
    return None


class TestFase2StructAttr(unittest.TestCase):
    """RC-fase2-struct-attr-e0277: every type with a #[wasm_bindgen] impl block
    under src/facade/ must carry #[wasm_bindgen] on its struct definition.
    Removing the struct attribute breaks every wasm signature mentioning the
    type (E0277: unsatisfied WasmDescribe/IntoWasmAbi trait bounds)."""

    def test_every_facade_impl_target_has_struct_attr(self):
        bad = []
        for name in sorted(facade_impl_targets()):
            status = struct_has_wasm_attr(name)
            if status is not True:
                bad.append((name, "missing" if status is None else "no-attr"))
        self.assertEqual(bad, [], f"impl targets without struct attr: {bad}")

    def test_all_13_fase2_structs_keep_attr(self):
        bad = [n for n in FASE2_STRUCTS if struct_has_wasm_attr(n) is not True]
        self.assertEqual(bad, [], f"Fase-2 structs missing #[wasm_bindgen]: {bad}")


class TestFase2ImplPlacement(unittest.TestCase):
    """RC-fase2-impl-block-placement: #[wasm_bindgen] impl blocks for the 13
    Fase-2 domain structs live in src/facade/, never in domain files.
    (Pre-existing wasm impls for other types, e.g. MultiInputGraphPlan in
    multi_input_graph.rs, are grandfathered and out of scope.)"""

    def test_no_wasm_impl_for_fase2_structs_outside_facade(self):
        bad = []
        for path in SRC.rglob("*.rs"):
            if path.is_relative_to(FACADE):
                continue
            src = path.read_text()
            for m in re.finditer(
                r"#\[wasm_bindgen[^\n]*\]\n(?:#\[[^\n]*\]\n)*impl (\w+)", src
            ):
                if m.group(1) in FASE2_STRUCTS:
                    bad.append((str(path.relative_to(REPO)), m.group(1)))
        self.assertEqual(bad, [], f"wasm impl blocks outside facade: {bad}")


if __name__ == "__main__":
    unittest.main()
