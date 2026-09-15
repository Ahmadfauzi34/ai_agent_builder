#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"


def run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(args)}"
        )
    return completed


def main() -> None:
    run(["cargo", "package", "--allow-dirty", "--no-verify"], cwd=REPO)

    crate_path = REPO / "target" / "package" / f"{PACKAGE_NAME}-{PACKAGE_VERSION}.crate"
    if not crate_path.is_file():
        raise SystemExit(f"missing packaged crate: {crate_path}")

    with tempfile.TemporaryDirectory(prefix="burn-research-rust-package-") as tmp:
        tmp_path = Path(tmp)
        extract_root = tmp_path / "package"
        extract_root.mkdir()

        with tarfile.open(crate_path, "r:gz") as archive:
            archive.extractall(extract_root)

        package_dir = extract_root / f"{PACKAGE_NAME}-{PACKAGE_VERSION}"
        if not (package_dir / "Cargo.toml").is_file():
            raise SystemExit("packaged crate did not contain Cargo.toml")

        consumer = tmp_path / "consumer"
        (consumer / "src").mkdir(parents=True)

        package_path = package_dir.as_posix().replace('"', '\\"')
        (consumer / "Cargo.toml").write_text(
            f'''[package]\nname = "burn-research-package-consumer"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\nburn-research = {{ path = "{package_path}" }}\n''',
            encoding="utf-8",
        )

        (consumer / "src" / "main.rs").write_text(
            r'''use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::program_bundle::{export_program_bundle, import_program_bundle};
use burn_research::registry::LayerRegistry;
use burn_research::WasmTensor;

fn main() {
    let mut registry = LayerRegistry::new();
    let linear = AgentLayerSpec::linear(42_001, 2, 1, true).expect("linear spec");
    registry.init_agent_layer(&linear).expect("initialize linear");

    let mut builder = AgentGraphBuilder::new(2).expect("graph builder");
    builder.add_unary(&linear, 0, 1).expect("add graph step");
    builder.set_output(1).expect("set output");
    let graph = builder.compile(&registry).expect("compile graph");

    let binding = GraphParameterBinding::build(&graph, &registry).expect("build binding");
    assert_eq!(binding.total_len(), 3, "2->1 Linear+bias should expose 3 coordinates");

    let candidate = vec![1.0f32, 1.0, 1.0];
    binding
        .apply_flat(&graph, &mut registry, &candidate)
        .expect("apply canonical candidate");
    assert_eq!(
        binding.read_flat(&graph, &registry).expect("read canonical parameters"),
        candidate
    );

    let input = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
    let output = graph.run(&registry, &input).expect("graph run");
    let values = output.to_array();
    assert_eq!(values.len(), 1);
    assert!((values[0] - 4.0).abs() <= 1e-6, "unexpected packaged graph output: {:?}", values);

    let program_identity = graph.program_identity();
    let binding_identity = binding.identity_json();
    let bundle = export_program_bundle(&graph, &registry, true).expect("export stateful bundle");

    let mut imported_registry = LayerRegistry::new();
    let imported_graph = import_program_bundle(&mut imported_registry, &bundle)
        .expect("import stateful bundle");
    let imported_binding = GraphParameterBinding::build(&imported_graph, &imported_registry)
        .expect("rebuild imported binding");

    assert_eq!(imported_graph.program_identity(), program_identity);
    assert_eq!(imported_binding.identity_json(), binding_identity);
    assert_eq!(
        imported_binding
            .read_flat(&imported_graph, &imported_registry)
            .expect("read imported parameters"),
        candidate
    );

    let replay = imported_graph
        .run(&imported_registry, &input)
        .expect("replay graph run")
        .to_array();
    assert_eq!(replay.len(), 1);
    assert!((replay[0] - values[0]).abs() <= 1e-6);

    println!(
        "{{\"verdict\":\"PASS\",\"package\":\"burn-research\",\"consumer\":\"external-cargo-project\",\"total_len\":{},\"output\":{}}}",
        binding.total_len(),
        values[0]
    );
}
''',
            encoding="utf-8",
        )

        run(["cargo", "generate-lockfile", "--manifest-path", str(consumer / "Cargo.toml")], cwd=consumer)
        completed = run(
            ["cargo", "run", "--quiet", "--locked", "--manifest-path", str(consumer / "Cargo.toml")],
            cwd=consumer,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("external Rust package consumer produced no audit output")
        result = json.loads(lines[-1])
        if result.get("verdict") != "PASS":
            raise SystemExit(f"unexpected package consumer verdict: {result}")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
