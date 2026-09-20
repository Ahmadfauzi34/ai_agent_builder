#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FFI_DIR = REPO / "ffi"
REPORT = REPO / "runtime-architecture-artifact-report.json"
PACKAGE_NAME = "burn-research"
PACKAGE_VERSION = "0.1.0"

PYTHON_CONSUMER = r'''
from __future__ import annotations

import json
import math
import os
from array import array
from contextlib import ExitStack
from pathlib import Path

import burn_research_ffi as br
from burn_research_ffi import host


def run_scalar(graph, registry, values):
    with host.Tensor.vector(values) as inp:
        with graph.run(registry, inp) as out:
            data = out.to_f32()
            assert len(data) == 1 and math.isfinite(data[0])
            return data[0]


repo_root = Path(os.environ["BR_REPO_ROOT"]).resolve()
module_path = Path(br.__file__).resolve()
if repo_root == module_path or repo_root in module_path.parents:
    raise AssertionError(f"wheel consumer imported from repository: {module_path}")

proof = {}

with ExitStack() as stack:
    registry = stack.enter_context(host.Registry())
    linear = stack.enter_context(host.LinearLayerSpec(52_001, 2, 1, bias=True))
    registry.init_layer(linear)
    builder = stack.enter_context(host.GraphBuilder(2))
    builder.add_unary(linear, 0, 1).set_output(1)
    graph = stack.enter_context(builder.compile(registry))
    binding = stack.enter_context(host.GraphParameterBinding.build(graph, registry))

    program_identity = graph.program_identity()
    binding_identity = binding.identity()
    initial = binding.read_flat(graph, registry)
    assert len(initial) == 3

    learned = array("f", [1.0, -0.5, 0.25])
    binding.apply_flat(graph, registry, learned)
    assert binding.read_flat(graph, registry) == list(learned)
    assert graph.program_identity() == program_identity
    assert binding.identity() == binding_identity
    probe_after_weight_change = run_scalar(graph, registry, [2.0, 4.0])

    mismatch_registry = stack.enter_context(host.Registry())
    mismatch_linear = stack.enter_context(host.LinearLayerSpec(52_001, 2, 2, bias=True))
    mismatch_registry.init_layer(mismatch_linear)
    try:
        with host.Tensor.vector([2.0, 4.0]) as inp:
            graph.run(mismatch_registry, inp)
    except host.BurnResearchError:
        structural_mismatch_rejected = True
    else:
        structural_mismatch_rejected = False
    assert structural_mismatch_rejected

    proof["structural_identity"] = {
        "program_identity_stable_across_weight_mutation": True,
        "binding_identity_stable_across_weight_mutation": True,
        "mismatched_registry_rejected": True,
        "probe_after_weight_change": probe_after_weight_change,
    }

    before_invalid = binding.read_flat(graph, registry)
    try:
        binding.apply_flat(graph, registry, array("f", [1.0, 2.0]))
    except host.BurnResearchError:
        pass
    else:
        raise AssertionError("wrong-length candidate unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_invalid

    poisoned = array("f", [before_invalid[0], math.nan, before_invalid[2]])
    try:
        binding.apply_flat(graph, registry, poisoned)
    except host.BurnResearchError:
        pass
    else:
        raise AssertionError("non-finite candidate unexpectedly succeeded")
    assert binding.read_flat(graph, registry) == before_invalid

    proof["atomic_parameter_apply"] = {
        "valid_candidate_applied": True,
        "wrong_length_rejected_atomically": True,
        "non_finite_rejected_atomically": True,
        "state_preserved_after_rejection": True,
    }

    bundle = host.ProgramBundle.export(graph, registry, include_state=True)
    assert bundle

    sentinel_registry = stack.enter_context(host.Registry())
    sentinel_linear = stack.enter_context(host.LinearLayerSpec(53_001, 1, 1, bias=True))
    sentinel_registry.init_layer(sentinel_linear)
    sentinel_builder = stack.enter_context(host.GraphBuilder(2))
    sentinel_builder.add_unary(sentinel_linear, 0, 1).set_output(1)
    sentinel_graph = stack.enter_context(sentinel_builder.compile(sentinel_registry))
    sentinel_binding = stack.enter_context(
        host.GraphParameterBinding.build(sentinel_graph, sentinel_registry)
    )
    sentinel_candidate = array("f", [2.0, 3.0])
    sentinel_binding.apply_flat(sentinel_graph, sentinel_registry, sentinel_candidate)
    sentinel_program_identity = sentinel_graph.program_identity()
    sentinel_binding_identity = sentinel_binding.identity()
    sentinel_before = sentinel_binding.read_flat(sentinel_graph, sentinel_registry)
    sentinel_output_before = run_scalar(sentinel_graph, sentinel_registry, [4.0])

    corrupt_bundle = bundle[:-1]
    try:
        host.ProgramBundle.import_graph(sentinel_registry, corrupt_bundle)
    except host.BurnResearchError:
        pass
    else:
        raise AssertionError("truncated ProgramBundle unexpectedly imported")

    assert sentinel_graph.program_identity() == sentinel_program_identity
    assert sentinel_binding.identity() == sentinel_binding_identity
    assert sentinel_binding.read_flat(sentinel_graph, sentinel_registry) == sentinel_before
    sentinel_output_after = run_scalar(sentinel_graph, sentinel_registry, [4.0])
    assert abs(sentinel_output_after - sentinel_output_before) <= 1e-7

    imported_registry = stack.enter_context(host.Registry())
    imported_graph = stack.enter_context(
        host.ProgramBundle.import_graph(imported_registry, bundle)
    )
    imported_binding = stack.enter_context(
        host.GraphParameterBinding.build(imported_graph, imported_registry)
    )
    assert imported_graph.program_identity() == program_identity
    assert imported_binding.identity() == binding_identity
    assert imported_binding.read_flat(imported_graph, imported_registry) == list(learned)
    replay_output = run_scalar(imported_graph, imported_registry, [2.0, 4.0])
    assert abs(replay_output - probe_after_weight_change) <= 1e-7

    proof["program_bundle_transaction"] = {
        "truncated_import_rejected": True,
        "populated_target_registry_preserved": True,
        "stateful_replay_identity_exact": True,
        "stateful_replay_parameters_exact": True,
        "stateful_replay_output_exact": True,
        "bundle_bytes": len(bundle),
    }

    retry = stack.enter_context(
        host.EsOptimizer.strict(
            3, strategy=0, seed=7001, population=4, sigma=0.08, learning_rate=0.10
        )
    )
    retry_batch = retry.ask_f32()
    assert len(retry_batch) == retry.batch_size * 3
    try:
        retry.tell([0.0, 0.0])
    except host.BurnResearchError:
        pass
    else:
        raise AssertionError("wrong-cardinality tell unexpectedly succeeded")
    retry_report = retry.tell([0.0] * retry.batch_size)
    assert int(retry_report["gen"]) == 1

    control = stack.enter_context(
        host.EsOptimizer.strict(
            3, strategy=0, seed=7011, population=4, sigma=0.08, learning_rate=0.10
        )
    )
    replaced = stack.enter_context(
        host.EsOptimizer.strict(
            3, strategy=0, seed=7011, population=4, sigma=0.08, learning_rate=0.10
        )
    )
    control_first = control.ask_f32()
    replaced_first = replaced.ask_f32()
    assert control_first.tobytes() == replaced_first.tobytes()

    control.tell([0.0] * control.batch_size)
    control_second = control.ask_f32()
    replaced_second = replaced.ask_f32()
    assert control_second.tobytes() == replaced_second.tobytes()
    assert control_first.tobytes() != control_second.tobytes()

    replaced_report = replaced.tell([0.0] * replaced.batch_size)
    control_report = control.tell([0.0] * control.batch_size)
    assert int(replaced_report["gen"]) == 1
    assert int(control_report["gen"]) == 2

    control_third = control.ask_f32()
    replaced_third = replaced.ask_f32()
    assert control_third.tobytes() == replaced_third.tobytes()

    proof["optimizer_lifecycle"] = {
        "invalid_tell_does_not_consume_pending_batch": True,
        "retry_valid_tell_succeeds": True,
        "ask_replaces_pending_batch": True,
        "replacement_uses_next_rng_batch": True,
        "replacement_generation_commits_once": True,
        "zero_update_future_candidate_trajectory_matches": True,
    }

    lr_control = stack.enter_context(
        host.EsOptimizer.strict(
            3, strategy=0, seed=19_917, population=4, sigma=0.08, learning_rate=0.10
        )
    )
    lr_changed = stack.enter_context(
        host.EsOptimizer.strict(
            3, strategy=0, seed=19_917, population=4, sigma=0.08, learning_rate=0.10
        )
    )

    first_control = lr_control.ask_f32()
    first_changed = lr_changed.ask_f32()
    assert first_control.tobytes() == first_changed.tobytes()
    fitness1 = [2.0, -2.0, 1.0, -1.0]
    lr_control.tell(fitness1)
    lr_changed.tell(fitness1)

    lr_changed.set_learning_rate(0.08)
    second_control = lr_control.ask_f32()
    second_changed = lr_changed.ask_f32()
    assert second_control.tobytes() == second_changed.tobytes()

    try:
        lr_changed.set_learning_rate(0.07)
    except host.BurnResearchError:
        pass
    else:
        raise AssertionError("learning-rate mutation while batch pending unexpectedly succeeded")

    fitness2 = [3.0, -3.0, 1.5, -1.5]
    report_control = lr_control.tell(fitness2)
    report_changed = lr_changed.tell(fitness2)
    assert abs(float(report_control["lr"]) - 0.10) <= 1e-6
    assert abs(float(report_changed["lr"]) - 0.08) <= 1e-6

    third_control = lr_control.ask_f32()
    third_changed = lr_changed.ask_f32()
    assert third_control.tobytes() != third_changed.tobytes()

    proof["optimizer_control"] = {
        "pre_mutation_trajectory_identical": True,
        "next_batch_identical_after_lr_mutation": True,
        "pending_batch_mutation_rejected": True,
        "trajectory_diverges_only_after_tell": True,
    }

print(json.dumps({
    "verdict": "PASS",
    "surface": "installed-python-wheel",
    "module_path": str(module_path),
    "abi_version": host.abi_version(),
    "proof": proof,
}, sort_keys=True))
'''

RUST_CONSUMER = r'''
use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::es::optimizer::EsOptimizer;
use burn_research::graph_parameters::GraphParameterBinding;
use burn_research::math::{MathProgramV9, MathProgramV9Builder};
use burn_research::math::program::{OP_ADD, OP_MUL, OP_SUB};
use burn_research::program_bundle::{export_program_bundle, import_program_bundle};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::WasmTensor;

fn main() {
    let mut math = MathProgramV9Builder::new(1, 9).expect("v9 builder");
    math.add_indices_like(0, 1, 1).expect("row indices");
    math.add_indices_like(0, 2, 2).expect("column indices");
    math.add_less_equal_01(2, 1, 3).expect("causal predicate");
    math.add_fill_like(0, 4, 1.0).expect("ones");
    math.add_binary(OP_SUB, 4, 3, 5).expect("invert predicate");
    math.add_fill_like(0, 6, -100.0).expect("negative fill");
    math.add_binary(OP_MUL, 5, 6, 7).expect("mask penalty");
    math.add_binary(OP_ADD, 0, 7, 8).expect("apply mask");
    math.set_output(8).expect("math output");
    let program = math.compile().expect("compile v9");

    let scores = WasmTensor::new(
        &[
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
        ],
        &[1, 4, 4, 1],
    );
    let expected = vec![
        0.0, -99.0, -98.0, -97.0,
        4.0, 5.0, -94.0, -93.0,
        8.0, 9.0, 10.0, -89.0,
        12.0, 13.0, 14.0, 15.0,
    ];
    let output = program.run_inputs(&[scores.clone()]).expect("v9 run").to_array();
    assert_eq!(output, expected);
    let plan = program.program_plan();
    let identity = program.program_identity();
    let replay = MathProgramV9::from_plan(&plan).expect("v9 replay");
    assert_eq!(replay.program_identity(), identity);
    assert_eq!(replay.program_plan(), plan);
    assert_eq!(
        replay.run_inputs(&[scores]).expect("v9 replay run").to_array(),
        expected
    );

    let mut registry = LayerRegistry::new();
    let linear = AgentLayerSpec::linear(42_001, 2, 1, true).expect("linear spec");
    registry.init_agent_layer(&linear).expect("init linear");
    let mut graph_builder = AgentGraphBuilder::new(2).expect("graph builder");
    graph_builder.add_unary(&linear, 0, 1).expect("graph step");
    graph_builder.set_output(1).expect("graph output");
    let graph = graph_builder.compile(&registry).expect("compile graph");
    let binding = GraphParameterBinding::build(&graph, &registry).expect("binding");
    let learned = vec![1.0f32, 1.0, 1.0];
    binding.apply_flat(&graph, &mut registry, &learned).expect("apply learned state");

    let mut workspace = AgentWorkspace::new(4).expect("workspace");
    workspace.put(
        "notes".to_string(),
        "session".to_string(),
        "artifact".to_string(),
        "active".to_string(),
        "workspace-only-marker".to_string(),
    ).expect("workspace put");
    assert!(workspace.get("notes".to_string(), "session".to_string()).contains("workspace-only-marker"));

    let mut optimizer = EsOptimizer::strict(
        binding.total_len() as u32,
        0,
        77,
        Some(4),
        Some(0.08),
        Some(0.10),
    ).expect("optimizer");
    let batch = optimizer.ask();
    assert_eq!(batch.len(), binding.total_len() * 4);
    optimizer.tell(&[2.0, -2.0, 1.0, -1.0]).expect("optimizer tell");
    assert_eq!(optimizer.generation(), 1);

    let bundle = export_program_bundle(&graph, &registry, true).expect("bundle export");
    let mut imported_registry = LayerRegistry::new();
    let imported_graph = import_program_bundle(&mut imported_registry, &bundle)
        .expect("bundle import");
    let imported_binding = GraphParameterBinding::build(&imported_graph, &imported_registry)
        .expect("imported binding");
    assert_eq!(imported_graph.program_identity(), graph.program_identity());
    assert_eq!(imported_binding.identity_json(), binding.identity_json());
    assert_eq!(
        imported_binding.read_flat(&imported_graph, &imported_registry).expect("read replay"),
        learned
    );

    let fresh_optimizer = EsOptimizer::strict(
        binding.total_len() as u32,
        0,
        77,
        Some(4),
        Some(0.08),
        Some(0.10),
    ).expect("fresh optimizer");
    assert_eq!(optimizer.generation(), 1);
    assert_eq!(fresh_optimizer.generation(), 0);

    let fresh_workspace = AgentWorkspace::new(4).expect("fresh workspace");
    assert_eq!(
        fresh_workspace.get("notes".to_string(), "session".to_string()),
        "null"
    );
    assert!(workspace.get("notes".to_string(), "session".to_string()).contains("workspace-only-marker"));

    println!(
        "{{\"verdict\":\"PASS\",\"surface\":\"external-packaged-rust\",\"math_v9\":{{\"causal_composition\":true,\"plan_replay_exact\":true,\"identity_replay_exact\":true,\"output_replay_exact\":true,\"steps\":8}},\"persistence_boundary\":{{\"program_bundle_graph_state_replay\":true,\"optimizer_generation_before\":1,\"fresh_optimizer_generation\":0,\"workspace_marker_preserved_only_in_original\":true,\"fresh_workspace_marker_absent\":true}},\"bundle_bytes\":{}}}",
        bundle.len()
    );
}
'''


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"cwd: {cwd}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def last_json(stdout: str, context: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{context} produced no output")
    payload = json.loads(lines[-1])
    if payload.get("verdict") != "PASS":
        raise SystemExit(f"{context} verdict was not PASS: {payload}")
    return payload


def run_python_artifacts() -> dict:
    with tempfile.TemporaryDirectory(prefix="burn-research-runtime-python-") as tmp:
        root = Path(tmp)
        wheels = root / "wheels"
        wheels.mkdir()

        run([sys.executable, "-m", "maturin", "build", "--out", str(wheels)], cwd=FFI_DIR)
        wheel_files = sorted(wheels.glob("*.whl"))
        if len(wheel_files) != 1:
            raise SystemExit(f"expected exactly one wheel, found {wheel_files}")

        venv = root / "venv"
        run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        venv_python = venv / "bin" / "python"
        run([
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            str(wheel_files[0]),
        ], cwd=root)

        consumer = root / "consumer.py"
        consumer.write_text(PYTHON_CONSUMER, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["BR_REPO_ROOT"] = str(REPO.resolve())
        completed = run([str(venv_python), str(consumer)], cwd=root, env=env)
        return last_json(completed.stdout, "installed Python wheel artifact consumer")


def run_rust_artifacts() -> dict:
    run(["cargo", "package", "--allow-dirty", "--no-verify"], cwd=REPO)
    crate_path = REPO / "target" / "package" / f"{PACKAGE_NAME}-{PACKAGE_VERSION}.crate"
    if not crate_path.is_file():
        raise SystemExit(f"missing packaged crate: {crate_path}")

    with tempfile.TemporaryDirectory(prefix="burn-research-runtime-rust-") as tmp:
        root = Path(tmp)
        extract_root = root / "package"
        extract_root.mkdir()
        with tarfile.open(crate_path, "r:gz") as archive:
            archive.extractall(extract_root)

        package_dir = extract_root / f"{PACKAGE_NAME}-{PACKAGE_VERSION}"
        consumer = root / "consumer"
        (consumer / "src").mkdir(parents=True)

        package_path = package_dir.as_posix().replace('"', '\\"')
        (consumer / "Cargo.toml").write_text(
            f'[package]\nname = "burn-research-runtime-artifact-consumer"\n'
            'version = "0.0.0"\nedition = "2021"\n\n'
            f'[dependencies]\nburn-research = {{ path = "{package_path}" }}\n',
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(RUST_CONSUMER, encoding="utf-8")

        run(["cargo", "generate-lockfile", "--manifest-path", str(consumer / "Cargo.toml")], cwd=consumer)
        completed = run([
            "cargo", "run", "--quiet", "--locked",
            "--manifest-path", str(consumer / "Cargo.toml"),
        ], cwd=consumer)
        return last_json(completed.stdout, "external Rust package artifact consumer")


def main() -> None:
    python_result = run_python_artifacts()
    rust_result = run_rust_artifacts()
    commit = run(["git", "rev-parse", "HEAD"], cwd=REPO).stdout.strip()

    report = {
        "schema": "burn-research.runtime-architecture-artifact-proof.v1",
        "verdict": "PASS",
        "commit": commit,
        "python_installed_wheel": python_result,
        "rust_external_package": rust_result,
        "boundaries": {
            "structure_not_state": True,
            "state_not_identity": True,
            "checkpoint_not_optimizer_state": True,
            "checkpoint_not_workspace_state": True,
            "optimizer_control_not_schedule_policy": True,
            "math_v9_composition_replay": True,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
