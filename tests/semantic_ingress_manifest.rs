use burn_research::input_port::workspace_bind_input_port_metadata;
use burn_research::semantic_ingress_manifest::{
    semantic_ingress_manifest_status, SemanticIngressManifest,
};
use burn_research::workspace::AgentWorkspace;

#[test]
fn required_deferred_memory_keeps_multi_source_manifest_runtime_incomplete() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "market-feed".into(),
        18,
        "obs:18".into(),
    )
    .unwrap();

    let mut manifest = SemanticIngressManifest::new();
    manifest
        .add_runtime_backed_port(
            "observation".into(),
            "observation".into(),
            "market-feed".into(),
            18,
            "obs:18".into(),
            0,
            true,
        )
        .unwrap();
    manifest
        .add_deferred_port(
            "memory".into(),
            "state".into(),
            "agent-memory".into(),
            4,
            "mem:4".into(),
            true,
        )
        .unwrap();
    manifest
        .add_deferred_port(
            "objective".into(),
            "context".into(),
            "objective-store".into(),
            2,
            "obj:2".into(),
            false,
        )
        .unwrap();
    manifest
        .add_deferred_port(
            "constraints".into(),
            "context".into(),
            "constraint-store".into(),
            3,
            "constraint:3".into(),
            false,
        )
        .unwrap();

    let status: serde_json::Value =
        serde_json::from_str(&semantic_ingress_manifest_status(&workspace, &manifest)).unwrap();

    assert_eq!(status["port_count"], 4);
    assert_eq!(status["runtime_backed_port_count"], 1);
    assert_eq!(status["deferred_port_count"], 3);
    assert_eq!(status["runtime_coverage_complete"], false);
    assert_eq!(status["required_uncovered_count"], 1);
    assert_eq!(status["execution_authorized"], false);
}

#[test]
fn logical_multi_source_manifest_never_claims_slot_one_as_runtime_backing() {
    let mut manifest = SemanticIngressManifest::new();

    let error = manifest
        .add_runtime_backed_port(
            "memory".into(),
            "state".into(),
            "agent-memory".into(),
            1,
            "mem:1".into(),
            1,
            true,
        )
        .unwrap_err();

    assert!(error.contains("only slot 0"));
    assert_eq!(manifest.port_count(), 0);
}

#[test]
fn deferred_ports_do_not_mutate_or_replace_real_slot_zero_metadata() {
    let mut workspace = AgentWorkspace::new(4).unwrap();
    workspace_bind_input_port_metadata(
        &mut workspace,
        "observation".into(),
        "sensor".into(),
        7,
        "obs:7".into(),
    )
    .unwrap();

    let before = workspace.snapshot();

    let mut manifest = SemanticIngressManifest::new();
    manifest
        .add_deferred_port(
            "memory".into(),
            "state".into(),
            "memory".into(),
            2,
            "mem:2".into(),
            true,
        )
        .unwrap();

    let _ = semantic_ingress_manifest_status(&workspace, &manifest);
    assert_eq!(workspace.snapshot(), before);
}
