//! Opsi F: setiap modul domain WAJIB memiliki blok kontrak `//! # Kontrak:`
//! di kepala file-nya. Test ini memastikan blok tersebut tidak hilang
//! diam-diam saat refactor.

use std::path::PathBuf;

const CONTRACT_MODULES: &[&str] = &[
    "src/protocol.rs",
    "src/registry.rs",
    "src/layers/mod.rs",
    "src/agent.rs",
    "src/graph/graph.rs",
    "src/math/mod.rs",
    "src/es/mod.rs",
    "src/contracts.rs",
    "src/semantic/semantic_lifecycle.rs",
    "src/resolution/resolution.rs",
    "src/ingress/input_port.rs",
    "src/evidence/proof_provenance.rs",
];

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn every_domain_module_declares_its_contract() {
    let mut missing = Vec::new();
    for rel in CONTRACT_MODULES {
        let path = manifest_dir().join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        // Blok kontrak harus berupa doc-comment level-modul di dekat kepala file.
        let head: String = src.lines().take(40).collect::<Vec<_>>().join("\n");
        if !head.contains("//! # Kontrak:") {
            missing.push(rel.to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "modul tanpa blok `//! # Kontrak:` di 40 baris pertama: {missing:?}"
    );
}

#[test]
fn contract_blocks_follow_the_four_section_format() {
    for rel in CONTRACT_MODULES {
        let path = manifest_dir().join(rel);
        let src = std::fs::read_to_string(&path).expect("readable");
        let head: String = src.lines().take(60).collect::<Vec<_>>().join("\n");
        for section in [
            "## Tanggung jawab",
            "## Invariant",
            "## Bukan tanggung jawab modul ini",
        ] {
            assert!(
                head.contains(section),
                "{rel}: blok kontrak kehilangan seksi wajib `{section}`"
            );
        }
    }
}
