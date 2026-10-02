# `scripts/` — indeks

Folder ini datar dengan sengaja: setiap script adalah satu unit yang dipanggil
langsung oleh workflow, packager, atau audit lain. Peta di bawah menjelaskan
peran masing-masing kelompok.

## Gerbang CI (`audit_*`, `check_*`)

Dijalankan oleh workflow sebagai penegak. Jangan dipindah tanpa
memperbarui workflow + `docs/contracts/REGISTRY.md` (check #5 memaksa path
yang terdaftar harus ada).

| Script | Workflow | Peran |
|---|---|---|
| `check_contract_registry.py` | `main.yml` | indeks kontrak kanonis (50 kontrak, `include_str!`, JSON valid, path enforcer ada) |
| `check_wasm_surface.py` | `main.yml` | permukaan WASM terpaket vs kontrak (terdaftar di REGISTRY.md) |
| `check_distribution_links.mjs` | `main.yml` | link distribusi `pkg/` |
| `audit_rust_package.py` | `main.yml` | kemasan Rust |
| `audit_python_ffi.py` | `main.yml` | FFI Python |
| `audit_wasm_artifact.mjs` | `main.yml` | artefak WASM (R-18) |
| `audit_interactive_multi_input_ingress.mjs` | `main.yml` | ingress multi-input interaktif (R-20) |
| `audit_signed_ingress_provenance.mjs` | `main.yml` | provenance ingress bertanda |
| `audit_durable_signed_ingress.mjs` | `main.yml` | ingress durable + ledger (R-19) |
| `audit_state_bound_signed_ingress.mjs` | `main.yml` | state-bound ingress |
| `audit_node_host.mjs` | `main.yml` | host Node |
| `audit_python_wheel.py` | `python-wheel.yml` | wheel Python |
| `audit_python_relu_layer_spec.py` | `python-relu-layer-spec-proof.yml` | spesifikasi layer ReLU |
| `audit_runtime_architecture_artifacts.py` | `runtime-architecture-artifact-proof.yml` | artefak arsitektur runtime |
| `audit_operation_contract_registry.mjs` | — (terdaftar di REGISTRY.md) | registry kontrak operasi |
| `audit_branch_promotion_lineage.mjs` | — | lineage promosi branch |
| `audit_weight_tracking_restore.mjs` | — | restore pelacakan bobot |

## Utilitas packaging / CI

| Script | Peran |
|---|---|
| `package_node_host.mjs` | membangun `pkg/` + zip `burn-research-pkg-<sha>.zip` (dipanggil `main.yml`) |
| `generate_wasm_surface_actual.mjs` | menghasilkan `wasm-surface.actual.json` (`--write` / `--check`) |
| `smoke_distribution.mjs` | smoke test distribusi `pkg/` |
| `repo_map.py` | memetakan repo → graph (diuji `tests/test_repo_map.py`) |

## Library host (`*.mjs` tanpa awalan)

Dipakai bersama oleh audit (`import ... from './ingress_*.mjs'`) **dan**
dicopy ke `pkg/` oleh `package_node_host.mjs`. Pindah file = perbaiki import
relatif + daftar copy packager:
`branch_promotion_lineage.mjs`, `ingress_execution_receipt.mjs`,
`ingress_provenance.mjs`, `ingress_replay_ledger.mjs`,
`init_ingress_replay_ledger.mjs`, `interactive_multi_input_ingress.mjs`,
`operation_contract_registry.mjs`.

## Riset (`research_*`)

Satu paket per topik: `scripts/research_<topik>.{mjs,py}` +
`.github/workflows/<topik>-research.yml` + `docs/<topik>-research.md`.
Workflow riset jalan di setiap PR yang menyentuh `src/` — ini CI yang hidup,
bukan script mati. 14 `research_*.mjs` juga dipanggil langsung dari `main.yml`.

## Arsip (`research/archive/`)

Script riset yang sudah tidak direferensikan workflow/dokumen mana pun.
