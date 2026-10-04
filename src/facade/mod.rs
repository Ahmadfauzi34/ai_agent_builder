//! Fasad WASM tunggal (Opsi C, Fase 1 + Fase 2 + Fase 3).
//!
//! Seluruh `#[wasm_bindgen]` yang *pindah murni* dikumpulkan di sini:
//! - free function `#[wasm_bindgen]` (nama export JS = nama Rust, jadi pindah
//!   modul tidak mengubah permukaan JS),
//! - struct `Wasm*` yang memang sudah merupakan adapter tipis WASM,
//! - item `#[wasm_bindgen]` yang sebelumnya didefinisikan di `src/lib.rs`,
//! - blok `#[wasm_bindgen] impl` untuk struct domain (Fase 2: 13 struct,
//!   Fase 3: 4 blok kakek terakhir — MultiInputGraphPlan,
//!   MultiInputInputBundle, SemanticIngressManifest, AgentWorkspace).
//!   Struct tetap di domain dengan `#[wasm_bindgen]` sebagai marker ABI.
//!
//! Kompatibilitas: setiap item yang pindah di-re-export rangkap sehingga
//! `burn_research::<item>` (root) dan `burn_research::<domain-lama>::<item>`
//! tetap hidup. API Rust nol berubah; permukaan JS identik.

pub mod agent;
pub mod contracts;
pub mod coprocessor;
pub mod es;
pub mod evidence;
pub mod graph;
pub mod ingress;
pub mod interaction;
pub mod introspection;
pub mod math;
pub mod protocol;
pub mod registry;
pub mod resolution;
pub mod semantic;
pub mod tensor;
pub mod wasm_types;
pub mod workspace;
