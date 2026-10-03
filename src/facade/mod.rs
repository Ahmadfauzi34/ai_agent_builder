//! Fasad WASM tunggal (Opsi C, Fase 1).
//!
//! Seluruh `#[wasm_bindgen]` yang *pindah murni* dikumpulkan di sini:
//! - free function `#[wasm_bindgen]` (nama export JS = nama Rust, jadi pindah
//!   modul tidak mengubah permukaan JS),
//! - struct `Wasm*` yang memang sudah merupakan adapter tipis WASM,
//! - item `#[wasm_bindgen]` yang sebelumnya didefinisikan di `src/lib.rs`.
//!
//! Yang BELUM pindah (Fase 2): 13 domain-struct + `#[wasm_bindgen] impl`
//! block — method-nya saling mengoper struct WASM lain by reference sehingga
//! butuh wrapper graph dengan desain per-tipe, bukan sekadar pindah file.
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
pub mod registry;
pub mod resolution;
pub mod semantic;
pub mod tensor;
pub mod wasm_types;
pub mod workspace;
