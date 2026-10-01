//! # Kontrak: `es`
//!
//! ## Tanggung jawab
//! Evolution strategies untuk optimasi: `rng` (keacakan), `strategy`
//! (varian ES), `objective` (fungsi tujuan), `optimizer` (loop ask/tell),
//! `diag` (diagnostik).
//!
//! ## Invariant
//! - `optimizer` tidak menyentuh I/O host; murni komputasi + state.
//! - Invariant diuji di `invariant_tests` (`#[cfg(test)]`).
//!
//! ## Bukan tanggung jawab modul ini
//! - Menjalankan model/layer → `layers/*`, `graph`.

pub mod rng;
pub mod diag;
pub mod strategy;
pub mod objective;
pub mod optimizer;

#[cfg(test)]
mod invariant_tests;
