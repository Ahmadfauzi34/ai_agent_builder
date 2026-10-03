//! Fasad WASM tunggal — domain `coprocessor` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::coprocessor::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::coprocessor::verify_vectors_report;

/// Compare an external implementation result with a trusted numerical reference.
///
/// This is intentionally dependency-free and returns compact JSON so an agent can
/// consume the proof result without coupling the produced artifact to this WASM runtime.
#[wasm_bindgen(js_name = mathVerifyVectors)]
pub fn math_verify_vectors(
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> Result<String, String> {
    verify_vectors_report(reference, candidate, abs_tol, rel_tol)
}
