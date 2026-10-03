//! Fasad WASM tunggal — domain `agent` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::agent::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::capability_manifest;

/// Return a compact, machine-readable description of the stable WASM capabilities.
///
/// Agents should call this once before planning numerical work instead of inferring
/// features from generated JS glue or repeatedly probing exports.
#[wasm_bindgen(js_name = agentCapabilities)]
pub fn agent_capabilities() -> String {
    capability_manifest()
}
