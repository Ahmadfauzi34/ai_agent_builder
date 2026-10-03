//! Fasad WASM tunggal — domain `registry` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::registry::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::registry::runtime_contract::binding::BINDING_CAPABILITIES_V1;
use crate::registry::runtime_contract::inventory::INVENTORY_CAPABILITIES_V1;

#[wasm_bindgen(js_name = layerRegistryOperationBindingCapabilities)]
pub fn layer_registry_operation_binding_capabilities() -> String {
    BINDING_CAPABILITIES_V1.to_string()
}

#[wasm_bindgen(js_name = layerRegistryInventoryCapabilities)]
pub fn layer_registry_inventory_capabilities() -> String {
    INVENTORY_CAPABILITIES_V1.to_string()
}
