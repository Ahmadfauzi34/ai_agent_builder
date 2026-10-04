//! Fasad WASM tunggal — domain `es` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::es::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

/// Machine-readable ES contracts for agent planning.
#[wasm_bindgen(js_name = esCapabilities)]
pub fn es_capabilities() -> String {
    concat!(
        "{",
        "\"entry\":\"EsOptimizer\",",
        "\"constructor\":{\"mode\":\"legacy_forgiving\",\"coercions\":[\"dim_zero_to_one\",\"pop_below_two_to_two\",\"unknown_strategy_to_openes\",\"openes_odd_pop_truncates_to_pairs\"]},",
        "\"strict_factory\":\"EsOptimizer.strict\",",
        "\"strategies\":{\"openes\":0,\"mu_lambda\":1},",
        "\"strict_contract\":{",
        "\"dim\":{\"min\":1},",
        "\"openes\":{\"pop_min\":2,\"pop_even\":true,\"sigma\":\"finite>0\",\"lr\":\"finite>0\"},",
        "\"mu_lambda\":{\"pop_min\":2,\"sigma\":\"finite>0\",\"lr\":\"omit\"}},",
        "\"lifecycle\":\"ask->tell\",",
        "\"controls\":{\"set_learning_rate\":{\"method\":\"setLearningRate\",\"strategy\":\"openes\",\"lr\":\"finite>0\",\"lifecycle\":\"between_completed_generations\"}},",
        "\"linear_demo\":{\"method\":\"runLinearDemo\",\"optimizer_dim\":6,\"gens_min\":1}",
        "}"
    )
    .to_string()
}
