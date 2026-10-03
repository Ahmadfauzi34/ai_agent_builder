//! Domain `graph`: rencana eksekusi multi-input yang dikompilasi, parameter,
//! trace eksekusi, verifikasi kandidat, transaksi mutasi, dan binding reverify.
//!
//! Struktur: `graph.rs` dimuat via `#[path]` sebagai `graph_inner` privat agar
//! path lama `crate::graph::<item>` tetap hidup lewat `pub use graph_inner::*`
//! di bawah. Child `mod` via `#[path]` di dalam `graph.rs`
//! (`plan_explain`, `execution_trace`, `candidate_verification`,
//! `mutation_transaction`) TIDAK disentuh — mereka resolve relatif terhadap
//! file target `#[path]` dan ikut pindah bersama sibling-nya.

#[path = "graph.rs"]
mod graph_inner;
pub use graph_inner::*;

pub mod graph_candidate_verification;
pub mod graph_execution_trace;
pub mod graph_mutation_transaction;
pub mod graph_parameters;
pub mod graph_parameters_wasm;
pub mod graph_plan;
pub mod graph_plan_explain;
pub mod graph_reverify_execution_adapter;
pub mod graph_reverify_runtime_binding;
pub mod multi_input_graph;
