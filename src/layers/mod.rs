//! # Kontrak: `layers`
//!
//! ## Tanggung jawab
//! Engine komputasi per tipe layer: 1 file = 1 engine = 1 byte `LAYER_*`.
//! Setiap engine mengimplementasikan init deterministik, forward, dan
//! ekspor/impor state.
//!
//! ## Invariant
//! - Init default DETERMINISTIK NOL: bobot awal identik antar registry dan
//!   antar proses (keluhan #15). Tidak ada bobot implisit nondeterministik.
//! - Ekspor state byte-stabil: `state_record::deterministic_record_bytes`
//!   me-rekey `ParamId` acak Burn agar checkpoint identik untuk bobot
//!   identik; format checkpoint lama tetap dapat di-load.
//! - Validasi shape-vs-data gagal-cepat ke dua arah SEBELUM alokasi; tidak
//!   ada alokasi buta di bawah ceiling protokol.
//! - `shape_contract` dan `state_record` adalah `pub(crate)`: detail
//!   internal, bukan API publik.
//!
//! ## Error yang dijamin
//! - `tensor_too_large:` bila output melebihi budget 64 MiB (keluhan #18).
//!
//! ## Bukan tanggung jawab modul ini
//! - Angka protokol dan budget → `protocol`.
//! - Siklus hidup instance → `registry`.

pub mod activation;
pub mod binary;
pub mod conv;
pub mod custom;
pub mod embedding;
pub mod layout;
pub mod linear;
pub mod norm;
pub mod pool;
pub(crate) mod shape_contract;
pub(crate) mod state_record;
