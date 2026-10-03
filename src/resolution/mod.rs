//! Resolution domain: resolution workflow, review, revision, runtime bridge, and subject projection.
//!
//! `resolution.rs` is re-exported at the domain root so `resolution::*` paths
//! keep resolving to the items that previously lived at the crate root.

#[path = "resolution.rs"]
mod resolution_inner;
pub use resolution_inner::*;

pub mod resolution_review;
pub mod resolution_revision;
pub mod resolution_runtime_bridge;
pub mod resolution_subject;
