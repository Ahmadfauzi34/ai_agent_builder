//! Callable tiered-limit guard for state-record decoding (complaint #14).
//!
//! The live `load_state` paths decode bincode under a byte-limit tier derived
//! from the input length
//! ([`crate::layers::state_record::tier_limit_for_len`]). That policy used to
//! be observable only by triggering it — a corrupt 226-byte payload once grew
//! WASM linear memory by 1 GiB before failing, because the tiered decoder was
//! dead code and the live path was unlimited.
//!
//! This module exposes the tier decision itself so agents, auditors, and the
//! `helper/` tooling can probe it directly: which tier a given input length
//! maps to, and whether it would be rejected outright — without performing
//! any decode and without risking the instance.

use wasm_bindgen::prelude::*;

use crate::layers::state_record::tier_limit_for_len;

/// Decide the bincode byte-limit tier for a state-record payload, without
/// decoding anything.
///
/// This is the guard's decision function: the same tier the live decode path
/// enforces. Only the input *length* is inspected, so probing is always safe.
///
/// Returns JSON:
/// `{"input_bytes":N,"tier_limit":M,"accepted":true,"reason":null}`
/// or `{"input_bytes":N,"tier_limit":null,"accepted":false,"reason":"..."}`.
///
/// An agent designing around untrusted checkpoints uses this to ask, before
/// touching `load_state`: "would this payload shape be accepted, and under
/// which tier?" A corrupt varint *inside* the payload is then bounded by that
/// tier at decode time (`DecodeError::LimitExceeded` instead of a wedged
/// instance).
///
/// Note: JSON is built by hand because `serde_json` is a non-wasm32-only
/// dependency in this crate; the `reason` string is fully controlled
/// (byte count + static text), so no escaping edge cases arise.
#[wasm_bindgen(js_name = decodeGuardCheck)]
pub fn decode_guard_check(data: &[u8]) -> String {
    let input_bytes = data.len();
    match tier_limit_for_len(input_bytes) {
        Ok(tier_limit) => format!(
            "{{\"input_bytes\":{input_bytes},\"tier_limit\":{tier_limit},\"accepted\":true,\"reason\":null}}"
        ),
        Err(reason) => format!(
            "{{\"input_bytes\":{input_bytes},\"tier_limit\":null,\"accepted\":false,\"reason\":\"{reason}\"}}"
        ),
    }
}
