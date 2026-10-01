use burn::record::{BurnRecord, FullPrecisionSettings, Record};
use burn::tensor::backend::Backend;

/// Decode a bincode payload with a byte limit derived from the input length.
///
/// Complaint #14: `bincode::config::standard()` has no byte limit, so a single
/// corrupt varint length prefix inside a state record makes `Vec::decode` call
/// `Vec::with_capacity(huge)`. Natively that aborts the process
/// (`memory allocation of N bytes failed`); on WASM the failed allocation
/// traps and permanently wedges the in-process instance. Bounding the decoder
/// turns those inputs into a plain per-call `DecodeError::LimitExceeded`
/// instead, and any allocation the decoder still attempts is O(input size),
/// which cannot fail in a functioning instance.
///
/// The limit is a const generic in bincode 2.0.1, so dispatch on the input
/// length to the smallest tier covering it with headroom. The 8x headroom
/// covers bincode's container claim accounting (`claim_container_read` books
/// `size_of::<T>()` per element, which over-estimates for nested containers).
/// Single state records beyond 128 MiB are rejected outright.
fn decode_from_slice_limited<'a, T>(data: &'a [u8], context: &str) -> Result<(T, usize), String>
where
    T: serde::de::DeserializeOwned,
{
    macro_rules! tier {
        ($limit:expr) => {
            bincode::serde::decode_from_slice::<T, _>(
                data,
                bincode::config::standard().with_limit::<$limit>(),
            )
            .map_err(|err| format!("{context}: invalid Burn bincode record: {err}"))
        };
    }

    let need = data.len().saturating_mul(8);
    if need <= (1_usize << 16) {
        tier! {65536}
    } else if need <= (1_usize << 20) {
        tier! {1048576}
    } else if need <= (1_usize << 24) {
        tier! {16777216}
    } else if need <= (1_usize << 30) {
        tier! {1073741824}
    } else {
        Err(format!(
            "{context}: state record too large: {} bytes (max 128 MiB)",
            data.len()
        ))
    }
}

/// Decode the same bincode payload used by Burn's `BinBytesRecorder`, while
/// keeping malformed or non-canonical bytes on our fallible boundary instead
/// of reaching recorder/load_record panic paths in Burn 0.20.1.
pub(crate) fn decode_bin_record<B, R>(
    data: &[u8],
    device: &B::Device,
    context: &str,
) -> Result<R, String>
where
    B: Backend,
    R: Record<B>,
{
    let (record, consumed): (BurnRecord<R::Item<FullPrecisionSettings>, B>, usize) =
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map_err(|err| format!("{context}: invalid Burn bincode record: {err}"))?;

    if consumed != data.len() {
        return Err(format!(
            "{context}: non-canonical Burn record: consumed {consumed} of {} bytes",
            data.len()
        ));
    }

    Ok(R::from_item::<FullPrecisionSettings>(record.item, device))
}
