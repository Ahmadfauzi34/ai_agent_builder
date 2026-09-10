use burn::record::{BurnRecord, FullPrecisionSettings, Record};
use burn::tensor::backend::Backend;

/// Decode the same bincode payload used by Burn's `BinBytesRecorder`, but keep
/// malformed/untrusted bytes on our fallible boundary instead of reaching the
/// recorder's internal `unwrap()` in Burn 0.20.1.
pub(crate) fn decode_bin_record<B, R>(
    data: &[u8],
    device: &B::Device,
    context: &str,
) -> Result<R, String>
where
    B: Backend,
    R: Record<B>,
{
    let (record, _consumed): (BurnRecord<R::Item<FullPrecisionSettings>, B>, usize) =
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map_err(|err| format!("{context}: invalid Burn bincode record: {err}"))?;

    Ok(R::from_item::<FullPrecisionSettings>(record.item, device))
}
