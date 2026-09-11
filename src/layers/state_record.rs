use burn::record::{BurnRecord, FullPrecisionSettings, Record};
use burn::tensor::backend::Backend;

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
