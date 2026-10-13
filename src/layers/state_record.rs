use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::record::{BinBytesRecorder, BurnRecord, FullPrecisionSettings, Record, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// Byte-limit tier for a state-record payload of `len` bytes.
///
/// The limit is a const generic in bincode 2.0.1, so callers dispatch on the
/// input length to the smallest tier covering it with headroom. The 8x
/// headroom covers bincode's container claim accounting (`claim_container_read`
/// books `size_of::<T>()` per element, which over-estimates for nested
/// containers). Single state records beyond 128 MiB are rejected outright.
///
/// This is the single source of truth for the tier policy: both the live
/// decode path and the callable guard below consult it.
pub(crate) fn tier_limit_for_len(len: usize) -> Result<usize, String> {
    let need = len.saturating_mul(8);
    if need <= (1_usize << 16) {
        Ok(65536)
    } else if need <= (1_usize << 20) {
        Ok(1048576)
    } else if need <= (1_usize << 24) {
        Ok(16777216)
    } else if need <= (1_usize << 30) {
        Ok(1073741824)
    } else {
        Err(format!(
            "state record too large: {} bytes (max 128 MiB)",
            len
        ))
    }
}

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

    match tier_limit_for_len(data.len()) {
        Ok(65536) => tier! {65536},
        Ok(1048576) => tier! {1048576},
        Ok(16777216) => tier! {16777216},
        Ok(1073741824) => tier! {1073741824},
        Ok(other) => Err(format!(
            "{context}: internal tier error: unexpected limit {other}"
        )),
        Err(e) => Err(format!("{context}: {e}")),
    }
}

/// Decode the same bincode payload used by Burn's `BinBytesRecorder`, while
/// keeping malformed or non-canonical bytes on our fallible boundary instead
/// of reaching recorder/load_record panic paths in Burn 0.20.1.
///
/// The decode runs under the tiered byte limit from
/// [`decode_from_slice_limited`]: a corrupt varint length prefix can no
/// longer make the decoder `with_capacity(huge)` (complaint #14 — a 226-byte
/// payload once grew WASM linear memory by 1 GiB before failing).
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
        decode_from_slice_limited(data, context)?;

    if consumed != data.len() {
        return Err(format!(
            "{context}: non-canonical Burn record: consumed {consumed} of {} bytes",
            data.len()
        ));
    }

    Ok(R::from_item::<FullPrecisionSettings>(record.item, device))
}

/// Re-keys every parameter of a cloned module with deterministic ids.
///
/// Root cause (complaint #15 follow-up, found via the durable-ingress CI
/// diagnostic): Burn assigns each `Param` a *random* `ParamId` at creation
/// (`burn_std::id::IdGenerator::generate`), and `into_record()` serializes
/// that id as the tensor's 13-character base32 name inside the bincode state
/// bytes. Two identically-initialized layers therefore exported different
/// checkpoint bytes even though weights, outputs and program identity were
/// all identical, breaking the deterministic-checkpoint contract that the
/// durable signed-ingress audit verifies.
///
/// The mapper walks the module in structural (field declaration) order and
/// assigns `ParamId::from(counter)`, so the id sequence is a pure function of
/// the module structure. Tensor values and initialization state are preserved
/// (`Param::from_mapped_value` keeps both); only the live tensor's random id
/// is replaced, and only on the clone used for export.
struct DeterministicParamIdMapper {
    counter: u64,
}

impl DeterministicParamIdMapper {
    fn new() -> Self {
        Self { counter: 0 }
    }

    fn next_id(&mut self) -> ParamId {
        let id = ParamId::from(self.counter);
        self.counter += 1;
        id
    }
}

impl<B: Backend> ModuleMapper<B> for DeterministicParamIdMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (_old_id, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next_id(), tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        let (_old_id, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next_id(), tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        let (_old_id, tensor, mapper) = param.consume();
        Param::from_mapped_value(self.next_id(), tensor, mapper)
    }
}

/// Serialize a module's record with deterministic tensor names.
///
/// Drop-in replacement for the
/// `BinBytesRecorder::default().record(module.clone().into_record(), ())`
/// pattern used by every layer `get_state()`: the clone is re-keyed through
/// [`DeterministicParamIdMapper`] before recording, so the exported bytes are
/// a pure function of (module structure, parameter values). The live module
/// is untouched, and the output remains a valid `BinBytesRecorder` payload
/// that `load_layer_state` accepts (old checkpoints with random ids still
/// load; only newly exported bytes are normalized).
pub(crate) fn deterministic_record_bytes<M, B>(module: &M) -> Result<Vec<u8>, String>
where
    B: Backend,
    M: Module<B> + Clone,
    M::Record: Record<B>,
{
    let mut mapper = DeterministicParamIdMapper::new();
    let rekeyed = module.clone().map(&mut mapper);
    let record = rekeyed.into_record();
    BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(record, ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tier_tests {
    use super::tier_limit_for_len;

    #[test]
    fn tier_boundaries_cover_input_lengths() {
        // need = len * 8; tier edges at need == 2^16 / 2^20 / 2^24 / 2^30.
        assert_eq!(tier_limit_for_len(0).unwrap(), 65536);
        assert_eq!(tier_limit_for_len(8192).unwrap(), 65536); // need == 65536
        assert_eq!(tier_limit_for_len(8193).unwrap(), 1048576);
        assert_eq!(tier_limit_for_len(131072).unwrap(), 1048576); // need == 2^20
        assert_eq!(tier_limit_for_len(131073).unwrap(), 16777216);
        assert_eq!(tier_limit_for_len(2097152).unwrap(), 16777216); // need == 2^24
        assert_eq!(tier_limit_for_len(2097153).unwrap(), 1073741824);
        assert_eq!(tier_limit_for_len(134217728).unwrap(), 1073741824); // 128 MiB
    }

    #[test]
    fn tier_rejects_beyond_128mib() {
        let err = tier_limit_for_len(134217729).unwrap_err();
        assert!(err.contains("too large"), "unexpected: {err}");
        // saturating_mul must not wrap into a small tier on huge inputs
        let err = tier_limit_for_len(usize::MAX).unwrap_err();
        assert!(err.contains("too large"), "unexpected: {err}");
    }

    #[test]
    fn killer_payload_shape_maps_to_smallest_tier() {
        // The Fase C killer: 226 corrupt bytes once grew WASM memory by 1 GiB.
        // The guard maps it to the 64 KiB tier, so any varint claim inside is
        // bounded there at decode time (LimitExceeded, no giant allocation).
        assert_eq!(tier_limit_for_len(226).unwrap(), 65536);
    }
}
