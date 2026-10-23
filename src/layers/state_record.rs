use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::record::{BinBytesRecorder, BurnRecord, FullPrecisionSettings, Record, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::quantization::{
    Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
    compute_q_params, compute_range,
};
use burn::tensor::{Bool, Int, Tensor};
use serde::Serialize;

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

// ============================================================
// QUANTIZED STATE (int8 PTQ) — opsi A, Fase 1.
//
// Bobot disimpan sebagai int8 simetris (burn 0.20.1; backend NdArray hanya
// mengimplementasikan Q8S/Q8F — skema lain panic `unimplemented!`) + scale
// f32 per blok. Live module TETAP f32; yang terkuantisasi hanya representasi
// serialisasinya. Komputasi tidak berubah (tidak ada integer compute di burn:
// tiap op otomatis dequantize -> float -> requantize).
//
// Skema string (kontrak agen):
//   "q8-per-tensor" — satu scale per param.
//   "q8-per-block"  — satu scale per blok kanonis (canonical_block_dims).
//
// Determinisme: kuantisasi adalah fungsi murni (min-max -> scale -> round),
// tanpa randomness; bytes output deterministik untuk (struktur, nilai) sama.
// Batas error per elemen: max|w - dequantize(w)| <= scale_max/2 per blok,
// dapat dihitung SEBELUM eksperimen dari data bobotnya.
// ============================================================

/// Envelope state terkuantisasi: magic 4 byte, version 1 byte,
/// inner_len u64 LE 8 byte, lalu inner bytes (BinBytesRecorder atas record
/// terkuantisasi).
/// `load_state` mendeteksi magic ini dan menormalisasi ke f32 SEBELUM jalur
/// validasi existing, jadi semua validasi struktur tetap berlaku.
pub(crate) const QUANTIZED_MAGIC: &[u8; 4] = b"BQZ1";
pub(crate) const QUANTIZED_VERSION: u8 = 1;
const QUANTIZED_HEADER_LEN: usize = 4 + 1 + 8;

/// Granularitas kuantisasi dari sudut pandang agen (tanpa detail BlockSize).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantGranularity {
    PerTensor,
    PerBlock,
}

pub(crate) fn parse_quant_scheme(s: &str) -> Result<QuantGranularity, String> {
    match s {
        "q8-per-tensor" => Ok(QuantGranularity::PerTensor),
        "q8-per-block" => Ok(QuantGranularity::PerBlock),
        other => Err(format!(
            "unknown quantization scheme '{other}' (expected 'q8-per-tensor' | 'q8-per-block')"
        )),
    }
}

fn quant_scheme_name(granularity: QuantGranularity) -> &'static str {
    match granularity {
        QuantGranularity::PerTensor => "q8-per-tensor",
        QuantGranularity::PerBlock => "q8-per-block",
    }
}

/// Geometri blok kanonis untuk `QuantGranularity::PerBlock`.
///
/// Aturan (deterministik, didokumentasikan, diuji):
/// - Param 1-D (bias, norm) -> per-tensor. Blok per-elemen = degenerate
///   (lebih banyak scale daripada nilai).
/// - Param D-dim (D >= 2) -> blok [1, b1, ..., b_{D-1}] dengan b_i = pembagi
///   terbesar dari d_i yang muat di u8 (<= 255; batas BlockSize burn).
///   Untuk semua dim <= 255 ini = satu slice dim-0 (mis. per-output-channel
///   untuk conv [o,i,h,w], per-baris untuk linear [in,out]).
/// - Tiap d_i habis dibagi b_i, sehingga numel habis dibagi block_elems
///   (syarat assert di burn) dan flat chunk selaras dengan slice dim-0.
///
/// Return `None` -> pakai per-tensor.
pub(crate) fn canonical_block_dims(shape: &[usize]) -> Option<Vec<u8>> {
    if shape.len() < 2 {
        return None;
    }
    let mut dims = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        if i == 0 {
            dims.push(1u8);
            continue;
        }
        // Pembagi terbesar dari d yang <= 255. Selalu ada (minimal 1).
        let b = (1..=d.min(255)).rev().find(|b| d % b == 0).unwrap_or(1);
        dims.push(b as u8);
    }
    Some(dims)
}

fn quant_scheme_for_shape(granularity: QuantGranularity, shape: &[usize]) -> QuantScheme {
    let base = || QuantScheme {
        value: QuantValue::Q8S,
        param: QuantParam::F32,
        store: QuantStore::Native,
        level: QuantLevel::Tensor,
        mode: QuantMode::Symmetric,
    };
    match granularity {
        QuantGranularity::PerTensor => base(),
        QuantGranularity::PerBlock => match canonical_block_dims(shape) {
            Some(dims) => QuantScheme {
                level: QuantLevel::block(dims),
                ..base()
            },
            None => base(),
        },
    }
}

/// Satu baris laporan per param terkuantisasi.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct QuantEntry {
    pub index: usize,
    pub shape: Vec<usize>,
    pub num_scales: usize,
    pub scale_max: f32,
    /// Batas error absolut per elemen: max|w - dequantize(w)| <= bound.
    pub bound: f32,
}

/// Laporan kuantisasi satu module (untuk agen, sebagai JSON).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct QuantReport {
    pub scheme: String,
    pub params: Vec<QuantEntry>,
    pub bytes_f32: usize,
    pub bytes_q8: usize,
    pub ratio: f32,
}

/// ModuleMapper yang mengkuantisasi tiap param float dengan algoritma burn
/// (compute_range + compute_q_params + Tensor::quantize) sekaligus mencatat
/// laporan. Dipakai pada clone; live module tidak disentuh.
struct QuantizingMapper {
    granularity: QuantGranularity,
    entries: Vec<QuantEntry>,
}

impl<B: Backend> ModuleMapper<B> for QuantizingMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let index = self.entries.len();
        let shape: Vec<usize> = tensor.dims().to_vec();
        let scheme = quant_scheme_for_shape(self.granularity, &shape);
        let range = compute_range(&scheme, &tensor, &Calibration::MinMax);
        let qparams = compute_q_params(&scheme, range);
        let scales: Vec<f32> = qparams
            .scales
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("quantize: scales readback");
        let scale_max = scales.iter().copied().fold(0.0f32, f32::max);
        self.entries.push(QuantEntry {
            index,
            shape,
            num_scales: scales.len().max(1),
            scale_max,
            bound: scale_max / 2.0,
        });
        let tensor = tensor.quantize(&scheme, qparams);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// ModuleMapper yang mengembalikan tiap param terkuantisasi ke f32.
/// Idempoten: param non-QFloat dikembalikan apa adanya
/// (Tensor::dequantize = no-op untuk non-quantized).
struct DequantizeMapper;

impl<B: Backend> ModuleMapper<B> for DequantizeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.dequantize();
        Param::from_mapped_value(id, tensor, mapper)
    }
}

pub(crate) fn dequantize_module<M, B>(module: &M) -> M
where
    B: Backend,
    M: Module<B> + Clone,
{
    module.clone().map(&mut DequantizeMapper)
}

/// Kuantisasi state sebuah module ke int8 (skema string kontrak agen).
///
/// Return `(enveloped_bytes, report_json)`. Live module tidak disentuh.
/// Bytes deterministik: fungsi murni dari (struktur module, nilai param).
pub(crate) fn quantized_record_bytes<M, B>(
    module: &M,
    scheme_str: &str,
) -> Result<(Vec<u8>, String), String>
where
    B: Backend,
    M: Module<B> + Clone,
    M::Record: Record<B>,
{
    let granularity = parse_quant_scheme(scheme_str)?;

    // Rekey deterministik dulu (jalur yang sama dengan state f32).
    let mut id_mapper = DeterministicParamIdMapper::new();
    let rekeyed = module.clone().map(&mut id_mapper);

    // Ukuran f32 untuk laporan (sudah rekeyed -> deterministik).
    let bytes_f32 = BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(rekeyed.clone().into_record(), ())
        .map_err(|e| e.to_string())?
        .len();

    // Kuantisasi + kumpulkan laporan.
    let mut qmapper = QuantizingMapper {
        granularity,
        entries: Vec::new(),
    };
    let quantized = rekeyed.map(&mut qmapper);
    let inner = BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(quantized.into_record(), ())
        .map_err(|e| e.to_string())?;

    // Envelope.
    let mut out = Vec::with_capacity(QUANTIZED_HEADER_LEN + inner.len());
    out.extend_from_slice(QUANTIZED_MAGIC);
    out.push(QUANTIZED_VERSION);
    out.extend_from_slice(&(inner.len() as u64).to_le_bytes());
    out.extend_from_slice(&inner);

    let report = QuantReport {
        scheme: quant_scheme_name(granularity).to_string(),
        params: qmapper.entries,
        bytes_f32,
        bytes_q8: out.len(),
        ratio: out.len() as f32 / bytes_f32.max(1) as f32,
    };
    let report_json =
        serde_json::to_string(&report).map_err(|e| format!("quant report: {e}"))?;
    Ok((out, report_json))
}

/// Kembalikan inner bytes jika `data` ber-envelope kuantisasi; else `None`.
/// Validasi ketat: magic + version + panjang cocok persis.
pub(crate) fn strip_quantized_envelope(data: &[u8]) -> Option<&[u8]> {
    if data.len() < QUANTIZED_HEADER_LEN {
        return None;
    }
    if &data[..4] != QUANTIZED_MAGIC {
        return None;
    }
    if data[4] != QUANTIZED_VERSION {
        return None;
    }
    let inner_len = u64::from_le_bytes(data[5..13].try_into().ok()?) as usize;
    if inner_len != data.len() - QUANTIZED_HEADER_LEN {
        return None;
    }
    Some(&data[QUANTIZED_HEADER_LEN..])
}

/// Normalisasi state bytes untuk `load_state`.
///
/// Envelope kuantisasi -> decode inner -> dekuantisasi -> f32 bytes.
/// Non-envelope dikembalikan apa adanya. F32 bytes hasil normalisasi lalu
/// mengalir ke jalur validasi + load existing TANPA perubahan, sehingga semua
/// invariant struktur per-type tetap berlaku.
pub(crate) fn normalize_state_bytes<B, M>(
    data: &[u8],
    template: &M,
    device: &B::Device,
    context: &str,
) -> Result<Vec<u8>, String>
where
    B: Backend,
    M: Module<B> + Clone,
    M::Record: Record<B>,
{
    let Some(inner) = strip_quantized_envelope(data) else {
        return Ok(data.to_vec());
    };
    let record: M::Record = decode_bin_record(inner, device, context)?;
    let quantized = template.clone().load_record(record);
    let f32_module = dequantize_module(&quantized);
    deterministic_record_bytes(&f32_module)
}

#[cfg(test)]
mod tier_tests {    use super::tier_limit_for_len;

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

#[cfg(test)]
mod quant_tests {
    use super::*;
    use burn::module::Module;
    use burn::nn::LinearConfig;
    use burn::tensor::TensorData;

    type B = crate::WasmBackend;

    fn test_linear() -> burn::nn::Linear<B> {
        let device = Default::default();
        let mut layer: burn::nn::Linear<B> =
            LinearConfig::new(8, 64).with_bias(true).init(&device);
        // Deterministic non-trivial weights (incl. an outlier row).
        let w: Vec<f32> = (0..8 * 64)
            .map(|i| {
                let r = (i / 64) as f32;
                let v = (((i * 29 + 7) % 61) as f32 - 30.0) * 0.002;
                if r == 0.0 { v * 10.0 } else { v }
            })
            .collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.001).collect();
        layer.weight = Param::from_data(TensorData::new(w, [8, 64]), &device);
        layer.bias = Some(Param::from_data(TensorData::new(b, [64]), &device));
        layer
    }

    #[test]
    fn scheme_strings_parse() {
        assert_eq!(
            parse_quant_scheme("q8-per-tensor").unwrap(),
            QuantGranularity::PerTensor
        );
        assert_eq!(
            parse_quant_scheme("q8-per-block").unwrap(),
            QuantGranularity::PerBlock
        );
        assert!(parse_quant_scheme("q4").is_err());
        assert!(parse_quant_scheme("").is_err());
    }

    #[test]
    fn canonical_block_dims_cases() {
        assert_eq!(canonical_block_dims(&[8, 64]), Some(vec![1, 64]));
        assert_eq!(canonical_block_dims(&[3]), None); // 1-D -> per-tensor
        assert_eq!(canonical_block_dims(&[]), None);
        assert_eq!(canonical_block_dims(&[4, 512]), Some(vec![1, 128]));
        assert_eq!(canonical_block_dims(&[2, 1000]), Some(vec![1, 250]));
        assert_eq!(canonical_block_dims(&[2, 3, 4]), Some(vec![1, 3, 4]));
        // Tiap dim habis dibagi bloknya.
        for shape in [&[8usize, 64][..], &[4, 512][..], &[2, 1000][..], &[2, 3, 4][..]] {
            let dims = canonical_block_dims(shape).unwrap();
            for (d, b) in shape.iter().zip(dims.iter()) {
                assert_eq!(d % (*b as usize), 0, "shape {shape:?}");
            }
        }
    }

    #[test]
    fn envelope_roundtrip_and_reject() {
        let inner = vec![1u8, 2, 3];
        let mut out = Vec::new();
        out.extend_from_slice(QUANTIZED_MAGIC);
        out.push(QUANTIZED_VERSION);
        out.extend_from_slice(&(inner.len() as u64).to_le_bytes());
        out.extend_from_slice(&inner);
        assert_eq!(strip_quantized_envelope(&out), Some(inner.as_slice()));
        // Magic rusak -> None (jalur f32 legacy).
        let mut bad = out.clone();
        bad[0] = 0xFF;
        assert_eq!(strip_quantized_envelope(&bad), None);
        // Panjang tidak cocok -> None.
        let mut bad2 = out.clone();
        bad2.pop();
        assert_eq!(strip_quantized_envelope(&bad2), None);
        // Terlalu pendek -> None.
        assert_eq!(strip_quantized_envelope(&[1, 2, 3]), None);
    }

    #[test]
    fn quantized_state_is_smaller_and_deterministic() {
        let layer = test_linear();
        let (b1, r1) = quantized_record_bytes(&layer, "q8-per-block").unwrap();
        let (b2, r2) = quantized_record_bytes(&layer, "q8-per-block").unwrap();
        assert_eq!(b1, b2, "quantized bytes must be byte-deterministic");
        assert_eq!(r1, r2);
        assert_eq!(&b1[..4], QUANTIZED_MAGIC);
        let rep: serde_json::Value = serde_json::from_str(&r1).unwrap();
        assert_eq!(rep["scheme"], "q8-per-block");
        assert!(rep["bytes_q8"].as_u64().unwrap() < rep["bytes_f32"].as_u64().unwrap());
        // Bobot dominan: ~4x (8*64+64 floats -> int8 + scales).
        let ratio = rep["ratio"].as_f64().unwrap();
        assert!(ratio < 0.5, "expected strong compression, got ratio {ratio}");
        assert_eq!(rep["params"].as_array().unwrap().len(), 2); // weight + bias
    }

    #[test]
    fn quantize_bound_holds_per_param() {
        let layer = test_linear();
        let (qbytes, report_json) = quantized_record_bytes(&layer, "q8-per-block").unwrap();
        let device: <B as Backend>::Device = Default::default();
        let inner = strip_quantized_envelope(&qbytes).unwrap();
        type Rec = <burn::nn::Linear<B> as Module<B>>::Record;
        let record: Rec = decode_bin_record(inner, &device, "test").unwrap();
        let qlayer = layer.clone().load_record(record);
        let f32layer = dequantize_module(&qlayer);

        let rep: serde_json::Value = serde_json::from_str(&report_json).unwrap();
        let params = rep["params"].as_array().unwrap();
        let bound_max = params
            .iter()
            .map(|p| p["bound"].as_f64().unwrap())
            .fold(0.0f64, f64::max);
        // weight [8,64] (2-D) dan bias [64] (1-D) dicek terpisah (tipe rank beda).
        let w_orig: Vec<f32> = layer.weight.val().into_data().into_vec::<f32>().unwrap();
        let w_dq: Vec<f32> = f32layer.weight.val().into_data().into_vec::<f32>().unwrap();
        let b_orig: Vec<f32> = layer
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let b_dq: Vec<f32> = f32layer
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        for (name, a, b) in [("weight", &w_orig, &w_dq), ("bias", &b_orig, &b_dq)] {
            let err = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                err as f64 <= bound_max * 1.001,
                "{name}: measured err {err} exceeds max reported bound {bound_max}"
            );
        }
    }

    #[test]
    fn normalize_load_roundtrip_forward_equivalence() {
        let layer = test_linear();
        let device: <B as Backend>::Device = Default::default();
        let (qbytes, report_json) = quantized_record_bytes(&layer, "q8-per-block").unwrap();

        // normalize -> f32 bytes -> decode -> load ke layer fresh.
        let norm = normalize_state_bytes(&qbytes, &layer, &device, "test").unwrap();
        assert_ne!(&norm[..4], QUANTIZED_MAGIC, "normalized must be plain f32");
        type Rec = <burn::nn::Linear<B> as Module<B>>::Record;
        let record: Rec = decode_bin_record(&norm, &device, "test").unwrap();
        let loaded = layer.clone().load_record(record);

        // Ekuivalensi forward dengan toleransi turunan (bukan asal).
        // y = Wx + b -> |dy_i| <= bound_w * sum|x| + bound_b.
        let rep: serde_json::Value = serde_json::from_str(&report_json).unwrap();
        let params = rep["params"].as_array().unwrap();
        let bound_of = |ndim: usize| -> f64 {
            params
                .iter()
                .find(|p| p["shape"].as_array().unwrap().len() == ndim)
                .unwrap()["bound"]
                .as_f64()
                .unwrap()
        };
        let bound_w = bound_of(2);
        let bound_b = bound_of(1);
        let x: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.25).collect();
        let sum_abs_x: f64 = x.iter().map(|v| v.abs() as f64).sum();
        let tol = bound_w * sum_abs_x + bound_b;

        let input: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(x, [1, 8]), &device);
        let out_orig: Vec<f32> = layer
            .forward(input.clone())
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let out_loaded: Vec<f32> = loaded
            .forward(input)
            .into_data()
            .into_vec::<f32>()
            .unwrap();
        let err = out_orig
            .iter()
            .zip(out_loaded.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0f64, f64::max);
        assert!(
            err <= tol * 1.01,
            "forward err {err} exceeds derived tolerance {tol}"
        );
    }

    #[test]
    fn per_tensor_scheme_also_works() {
        let layer = test_linear();
        let (b, r) = quantized_record_bytes(&layer, "q8-per-tensor").unwrap();
        assert_eq!(&b[..4], QUANTIZED_MAGIC);
        let rep: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert_eq!(rep["scheme"], "q8-per-tensor");
        for p in rep["params"].as_array().unwrap() {
            assert_eq!(p["num_scales"], 1);
        }
    }
}
