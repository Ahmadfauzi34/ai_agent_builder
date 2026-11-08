use super::conv::push_param;
use super::state_record::deterministic_record_bytes;
pub use crate::facade::wasm_types::WasmLinear;
use crate::layers::shape_contract::{require_axis_size, require_singleton_spatial};
use crate::{WasmBackend, WasmTensor};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use wasm_bindgen::prelude::*;

// --- CONFIG & MODULE ---
#[derive(Config, Debug)]
pub struct LinearLayerConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl LinearLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearLayer<B> {
        let linear = LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .with_initializer(burn::module::Initializer::Zeros)
            .init(device);
        LinearLayer { inner: linear }
    }
}

#[derive(Module, Debug)]
pub struct LinearLayer<B: Backend> {
    pub(crate) inner: Linear<B>,
}

impl<B: Backend> LinearLayer<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }
}

// --- WASM WRAPPER ---

// ============================================================
// FLOAT-BRIDGE (B) — baca/tulis bobot sebagai Vec<f32> flat.
// Ini kabel yang membuat ES (gradient-free) bisa menyentuh bobot,
// sehingga ES bisa melatih layer nyata, bukan cuma vektor abstrak.
//
// Urutan flat: weight row-major [in_dim * out_dim], lalu bias [out_dim] (kalau ada).
// Implementasi via Module Record (bobot jadi tensor plain) -> menghindari Parameter API.
// ============================================================

// ============================================================
// WEIGHT LAYOUT (M2) — linear. Mirror urutan getWeightsFlat.
// ============================================================

// ============================================================
// FLAT-BRIDGE HELPERS — composite layers (Tahap 7b).
// A burn Linear record's flat segment is weight row-major [in, out]
// then bias [out] (iff present) — the same order WasmLinear uses.
// SwiGlu (linear_inner/linear_outer) and SeBlock (fc1/fc2) embed
// LinearRecords, so they share these offset-tracking helpers instead
// of duplicating the slice arithmetic (drift here silently corrupts
// parameter slicing, same invariant as getWeightsFlat/weight_segs).
// ============================================================

/// Append one LinearRecord's flat segment (weight, then bias if present).
pub(crate) fn push_linear_record<B: Backend>(
    rec: &burn::nn::LinearRecord<B>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    push_param(&rec.weight, out)?;
    if let Some(b) = &rec.bias {
        push_param(b, out)?;
    }
    Ok(())
}

/// Write one LinearRecord's flat segment from `data` at `offset`.
/// Returns the new offset. Validates bounds (never panics on short data);
/// the caller checks the final offset == data.len().
pub(crate) fn set_linear_record<B: Backend>(
    rec: &mut burn::nn::LinearRecord<B>,
    data: &[f32],
    offset: usize,
) -> Result<usize, String> {
    let wd = rec.weight.dims(); // [in, out]
    let wlen = wd.iter().product::<usize>();
    let has_bias = rec.bias.is_some();
    let blen = if has_bias {
        rec.bias.as_ref().unwrap().dims().iter().product::<usize>()
    } else {
        0
    };
    if data.len() < offset + wlen + blen {
        return Err(format!(
            "setWeightsFlat: linear segment expected {} floats at offset {}, got {} total",
            wlen + blen,
            offset,
            data.len()
        ));
    }
    let device: <B as Backend>::Device = Default::default();
    rec.weight = burn::module::Param::from_data(
        burn::tensor::TensorData::new(data[offset..offset + wlen].to_vec(), wd),
        &device,
    );
    if has_bias {
        rec.bias = Some(burn::module::Param::from_data(
            burn::tensor::TensorData::new(
                data[offset + wlen..offset + wlen + blen].to_vec(),
                [blen],
            ),
            &device,
        ));
    }
    Ok(offset + wlen + blen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_weights(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let bucket = ((index * 29 + 7) % 61) as i32 - 30;
                bucket as f32 * 0.002
            })
            .collect()
    }

    #[test]
    fn immutable_dims_survive_weight_and_state_updates() {
        let mut layer = WasmLinear::new(3, 2, true);
        assert_eq!(layer.weight_dims(), vec![3, 2]);

        let weights = deterministic_weights(3 * 2 + 2);
        layer.set_weights_flat(&weights).expect("set weights");
        assert_eq!(layer.weight_dims(), vec![3, 2]);
        assert_eq!(layer.get_weights_flat().expect("read weights"), weights);

        let input = WasmTensor::new(&[0.25, -0.5, 0.75], &[1, 3, 1, 1]);
        let expected_output = layer.forward(&input).to_array();
        let state = layer.get_state().expect("serialize state");

        layer
            .set_weights_flat(&vec![0.0; 3 * 2 + 2])
            .expect("replace weights");
        layer.load_state(&state).expect("restore valid state");

        assert_eq!(layer.weight_dims(), vec![3, 2]);
        assert_eq!(layer.get_weights_flat().expect("restored weights"), weights);
        assert_eq!(layer.forward(&input).to_array(), expected_output);
    }

    #[test]
    fn mismatched_state_is_rejected_without_mutation() {
        let mut target = WasmLinear::new(3, 2, true);
        let weights = deterministic_weights(3 * 2 + 2);
        target
            .set_weights_flat(&weights)
            .expect("set target weights");
        let before = target.get_weights_flat().expect("read target weights");

        let mismatched_shape = WasmLinear::new(4, 2, true)
            .get_state()
            .expect("serialize mismatched state");
        assert!(target.load_state(&mismatched_shape).is_err());
        assert_eq!(target.weight_dims(), vec![3, 2]);
        assert_eq!(
            target.get_weights_flat().expect("weights after rejection"),
            before
        );

        let mismatched_bias = WasmLinear::new(3, 2, false)
            .get_state()
            .expect("serialize bias-mismatched state");
        assert!(target.load_state(&mismatched_bias).is_err());
        assert_eq!(target.weight_dims(), vec![3, 2]);
        assert_eq!(
            target
                .get_weights_flat()
                .expect("weights after bias rejection"),
            before
        );
    }

    #[test]
    fn corrupt_length_prefix_is_rejected_without_huge_allocation() {
        // Complaint #14 regression: a corrupt varint length prefix must not
        // make the decoder attempt a huge allocation. The tiered byte limit
        // (64 KiB for this ~200-byte state) rejects the 1 GiB claim via
        // LimitExceeded instead of Vec::with_capacity(1 GiB).
        let mut layer = WasmLinear::new(3, 2, true);
        let weights = deterministic_weights(3 * 2 + 2);
        layer.set_weights_flat(&weights).expect("set weights");
        let before = layer.get_weights_flat().expect("read weights");
        let state = layer.get_state().expect("serialize state");

        let mut corrupt = vec![0xFD]; // bincode varint u64 marker
        corrupt.extend_from_slice(&0x4000_0000u64.to_le_bytes()); // 1 GiB
        corrupt.extend_from_slice(&state[1..]);

        let err = layer
            .load_state(&corrupt)
            .expect_err("corrupt length prefix must be rejected");
        assert!(
            err.contains("LimitExceeded"),
            "expected LimitExceeded, got: {err}"
        );
        assert_eq!(
            layer.get_weights_flat().expect("weights after rejection"),
            before
        );
    }
}
