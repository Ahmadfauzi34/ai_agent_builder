//! Canonical identity-bound value sources for Math Program plans.
//!
//! This module deliberately does not implement implicit broadcasting. `fillLike` copies only the
//! rank-4 shape of a reference tensor and materializes one finite scalar value over that shape.
//! The scalar is canonical plan metadata, so it can participate in deterministic program identity.

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::{WasmBackend, WasmTensor};

pub(crate) const PARAM_FILL_LIKE: u8 = 7;
pub(crate) const FILL_LIKE_PARAM_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FillLikeParams {
    scalar_bits: u32,
}

impl FillLikeParams {
    pub(crate) fn new(value: f32) -> Result<Self, String> {
        if !value.is_finite() {
            return Err(format!(
                "MathProgram.fillLike: scalar must be finite, got {value}"
            ));
        }
        Ok(Self {
            scalar_bits: canonical_f32_bits(value),
        })
    }

    pub(crate) fn decode(payload: &[u8]) -> Result<Self, String> {
        if payload.len() != FILL_LIKE_PARAM_BYTES {
            return Err(format!(
                "MathProgram.fillLike: payload must contain exactly {FILL_LIKE_PARAM_BYTES} bytes, got {}",
                payload.len()
            ));
        }
        let bits = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let value = f32::from_bits(bits);
        let decoded = Self::new(value)?;
        if decoded.scalar_bits != bits {
            return Err("MathProgram.fillLike: scalar payload is not canonically encoded".into());
        }
        Ok(decoded)
    }

    pub(crate) fn encode(self) -> [u8; FILL_LIKE_PARAM_BYTES] {
        self.scalar_bits.to_le_bytes()
    }

    pub(crate) fn value(self) -> f32 {
        f32::from_bits(self.scalar_bits)
    }
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

pub(crate) fn fill_like(
    reference: &WasmTensor,
    params: FillLikeParams,
) -> Result<WasmTensor, String> {
    let dims = reference.inner.dims();
    let count = dims.into_iter().try_fold(1usize, |count, dim| {
        count.checked_mul(dim).ok_or_else(|| {
            format!(
                "MathProgram.fillLike: element-count overflow for reference shape {dims:?}"
            )
        })
    })?;
    let device = reference.inner.device();
    let data = TensorData::new(vec![params.value(); count], dims);
    Ok(WasmTensor {
        inner: Tensor::<WasmBackend, 4>::from_data(data, &device),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_codec_is_finite_and_canonicalizes_negative_zero() {
        let positive = FillLikeParams::new(0.0).unwrap();
        let negative = FillLikeParams::new(-0.0).unwrap();
        assert_eq!(positive, negative);
        assert_eq!(positive.encode(), [0; 4]);
        assert_eq!(FillLikeParams::decode(&negative.encode()).unwrap(), positive);
        assert!(FillLikeParams::new(f32::NAN).is_err());
        assert!(FillLikeParams::new(f32::INFINITY).is_err());
        assert!(FillLikeParams::new(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn noncanonical_negative_zero_payload_is_rejected() {
        assert!(FillLikeParams::decode(&(-0.0f32).to_bits().to_le_bytes()).is_err());
    }

    #[test]
    fn fill_like_preserves_exact_reference_shape_and_materializes_value() {
        let reference = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let output = fill_like(&reference, FillLikeParams::new(2.5).unwrap()).unwrap();
        assert_eq!(output.shape(), reference.shape());
        assert_eq!(output.to_array(), vec![2.5, 2.5, 2.5, 2.5]);
    }
}
