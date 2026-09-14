//! Deterministic positional value sources for rank-4 tensors.
//!
//! `indicesLike` uses a reference tensor only for its rank-4 shape and device. Reference values do
//! not participate in the result. The generated coordinate values are finite, non-negative exact
//! f32 integers within the explicit v1 precision boundary.

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::{WasmBackend, WasmTensor};

/// Largest coordinate up to which every non-negative integer is exactly representable by f32.
pub const MAX_EXACT_F32_COORDINATE: usize = 1 << 24;

/// Largest supported axis length for `indicesLike` v1.
///
/// Coordinates span `0..axis_length`, so an axis of this length has the largest coordinate
/// `2^24`, which is still exactly representable by f32.
pub const MAX_INDICES_LIKE_AXIS_LENGTH: usize = MAX_EXACT_F32_COORDINATE + 1;

fn checked_axis(axis: u32, context: &str) -> Result<usize, String> {
    if axis >= 4 {
        return Err(format!("{context}: axis must be in 0..4, got {axis}"));
    }
    Ok(axis as usize)
}

pub(crate) fn validate_indices_like_shape(
    shape: [usize; 4],
    axis: usize,
) -> Result<usize, String> {
    for (shape_axis, dim) in shape.iter().copied().enumerate() {
        if dim == 0 {
            return Err(format!(
                "IndexSource.indicesLike: zero-sized dimensions are not supported in v1; axis {shape_axis} has length 0"
            ));
        }
    }

    let axis_length = shape[axis];
    if axis_length > MAX_INDICES_LIKE_AXIS_LENGTH {
        return Err(format!(
            "IndexSource.indicesLike: axis {axis} length {axis_length} exceeds exact-f32 coordinate bound {MAX_INDICES_LIKE_AXIS_LENGTH}"
        ));
    }

    shape.into_iter().try_fold(1usize, |count, dim| {
        count.checked_mul(dim).ok_or_else(|| {
            format!(
                "IndexSource.indicesLike: element-count overflow for reference shape {shape:?}"
            )
        })
    })
}

pub fn index_source_capabilities() -> String {
    format!(
        concat!(
            "{{",
            "\"schema\":\"burn-research.index-source.v1\",",
            "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
            "\"rank\":4,",
            "\"sources\":[\"indicesLike\"],",
            "\"axis\":\"runtime_0_to_3\",",
            "\"max_exact_f32_coordinate\":{},",
            "\"max_supported_axis_length\":{},",
            "\"contracts\":{{",
            "\"reference_values_ignored\":true,",
            "\"shape_preserved\":true,",
            "\"finite_outputs\":true,",
            "\"zero_sized_dimensions\":false,",
            "\"implicit_broadcasting\":false,",
            "\"stateless\":true,",
            "\"registry_independent\":true,",
            "\"grants_authority\":false",
            "}}",
            "}}"
        ),
        MAX_EXACT_F32_COORDINATE, MAX_INDICES_LIKE_AXIS_LENGTH
    )
}

/// Stateless deterministic rank-4 positional source.
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorIndexSource;

impl TensorIndexSource {
    pub fn new() -> Self {
        Self
    }

    /// Materialize exact f32 coordinates along one axis while preserving the reference shape.
    pub fn indices_like(
        &self,
        reference: &WasmTensor,
        axis: u32,
    ) -> Result<WasmTensor, String> {
        // Validate the scalar parameter before consulting tensor metadata or executing backend work.
        let axis = checked_axis(axis, "IndexSource.indicesLike")?;
        let shape = reference.inner.dims();
        let count = validate_indices_like_shape(shape, axis)?;

        let stride = shape[(axis + 1)..]
            .iter()
            .copied()
            .try_fold(1usize, |stride, dim| {
                stride.checked_mul(dim).ok_or_else(|| {
                    format!(
                        "IndexSource.indicesLike: stride overflow for reference shape {shape:?}"
                    )
                })
            })?;

        let axis_length = shape[axis];
        let mut values = Vec::with_capacity(count);
        for flat_index in 0..count {
            let coordinate = (flat_index / stride) % axis_length;
            debug_assert!(coordinate <= MAX_EXACT_F32_COORDINATE);
            values.push(coordinate as f32);
        }

        let device = reference.inner.device();
        let output = WasmTensor {
            inner: Tensor::<WasmBackend, 4>::from_data(TensorData::new(values, shape), &device),
        };

        if output.inner.dims() != shape {
            return Err(format!(
                "IndexSource.indicesLike: internal shape invariant failed; expected {shape:?}, got {:?}",
                output.inner.dims()
            ));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> TensorIndexSource {
        TensorIndexSource::new()
    }

    fn bits(values: Vec<f32>) -> Vec<u32> {
        values.into_iter().map(f32::to_bits).collect()
    }

    #[test]
    fn all_axes_produce_exact_coordinate_grids_and_preserve_shape() {
        let reference = WasmTensor::new(&[9.0; 24], &[2, 3, 2, 2]);

        let axis0 = source().indices_like(&reference, 0).unwrap();
        assert_eq!(axis0.shape(), vec![2, 3, 2, 2]);
        assert_eq!(
            axis0.to_array(),
            [vec![0.0; 12], vec![1.0; 12]].concat()
        );

        let axis1 = source().indices_like(&reference, 1).unwrap();
        assert_eq!(
            axis1.to_array(),
            [
                vec![0.0; 4],
                vec![1.0; 4],
                vec![2.0; 4],
                vec![0.0; 4],
                vec![1.0; 4],
                vec![2.0; 4],
            ]
            .concat()
        );

        let axis2 = source().indices_like(&reference, 2).unwrap();
        assert_eq!(
            axis2.to_array(),
            vec![0.0, 0.0, 1.0, 1.0].repeat(6)
        );

        let axis3 = source().indices_like(&reference, 3).unwrap();
        assert_eq!(axis3.to_array(), vec![0.0, 1.0].repeat(12));
    }

    #[test]
    fn reference_values_do_not_affect_output_bits() {
        let a = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let b = WasmTensor::new(&[-7.0, 99.0, 0.25, -0.0], &[1, 2, 2, 1]);

        let out_a = source().indices_like(&a, 2).unwrap();
        let out_b = source().indices_like(&b, 2).unwrap();
        assert_eq!(out_a.shape(), out_b.shape());
        assert_eq!(bits(out_a.to_array()), bits(out_b.to_array()));
    }

    #[test]
    fn invalid_axis_is_rejected_before_shape_validation() {
        let empty = WasmTensor::new(&[], &[1, 0, 1, 1]);
        let err = source().indices_like(&empty, 4).err().unwrap();
        assert!(err.contains("axis must be in 0..4"));
    }

    #[test]
    fn zero_sized_shape_fails_closed() {
        let empty = WasmTensor::new(&[], &[1, 0, 1, 1]);
        let err = source().indices_like(&empty, 1).err().unwrap();
        assert!(err.contains("zero-sized dimensions"));
    }

    #[test]
    fn coordinate_precision_boundary_is_explicit_and_fail_closed() {
        assert_eq!(MAX_EXACT_F32_COORDINATE, 16_777_216);
        assert_eq!(MAX_INDICES_LIKE_AXIS_LENGTH, 16_777_217);
        assert_eq!(MAX_EXACT_F32_COORDINATE as f32, 16_777_216.0);
        assert!(validate_indices_like_shape(
            [1, MAX_INDICES_LIKE_AXIS_LENGTH, 1, 1],
            1
        )
        .is_ok());
        assert!(validate_indices_like_shape(
            [1, MAX_INDICES_LIKE_AXIS_LENGTH + 1, 1, 1],
            1
        )
        .is_err());
    }

    #[test]
    fn produced_values_are_finite_and_repeated_execution_is_deterministic() {
        let reference = WasmTensor::new(&[3.0; 12], &[2, 3, 2, 1]);
        let first = source().indices_like(&reference, 1).unwrap();
        let second = source().indices_like(&reference, 1).unwrap();

        assert!(first.to_array().iter().all(|value| value.is_finite()));
        assert_eq!(bits(first.to_array()), bits(second.to_array()));
    }

    #[test]
    fn capabilities_publish_precision_and_authority_boundaries() {
        let caps = index_source_capabilities();
        assert!(caps.contains("burn-research.index-source.v1"));
        assert!(caps.contains("\"reference_values_ignored\":true"));
        assert!(caps.contains("\"max_exact_f32_coordinate\":16777216"));
        assert!(caps.contains("\"max_supported_axis_length\":16777217"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
        assert!(caps.contains("\"grants_authority\":false"));
    }
}
