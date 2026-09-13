use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use wasm_bindgen::prelude::*;

use crate::{WasmBackend, WasmTensor};

fn checked_element_count(dims: [usize; 4], context: &str) -> Result<usize, String> {
    dims.into_iter().try_fold(1usize, |count, dim| {
        if dim == 0 {
            return Err(format!("{context}: zero-sized dimensions are not allowed in v1"));
        }
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("{context}: element-count overflow for shape {dims:?}"))
    })
}

fn parse_rank4_shape(shape: &[usize], context: &str) -> Result<[usize; 4], String> {
    if shape.len() != 4 {
        return Err(format!(
            "{context}: expected exactly 4 dimensions, got {}",
            shape.len()
        ));
    }
    let dims = [shape[0], shape[1], shape[2], shape[3]];
    checked_element_count(dims, context)?;
    Ok(dims)
}

fn parse_permutation(axes: &[usize]) -> Result<[usize; 4], String> {
    if axes.len() != 4 {
        return Err(format!(
            "TensorTransform.permute: expected exactly 4 axes, got {}",
            axes.len()
        ));
    }

    let mut seen = [false; 4];
    for (position, &axis) in axes.iter().enumerate() {
        if axis >= 4 {
            return Err(format!(
                "TensorTransform.permute: axis at position {position} is out of range: {axis}"
            ));
        }
        if seen[axis] {
            return Err(format!(
                "TensorTransform.permute: duplicate axis {axis} at position {position}"
            ));
        }
        seen[axis] = true;
    }

    Ok([axes[0], axes[1], axes[2], axes[3]])
}

fn parse_slice_ranges(
    dims: [usize; 4],
    starts: &[usize],
    ends: &[usize],
) -> Result<([usize; 4], [usize; 4]), String> {
    if starts.len() != 4 || ends.len() != 4 {
        return Err(format!(
            "TensorTransform.slice: starts and ends must each contain 4 values, got {} and {}",
            starts.len(),
            ends.len()
        ));
    }

    let starts = [starts[0], starts[1], starts[2], starts[3]];
    let ends = [ends[0], ends[1], ends[2], ends[3]];
    for axis in 0..4 {
        if starts[axis] >= ends[axis] {
            return Err(format!(
                "TensorTransform.slice: axis {axis} requires start < end, got {}..{}",
                starts[axis], ends[axis]
            ));
        }
        if ends[axis] > dims[axis] {
            return Err(format!(
                "TensorTransform.slice: axis {axis} end {} exceeds dimension {}",
                ends[axis], dims[axis]
            ));
        }
    }
    Ok((starts, ends))
}

fn validate_select_indices(dims: [usize; 4], axis: usize, indices: &[usize]) -> Result<(), String> {
    if axis >= 4 {
        return Err(format!(
            "TensorTransform.selectAxis: axis {axis} is out of range for rank 4"
        ));
    }
    if indices.is_empty() {
        return Err("TensorTransform.selectAxis: indices must not be empty in v1".into());
    }
    for (position, &index) in indices.iter().enumerate() {
        if index >= dims[axis] {
            return Err(format!(
                "TensorTransform.selectAxis: index at position {position} is out of bounds: {index} >= {} on axis {axis}",
                dims[axis]
            ));
        }
        if index > i64::MAX as usize {
            return Err(format!(
                "TensorTransform.selectAxis: index at position {position} exceeds i64 range: {index}"
            ));
        }
    }
    Ok(())
}

#[wasm_bindgen(js_name = tensorTransformCapabilities)]
pub fn tensor_transform_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.tensor-transform.v1\",",
        "\"rank\":4,",
        "\"ops\":[\"reshape\",\"transpose\",\"permute\",\"slice\",\"selectAxis\"],",
        "\"contracts\":{",
        "\"reshape\":\"exact_rank4_equal_element_count\",",
        "\"permute\":\"complete_unique_axes\",",
        "\"slice\":\"bounded_nonempty_ranges\",",
        "\"selectAxis\":\"validated_axis_and_indices\"",
        "}",
        "}"
    )
    .to_string()
}

/// Stateless rank-4 tensor/layout transform surface.
///
/// Every operation validates shape/axis/range/index metadata before calling Burn so malformed
/// requests fail as controlled errors rather than backend panics or unchecked indexing behavior.
#[wasm_bindgen]
pub struct WasmTensorTransform;

#[wasm_bindgen]
impl WasmTensorTransform {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmTensorTransform {
        WasmTensorTransform
    }

    pub fn reshape(&self, input: &WasmTensor, shape: &[usize]) -> Result<WasmTensor, String> {
        let target = parse_rank4_shape(shape, "TensorTransform.reshape")?;
        let source = input.inner.dims();
        let source_count = checked_element_count(source, "TensorTransform.reshape source")?;
        let target_count = checked_element_count(target, "TensorTransform.reshape target")?;
        if source_count != target_count {
            return Err(format!(
                "TensorTransform.reshape: element-count mismatch: source {source:?} has {source_count}, target {target:?} has {target_count}"
            ));
        }
        Ok(WasmTensor {
            inner: input.inner.clone().reshape(target),
        })
    }

    pub fn transpose(&self, input: &WasmTensor) -> WasmTensor {
        WasmTensor {
            inner: input.inner.clone().swap_dims(2, 3),
        }
    }

    pub fn permute(&self, input: &WasmTensor, axes: &[usize]) -> Result<WasmTensor, String> {
        let axes = parse_permutation(axes)?;
        Ok(WasmTensor {
            inner: input.inner.clone().permute(axes),
        })
    }

    pub fn slice(
        &self,
        input: &WasmTensor,
        starts: &[usize],
        ends: &[usize],
    ) -> Result<WasmTensor, String> {
        let dims = input.inner.dims();
        let (starts, ends) = parse_slice_ranges(dims, starts, ends)?;
        Ok(WasmTensor {
            inner: input.inner.clone().slice([
                starts[0]..ends[0],
                starts[1]..ends[1],
                starts[2]..ends[2],
                starts[3]..ends[3],
            ]),
        })
    }

    #[wasm_bindgen(js_name = selectAxis)]
    pub fn select_axis(
        &self,
        input: &WasmTensor,
        axis: usize,
        indices: &[usize],
    ) -> Result<WasmTensor, String> {
        let dims = input.inner.dims();
        validate_select_indices(dims, axis, indices)?;

        let device = input.inner.device();
        let index_values: Vec<i64> = indices.iter().map(|&value| value as i64).collect();
        let index_tensor = Tensor::<WasmBackend, 1, Int>::from_data(
            TensorData::new(index_values, [indices.len()]),
            &device,
        );

        Ok(WasmTensor {
            inner: input.inner.clone().select(axis, index_tensor),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{tensor_transform_capabilities, WasmTensorTransform};
    use crate::WasmTensor;

    #[test]
    fn reshape_preserves_values_and_requires_exact_element_count() {
        let ops = WasmTensorTransform::new();
        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]);
        let output = ops.reshape(&input, &[1, 1, 3, 2]).unwrap();
        assert_eq!(output.shape(), vec![1, 1, 3, 2]);
        assert_eq!(output.to_array(), input.to_array());
        assert!(ops.reshape(&input, &[1, 1, 2, 2]).is_err());
        assert!(ops.reshape(&input, &[1, 6, 1]).is_err());
        assert!(ops.reshape(&input, &[1, 0, 2, 3]).is_err());
    }

    #[test]
    fn transpose_and_permute_reorder_axes_deterministically() {
        let ops = WasmTensorTransform::new();
        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]);

        let transposed = ops.transpose(&input);
        assert_eq!(transposed.shape(), vec![1, 2, 3, 1]);
        assert_eq!(transposed.to_array(), input.to_array());

        let permuted = ops.permute(&input, &[0, 2, 3, 1]).unwrap();
        assert_eq!(permuted.shape(), vec![1, 1, 3, 2]);
        assert_eq!(permuted.to_array(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(ops.permute(&input, &[0, 1, 1, 3]).is_err());
        assert!(ops.permute(&input, &[0, 1, 2, 4]).is_err());
    }

    #[test]
    fn slice_validates_nonempty_in_bounds_ranges() {
        let ops = WasmTensorTransform::new();
        let values: Vec<f32> = (1..=12).map(|value| value as f32).collect();
        let input = WasmTensor::new(&values, &[1, 2, 2, 3]);
        let output = ops
            .slice(&input, &[0, 0, 0, 1], &[1, 2, 2, 3])
            .unwrap();
        assert_eq!(output.shape(), vec![1, 2, 2, 2]);
        assert_eq!(output.to_array(), vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]);
        assert!(ops.slice(&input, &[0, 0, 0, 2], &[1, 2, 2, 2]).is_err());
        assert!(ops.slice(&input, &[0, 0, 0, 0], &[1, 2, 2, 4]).is_err());
    }

    #[test]
    fn select_axis_validates_indices_before_backend_execution() {
        let ops = WasmTensorTransform::new();
        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]);
        let output = ops.select_axis(&input, 1, &[1, 0]).unwrap();
        assert_eq!(output.shape(), vec![1, 2, 1, 3]);
        assert_eq!(output.to_array(), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert!(ops.select_axis(&input, 4, &[0]).is_err());
        assert!(ops.select_axis(&input, 1, &[]).is_err());
        assert!(ops.select_axis(&input, 1, &[2]).is_err());
    }

    #[test]
    fn capabilities_are_machine_discoverable() {
        let caps = tensor_transform_capabilities();
        assert!(caps.contains("burn-research.tensor-transform.v1"));
        assert!(caps.contains("\"rank\":4"));
        assert!(caps.contains("\"selectAxis\""));
    }
}
