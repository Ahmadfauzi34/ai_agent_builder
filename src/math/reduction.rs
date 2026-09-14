use burn::prelude::*;

use crate::WasmTensor;

fn validate_input(input: &WasmTensor, context: &str) -> Result<[usize; 4], String> {
    let shape = input.inner.dims();
    for (axis, dim) in shape.iter().copied().enumerate() {
        if dim == 0 {
            return Err(format!(
                "{context}: zero-sized dimension at axis {axis} is not supported in reduction v1"
            ));
        }
    }
    for (index, value) in input.to_array().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite value at index {index}: {value}"
            ));
        }
    }
    Ok(shape)
}

fn checked_axis(axis: u32, context: &str) -> Result<usize, String> {
    if axis >= 4 {
        return Err(format!("{context}: axis must be in 0..4, got {axis}"));
    }
    Ok(axis as usize)
}

fn checked_output(
    inner: Tensor<crate::WasmBackend, 4>,
    input_shape: [usize; 4],
    axis: usize,
    context: &str,
) -> Result<WasmTensor, String> {
    let output = WasmTensor { inner };
    let mut expected = input_shape;
    expected[axis] = 1;
    let actual = output.inner.dims();
    if actual != expected {
        return Err(format!(
            "{context}: keepdim shape invariant failed, expected {expected:?}, got {actual:?}"
        ));
    }
    for (index, value) in output.to_array().into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{context}: non-finite output at index {index}: {value}"
            ));
        }
    }
    Ok(output)
}

pub fn reduction_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.reduction.v1\",",
        "\"backend\":\"Burn Tensor<WasmBackend,4>\",",
        "\"rank\":4,",
        "\"axis\":\"runtime_0_to_3\",",
        "\"keepdim\":true,",
        "\"reducers\":[\"sumAxis\",\"meanAxis\",\"minAxis\",\"maxAxis\"],",
        "\"contracts\":{",
        "\"finite_inputs\":true,",
        "\"finite_outputs\":true,",
        "\"zero_sized_dimensions\":false,",
        "\"statistics_v1_independent\":true,",
        "\"implicit_broadcasting\":false",
        "}",
        "}"
    )
    .to_string()
}

/// Generic stateless rank-4 reduction primitive.
///
/// Unlike `Statistics v1`, this surface carries no feature-vector interpretation. The caller
/// supplies an explicit runtime axis and the result always retains rank 4 with the reduced
/// dimension set to 1. Variance/std are intentionally not part of reduction v1.
#[derive(Clone, Copy, Debug, Default)]
pub struct TensorReduction;

impl TensorReduction {
    pub fn new() -> Self {
        Self
    }

    pub fn sum_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        let shape = validate_input(input, "Reduction.sumAxis input")?;
        let axis = checked_axis(axis, "Reduction.sumAxis")?;
        checked_output(
            input.inner.clone().sum_dim(axis),
            shape,
            axis,
            "Reduction.sumAxis output",
        )
    }

    pub fn mean_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        let shape = validate_input(input, "Reduction.meanAxis input")?;
        let axis = checked_axis(axis, "Reduction.meanAxis")?;
        checked_output(
            input.inner.clone().mean_dim(axis),
            shape,
            axis,
            "Reduction.meanAxis output",
        )
    }

    pub fn min_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        let shape = validate_input(input, "Reduction.minAxis input")?;
        let axis = checked_axis(axis, "Reduction.minAxis")?;
        checked_output(
            input.inner.clone().min_dim(axis),
            shape,
            axis,
            "Reduction.minAxis output",
        )
    }

    pub fn max_axis(&self, input: &WasmTensor, axis: u32) -> Result<WasmTensor, String> {
        let shape = validate_input(input, "Reduction.maxAxis input")?;
        let axis = checked_axis(axis, "Reduction.maxAxis")?;
        checked_output(
            input.inner.clone().max_dim(axis),
            shape,
            axis,
            "Reduction.maxAxis output",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{reduction_capabilities, TensorReduction};
    use crate::math::numeric::WasmNumericKernel;
    use crate::math::program_runtime_shape::expand_like;
    use crate::WasmTensor;

    fn tensor(values: &[f32], shape: &[usize]) -> WasmTensor {
        WasmTensor::new(values, shape)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected} within {tolerance}"
            );
        }
    }

    #[test]
    fn reducers_support_all_axes_with_rank4_keepdim_semantics() {
        let reduction = TensorReduction::new();
        let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2, 1]);

        let sum0 = reduction.sum_axis(&input, 0).unwrap();
        assert_eq!(sum0.shape(), vec![1, 2, 2, 1]);
        assert_eq!(sum0.to_array(), vec![6.0, 8.0, 10.0, 12.0]);

        let sum1 = reduction.sum_axis(&input, 1).unwrap();
        assert_eq!(sum1.shape(), vec![2, 1, 2, 1]);
        assert_eq!(sum1.to_array(), vec![4.0, 6.0, 12.0, 14.0]);

        let sum2 = reduction.sum_axis(&input, 2).unwrap();
        assert_eq!(sum2.shape(), vec![2, 2, 1, 1]);
        assert_eq!(sum2.to_array(), vec![3.0, 7.0, 11.0, 15.0]);

        let sum3 = reduction.sum_axis(&input, 3).unwrap();
        assert_eq!(sum3.shape(), vec![2, 2, 2, 1]);
        assert_eq!(sum3.to_array(), input.to_array());

        let mean0 = reduction.mean_axis(&input, 0).unwrap();
        assert_close(&mean0.to_array(), &[3.0, 4.0, 5.0, 6.0], 1e-6);

        let min1 = reduction.min_axis(&input, 1).unwrap();
        assert_eq!(min1.to_array(), vec![1.0, 2.0, 5.0, 6.0]);

        let max2 = reduction.max_axis(&input, 2).unwrap();
        assert_eq!(max2.to_array(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn invalid_axis_zero_shape_and_nonfinite_input_fail_closed() {
        let reduction = TensorReduction::new();
        let input = tensor(&[1.0, 2.0], &[1, 2, 1, 1]);
        assert!(reduction.sum_axis(&input, 4).is_err());

        let empty = tensor(&[], &[1, 0, 1, 1]);
        assert!(reduction.sum_axis(&empty, 1).is_err());
        assert!(reduction.min_axis(&empty, 1).is_err());

        let nonfinite = tensor(&[1.0, f32::INFINITY], &[1, 2, 1, 1]);
        assert!(reduction.mean_axis(&nonfinite, 1).is_err());
        assert!(reduction.max_axis(&nonfinite, 1).is_err());
    }

    #[test]
    fn stable_softmax_composes_on_non_feature_axis() {
        let reduction = TensorReduction::new();
        let kernel = WasmNumericKernel::new();
        let input = tensor(&[1000.0, 1001.0, 1002.0, 1.0, 2.0, 3.0], &[1, 2, 3, 1]);

        let max = reduction.max_axis(&input, 2).unwrap();
        assert_eq!(max.shape(), vec![1, 2, 1, 1]);
        assert_eq!(max.to_array(), vec![1002.0, 3.0]);
        let max_full = expand_like(&max, &input).unwrap();
        let shifted = kernel.sub(&input, &max_full).unwrap();
        let exp = kernel.exp(&shifted).unwrap();
        let den = reduction.sum_axis(&exp, 2).unwrap();
        let den_full = expand_like(&den, &exp).unwrap();
        let output = kernel.div(&exp, &den_full).unwrap();

        let values = output.to_array();
        assert!((values[0..3].iter().sum::<f32>() - 1.0).abs() <= 2e-6);
        assert!((values[3..6].iter().sum::<f32>() - 1.0).abs() <= 2e-6);
        assert!(values.iter().all(|value| value.is_finite()));
        assert_close(&values[0..3], &values[3..6], 2e-6);
    }

    #[test]
    fn capabilities_keep_generic_reduction_separate_from_statistics() {
        let caps = reduction_capabilities();
        assert!(caps.contains("burn-research.reduction.v1"));
        assert!(caps.contains("\"axis\":\"runtime_0_to_3\""));
        assert!(caps.contains("\"keepdim\":true"));
        assert!(caps.contains("\"statistics_v1_independent\":true"));
        assert!(caps.contains("\"implicit_broadcasting\":false"));
    }
}
