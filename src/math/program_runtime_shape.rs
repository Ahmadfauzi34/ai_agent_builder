//! Explicit runtime shape operations for Math Program plans.
//!
//! `expandLike` is intentionally visible and fail-closed. It never changes Numeric Kernel binary
//! semantics: add/sub/mul/div still require exact shape equality and never broadcast implicitly.

use crate::WasmTensor;

pub(crate) fn validate_expand_like_shape(
    source: [usize; 4],
    reference: [usize; 4],
) -> Result<(), String> {
    for axis in 0..4 {
        let src = source[axis];
        let target = reference[axis];
        if src == 0 || target == 0 {
            return Err(format!(
                "MathProgram.expandLike: zero-sized dimensions are not allowed in v1; axis {axis} has source={src}, reference={target}"
            ));
        }
        if src != target && src != 1 {
            return Err(format!(
                "MathProgram.expandLike: axis {axis} cannot expand source dimension {src} to reference dimension {target}; source must match target or be singleton"
            ));
        }
    }
    Ok(())
}

pub(crate) fn expand_like(
    source: &WasmTensor,
    reference: &WasmTensor,
) -> Result<WasmTensor, String> {
    let source_dims = source.inner.dims();
    let reference_dims = reference.inner.dims();
    validate_expand_like_shape(source_dims, reference_dims)?;

    if source_dims == reference_dims {
        return Ok(source.clone());
    }

    let output = WasmTensor {
        inner: source.inner.clone().expand(reference_dims),
    };
    if output.inner.dims() != reference_dims {
        return Err(format!(
            "MathProgram.expandLike: internal shape invariant failed; expected {reference_dims:?}, got {:?}",
            output.inner.dims()
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_singleton_feature_axis_per_batch() {
        let source = WasmTensor::new(&[10.0, 20.0], &[2, 1, 1, 1]);
        let reference = WasmTensor::new(&[0.0; 6], &[2, 3, 1, 1]);
        let output = expand_like(&source, &reference).unwrap();
        assert_eq!(output.shape(), vec![2, 3, 1, 1]);
        assert_eq!(
            output.to_array(),
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );
    }

    #[test]
    fn equal_shape_is_semantic_identity_copy() {
        let source = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        let reference = WasmTensor::new(&[9.0, 9.0, 9.0, 9.0], &[1, 2, 2, 1]);
        let output = expand_like(&source, &reference).unwrap();
        assert_eq!(output.shape(), source.shape());
        assert_eq!(output.to_array(), source.to_array());
    }

    #[test]
    fn scalar_like_source_can_expand_on_multiple_axes() {
        let source = WasmTensor::new(&[7.0], &[1, 1, 1, 1]);
        let reference = WasmTensor::new(&[0.0; 8], &[2, 2, 2, 1]);
        let output = expand_like(&source, &reference).unwrap();
        assert_eq!(output.shape(), vec![2, 2, 2, 1]);
        assert_eq!(output.to_array(), vec![7.0; 8]);
    }

    #[test]
    fn rejects_non_singleton_mismatch_shrink_and_zero_dimensions() {
        assert!(validate_expand_like_shape([2, 2, 1, 1], [2, 3, 1, 1]).is_err());
        assert!(validate_expand_like_shape([2, 3, 1, 1], [2, 1, 1, 1]).is_err());
        assert!(validate_expand_like_shape([1, 0, 1, 1], [1, 3, 1, 1]).is_err());
        assert!(validate_expand_like_shape([1, 1, 1, 1], [1, 0, 1, 1]).is_err());
    }
}
