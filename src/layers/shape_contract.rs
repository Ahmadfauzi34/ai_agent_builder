pub(crate) fn require_singleton_axis(
    shape: [usize; 4],
    axis: usize,
    context: &str,
) -> Result<(), String> {
    if axis >= shape.len() {
        return Err(format!("{context}: invalid shape axis {axis}"));
    }
    if shape[axis] != 1 {
        return Err(format!(
            "{context}: expected axis {axis} to be 1, got {} for shape {:?}",
            shape[axis], shape
        ));
    }
    Ok(())
}

pub(crate) fn require_singleton_spatial(
    shape: [usize; 4],
    context: &str,
) -> Result<(), String> {
    require_singleton_axis(shape, 2, context)?;
    require_singleton_axis(shape, 3, context)
}

pub(crate) fn require_axis_size(
    shape: [usize; 4],
    axis: usize,
    expected: usize,
    context: &str,
) -> Result<(), String> {
    if axis >= shape.len() {
        return Err(format!("{context}: invalid shape axis {axis}"));
    }
    if shape[axis] != expected {
        return Err(format!(
            "{context}: expected axis {axis} size {expected}, got {} for shape {:?}",
            shape[axis], shape
        ));
    }
    Ok(())
}

pub(crate) fn forward_fail<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_spatial_accepts_canonical_linear_shape() {
        assert!(require_singleton_spatial([2, 8, 1, 1], "linear").is_ok());
    }

    #[test]
    fn singleton_spatial_rejects_hidden_height_elements() {
        let err = require_singleton_spatial([2, 8, 3, 1], "linear").unwrap_err();
        assert!(err.contains("axis 2"));
    }

    #[test]
    fn singleton_width_rejects_hidden_width_elements() {
        let err = require_singleton_axis([2, 4, 6, 3], 3, "conv1d").unwrap_err();
        assert!(err.contains("axis 3"));
    }

    #[test]
    fn axis_size_rejects_linear_feature_mismatch() {
        let err = require_axis_size([4, 7, 1, 1], 1, 8, "linear").unwrap_err();
        assert!(err.contains("size 8"));
    }

    #[test]
    fn axis_size_accepts_expected_dimension() {
        assert!(require_axis_size([4, 8, 1, 1], 1, 8, "linear").is_ok());
    }
}
