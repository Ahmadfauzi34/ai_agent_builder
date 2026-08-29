#[cfg(test)]
mod tests {
    use crate::{normalize_4d_shape, tensor_view_byte_len};

    #[test]
    fn shape_is_padded_to_four_dimensions() {
        assert_eq!(normalize_4d_shape(&[2, 3], 6).unwrap(), [2, 3, 1, 1]);
    }

    #[test]
    fn shape_data_mismatch_is_rejected_before_backend() {
        assert!(normalize_4d_shape(&[2, 3], 5).is_err());
    }

    #[test]
    fn rank_above_four_is_rejected_instead_of_truncated() {
        assert!(normalize_4d_shape(&[1, 1, 1, 1, 1], 1).is_err());
    }

    #[test]
    fn shape_element_overflow_is_rejected() {
        assert!(normalize_4d_shape(&[usize::MAX, 2], 0).is_err());
    }

    #[test]
    fn zero_sized_layout_remains_representable() {
        assert_eq!(normalize_4d_shape(&[0, 4], 0).unwrap(), [0, 4, 1, 1]);
    }

    #[test]
    fn tensor_view_byte_length_is_checked() {
        assert_eq!(tensor_view_byte_len(8).unwrap(), 32);
        if usize::BITS > 32 {
            let too_many = (u32::MAX as usize / 4) + 1;
            assert!(tensor_view_byte_len(too_many).is_err());
        }
    }
}
