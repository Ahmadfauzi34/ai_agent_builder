#[cfg(test)]
mod tests {
    use crate::protocol::{
        read_f64, read_option_f64, read_option_u32, read_u32, PayloadCursor,
    };

    #[test]
    fn read_u32_rejects_offset_overflow_without_panicking() {
        assert!(read_u32(&[], usize::MAX).is_err());
    }

    #[test]
    fn read_f64_rejects_offset_overflow_without_panicking() {
        assert!(read_f64(&[], usize::MAX).is_err());
    }

    #[test]
    fn free_option_u32_uses_same_fixed_width_as_cursor() {
        assert!(read_option_u32(&[0], 0).is_err());

        let bytes = [0, 0, 0, 0, 0];
        assert_eq!(read_option_u32(&bytes, 0).unwrap(), None);

        let mut cursor = PayloadCursor::new(&bytes);
        assert_eq!(cursor.read_option_u32().unwrap(), None);
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn free_option_f64_uses_same_fixed_width_as_cursor() {
        assert!(read_option_f64(&[0], 0).is_err());

        let bytes = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(read_option_f64(&bytes, 0).unwrap(), None);

        let mut cursor = PayloadCursor::new(&bytes);
        assert_eq!(cursor.read_option_f64().unwrap(), None);
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn free_option_u32_decodes_present_value() {
        let mut bytes = vec![1];
        bytes.extend_from_slice(&42u32.to_le_bytes());
        assert_eq!(read_option_u32(&bytes, 0).unwrap(), Some(42));
    }

    #[test]
    fn free_option_f64_decodes_present_value() {
        let mut bytes = vec![1];
        bytes.extend_from_slice(&3.5f64.to_le_bytes());
        assert_eq!(read_option_f64(&bytes, 0).unwrap(), Some(3.5));
    }
}
