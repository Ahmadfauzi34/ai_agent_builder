#[cfg(test)]
mod tests {
    use crate::protocol::{PacketHeader, PayloadCursor};
    use crate::registry::LayerRegistry;

    #[test]
    fn test_payload_cursor_basic() {
        // Construct a sample payload:
        // u32 = 42
        // f64 = 3.14
        // bool = true (1)
        // Option<u32> = Some(100) -> 1, 100
        // Option<f64> = None -> 0, 0.0
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&3.14f64.to_le_bytes());
        data.push(1); // true
        data.push(1); // Some
        data.extend_from_slice(&100u32.to_le_bytes());
        data.push(0); // None
        data.extend_from_slice(&0.0f64.to_le_bytes());

        let mut cursor = PayloadCursor::new(&data);
        assert_eq!(cursor.read_u32().unwrap(), 42);
        assert_eq!(cursor.read_f64().unwrap(), 3.14);
        assert!(cursor.read_bool().unwrap());
        assert_eq!(cursor.read_option_u32().unwrap(), Some(100));
        assert_eq!(cursor.read_option_f64().unwrap(), None);
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn test_packet_header_validate() {
        let mut header_bytes = [0u8; 8];
        header_bytes[0] = 0x01; // opcode
        header_bytes[1] = 0x02; // layer_type
        header_bytes[2] = 0x03; // variant
        header_bytes[3] = 0x04; // flags
        let payload_len: u32 = 12;
        header_bytes[4..8].copy_from_slice(&payload_len.to_le_bytes());

        let header = PacketHeader::from_bytes(&header_bytes).unwrap();
        assert_eq!(header.opcode, 0x01);
        assert_eq!(header.layer_type, 0x02);
        assert_eq!(header.variant, 0x03);
        assert_eq!(header.flags, 0x04);
        assert_eq!(header.payload_len, 12);

        let valid_payload = vec![0u8; 12];
        let validated = header.validate_payload(&valid_payload).unwrap();
        assert_eq!(validated.len(), 12);

        let short_payload = vec![0u8; 10];
        assert!(header.validate_payload(&short_payload).is_err());
    }

    #[test]
    fn test_layer_registry_param_cache() {
        let mut registry = LayerRegistry::new();
        assert_eq!(registry.total_params(), 0);

        // We can check parameter calculations on initialization and caching.
        // Let's create a PacketHeader for Linear layer
        // Linear payload format: id (u32), in_dim (usize/u32), out_dim (usize/u32), bias (bool/u8)
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes()); // id = 1
        payload.extend_from_slice(&10u32.to_le_bytes()); // in_dim = 10
        payload.extend_from_slice(&5u32.to_le_bytes());  // out_dim = 5
        payload.push(1); // bias = true

        let mut header_bytes = [0u8; 8];
        header_bytes[0] = crate::protocol::OP_INIT;
        header_bytes[1] = crate::protocol::LAYER_LINEAR;
        header_bytes[2] = 0xFF; // variant
        header_bytes[3] = 0; // flags
        let payload_len = payload.len() as u32;
        header_bytes[4..8].copy_from_slice(&payload_len.to_le_bytes());

        let header = PacketHeader::from_bytes(&header_bytes).unwrap();
        registry.init_layer(&header, &payload).unwrap();

        // 10 * 5 (weights) + 5 (bias) = 55 parameters
        let params = registry.total_params();
        assert_eq!(params, 55);

        // Delete the layer
        let destroyed = registry.destroy_layer(1, crate::protocol::LAYER_LINEAR);
        assert!(destroyed);
        assert_eq!(registry.total_params(), 0);
    }
}
