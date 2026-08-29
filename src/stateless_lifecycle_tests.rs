#[cfg(test)]
mod tests {
    use crate::protocol::{
        PacketHeader, BINARY_ADD, LAYER_BINARY, LAYER_POOL, LAYER_SHIFT, OP_INIT,
        POOL_MAXPOOL1D, SHIFT_UP,
    };
    use crate::registry::LayerRegistry;

    fn header(layer_type: u8, variant: u8, payload_len: usize) -> PacketHeader {
        let mut bytes = [0u8; 8];
        bytes[0] = OP_INIT;
        bytes[1] = layer_type;
        bytes[2] = variant;
        bytes[4..8].copy_from_slice(&(payload_len as u32).to_le_bytes());
        PacketHeader::from_bytes(&bytes).unwrap()
    }

    fn assert_missing_state(reg: &mut LayerRegistry, id: u32, layer_type: u8) {
        assert!(reg.get_layer_state(id, layer_type).is_err());
        assert!(reg.load_layer_state(id, layer_type, &[9, 8, 7]).is_err());
    }

    fn assert_present_stateless_state(reg: &mut LayerRegistry, id: u32, layer_type: u8) {
        assert_eq!(reg.get_layer_state(id, layer_type).unwrap(), Vec::<u8>::new());
        // Preserve the existing stateless contract: state bytes are ignored.
        assert!(reg.load_layer_state(id, layer_type, &[9, 8, 7]).is_ok());
    }

    #[test]
    fn pool_state_api_tracks_pool_lifecycle() {
        let id = 41u32;
        let mut reg = LayerRegistry::new();
        assert_missing_state(&mut reg, id, LAYER_POOL);

        let mut payload = Vec::new();
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(&2u32.to_le_bytes()); // kernel
        payload.push(0); // stride None, fixed-width option
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.push(0); // padding None, fixed-width option
        payload.extend_from_slice(&0u32.to_le_bytes());
        reg.init_layer(&header(LAYER_POOL, POOL_MAXPOOL1D, payload.len()), &payload)
            .unwrap();

        assert!(reg.layer_exists(LAYER_POOL, id));
        assert_present_stateless_state(&mut reg, id, LAYER_POOL);
        assert!(reg.destroy_layer(id, LAYER_POOL));
        assert_missing_state(&mut reg, id, LAYER_POOL);
    }

    #[test]
    fn shift_state_api_tracks_shift_lifecycle() {
        let id = 42u32;
        let mut reg = LayerRegistry::new();
        assert_missing_state(&mut reg, id, LAYER_SHIFT);

        let mut payload = Vec::new();
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        reg.init_layer(&header(LAYER_SHIFT, SHIFT_UP, payload.len()), &payload)
            .unwrap();

        assert!(reg.layer_exists(LAYER_SHIFT, id));
        assert_present_stateless_state(&mut reg, id, LAYER_SHIFT);
        assert!(reg.destroy_layer(id, LAYER_SHIFT));
        assert_missing_state(&mut reg, id, LAYER_SHIFT);
    }

    #[test]
    fn binary_state_api_tracks_binary_lifecycle() {
        let id = 43u32;
        let mut reg = LayerRegistry::new();
        assert_missing_state(&mut reg, id, LAYER_BINARY);

        let mut payload = Vec::new();
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes()); // concat dim unused by ADD
        reg.init_layer(&header(LAYER_BINARY, BINARY_ADD, payload.len()), &payload)
            .unwrap();

        assert!(reg.layer_exists(LAYER_BINARY, id));
        assert_present_stateless_state(&mut reg, id, LAYER_BINARY);
        assert!(reg.destroy_layer(id, LAYER_BINARY));
        assert_missing_state(&mut reg, id, LAYER_BINARY);
    }
}
