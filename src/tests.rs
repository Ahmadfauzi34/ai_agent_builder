#[cfg(test)]
mod tests {
    use crate::protocol::{
        PacketHeader, PayloadCursor, ACT_RELU, LAYER_ACTIVATION, LAYER_LINEAR, OP_INIT,
        VARIANT_NONE,
    };
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn header(opcode: u8, layer_type: u8, variant: u8, flags: u8, payload_len: u32) -> [u8; 8] {
        let mut h = [0u8; 8];
        h[0] = opcode;
        h[1] = layer_type;
        h[2] = variant;
        h[3] = flags;
        h[4..8].copy_from_slice(&payload_len.to_le_bytes());
        h
    }

    fn init_header(layer_type: u8, variant: u8, payload_len: u32) -> PacketHeader {
        PacketHeader::from_bytes(&header(OP_INIT, layer_type, variant, 0, payload_len)).unwrap()
    }

    #[test]
    fn test_payload_cursor_basic() {
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
        header_bytes[0] = 0x01;
        header_bytes[1] = 0x02;
        header_bytes[2] = 0x03;
        header_bytes[3] = 0x04;
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

        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes()); // id = 1
        payload.extend_from_slice(&10u32.to_le_bytes()); // in_dim = 10
        payload.extend_from_slice(&5u32.to_le_bytes()); // out_dim = 5
        payload.push(1); // bias = true

        let h = init_header(LAYER_LINEAR, VARIANT_NONE, payload.len() as u32);
        registry.init_layer(&h, &payload).unwrap();

        // 10 * 5 (weights) + 5 (bias) = 55 parameters
        assert_eq!(registry.total_params(), 55);

        let destroyed = registry.destroy_layer(1, LAYER_LINEAR);
        assert!(destroyed);
        assert_eq!(registry.total_params(), 0);
    }

    // ------------------------------------------------------------
    // GRAPH EXECUTOR TESTS
    // ------------------------------------------------------------
    fn build_linear_relu_registry() -> (LayerRegistry, WasmTensor) {
        let mut reg = LayerRegistry::new();

        // linear id=1 : in=3, out=4, bias=true
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(&4u32.to_le_bytes());
        p.push(1);
        reg.init_layer(&init_header(LAYER_LINEAR, VARIANT_NONE, p.len() as u32), &p)
            .unwrap();

        // activation id=2 : relu
        let mut p2 = Vec::new();
        p2.extend_from_slice(&2u32.to_le_bytes());
        reg.init_layer(&init_header(LAYER_ACTIVATION, ACT_RELU, p2.len() as u32), &p2)
            .unwrap();

        let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        (reg, input)
    }

    fn push_step(plan: &mut Vec<u8>, layer_type: u8, layer_id: u32, in_slot: u8, out_slot: u8) {
        plan.push(layer_type);
        plan.extend_from_slice(&layer_id.to_le_bytes());
        plan.push(in_slot);
        plan.push(out_slot);
    }

    #[test]
    fn test_run_graph_matches_manual_chain() {
        let (reg, input) = build_linear_relu_registry();

        // plan: slot0=input -> linear -> slot1 -> relu -> slot2 ; return slot2
        let mut plan = Vec::new();
        plan.extend_from_slice(&2u32.to_le_bytes()); // num_steps
        plan.extend_from_slice(&3u32.to_le_bytes()); // num_slots
        push_step(&mut plan, LAYER_LINEAR, 1, 0, 1);
        push_step(&mut plan, LAYER_ACTIVATION, 2, 1, 2);
        plan.push(2); // out_slot

        let run = reg.run_graph(&plan, &input).unwrap();

        // jalur manual lewat forward_layer (registry sama => bobot sama)
        let t1 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap();
        let t2 = reg.forward_layer(2, LAYER_ACTIVATION, &t1).unwrap();

        assert_eq!(run.to_array(), t2.to_array());
    }

    #[test]
    fn test_run_graph_rejects_empty_input_slot() {
        let (reg, input) = build_linear_relu_registry();

        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&3u32.to_le_bytes());
        push_step(&mut plan, LAYER_LINEAR, 1, 2, 1); // in_slot=2 belum terisi
        plan.push(1);

        assert!(reg.run_graph(&plan, &input).is_err());
    }

    #[test]
    fn test_run_graph_rejects_unknown_layer() {
        let (reg, input) = build_linear_relu_registry();

        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&2u32.to_le_bytes());
        push_step(&mut plan, LAYER_LINEAR, 999, 0, 1); // id tidak ada
        plan.push(1);

        assert!(reg.run_graph(&plan, &input).is_err());
    }

    // ------------------------------------------------------------
    // COMPILED GRAPH TESTS
    // ------------------------------------------------------------
    #[test]
    fn test_compiled_graph_matches_manual_chain() {
        let (reg, input) = build_linear_relu_registry();

        // plan: slot0=input -> linear -> slot1 -> relu -> slot2 ; return slot2
        let mut plan = Vec::new();
        plan.extend_from_slice(&2u32.to_le_bytes()); // num_steps
        plan.extend_from_slice(&3u32.to_le_bytes()); // num_slots
        push_step(&mut plan, LAYER_LINEAR, 1, 0, 1);
        push_step(&mut plan, LAYER_ACTIVATION, 2, 1, 2);
        plan.push(2); // out_slot

        let compiled = reg.compile_graph(&plan).unwrap();
        assert_eq!(compiled.step_count(), 2);
        assert_eq!(compiled.slot_count(), 3);
        assert_eq!(compiled.output_slot(), 2);

        let run = compiled.run(&reg, &input).unwrap();

        // jalur manual lewat forward_layer
        let t1 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap();
        let t2 = reg.forward_layer(2, LAYER_ACTIVATION, &t1).unwrap();

        assert_eq!(run.to_array(), t2.to_array());
    }

    #[test]
    fn test_compiled_graph_rejects_empty_input_slot() {
        let (reg, _input) = build_linear_relu_registry();

        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&3u32.to_le_bytes());
        push_step(&mut plan, LAYER_LINEAR, 1, 2, 1); // in_slot=2 belum terisi
        plan.push(1);

        assert!(reg.compile_graph(&plan).is_err());
    }
}
