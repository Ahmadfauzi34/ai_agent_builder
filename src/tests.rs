#[cfg(test)]
mod tests {
    use crate::protocol::{
        PacketHeader, PayloadCursor, ACT_RELU, BINARY_ADD, LAYER_ACTIVATION, LAYER_BINARY,
        LAYER_LINEAR, OP_INIT, VARIANT_NONE,
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

    // --- plan builder v2 (9 byte/step) ---
    fn push_unary(
        plan: &mut Vec<u8>,
        layer_type: u8,
        layer_id: u32,
        in_slot: u8,
        out_slot: u8,
    ) {
        plan.push(1); // arity = unary
        plan.push(layer_type);
        plan.extend_from_slice(&layer_id.to_le_bytes());
        plan.push(in_slot);
        plan.push(0); // in_slot2 = don't-care
        plan.push(out_slot);
    }

    fn push_binary(
        plan: &mut Vec<u8>,
        layer_type: u8,
        layer_id: u32,
        in_slot: u8,
        in_slot2: u8,
        out_slot: u8,
    ) {
        plan.push(2); // arity = binary
        plan.push(layer_type);
        plan.extend_from_slice(&layer_id.to_le_bytes());
        plan.push(in_slot);
        plan.push(in_slot2);
        plan.push(out_slot);
    }

    // ------------------------------------------------------------
    // NON-GRAPH TESTS (direkonstruksi utuh)
    // ------------------------------------------------------------
    #[test]
    fn test_payload_cursor_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&3.14f64.to_le_bytes());
        data.push(1);
        data.push(1);
        data.extend_from_slice(&100u32.to_le_bytes());
        data.push(0);
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
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&10u32.to_le_bytes());
        payload.extend_from_slice(&5u32.to_le_bytes());
        payload.push(1);

        let h = init_header(LAYER_LINEAR, VARIANT_NONE, payload.len() as u32);
        registry.init_layer(&h, &payload).unwrap();
        assert_eq!(registry.total_params(), 55);

        let destroyed = registry.destroy_layer(1, LAYER_LINEAR);
        assert!(destroyed);
        assert_eq!(registry.total_params(), 0);
    }

    // ------------------------------------------------------------
    // GRAPH TESTS — unary (di-adaptasi ke format 9-byte)
    // ------------------------------------------------------------
    fn build_linear_relu_registry() -> (LayerRegistry, WasmTensor) {
        let mut reg = LayerRegistry::new();

        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(&4u32.to_le_bytes());
        p.push(1);
        reg.init_layer(&init_header(LAYER_LINEAR, VARIANT_NONE, p.len() as u32), &p)
            .unwrap();

        let mut p2 = Vec::new();
        p2.extend_from_slice(&2u32.to_le_bytes());
        reg.init_layer(&init_header(LAYER_ACTIVATION, ACT_RELU, p2.len() as u32), &p2)
            .unwrap();

        let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        (reg, input)
    }

    #[test]
    fn test_run_graph_matches_manual_chain() {
        let (reg, input) = build_linear_relu_registry();

        let mut plan = Vec::new();
        plan.extend_from_slice(&2u32.to_le_bytes());
        plan.extend_from_slice(&3u32.to_le_bytes());
        push_unary(&mut plan, LAYER_LINEAR, 1, 0, 1);
        push_unary(&mut plan, LAYER_ACTIVATION, 2, 1, 2);
        plan.push(2);

        let run = reg.run_graph(&plan, &input).unwrap();
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
        push_unary(&mut plan, LAYER_LINEAR, 1, 2, 1); // in_slot 2 kosong
        plan.push(1);
        assert!(reg.run_graph(&plan, &input).is_err());
    }

    #[test]
    fn test_run_graph_rejects_unknown_layer() {
        let (reg, input) = build_linear_relu_registry();

        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&2u32.to_le_bytes());
        push_unary(&mut plan, LAYER_LINEAR, 999, 0, 1); // id tidak ada
        plan.push(1);
        assert!(reg.run_graph(&plan, &input).is_err());
    }

    // ------------------------------------------------------------
    // GRAPH TESTS — binary (BARU): dua cabang linear di-add
    // ------------------------------------------------------------
    fn build_binary_registry() -> (LayerRegistry, WasmTensor) {
        let mut reg = LayerRegistry::new();

        // linear id=1 (in3,out4,bias)
        let mut p1 = Vec::new();
        p1.extend_from_slice(&1u32.to_le_bytes());
        p1.extend_from_slice(&3u32.to_le_bytes());
        p1.extend_from_slice(&4u32.to_le_bytes());
        p1.push(1);
        reg.init_layer(&init_header(LAYER_LINEAR, VARIANT_NONE, p1.len() as u32), &p1)
            .unwrap();

        // linear id=2 (in3,out4,bias)
        let mut p2 = Vec::new();
        p2.extend_from_slice(&2u32.to_le_bytes());
        p2.extend_from_slice(&3u32.to_le_bytes());
        p2.extend_from_slice(&4u32.to_le_bytes());
        p2.push(1);
        reg.init_layer(&init_header(LAYER_LINEAR, VARIANT_NONE, p2.len() as u32), &p2)
            .unwrap();

        // binary add id=3 (dim don't-care = 0)
        let mut pb = Vec::new();
        pb.extend_from_slice(&3u32.to_le_bytes());
        pb.extend_from_slice(&0u32.to_le_bytes());
        reg.init_layer(&init_header(LAYER_BINARY, BINARY_ADD, pb.len() as u32), &pb)
            .unwrap();

        let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        (reg, input)
    }

    fn binary_plan() -> Vec<u8> {
        // slot0=input -> linear1 -> slot1 ; slot0 -> linear2 -> slot2 ;
        // add(slot1,slot2) -> slot3 ; return slot3
        let mut plan = Vec::new();
        plan.extend_from_slice(&3u32.to_le_bytes());
        plan.extend_from_slice(&4u32.to_le_bytes());
        push_unary(&mut plan, LAYER_LINEAR, 1, 0, 1);
        push_unary(&mut plan, LAYER_LINEAR, 2, 0, 2);
        push_binary(&mut plan, LAYER_BINARY, 3, 1, 2, 3);
        plan.push(3);
        plan
    }

    fn binary_manual(reg: &LayerRegistry, input: &WasmTensor) -> Vec<f32> {
        let t1 = reg.forward_layer(1, LAYER_LINEAR, input).unwrap();
        let t2 = reg.forward_layer(2, LAYER_LINEAR, input).unwrap();
        let t3 = reg.forward_binary_layer(3, &t1, &t2).unwrap();
        t3.to_array()
    }

    #[test]
    fn test_run_graph_binary_add() {
        let (reg, input) = build_binary_registry();
        let run = reg.run_graph(&binary_plan(), &input).unwrap();
        assert_eq!(run.to_array(), binary_manual(&reg, &input));
    }

    #[test]
    fn test_compile_graph_binary_add() {
        let (reg, input) = build_binary_registry();
        let compiled = reg.compile_graph(&binary_plan()).unwrap();
        assert_eq!(compiled.step_count(), 3);
        assert_eq!(compiled.slot_count(), 4);
        assert_eq!(compiled.output_slot(), 3);
        let run = compiled.run(&reg, &input).unwrap();
        assert_eq!(run.to_array(), binary_manual(&reg, &input));
    }
}
