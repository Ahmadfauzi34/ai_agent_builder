#[cfg(test)]
mod tests {
    use crate::protocol::{
        PacketHeader, PayloadCursor, OP_INIT, VARIANT_NONE, LAYER_LINEAR, LAYER_ACTIVATION,
        ACT_RELU, LAYER_BINARY, BINARY_ADD, LAYER_EMBEDDING, LAYER_CONV, CONV_CONV2D,
    };
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;
    use crate::es::rng::Rng;
    use crate::es::diag::{mean_std, diversity};
    use crate::es::objective::{LinearMseObjective, Objective};
    use crate::es::optimizer::EsOptimizer;

    fn mk_header(layer_type: u8, variant: u8, payload_len: usize) -> PacketHeader {
        let mut h = [0u8; 8];
        h[0] = OP_INIT; h[1] = layer_type; h[2] = variant; h[3] = 0;
        h[4..8].copy_from_slice(&(payload_len as u32).to_le_bytes());
        PacketHeader::from_bytes(&h).unwrap()
    }
    fn push_unary(plan: &mut Vec<u8>, lt: u8, id: u32, in_slot: u8, out_slot: u8) {
        plan.push(1); plan.push(lt); plan.extend_from_slice(&id.to_le_bytes());
        plan.push(in_slot); plan.push(0); plan.push(out_slot);
    }
    fn push_binary(plan: &mut Vec<u8>, lt: u8, id: u32, a: u8, b: u8, out_slot: u8) {
        plan.push(2); plan.push(lt); plan.extend_from_slice(&id.to_le_bytes());
        plan.push(a); plan.push(b); plan.push(out_slot);
    }

    // ---- jangkar lama (direkonstruksi utuh) ----
    #[test]
    fn test_payload_cursor_basic() {
        let mut data = Vec::new();
        data.extend_from_slice(&42u32.to_le_bytes());
        data.extend_from_slice(&3.14f64.to_le_bytes());
        data.push(1); data.push(1);
        data.extend_from_slice(&100u32.to_le_bytes());
        data.push(0);
        data.extend_from_slice(&0.0f64.to_le_bytes());
        let mut c = PayloadCursor::new(&data);
        assert_eq!(c.read_u32().unwrap(), 42);
        assert_eq!(c.read_f64().unwrap(), 3.14);
        assert!(c.read_bool().unwrap());
        assert_eq!(c.read_option_u32().unwrap(), Some(100));
        assert_eq!(c.read_option_f64().unwrap(), None);
        assert_eq!(c.remaining(), 0);
    }
    #[test]
    fn test_packet_header_validate() {
        let mut hb = [0u8; 8];
        hb[0] = 0x01; hb[1] = 0x02; hb[2] = 0x03; hb[3] = 0x04;
        hb[4..8].copy_from_slice(&12u32.to_le_bytes());
        let h = PacketHeader::from_bytes(&hb).unwrap();
        assert_eq!(h.opcode, 0x01); assert_eq!(h.layer_type, 0x02);
        assert_eq!(h.variant, 0x03); assert_eq!(h.flags, 0x04); assert_eq!(h.payload_len, 12);
        assert_eq!(h.validate_payload(&vec![0u8; 12]).unwrap().len(), 12);
        assert!(h.validate_payload(&vec![0u8; 10]).is_err());
    }
    #[test]
    fn test_layer_registry_param_cache() {
        let mut reg = LayerRegistry::new();
        assert_eq!(reg.total_params(), 0);
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&10u32.to_le_bytes());
        p.extend_from_slice(&5u32.to_le_bytes());
        p.push(1);
        reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
        assert_eq!(reg.total_params(), 55);
        assert!(reg.destroy_layer(1, LAYER_LINEAR));
        assert_eq!(reg.total_params(), 0);
    }
    fn build_linear_relu() -> (LayerRegistry, WasmTensor) {
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(&4u32.to_le_bytes());
        p.push(1);
        reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
        let mut p2 = Vec::new();
        p2.extend_from_slice(&2u32.to_le_bytes());
        reg.init_layer(&mk_header(LAYER_ACTIVATION, ACT_RELU, p2.len()), &p2).unwrap();
        (reg, WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]))
    }
    #[test]
    fn test_run_graph_unary_matches_manual() {
        let (reg, input) = build_linear_relu();
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
    fn test_run_graph_rejects_empty_slot() {
        let (reg, input) = build_linear_relu();
        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&3u32.to_le_bytes());
        push_unary(&mut plan, LAYER_LINEAR, 1, 2, 1);
        plan.push(1);
        assert!(reg.run_graph(&plan, &input).is_err());
    }
    #[test]
    fn test_run_graph_rejects_unknown_layer() {
        let (reg, input) = build_linear_relu();
        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&2u32.to_le_bytes());
        push_unary(&mut plan, LAYER_LINEAR, 999, 0, 1);
        plan.push(1);
        assert!(reg.run_graph(&plan, &input).is_err());
    }
    fn build_binary() -> (LayerRegistry, WasmTensor) {
        let mut reg = LayerRegistry::new();
        for id in [1u32, 2] {
            let mut p = Vec::new();
            p.extend_from_slice(&id.to_le_bytes());
            p.extend_from_slice(&3u32.to_le_bytes());
            p.extend_from_slice(&4u32.to_le_bytes());
            p.push(1);
            reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
        }
        let mut pb = Vec::new();
        pb.extend_from_slice(&3u32.to_le_bytes());
        pb.extend_from_slice(&0u32.to_le_bytes());
        reg.init_layer(&mk_header(LAYER_BINARY, BINARY_ADD, pb.len()), &pb).unwrap();
        (reg, WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]))
    }
    fn binary_plan() -> Vec<u8> {
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
        reg.forward_binary_layer(3, &t1, &t2).unwrap().to_array()
    }
    #[test]
    fn test_run_graph_binary_add() {
        let (reg, input) = build_binary();
        assert_eq!(reg.run_graph(&binary_plan(), &input).unwrap().to_array(), binary_manual(&reg, &input));
    }
    #[test]
    fn test_compile_graph_binary_add() {
        let (reg, input) = build_binary();
        let c = reg.compile_graph(&binary_plan()).unwrap();
        assert_eq!(c.step_count(), 3);
        assert_eq!(c.slot_count(), 4);
        assert_eq!(c.output_slot(), 3);
        assert_eq!(c.run(&reg, &input).unwrap().to_array(), binary_manual(&reg, &input));
    }
    #[test]
    fn es_rng_is_deterministic() {
        let mut a = Rng::new(42); let mut b = Rng::new(42);
        let va: Vec<f32> = (0..100).map(|_| a.gaussian()).collect();
        let vb: Vec<f32> = (0..100).map(|_| b.gaussian()).collect();
        assert_eq!(va, vb);
    }
    #[test]
    fn es_diag_mean_std_known() {
        let (m, s) = mean_std(&[1.0, 2.0, 3.0, 4.0]);
        assert!((m - 2.5).abs() < 1e-12);
        assert!((s - 1.1180339887498949).abs() < 1e-9);
    }
    #[test]
    fn es_diag_diversity_identical_is_zero() {
        let cands = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_eq!(diversity(&cands), 0.0);
    }
    fn make_obj() -> LinearMseObjective {
        LinearMseObjective::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.0, 1.5, 2.0], 4, 1, 1)
    }
    fn run_es(seed: u32) -> (Vec<f32>, String) {
        let obj = make_obj();
        let mut es = EsOptimizer::new(1, 0, seed, Some(16), Some(0.2), Some(0.1));
        for _ in 0..5 {
            let flat = es.ask();
            let n = es.batch_size() as usize;
            let d = es.dim() as usize;
            let mut f = Vec::with_capacity(n);
            for i in 0..n { f.push(obj.fitness(&flat[i * d..(i + 1) * d]) as f32); }
            let _ = es.tell(&f);
        }
        (es.best(), es.report())
    }
    #[test]
    fn es_optimizer_is_deterministic_end_to_end() {
        let (b1, r1) = run_es(42);
        let (b2, r2) = run_es(42);
        assert_eq!(b1.len(), 1);
        assert_eq!(b1, b2);
        assert_eq!(r1, r2);
    }
    #[test]
    fn es_ask_batch_shape_contract() {
        let mut es = EsOptimizer::new(3, 0, 7, Some(8), Some(0.2), Some(0.1));
        let flat = es.ask();
        let n = es.batch_size() as usize;
        let d = es.dim() as usize;
        assert!(n > 0);
        assert_eq!(flat.len(), n * d);
    }
    #[test]
    fn float_bridge_linear_roundtrip_and_affects_forward() {
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&3u32.to_le_bytes());
        p.extend_from_slice(&2u32.to_le_bytes());
        p.push(1);
        reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
        let w = reg.get_weights_flat(1, LAYER_LINEAR).unwrap();
        assert_eq!(w.len(), 3 * 2 + 2);
        reg.set_weights_flat(1, LAYER_LINEAR, &w).unwrap();
        assert_eq!(reg.get_weights_flat(1, LAYER_LINEAR).unwrap(), w);
        let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
        let out1 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap().to_array();
        let w2: Vec<f32> = w.iter().map(|v| v + 1.0).collect();
        reg.set_weights_flat(1, LAYER_LINEAR, &w2).unwrap();
        let out2 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap().to_array();
        assert_ne!(out1, out2);
    }
    #[test]
    fn float_bridge_unknown_type_is_err() {
        let mut reg = LayerRegistry::new();
        assert!(reg.get_weights_flat(1, 0xFE).is_err());
        assert!(reg.set_weights_flat(1, 0xFE, &[]).is_err());
    }

    // ---- JANGKAR M1 BARU: float-bridge embedding & conv ----
    #[test]
    fn float_bridge_embedding_roundtrip_and_affects_forward() {
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes()); // id
        p.extend_from_slice(&3u32.to_le_bytes()); // vocab
        p.extend_from_slice(&2u32.to_le_bytes()); // d_model
        reg.init_layer(&mk_header(LAYER_EMBEDDING, VARIANT_NONE, p.len()), &p).unwrap();

        let w = reg.get_weights_flat(1, LAYER_EMBEDDING).unwrap();
        assert_eq!(w.len(), 3 * 2); // vocab * d_model, tanpa bias
        reg.set_weights_flat(1, LAYER_EMBEDDING, &w).unwrap();
        assert_eq!(reg.get_weights_flat(1, LAYER_EMBEDDING).unwrap(), w);

        let input = WasmTensor::new(&[0.0, 1.0, 2.0, 0.0], &[1, 4, 1, 1]);
        let out1 = reg.forward_layer(1, LAYER_EMBEDDING, &input).unwrap().to_array();
        assert_eq!(out1.len(), 4 * 2); // seq * d_model
        let w2: Vec<f32> = w.iter().map(|v| v + 1.0).collect();
        reg.set_weights_flat(1, LAYER_EMBEDDING, &w2).unwrap();
        let out2 = reg.forward_layer(1, LAYER_EMBEDDING, &input).unwrap().to_array();
        assert_ne!(out1, out2);
    }
    #[test]
    fn float_bridge_conv_roundtrip_and_affects_forward() {
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes()); // id
        p.extend_from_slice(&1u32.to_le_bytes()); // in_ch
        p.extend_from_slice(&1u32.to_le_bytes()); // out_ch
        p.extend_from_slice(&1u32.to_le_bytes()); // kh
        p.extend_from_slice(&1u32.to_le_bytes()); // kw
        for _ in 0..4 { p.push(0); p.extend_from_slice(&0u32.to_le_bytes()); } // sh,sw,ph,pw = None
        reg.init_layer(&mk_header(LAYER_CONV, CONV_CONV2D, p.len()), &p).unwrap();

        let w = reg.get_weights_flat(1, LAYER_CONV).unwrap();
        assert_eq!(w.len(), 1 * 1 * 1 * 1 + 1); // weight(1) + bias(1)
        reg.set_weights_flat(1, LAYER_CONV, &w).unwrap();
        assert_eq!(reg.get_weights_flat(1, LAYER_CONV).unwrap(), w);

        let input = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out1 = reg.forward_layer(1, LAYER_CONV, &input).unwrap().to_array();
        let w2: Vec<f32> = w.iter().map(|v| v + 1.0).collect();
        reg.set_weights_flat(1, LAYER_CONV, &w2).unwrap();
        let out2 = reg.forward_layer(1, LAYER_CONV, &input).unwrap().to_array();
        assert_ne!(out1, out2);
    }
}