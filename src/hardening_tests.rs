#[cfg(test)]
mod hardening_tests {
    use crate::es::optimizer::EsOptimizer;
    use crate::registry::LayerRegistry;
    use crate::agent::AgentLayerSpec;
    use crate::layers::binary::WasmBinary;
    use crate::protocol::{
        LAYER_EMBEDDING, LAYER_LINEAR, MAX_ALLOC_ELEMENTS, OP_INIT, PacketHeader,
        VARIANT_NONE,
    };
    use crate::WasmTensor;

    fn mk_init_header(layer_type: u8, payload_len: usize) -> PacketHeader {
        let mut h = [0u8; 8];
        h[0] = OP_INIT;
        h[1] = layer_type;
        h[2] = VARIANT_NONE;
        h[3] = 0;
        h[4..8].copy_from_slice(&(payload_len as u32).to_le_bytes());
        PacketHeader::from_bytes(&h).unwrap()
    }

    fn linear_payload(id: u32, in_dim: u32, out_dim: u32, bias: bool) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&id.to_le_bytes());
        p.extend_from_slice(&in_dim.to_le_bytes());
        p.extend_from_slice(&out_dim.to_le_bytes());
        p.push(u8::from(bias));
        p
    }

    // ---- Complaint #17: blind allocation under the 2^28 cap ----

    #[test]
    fn r17_init_agent_layer_10000x10000_rejected_before_allocation() {
        // Complaint #17 evidence (build 631d75a): linear(0,10000,10000) =
        // 1e8 elements = 381 MiB, under the old 2^28 cap, allocated blindly
        // (12.8 s, 1.9 GB RSS spike, no warning). It must now be rejected
        // BEFORE any allocation: fast and structured, via the real facade.
        let mut reg = LayerRegistry::new();
        let spec = AgentLayerSpec::linear(0, 10_000, 10_000, false).unwrap();
        let start = std::time::Instant::now();
        let err = reg.init_agent_layer(&spec).unwrap_err();
        let elapsed = start.elapsed();
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(
            err.contains("100000000"),
            "error must state the requested size: {err}"
        );
        assert!(
            err.contains(&MAX_ALLOC_ELEMENTS.to_string()),
            "error must state the budget: {err}"
        );
        assert!(
            elapsed.as_millis() < 100,
            "rejection must be fast (<100 ms), took {elapsed:?}"
        );
        // Nothing was allocated and the registry is healthy.
        assert_eq!(reg.total_params(), 0);
        let p = linear_payload(9, 4, 4, false);
        reg.init_layer(&mk_init_header(LAYER_LINEAR, p.len()), &p)
            .unwrap();
        assert_eq!(reg.total_params(), 16);
    }

    #[test]
    fn r17_init_budget_boundary() {
        // Exactly at the 64 MiB budget: accepted. Just over: rejected.
        let mut reg = LayerRegistry::new();
        // 16384 * 1024 = 2^24 elements = 64 MiB.
        let p_ok = linear_payload(1, 16384, 1024, false);
        reg.init_layer(&mk_init_header(LAYER_LINEAR, p_ok.len()), &p_ok)
            .unwrap();
        assert!(reg.destroy_layer(1, LAYER_LINEAR));
        // 16384 * 1025 = 16_793_600 > 2^24: one step over the budget.
        let p_over = linear_payload(2, 16384, 1025, false);
        let err = reg
            .init_layer(&mk_init_header(LAYER_LINEAR, p_over.len()), &p_over)
            .unwrap_err();
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
    }

    #[test]
    fn r17_init_embedding_rejects_oversized_vocab() {
        // The budget applies to every init path, not just linear.
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&50000u32.to_le_bytes());
        p.extend_from_slice(&1024u32.to_le_bytes());
        let start = std::time::Instant::now();
        let err = reg
            .init_layer(&mk_init_header(LAYER_EMBEDDING, p.len()), &p)
            .unwrap_err();
        assert!(
            start.elapsed().as_millis() < 100,
            "rejection must be fast (<100 ms)"
        );
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(err.contains("embedding weight"), "must name the stage: {err}");
    }

    // ---- Complaint #18: raw unreachable trap on giant tensor runs ----

    #[test]
    fn r18_forward_rejects_oversized_input_structured() {
        let mut reg = LayerRegistry::new();
        let p = linear_payload(1, 4, 4, false);
        reg.init_layer(&mk_init_header(LAYER_LINEAR, p.len()), &p)
            .unwrap();
        // Input one element over the budget.
        let big = WasmTensor::new(&vec![0.0f32; MAX_ALLOC_ELEMENTS + 1], &[
            MAX_ALLOC_ELEMENTS + 1,
            1,
            1,
            1,
        ]);
        let err = match reg.forward_layer(1, LAYER_LINEAR, &big) {
            Ok(_) => panic!("expected forward_layer to reject the oversized input"),
            Err(err) => err,
        };
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(
            err.contains("forwardLayer input"),
            "error must name the stage: {err}"
        );
        // Per-call containment: the registry still serves honest traffic,
        // i.e. the rejection did not wedge anything.
        let small = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4, 1, 1]);
        let out = reg.forward_layer(1, LAYER_LINEAR, &small).unwrap();
        assert_eq!(out.shape(), vec![1, 4, 1, 1]);
    }

    #[test]
    fn r18_linear_output_blowup_fails_structured() {
        // Weights within budget, but the run would materialize an output
        // above it: linear(1, 2048, 8192) weights = 2^24 (at budget, init OK),
        // input [4096, 2048] is fine, output [4096, 8192] = 2^25 > budget.
        let mut reg = LayerRegistry::new();
        let p = linear_payload(1, 2048, 8192, false);
        reg.init_layer(&mk_init_header(LAYER_LINEAR, p.len()), &p)
            .unwrap();
        let input = WasmTensor::new(&vec![0.0f32; 4096 * 2048], &[4096, 2048, 1, 1]);
        let start = std::time::Instant::now();
        let err = match reg.forward_layer(1, LAYER_LINEAR, &input) {
            Ok(_) => panic!("expected forward_layer to reject the oversized output"),
            Err(err) => err,
        };
        assert!(
            start.elapsed().as_millis() < 1000,
            "rejection must precede the matmul, took {:?}",
            start.elapsed()
        );
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(
            err.contains("Linear forward output"),
            "error must name the stage: {err}"
        );
        // No wedge: an in-budget run on the same layer still works.
        let small = WasmTensor::new(&vec![1.0f32; 2048], &[1, 2048, 1, 1]);
        let out = reg.forward_layer(1, LAYER_LINEAR, &small).unwrap();
        assert_eq!(out.shape(), vec![1, 8192, 1, 1]);
    }

    #[test]
    fn r18_matmul_output_blowup_fails_structured_not_unreachable() {
        // The #18 evidence shape: both inputs within budget, but the matmul
        // output [1,1,16384,16384] = 2^28 elements = 1 GiB. Must fail as a
        // structured per-call error, never a raw `unreachable` trap.
        let mm = WasmBinary::new_matmul();
        let a = WasmTensor::new(&vec![0.0f32; 1 << 24], &[1, 1, 16384, 1024]);
        let b = WasmTensor::new(&vec![0.0f32; 1 << 24], &[1, 1, 1024, 16384]);
        let start = std::time::Instant::now();
        let err = match mm.forward_binary(&a, &b) {
            Ok(_) => panic!("expected forward_binary to reject the oversized output"),
            Err(err) => err,
        };
        assert!(
            start.elapsed().as_millis() < 1000,
            "rejection must precede the matmul, took {:?}",
            start.elapsed()
        );
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(
            err.contains("268435456"),
            "error must state the requested size: {err}"
        );
        assert!(
            err.contains("binary matmul output"),
            "error must name the stage: {err}"
        );
        // No wedge: an in-budget matmul still computes correctly.
        let a2 = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let b2 = WasmTensor::new(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let out = mm.forward_binary(&a2, &b2).unwrap();
        assert_eq!(out.to_array(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn r18_embedding_output_blowup_fails_structured() {
        // embedding(1, vocab=1024, d_model=16384): weights = 2^24, at budget
        // (init OK). Input [1,2048] indices are fine, but the output
        // [1,2048,16384] = 2^25 elements exceeds the budget.
        let mut reg = LayerRegistry::new();
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&1024u32.to_le_bytes());
        p.extend_from_slice(&16384u32.to_le_bytes());
        reg.init_layer(&mk_init_header(LAYER_EMBEDDING, p.len()), &p)
            .unwrap();
        let input = WasmTensor::new(&vec![0.0f32; 2048], &[1, 2048, 1, 1]);
        let err = match reg.forward_layer(1, LAYER_EMBEDDING, &input) {
            Ok(_) => panic!("expected forward_layer to reject the oversized output"),
            Err(err) => err,
        };
        assert!(err.contains("tensor_too_large"), "unexpected error: {err}");
        assert!(
            err.contains("Embedding forward output"),
            "error must name the stage: {err}"
        );
    }

    #[test]
    fn compile_graph_rejects_untrusted_step_count_before_large_allocation() {
        let reg = LayerRegistry::new();
        let mut plan = Vec::new();
        plan.extend_from_slice(&u32::MAX.to_le_bytes());
        plan.extend_from_slice(&1u32.to_le_bytes());

        let err = match reg.compile_graph(&plan) {
            Ok(_) => panic!("expected compile_graph to reject an impossible step envelope"),
            Err(err) => err,
        };
        assert!(
            err.contains("malformed plan length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn compile_graph_rejects_trailing_garbage() {
        let reg = LayerRegistry::new();
        let mut plan = Vec::new();
        plan.extend_from_slice(&1u32.to_le_bytes());
        plan.extend_from_slice(&1u32.to_le_bytes());
        // One canonical 9-byte step, followed by output slot and one trash byte.
        plan.extend_from_slice(&[1, 0x01, 0, 0, 0, 0, 0, 0, 0]);
        plan.push(0);
        plan.push(0xAA);

        let err = match reg.compile_graph(&plan) {
            Ok(_) => panic!("expected compile_graph to reject trailing plan bytes"),
            Err(err) => err,
        };
        assert!(
            err.contains("malformed plan length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn es_tell_requires_pending_ask() {
        let mut es = EsOptimizer::new(2, 0, 42, Some(8), Some(0.2), Some(0.1));
        let err = es.tell(&[]).unwrap_err();
        assert!(err.contains("call ask() first"));
        assert_eq!(es.generation(), 0);
    }

    #[test]
    fn es_tell_cardinality_error_does_not_consume_batch() {
        let mut es = EsOptimizer::new(2, 0, 42, Some(8), Some(0.2), Some(0.1));
        let _ = es.ask();
        let expected = es.batch_size() as usize;
        assert!(expected > 1);

        let err = es.tell(&vec![0.0; expected - 1]).unwrap_err();
        assert!(err.contains("fitness length mismatch"));
        assert_eq!(es.generation(), 0);

        // The same pending batch remains valid after a rejected tell.
        let report = es.tell(&vec![0.0; expected]).unwrap();
        assert!(report.contains("\"gen\":1"));
        assert_eq!(es.generation(), 1);

        // A batch can be consumed exactly once.
        let err = es.tell(&vec![0.0; expected]).unwrap_err();
        assert!(err.contains("call ask() first"));
        assert_eq!(es.generation(), 1);
    }

    #[test]
    fn es_tell_non_finite_error_does_not_consume_or_mutate_batch() {
        let mut es = EsOptimizer::new(2, 0, 42, Some(8), Some(0.2), Some(0.1));
        let _ = es.ask();
        let expected = es.batch_size() as usize;
        let mean_before = es.mean();

        let mut invalid = vec![0.0; expected];
        invalid[0] = f32::NAN;
        let err = es.tell(&invalid).unwrap_err();
        assert!(err.contains("non-finite"));
        assert_eq!(es.generation(), 0);
        assert_eq!(es.mean(), mean_before);

        // Rejected fitness does not consume the pending candidate batch.
        let report = es.tell(&vec![0.0; expected]).unwrap();
        assert!(report.contains("\"gen\":1"));
        assert_eq!(es.generation(), 1);
    }
}
