#[cfg(test)]
mod hardening_tests {
    use crate::es::optimizer::EsOptimizer;
    use crate::registry::LayerRegistry;

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
