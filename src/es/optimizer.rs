use super::rng::Rng;
use super::strategy::Strategy;
pub use crate::facade::es::es_capabilities;
use wasm_bindgen::prelude::*;

pub(crate) const LINEAR_DEMO_IN_DIM: usize = 3;
pub(crate) const LINEAR_DEMO_OUT_DIM: usize = 2;
pub(crate) const LINEAR_DEMO_PARAM_DIM: usize = LINEAR_DEMO_IN_DIM * LINEAR_DEMO_OUT_DIM;
pub(crate) const DEFAULT_POP: u32 = 64;
pub(crate) const DEFAULT_SIGMA: f32 = 0.1;
pub(crate) const DEFAULT_LR: f32 = 0.05;

pub(crate) fn strict_config(
    dim: u32,
    strategy: u8,
    pop: Option<u32>,
    sigma: Option<f32>,
    lr: Option<f32>,
) -> Result<(u32, f32, Option<f32>), String> {
    if dim == 0 {
        return Err("EsOptimizer.strict: dim must be > 0".into());
    }
    if strategy > 1 {
        return Err(format!(
            "EsOptimizer.strict: strategy must be 0 (OpenES) or 1 (mu,lambda), got {strategy}"
        ));
    }

    let pop = pop.unwrap_or(DEFAULT_POP);
    if pop < 2 {
        return Err(format!("EsOptimizer.strict: pop must be >= 2, got {pop}"));
    }
    if strategy == 0 && !pop.is_multiple_of(2) {
        return Err(format!(
            "EsOptimizer.strict: OpenES pop must be even for antithetic pairs, got {pop}"
        ));
    }

    let sigma = sigma.unwrap_or(DEFAULT_SIGMA);
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(format!(
            "EsOptimizer.strict: sigma must be finite and > 0, got {sigma}"
        ));
    }

    if strategy == 0 {
        let lr = lr.unwrap_or(DEFAULT_LR);
        if !lr.is_finite() || lr <= 0.0 {
            return Err(format!(
                "EsOptimizer.strict: OpenES lr must be finite and > 0, got {lr}"
            ));
        }
        Ok((pop, sigma, Some(lr)))
    } else {
        if lr.is_some() {
            return Err(
                "EsOptimizer.strict: lr is not used by mu_lambda; omit the lr argument".into(),
            );
        }
        Ok((pop, sigma, None))
    }
}

#[wasm_bindgen]
pub struct EsOptimizer {
    pub(crate) strategy: Strategy,
    pub(crate) rng: Rng,
    pub(crate) dim: usize,
    pub(crate) last_candidates: Vec<Vec<f32>>,
    pub(crate) last_report: String,
    pub(crate) gen: u32,
    pub(crate) best_fitness: f64,
    pub(crate) best_params: Vec<f32>,
    pub(crate) stagnation: u32,
    pub(crate) awaiting_fitness: bool,
}

// #[wasm_bindgen] impl EsOptimizer — dipindah ke src/facade/coprocessor.rs (Opsi C Fase 2).

#[cfg(test)]
mod tests {
    use super::{es_capabilities, EsOptimizer};
    use crate::es::strategy::EsStrategy;

    fn optimizer(dim: u32) -> EsOptimizer {
        EsOptimizer::new(dim, 0, 123, Some(8), Some(0.1), Some(0.05))
    }

    #[test]
    fn strict_factory_rejects_legacy_coercions_and_invalid_hyperparameters() {
        assert!(EsOptimizer::strict(0, 0, 1, Some(8), Some(0.1), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 9, 1, Some(8), Some(0.1), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(1), Some(0.1), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(3), Some(0.1), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(8), Some(0.0), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(8), Some(f32::NAN), Some(0.05)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(8), Some(0.1), Some(0.0)).is_err());
        assert!(EsOptimizer::strict(2, 0, 1, Some(8), Some(0.1), Some(f32::INFINITY)).is_err());
        assert!(EsOptimizer::strict(2, 1, 1, Some(8), Some(0.1), Some(0.05)).is_err());
    }

    #[test]
    fn learning_rate_mutation_rejects_invalid_values_and_non_openes_strategy() {
        let mut openes = EsOptimizer::strict(3, 0, 7, Some(8), Some(0.08), Some(0.10)).unwrap();
        for lr in [0.0, -0.01, f32::NAN, f32::INFINITY] {
            assert!(openes.set_learning_rate(lr).is_err());
            assert_eq!(openes.generation(), 0);
        }

        let mut mu_lambda = EsOptimizer::strict(3, 1, 7, Some(8), Some(0.08), None).unwrap();
        let err = mu_lambda.set_learning_rate(0.10).unwrap_err();
        assert!(err.contains("only supported by OpenES"));
        assert_eq!(mu_lambda.generation(), 0);
    }

    #[test]
    fn learning_rate_mutation_rejects_pending_batch_without_consuming_it() {
        let mut optimizer = EsOptimizer::strict(3, 0, 11, Some(8), Some(0.08), Some(0.10)).unwrap();

        let pending = optimizer.ask();
        assert_eq!(pending.len(), 24);
        let before_best = optimizer.best();
        let before_generation = optimizer.generation();

        let err = optimizer.set_learning_rate(0.08).unwrap_err();
        assert!(err.contains("candidate batch is pending"));
        assert_eq!(optimizer.generation(), before_generation);
        assert_eq!(optimizer.best(), before_best);
        assert_eq!(optimizer.batch_size(), 8);

        let fitness = [2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.5, -0.5];
        optimizer.tell(&fitness).unwrap();
        assert_eq!(optimizer.generation(), 1);
    }

    #[test]
    fn learning_rate_mutation_preserves_search_state_until_next_tell() {
        let mut control = EsOptimizer::strict(3, 0, 19, Some(8), Some(0.08), Some(0.10)).unwrap();
        let mut changed = EsOptimizer::strict(3, 0, 19, Some(8), Some(0.08), Some(0.10)).unwrap();

        let first_control = control.ask();
        let first_changed = changed.ask();
        assert_eq!(first_control, first_changed);

        let first_fitness = [2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.5, -0.5];
        control.tell(&first_fitness).unwrap();
        changed.tell(&first_fitness).unwrap();

        let mean_before = control.mean();
        assert_eq!(changed.mean(), mean_before);
        assert_eq!(changed.best(), control.best());
        assert_eq!(changed.generation(), control.generation());

        changed.set_learning_rate(0.08).unwrap();

        assert_eq!(changed.mean(), mean_before);
        assert_eq!(changed.best(), control.best());
        assert_eq!(changed.generation(), control.generation());

        // LR does not participate in ask(), so preserving mean + RNG yields
        // an exactly identical next candidate batch.
        let second_control = control.ask();
        let second_changed = changed.ask();
        assert_eq!(second_control, second_changed);

        let second_fitness = [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.25, -0.25];
        control.tell(&second_fitness).unwrap();
        changed.tell(&second_fitness).unwrap();

        assert_eq!(control.generation(), 2);
        assert_eq!(changed.generation(), 2);
        assert_ne!(control.mean(), changed.mean());
        assert!((changed.strategy.lr() - 0.08).abs() <= f32::EPSILON);
    }

    #[test]
    fn strict_openes_preserves_requested_dimension_and_population() {
        let mut opt = EsOptimizer::strict(3, 0, 7, Some(8), Some(0.2), Some(0.1)).unwrap();
        assert_eq!(opt.dim(), 3);
        let flat = opt.ask();
        assert_eq!(opt.batch_size(), 8);
        assert_eq!(flat.len(), 24);
    }

    #[test]
    fn strict_mu_lambda_accepts_odd_lambda_but_requires_lr_omitted() {
        let mut opt = EsOptimizer::strict(3, 1, 7, Some(5), Some(0.2), None).unwrap();
        assert_eq!(opt.dim(), 3);
        let flat = opt.ask();
        assert_eq!(opt.batch_size(), 5);
        assert_eq!(flat.len(), 15);
    }

    #[test]
    fn legacy_constructor_remains_forgiving_for_compatibility() {
        let opt = EsOptimizer::new(0, 99, 7, Some(1), Some(0.1), Some(0.05));
        assert_eq!(opt.dim(), 1);
    }

    #[test]
    fn es_capabilities_exposes_strict_and_legacy_modes() {
        let manifest = es_capabilities();
        assert!(manifest.contains("\"strict_factory\":\"EsOptimizer.strict\""));
        assert!(manifest.contains("legacy_forgiving"));
        assert!(manifest.contains("openes_odd_pop_truncates_to_pairs"));
        assert!(manifest.contains("\"set_learning_rate\""));
        assert!(manifest.contains("\"method\":\"setLearningRate\""));
        assert!(manifest.contains("\"between_completed_generations\""));
        assert!(manifest.contains("\"optimizer_dim\":6"));
    }

    #[test]
    fn linear_demo_rejects_every_non_six_dimension_in_matrix() {
        for dim in 1..=10 {
            if dim == 6 {
                continue;
            }
            let mut opt = optimizer(dim);
            let err = opt.run_linear_demo(1).unwrap_err();
            assert!(err.contains("requires optimizer dim=6"));
            assert_eq!(opt.generation(), 0);
            assert_eq!(opt.report(), "{}");
        }
    }

    #[test]
    fn linear_demo_requires_at_least_one_generation() {
        let mut opt = optimizer(6);
        let err = opt.run_linear_demo(0).unwrap_err();
        assert!(err.contains("gens must be > 0"));
        assert_eq!(opt.generation(), 0);
        assert_eq!(opt.report(), "{}");
    }

    #[test]
    fn linear_demo_succeeds_only_for_six_dimensions_and_advances_generation() {
        let mut opt = optimizer(6);
        let report = opt.run_linear_demo(2).unwrap();
        assert_ne!(report, "{}");
        assert!(report.contains("\"dim\":6"));
        assert_eq!(opt.generation(), 2);
    }

    #[test]
    fn invalid_linear_demo_cannot_leak_a_stale_prior_report() {
        let mut opt = optimizer(5);
        let candidates = opt.ask();
        assert!(!candidates.is_empty());
        let fitness = vec![0.0f32; opt.batch_size() as usize];
        let previous = opt.tell(&fitness).unwrap();
        assert_ne!(previous, "{}");
        assert_eq!(opt.generation(), 1);

        let err = opt.run_linear_demo(2).unwrap_err();
        assert!(err.contains("requires optimizer dim=6"));
        assert_eq!(opt.generation(), 1);
        assert_eq!(opt.report(), previous);
    }
}
