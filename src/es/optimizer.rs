use wasm_bindgen::prelude::*;
use super::diag::{diversity, mean_std, EsReport};
use super::objective::{LinearMseObjective, Objective};
use super::rng::Rng;
use super::strategy::{EsStrategy, Strategy};

const LINEAR_DEMO_IN_DIM: usize = 3;
const LINEAR_DEMO_OUT_DIM: usize = 2;
const LINEAR_DEMO_PARAM_DIM: usize = LINEAR_DEMO_IN_DIM * LINEAR_DEMO_OUT_DIM;
const DEFAULT_POP: u32 = 64;
const DEFAULT_SIGMA: f32 = 0.1;
const DEFAULT_LR: f32 = 0.05;

fn strict_config(
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
        return Err(format!(
            "EsOptimizer.strict: pop must be >= 2, got {pop}"
        ));
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

/// Machine-readable ES contracts for agent planning.
#[wasm_bindgen(js_name = esCapabilities)]
pub fn es_capabilities() -> String {
    concat!(
        "{",
        "\"entry\":\"EsOptimizer\",",
        "\"constructor\":{\"mode\":\"legacy_forgiving\",\"coercions\":[\"dim_zero_to_one\",\"pop_below_two_to_two\",\"unknown_strategy_to_openes\",\"openes_odd_pop_truncates_to_pairs\"]},",
        "\"strict_factory\":\"EsOptimizer.strict\",",
        "\"strategies\":{\"openes\":0,\"mu_lambda\":1},",
        "\"strict_contract\":{",
        "\"dim\":{\"min\":1},",
        "\"openes\":{\"pop_min\":2,\"pop_even\":true,\"sigma\":\"finite>0\",\"lr\":\"finite>0\"},",
        "\"mu_lambda\":{\"pop_min\":2,\"sigma\":\"finite>0\",\"lr\":\"omit\"}},",
        "\"lifecycle\":\"ask->tell\",",
        "\"linear_demo\":{\"method\":\"runLinearDemo\",\"optimizer_dim\":6,\"gens_min\":1}",
        "}"
    )
    .to_string()
}

#[wasm_bindgen]
pub struct EsOptimizer {
    strategy: Strategy,
    rng: Rng,
    dim: usize,
    last_candidates: Vec<Vec<f32>>,
    last_report: String,
    gen: u32,
    best_fitness: f64,
    best_params: Vec<f32>,
    stagnation: u32,
    awaiting_fitness: bool,
}

#[wasm_bindgen]
impl EsOptimizer {
    /// Legacy forgiving constructor retained for compatibility.
    ///
    /// It clamps dim/pop and falls back unknown strategies to OpenES. Agent/proof workflows
    /// should prefer `EsOptimizer.strict(...)`, whose invalid configurations are controlled errors.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: u32,
        strategy: u8,
        seed: u32,
        pop: Option<u32>,
        sigma: Option<f32>,
        lr: Option<f32>,
    ) -> EsOptimizer {
        let dim = dim.max(1) as usize;
        let pop = pop.unwrap_or(DEFAULT_POP).max(2) as usize;
        let sigma = sigma.unwrap_or(DEFAULT_SIGMA);
        let lr = lr.unwrap_or(DEFAULT_LR);
        let mut rng = Rng::new(seed);
        let strat = match strategy {
            1 => Strategy::mu_lambda(dim, (pop / 2).max(1), pop, sigma, &mut rng),
            _ => Strategy::openes(dim, pop / 2, sigma, lr, &mut rng),
        };
        EsOptimizer {
            strategy: strat,
            rng,
            dim,
            last_candidates: Vec::new(),
            last_report: String::from("{}"),
            gen: 0,
            best_fitness: f64::NEG_INFINITY,
            best_params: Vec::new(),
            stagnation: 0,
            awaiting_fitness: false,
        }
    }

    /// Strict proof-boundary factory. Unlike the legacy constructor, this never silently
    /// repairs dimensions/population, never falls back an unknown strategy, and rejects
    /// non-finite/non-positive numerical hyperparameters before optimizer state is created.
    #[wasm_bindgen(js_name = strict)]
    pub fn strict(
        dim: u32,
        strategy: u8,
        seed: u32,
        pop: Option<u32>,
        sigma: Option<f32>,
        lr: Option<f32>,
    ) -> Result<EsOptimizer, String> {
        let (pop, sigma, strict_lr) = strict_config(dim, strategy, pop, sigma, lr)?;
        Ok(EsOptimizer::new(
            dim,
            strategy,
            seed,
            Some(pop),
            Some(sigma),
            strict_lr,
        ))
    }

    #[wasm_bindgen(js_name = dim)]
    pub fn dim(&self) -> u32 { self.dim as u32 }

    #[wasm_bindgen(js_name = generation)]
    pub fn generation(&self) -> u32 { self.gen }

    #[wasm_bindgen(js_name = batchSize)]
    pub fn batch_size(&self) -> u32 { self.last_candidates.len() as u32 }

    /// Minta kandidat generasi ini. Mengembalikan Float32Array flat (n_kandidat * dim).
    /// JS slice per `dim()`. Panggil `tell()` sesudahnya dengan fitness seurut kandidat.
    /// Memanggil ask lagi sebelum tell diperbolehkan: batch sebelumnya dianggap dibatalkan.
    pub fn ask(&mut self) -> Vec<f32> {
        let cands = self.strategy.ask(&mut self.rng);
        let mut flat = Vec::with_capacity(cands.len() * self.dim);
        for c in &cands { flat.extend_from_slice(c); }
        self.last_candidates = cands;
        self.awaiting_fitness = true;
        flat
    }

    /// Serahkan fitness (seurut kandidat ask). Mengembalikan laporan JSON generasi ini.
    /// Boundary contract: tell hanya sah setelah ask, cardinality harus tepat, dan semua fitness finite.
    pub fn tell(&mut self, fitnesses: &[f32]) -> Result<String, String> {
        if !self.awaiting_fitness {
            return Err("tell: no pending candidate batch; call ask() first".into());
        }
        let expected = self.last_candidates.len();
        if fitnesses.len() != expected {
            return Err(format!(
                "tell: fitness length mismatch: expected {}, got {}",
                expected,
                fitnesses.len()
            ));
        }
        if let Some((index, value)) = fitnesses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "tell: non-finite fitness at index {}: {}",
                index, value
            ));
        }

        let f64s: Vec<f64> = fitnesses.iter().map(|&v| v as f64).collect();

        // Update strategy only after the public contract is fully validated.
        self.strategy.tell(&f64s);

        // statistik fitness
        let (mean, std) = mean_std(&f64s);
        let mut best = f64::NEG_INFINITY;
        let mut worst = f64::INFINITY;
        for &v in &f64s {
            if v > best { best = v; }
            if v < worst { worst = v; }
        }

        // global best + stagnation (scan kandidat vs fitness)
        let mut gen_best = f64::NEG_INFINITY;
        let mut gen_best_params: Vec<f32> = Vec::new();
        for (c, &v) in self.last_candidates.iter().zip(f64s.iter()) {
            if v > gen_best { gen_best = v; gen_best_params = c.clone(); }
        }
        let prev_best = self.best_fitness;
        if gen_best > self.best_fitness + 1e-8 {
            self.best_fitness = gen_best;
            self.best_params = gen_best_params;
            self.stagnation = 0;
        } else {
            self.stagnation = self.stagnation.saturating_add(1);
        }
        let improvement = self.best_fitness - prev_best;

        // diagnosa populasi
        let div = diversity(&self.last_candidates);
        let mean_vec = self.strategy.mean();
        let mean_norm = (mean_vec.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt();
        let best_norm = (self.best_params.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt();

        // flags
        let mut flags: Vec<String> = Vec::new();
        if div < 1e-6 { flags.push("DIVERSITY_COLLAPSE".into()); }
        if improvement <= 1e-8 { flags.push("NO_IMPROVEMENT".into()); }
        if std < 1e-9 { flags.push("ALL_FITNESS_EQUAL".into()); }

        self.gen = self.gen.saturating_add(1);
        self.awaiting_fitness = false;

        let rep = EsReport {
            gen: self.gen,
            strategy: self.strategy.name(),
            evals: f64s.len(),
            dim: self.dim,
            pop: self.last_candidates.len(),
            best, worst, mean, std,
            improvement,
            stagnation: self.stagnation,
            diversity: div,
            sigma: self.strategy.sigma(),
            lr: self.strategy.lr(),
            mean_norm, best_norm,
            flags,
        };
        let json = rep.to_json();
        self.last_report = json.clone();
        Ok(json)
    }

    /// Vektor terbaik sepanjang pelatihan (Float32Array).
    pub fn best(&self) -> Vec<f32> { self.best_params.clone() }

    /// Rata-rata populasi saat ini (Float32Array).
    pub fn mean(&self) -> Vec<f32> { self.strategy.mean() }

    /// Laporan JSON generasi terakhir.
    pub fn report(&self) -> String { self.last_report.clone() }

    /// Proof-of-life mandiri untuk problem linear tetap `in=3, out=2`.
    ///
    /// Contract: optimizer harus dibuat dengan `dim == 6` dan `gens > 0`.
    /// Pelanggaran contract atau kegagalan internal `tell()` dikembalikan sebagai error
    /// (menjadi exception terkontrol pada boundary JavaScript/WASM), bukan report kosong/stale.
    #[wasm_bindgen(js_name = runLinearDemo)]
    pub fn run_linear_demo(&mut self, gens: u32) -> Result<String, String> {
        if self.dim != LINEAR_DEMO_PARAM_DIM {
            return Err(format!(
                "runLinearDemo: fixed 3x2 linear demo requires optimizer dim={}, got {}",
                LINEAR_DEMO_PARAM_DIM, self.dim
            ));
        }
        if gens == 0 {
            return Err("runLinearDemo: gens must be > 0".into());
        }

        // masalah kecil deterministik: in=3, out=2, n=8
        let in_dim = LINEAR_DEMO_IN_DIM;
        let out_dim = LINEAR_DEMO_OUT_DIM;
        let n = 8usize;
        let w_true: [f32; LINEAR_DEMO_PARAM_DIM] = [0.7, -0.3, 0.2, 0.5, -0.8, 0.4];
        let mut x = Vec::with_capacity(n * in_dim);
        let mut y = Vec::with_capacity(n * out_dim);
        let mut r = Rng::new(12345); // rng terpisah & tetap untuk data
        for _ in 0..n {
            let row: Vec<f32> = (0..in_dim).map(|_| r.gaussian()).collect();
            let mut yo = vec![0.0f32; out_dim];
            for k in 0..in_dim {
                for o in 0..out_dim { yo[o] += row[k] * w_true[k * out_dim + o]; }
            }
            x.extend_from_slice(&row);
            y.extend_from_slice(&yo);
        }
        let obj = LinearMseObjective::new(x, y, n, in_dim, out_dim);

        for _ in 0..gens {
            let flat = self.ask();
            let nb = self.batch_size() as usize;
            let mut f = Vec::with_capacity(nb);
            for i in 0..nb {
                let cand = &flat[i * self.dim..(i + 1) * self.dim];
                f.push(obj.fitness(cand) as f32);
            }
            // Internal demo evaluates exactly the pending batch and must never swallow tell failures.
            self.tell(&f)?;
        }
        Ok(self.report())
    }
}

/// Core OpenES control boundary.
///
/// This method is intentionally outside the wasm-bindgen impl. Surface exposure
/// is staged separately after the state-continuity proof is green.
impl EsOptimizer {
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if self.awaiting_fitness {
            return Err(
                "set_learning_rate: cannot change learning rate while a candidate batch is pending; call tell() first"
                    .into(),
            );
        }
        self.strategy.set_learning_rate(lr)
    }
}

#[cfg(test)]
mod tests {
    use super::{es_capabilities, EsOptimizer};

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
        let mut openes =
            EsOptimizer::strict(3, 0, 7, Some(8), Some(0.08), Some(0.10)).unwrap();
        for lr in [0.0, -0.01, f32::NAN, f32::INFINITY] {
            assert!(openes.set_learning_rate(lr).is_err());
            assert_eq!(openes.generation(), 0);
        }

        let mut mu_lambda =
            EsOptimizer::strict(3, 1, 7, Some(8), Some(0.08), None).unwrap();
        let err = mu_lambda.set_learning_rate(0.10).unwrap_err();
        assert!(err.contains("only supported by OpenES"));
        assert_eq!(mu_lambda.generation(), 0);
    }

    #[test]
    fn learning_rate_mutation_rejects_pending_batch_without_consuming_it() {
        let mut optimizer =
            EsOptimizer::strict(3, 0, 11, Some(8), Some(0.08), Some(0.10)).unwrap();

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
        let mut control =
            EsOptimizer::strict(3, 0, 19, Some(8), Some(0.08), Some(0.10)).unwrap();
        let mut changed =
            EsOptimizer::strict(3, 0, 19, Some(8), Some(0.08), Some(0.10)).unwrap();

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

        // Learning rate does not participate in ask(). Preserved mean + RNG must
        // therefore produce an exactly identical next candidate batch.
        let second_control = control.ask();
        let second_changed = changed.ask();
        assert_eq!(second_control, second_changed);

        let second_fitness = [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.25, -0.25];
        control.tell(&second_fitness).unwrap();
        changed.tell(&second_fitness).unwrap();

        assert_eq!(control.generation(), 2);
        assert_eq!(changed.generation(), 2);
        assert_ne!(control.mean(), changed.mean());
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