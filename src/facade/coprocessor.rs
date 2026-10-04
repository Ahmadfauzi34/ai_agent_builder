//! Fasad WASM tunggal — domain `coprocessor` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::coprocessor::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::coprocessor::verify_vectors_report;

// Opsi C Fase 2: imports untuk #[wasm_bindgen] impl EsOptimizer yang pindah ke sini.
use crate::es::diag::{diversity, mean_std, EsReport};
use crate::es::objective::{LinearMseObjective, Objective};
use crate::es::optimizer::{
    strict_config, EsOptimizer, DEFAULT_LR, DEFAULT_POP, DEFAULT_SIGMA, LINEAR_DEMO_IN_DIM,
    LINEAR_DEMO_OUT_DIM, LINEAR_DEMO_PARAM_DIM,
};
use crate::es::rng::Rng;
use crate::es::strategy::{EsStrategy, Strategy};

/// Compare an external implementation result with a trusted numerical reference.
///
/// This is intentionally dependency-free and returns compact JSON so an agent can
/// consume the proof result without coupling the produced artifact to this WASM runtime.
#[wasm_bindgen(js_name = mathVerifyVectors)]
pub fn math_verify_vectors(
    reference: &[f32],
    candidate: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> Result<String, String> {
    verify_vectors_report(reference, candidate, abs_tol, rel_tol)
}

// ============================================================
// Opsi C Fase 2 — pindahan murni dari `src/es/optimizer.rs`:
// #[wasm_bindgen] impl EsOptimizer (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

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
    pub fn dim(&self) -> u32 {
        self.dim as u32
    }

    #[wasm_bindgen(js_name = generation)]
    pub fn generation(&self) -> u32 {
        self.gen
    }

    #[wasm_bindgen(js_name = batchSize)]
    pub fn batch_size(&self) -> u32 {
        self.last_candidates.len() as u32
    }

    /// Change OpenES learning rate without resetting search state.
    ///
    /// Mutation is valid only between completed generations. If an ask() batch
    /// is still awaiting fitness, reject without consuming or replacing it.
    #[wasm_bindgen(js_name = setLearningRate)]
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), String> {
        if self.awaiting_fitness {
            return Err(
                "setLearningRate: cannot change learning rate while a candidate batch is pending; call tell() first"
                    .into(),
            );
        }
        self.strategy.set_learning_rate(lr)
    }

    /// Minta kandidat generasi ini. Mengembalikan Float32Array flat (n_kandidat * dim).
    /// JS slice per `dim()`. Panggil `tell()` sesudahnya dengan fitness seurut kandidat.
    /// Memanggil ask lagi sebelum tell diperbolehkan: batch sebelumnya dianggap dibatalkan.
    pub fn ask(&mut self) -> Vec<f32> {
        let cands = self.strategy.ask(&mut self.rng);
        let mut flat = Vec::with_capacity(cands.len() * self.dim);
        for c in &cands {
            flat.extend_from_slice(c);
        }
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
            if v > best {
                best = v;
            }
            if v < worst {
                worst = v;
            }
        }

        // global best + stagnation (scan kandidat vs fitness)
        let mut gen_best = f64::NEG_INFINITY;
        let mut gen_best_params: Vec<f32> = Vec::new();
        for (c, &v) in self.last_candidates.iter().zip(f64s.iter()) {
            if v > gen_best {
                gen_best = v;
                gen_best_params = c.clone();
            }
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
        let mean_norm = (mean_vec
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>())
        .sqrt();
        let best_norm = (self
            .best_params
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>())
        .sqrt();

        // flags
        let mut flags: Vec<String> = Vec::new();
        if div < 1e-6 {
            flags.push("DIVERSITY_COLLAPSE".into());
        }
        if improvement <= 1e-8 {
            flags.push("NO_IMPROVEMENT".into());
        }
        if std < 1e-9 {
            flags.push("ALL_FITNESS_EQUAL".into());
        }

        self.gen = self.gen.saturating_add(1);
        self.awaiting_fitness = false;

        let rep = EsReport {
            gen: self.gen,
            strategy: self.strategy.name(),
            evals: f64s.len(),
            dim: self.dim,
            pop: self.last_candidates.len(),
            best,
            worst,
            mean,
            std,
            improvement,
            stagnation: self.stagnation,
            diversity: div,
            sigma: self.strategy.sigma(),
            lr: self.strategy.lr(),
            mean_norm,
            best_norm,
            flags,
        };
        let json = rep.to_json();
        self.last_report = json.clone();
        Ok(json)
    }

    /// Vektor terbaik sepanjang pelatihan (Float32Array).
    pub fn best(&self) -> Vec<f32> {
        self.best_params.clone()
    }

    /// Rata-rata populasi saat ini (Float32Array).
    pub fn mean(&self) -> Vec<f32> {
        self.strategy.mean()
    }

    /// Laporan JSON generasi terakhir.
    pub fn report(&self) -> String {
        self.last_report.clone()
    }

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
                for o in 0..out_dim {
                    yo[o] += row[k] * w_true[k * out_dim + o];
                }
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
