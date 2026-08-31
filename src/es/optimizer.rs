use wasm_bindgen::prelude::*;
use super::diag::{diversity, mean_std, EsReport};
use super::objective::{LinearMseObjective, Objective};
use super::rng::Rng;
use super::strategy::{EsStrategy, Strategy};

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
    /// strategy: 0 = OpenEs antithetic, 1 = (mu,lambda).
    /// `pop` = jumlah pasangan (OpenEs) ATAU lambda (MuLambda); mu = pop/2 untuk MuLambda.
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
        let pop = pop.unwrap_or(64).max(2) as usize;
        let sigma = sigma.unwrap_or(0.1);
        let lr = lr.unwrap_or(0.05);
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
        for &v in &f64s { if v > best { best = v; } if v < worst { worst = v; } }

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

    /// Proof-of-life mandiri: latih W supaya X*W ≈ Y (plain Rust), kembalikan laporan akhir.
    /// Tidak butuh JS objective, tidak butuh burn. Berguna untuk "lihat ES bekerja" instan.
    #[wasm_bindgen(js_name = runLinearDemo)]
    pub fn run_linear_demo(&mut self, gens: u32) -> String {
        // masalah kecil deterministik: in=3, out=2, n=8
        let in_dim = 3usize;
        let out_dim = 2usize;
        let n = 8usize;
        let w_true: [f32; 6] = [0.7, -0.3, 0.2, 0.5, -0.8, 0.4];
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
            // Internal demo always evaluates exactly the batch returned by ask().
            let _ = self.tell(&f);
        }
        self.report()
    }
}
