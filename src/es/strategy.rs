use super::rng::Rng;

pub trait EsStrategy {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn sigma(&self) -> f32;
    fn lr(&self) -> f32;
    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>>;
    fn tell(&mut self, fitness: &[f64]);
    fn mean(&self) -> Vec<f32>;
}

// ---------------- OpenES (antithetic Gaussian, centered fitness shaping) ----------------
pub struct OpenEs {
    dim: usize,
    half: usize,
    sigma: f32,
    lr: f32,
    mean: Vec<f32>,
    eps: Vec<Vec<f32>>,
}

impl OpenEs {
    pub fn new(dim: usize, half: usize, sigma: f32, lr: f32, rng: &mut Rng) -> Self {
        let mean = (0..dim).map(|_| rng.gaussian() * 0.1).collect();
        Self { dim, half, sigma, lr, mean, eps: Vec::new() }
    }
}

impl EsStrategy for OpenEs {
    fn name(&self) -> &'static str { "openes_antithetic" }
    fn dim(&self) -> usize { self.dim }
    fn sigma(&self) -> f32 { self.sigma }
    fn lr(&self) -> f32 { self.lr }

    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>> {
        self.eps.clear();
        let mut cands = Vec::with_capacity(self.half * 2);
        for _ in 0..self.half {
            let e: Vec<f32> = (0..self.dim).map(|_| rng.gaussian() * self.sigma).collect();
            let plus: Vec<f32> = self.mean.iter().zip(e.iter()).map(|(m, &ei)| m + ei).collect();
            let minus: Vec<f32> = self.mean.iter().zip(e.iter()).map(|(m, &ei)| m - ei).collect();
            self.eps.push(e);
            cands.push(plus);
            cands.push(minus);
        }
        cands
    }

    fn tell(&mut self, fitness: &[f64]) {
        debug_assert_eq!(fitness.len(), self.half * 2);
        let (m, s) = super::diag::mean_std(fitness);
        let std = if s > 1e-8 { s } else { 1e-8 };
        let centered: Vec<f64> = fitness.iter().map(|&r| (r - m) / std).collect();
        let denom = 2.0 * self.half as f64 * self.sigma as f64;
        let mut g = vec![0.0f64; self.dim];
        for j in 0..self.half {
            let diff = centered[2 * j] - centered[2 * j + 1];
            for d in 0..self.dim { g[d] += diff * self.eps[j][d] as f64; }
        }
        for d in 0..self.dim { self.mean[d] += (self.lr as f64) * (g[d] / denom) as f64; }
    }

    fn mean(&self) -> Vec<f32> { self.mean.clone() }
}

// ---------------- (mu, lambda) elitist Gaussian mutation ----------------
pub struct MuLambda {
    dim: usize,
    mu: usize,
    lambda: usize,
    sigma: f32,
    parents: Vec<Vec<f32>>,
    last_children: Vec<Vec<f32>>,
}

impl MuLambda {
    pub fn new(dim: usize, mu: usize, lambda: usize, sigma: f32, rng: &mut Rng) -> Self {
        let parents = (0..mu)
            .map(|_| (0..dim).map(|_| rng.gaussian() * 0.5).collect())
            .collect();
        Self { dim, mu, lambda, sigma, parents, last_children: Vec::new() }
    }
}

impl EsStrategy for MuLambda {
    fn name(&self) -> &'static str { "mu_lambda" }
    fn dim(&self) -> usize { self.dim }
    fn sigma(&self) -> f32 { self.sigma }
    fn lr(&self) -> f32 { 0.0 } // tidak ada lr eksplisit; dilaporkan 0 supaya JSON seragam

    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>> {
        let mut cands = Vec::with_capacity(self.lambda);
        for i in 0..self.lambda {
            let p = &self.parents[i % self.mu];
            let child: Vec<f32> = p.iter().map(|&v| v + rng.gaussian() * self.sigma).collect();
            cands.push(child);
        }
        self.last_children = cands.clone();
        cands
    }

    fn tell(&mut self, fitness: &[f64]) {
        debug_assert_eq!(fitness.len(), self.lambda);
        let mut pairs: Vec<(f64, Vec<f32>)> =
            fitness.iter().copied().zip(self.last_children.iter().cloned()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal)); // desc
        self.parents = pairs.into_iter().take(self.mu).map(|(_, c)| c).collect();
    }

    fn mean(&self) -> Vec<f32> {
        if self.parents.is_empty() { return vec![0.0; self.dim]; }
        let mut m = vec![0.0f64; self.dim];
        for p in &self.parents { for d in 0..self.dim { m[d] += p[d] as f64; } }
        for d in 0..self.dim { m[d] /= self.parents.len() as f64; }
        m.into_iter().map(|v| v as f32).collect()
    }
}

// ---------------- enum dispatch (wasm-safe, tanpa dyn) ----------------
pub enum Strategy {
    OpenEs(OpenEs),
    MuLambda(MuLambda),
}

impl Strategy {
    pub fn openes(dim: usize, half: usize, sigma: f32, lr: f32, rng: &mut Rng) -> Self {
        Strategy::OpenEs(OpenEs::new(dim, half, sigma, lr, rng))
    }
    pub fn mu_lambda(dim: usize, mu: usize, lambda: usize, sigma: f32, rng: &mut Rng) -> Self {
        Strategy::MuLambda(MuLambda::new(dim, mu, lambda, sigma, rng))
    }
}

impl EsStrategy for Strategy {
    fn name(&self) -> &'static str { match self { Strategy::OpenEs(s) => s.name(), Strategy::MuLambda(s) => s.name() } }
    fn dim(&self) -> usize { match self { Strategy::OpenEs(s) => s.dim(), Strategy::MuLambda(s) => s.dim() } }
    fn sigma(&self) -> f32 { match self { Strategy::OpenEs(s) => s.sigma(), Strategy::MuLambda(s) => s.sigma() } }
    fn lr(&self) -> f32 { match self { Strategy::OpenEs(s) => s.lr(), Strategy::MuLambda(s) => s.lr() } }
    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>> { match self { Strategy::OpenEs(s) => s.ask(rng), Strategy::MuLambda(s) => s.ask(rng) } }
    fn tell(&mut self, fitness: &[f64]) { match self { Strategy::OpenEs(s) => s.tell(fitness), Strategy::MuLambda(s) => s.tell(fitness) } }
    fn mean(&self) -> Vec<f32> { match self { Strategy::OpenEs(s) => s.mean(), Strategy::MuLambda(s) => s.mean() } }
        }
