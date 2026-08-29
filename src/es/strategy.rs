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
        let expected = self.half.saturating_mul(2);
        if fitness.len() != expected
            || self.eps.len() != self.half
            || self.eps.iter().any(|e| e.len() != self.dim)
            || fitness.iter().any(|v| !v.is_finite())
            || !self.sigma.is_finite()
            || self.sigma <= 0.0
            || !self.lr.is_finite()
        {
            return;
        }

        let (m, s) = super::diag::mean_std(fitness);
        let std = if s > 1e-8 { s } else { 1e-8 };
        let centered: Vec<f64> = fitness.iter().map(|&r| (r - m) / std).collect();
        let denom = 2.0 * self.half as f64 * self.sigma as f64;
        let mut g = vec![0.0f64; self.dim];
        for j in 0..self.half {
            let diff = centered[2 * j] - centered[2 * j + 1];
            for d in 0..self.dim { g[d] += diff * self.eps[j][d] as f64; }
        }
        for d in 0..self.dim { self.mean[d] += ((self.lr as f64) * (g[d] / denom)) as f32; }

        // One ask batch may be consumed only once. Invalid tell calls above do not consume it,
        // so callers can correct the fitness vector and retry safely.
        self.eps.clear();
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
        if self.mu == 0 || self.parents.len() != self.mu || !self.sigma.is_finite() {
            self.last_children.clear();
            return Vec::new();
        }

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
        if fitness.len() != self.lambda
            || self.last_children.len() != self.lambda
            || self.last_children.iter().any(|c| c.len() != self.dim)
            || fitness.iter().any(|v| !v.is_finite())
        {
            return;
        }

        let mut pairs: Vec<(f64, Vec<f32>)> =
            fitness.iter().copied().zip(self.last_children.iter().cloned()).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal)); // desc
        self.parents = pairs.into_iter().take(self.mu).map(|(_, c)| c).collect();
        self.last_children.clear();
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

#[cfg(test)]
mod tests {
    use super::{EsStrategy, MuLambda, OpenEs};
    use crate::es::rng::Rng;

    #[test]
    fn openes_tell_before_ask_is_noop() {
        let mut rng = Rng::new(7);
        let mut strategy = OpenEs::new(3, 2, 0.2, 0.1, &mut rng);
        let before = strategy.mean();
        strategy.tell(&[1.0, 0.0, -1.0, -2.0]);
        assert_eq!(strategy.mean(), before);
    }

    #[test]
    fn openes_non_finite_fitness_does_not_mutate_or_consume_batch() {
        let mut rng = Rng::new(8);
        let mut strategy = OpenEs::new(2, 2, 0.2, 0.1, &mut rng);
        let _ = strategy.ask(&mut rng);
        let before = strategy.mean();
        strategy.tell(&[1.0, f64::NAN, 0.0, -1.0]);
        assert_eq!(strategy.mean(), before);
        assert_eq!(strategy.eps.len(), 2);

        strategy.tell(&[2.0, -2.0, 1.0, -1.0]);
        assert_ne!(strategy.mean(), before);
        assert!(strategy.eps.is_empty());
    }

    #[test]
    fn openes_valid_batch_is_single_use() {
        let mut rng = Rng::new(9);
        let mut strategy = OpenEs::new(2, 2, 0.2, 0.1, &mut rng);
        let _ = strategy.ask(&mut rng);
        strategy.tell(&[2.0, -2.0, 1.0, -1.0]);
        let after_first = strategy.mean();
        strategy.tell(&[2.0, -2.0, 1.0, -1.0]);
        assert_eq!(strategy.mean(), after_first);
    }

    #[test]
    fn mu_lambda_tell_before_ask_is_noop() {
        let mut rng = Rng::new(10);
        let mut strategy = MuLambda::new(3, 2, 4, 0.2, &mut rng);
        let before = strategy.mean();
        strategy.tell(&[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(strategy.mean(), before);
    }

    #[test]
    fn mu_lambda_non_finite_fitness_does_not_mutate_or_consume_batch() {
        let mut rng = Rng::new(11);
        let mut strategy = MuLambda::new(2, 2, 4, 0.2, &mut rng);
        let _ = strategy.ask(&mut rng);
        let before = strategy.mean();
        strategy.tell(&[4.0, f64::INFINITY, 2.0, 1.0]);
        assert_eq!(strategy.mean(), before);
        assert_eq!(strategy.last_children.len(), 4);

        strategy.tell(&[4.0, 3.0, 2.0, 1.0]);
        assert!(strategy.last_children.is_empty());
    }

    #[test]
    fn mu_lambda_valid_batch_is_single_use() {
        let mut rng = Rng::new(12);
        let mut strategy = MuLambda::new(2, 2, 4, 0.2, &mut rng);
        let _ = strategy.ask(&mut rng);
        strategy.tell(&[4.0, 3.0, 2.0, 1.0]);
        let after_first = strategy.mean();
        strategy.tell(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(strategy.mean(), after_first);
    }

    #[test]
    fn mu_lambda_zero_parent_configuration_does_not_panic_on_ask() {
        let mut rng = Rng::new(13);
        let mut strategy = MuLambda::new(2, 0, 4, 0.2, &mut rng);
        assert!(strategy.ask(&mut rng).is_empty());
    }
}
