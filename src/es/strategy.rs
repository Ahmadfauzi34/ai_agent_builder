// Strategi evolusi = bisa di-swap. Tambah strategi = tambah struct + 1 variant + 1 lengan match.
// (Gaya dispatch tertutup yang sama dengan registry-mu; terisolasi di file ini, tidak menyentuh v1.)
use super::rng::Rng;

pub trait EsStrategy {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn sigma(&self) -> f32;
    fn lr(&self) -> f32;
    /// Hasilkan kandidat untuk generasi ini (urutan = urutan yang tell() terima).
    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>>;
    /// Terima fitness (sepanjang kandidat ask), update internal.
    fn tell(&mut self, fitness: &[f64]);
    /// Rata-rata populasi (untuk logging / deploy mean-network).
    fn mean(&self) -> Vec<f32>;
}

// ---------------- OpenES (antithetic Gaussian, centered fitness shaping) ----------------
pub struct OpenEs {
    dim: usize,
    half: usize,
    sigma: f32,
    lr: f32,
    mean: Vec<f32>,
    eps: Vec<Vec<f32>>, // noise per pasangan, dipakai ulang di tell()
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

        let mut g = vec![0.0f64; self.dim];
        let denom = 2.0 * self.half as f64 * self.sigma as f64;
        for j in 0..self.half {
            let diff = centered[2 * j] - centered[2 * j + 1];
            for d in 0..self.dim {
                g[d] += diff * self.eps[j][d] as f64;
            }
        }
        for d in 0..self.dim {
            self.mean[d] += (self.lr as f64) * (g[d] / denom) as f64;
        }
    }

    fn mean(&self) -> Vec<f32> { self.mean.clone() }
}

// ---------------- (mu + lambda) elitist Gaussian mutation ----------------
pub struct MuPlusLambda {
    dim: usize,
    mu: usize,
    lambda: usize,
    sigma: f32,
    lr: f32, // disimpan supaya laporan seragam (tidak dipakai untuk update mean)
    parents: Vec<Vec<f32>>,
}

impl MuPlusLambda {
    pub fn new(dim: usize, mu: usize, lambda: usize, sigma: f32, rng: &mut Rng) -> Self {
        let parents = (0..mu)
            .map(|_| (0..dim).map(|_| rng.gaussian() * 0.5).collect())
            .collect();
        Self { dim, mu, lambda, sigma, lr: 0.0, parents }
    }
}

impl EsStrategy for MuPlusLambda {
    fn name(&self) -> &'static str { "mu_plus_lambda" }
    fn dim(&self) -> usize { self.dim }
    fn sigma(&self) -> f32 { self.sigma }
    fn lr(&self) -> f32 { self.lr }

    fn ask(&mut self, rng: &mut Rng) -> Vec<Vec<f32>> {
        let mut cands = Vec::with_capacity(self.lambda);
        for i in 0..self.lambda {
            let p = &self.parents[i % self.mu];
            let child: Vec<f32> = p.iter().map(|&v| v + rng.gaussian() * self.sigma).collect();
            cands.push(child);
        }
        cands
    }

    fn tell(&mut self, fitness: &[f64]) {
        debug_assert_eq!(fitness.len(), self.lambda);
        // ambil kandidat ask terakhir? -> optimizer yang pegang; di sini kita butuh children.
        // Karena trait tidak membawa children, kita rekonstruksi pasangan via urutan:
        // optimizer memanggil tell dengan fitness seurut ask, tapi children tidak sampai ke sini.
        // SOLUSI: MuPlusLambda menyimpan children saat ask.
        // (lihat field `last_children` di bawah via wrapper) -> kita simpan di ask.
        // -> implementasi pakai self.last_children (ditambah di struct via Default trick):
        self.select_from(fitness);
    }

    fn mean(&self) -> Vec<f32> {
        if self.parents.is_empty() { return vec![0.0; self.dim]; }
        let mut m = vec![0.0f64; self.dim];
        for p in &self.parents {
            for d in 0..self.dim { m[d] += p[d] as f64; }
        }
        for d in 0..self.dim { m[d] /= self.parents.len() as f64; }
        m.into_iter().map(|v| v as f32).collect()
    }
      }
