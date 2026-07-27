// Objective = "seberapa bagus satu vektor bobot". ES tidak peduli objective-nya apa.
// Slice ini: objective bawaan plain-Rust (MSE linear) untuk proof-of-life & test.
// Slice compose nanti: impl Objective yang menjalankan graph burn-mu (lihat peta di akhir).

pub trait Objective {
    /// Lebih besar = lebih baik (ES memaksimalkan).
    fn fitness(&self, params: &[f32]) -> f64;
}

/// y = X * W ; fitness = -mean((pred - y)^2). Plain Rust, deterministik, tanpa burn.
pub struct LinearMseObjective {
    pub x: Vec<f32>,          // [n * in_dim]
    pub y: Vec<f32>,          // [n * out_dim]
    pub n: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LinearMseObjective {
    pub fn new(x: Vec<f32>, y: Vec<f32>, n: usize, in_dim: usize, out_dim: usize) -> Self {
        Self { x, y, n, in_dim, out_dim }
    }
}

impl Objective for LinearMseObjective {
    fn fitness(&self, w: &[f32]) -> f64 {
        if w.len() != self.in_dim * self.out_dim {
            return f64::NEG_INFINITY;
        }
        let mut mse = 0.0f64;
        let mut count = 0usize;
        for i in 0..self.n {
            for o in 0..self.out_dim {
                let mut pred = 0.0f64;
                for k in 0..self.in_dim {
                    pred += self.x[i * self.in_dim + k] as f64 * w[k * self.out_dim + o] as f64;
                }
                let target = self.y[i * self.out_dim + o] as f64;
                let err = pred - target;
                mse += err * err;
                count += 1;
            }
        }
        if count == 0 { return f64::NEG_INFINITY; }
        -(mse / count as f64) // negasi: ES memaksimalkan
    }
              }
