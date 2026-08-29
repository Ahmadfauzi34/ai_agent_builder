// Diagnosa = pure Rust + JSON string. Menambah metrik = tambah field + 1 baris di to_json().

pub fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut s = 0.0f64;
    for &x in v {
        s += x;
    }
    let m = s / n as f64;
    let mut var = 0.0f64;
    for &x in v {
        let d = x - m;
        var += d * d;
    }
    (m, (var / n as f64).sqrt())
}

/// Rata-rata std per-dimensi antar kandidat. Mengukur "collapse" populasi.
///
/// Populasi ragged bukan state yang valid untuk ES. Alih-alih indexing panic,
/// kembalikan NaN agar lapisan laporan dapat merepresentasikannya sebagai null.
pub fn diversity(cands: &[Vec<f32>]) -> f64 {
    let n = cands.len();
    if n < 2 {
        return 0.0;
    }
    let dim = cands[0].len();
    if dim == 0 {
        return 0.0;
    }
    if cands.iter().any(|c| c.len() != dim) {
        return f64::NAN;
    }

    let mut sum_std = 0.0f64;
    for d in 0..dim {
        let mut s = 0.0f64;
        for c in cands {
            s += c[d] as f64;
        }
        let m = s / n as f64;
        let mut var = 0.0f64;
        for c in cands {
            let diff = c[d] as f64 - m;
            var += diff * diff;
        }
        sum_std += (var / n as f64).sqrt();
    }
    sum_std / dim as f64
}

fn fjson(f: f64) -> String {
    if f.is_finite() {
        format!("{f}")
    } else {
        "null".into()
    }
}

pub struct EsReport {
    pub gen: u32,
    pub strategy: &'static str,
    pub evals: usize,
    pub dim: usize,
    pub pop: usize,
    pub best: f64,
    pub worst: f64,
    pub mean: f64,
    pub std: f64,
    pub improvement: f64,
    pub stagnation: u32,
    pub diversity: f64,
    pub sigma: f32,
    pub lr: f32,
    pub mean_norm: f64,
    pub best_norm: f64,
    pub flags: Vec<String>,
}

impl EsReport {
    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        s.push_str(&format!("\"gen\":{},", self.gen));
        s.push_str(&format!("\"strategy\":\"{}\",", self.strategy));
        s.push_str(&format!("\"evals\":{},", self.evals));
        s.push_str(&format!("\"dim\":{},", self.dim));
        s.push_str(&format!("\"pop\":{},", self.pop));
        s.push_str(&format!("\"best\":{},", fjson(self.best)));
        s.push_str(&format!("\"worst\":{},", fjson(self.worst)));
        s.push_str(&format!("\"mean\":{},", fjson(self.mean)));
        s.push_str(&format!("\"std\":{},", fjson(self.std)));
        s.push_str(&format!("\"improvement\":{},", fjson(self.improvement)));
        s.push_str(&format!("\"stagnation\":{},", self.stagnation));
        s.push_str(&format!("\"diversity\":{},", fjson(self.diversity)));
        s.push_str(&format!("\"sigma\":{},", fjson(self.sigma as f64)));
        s.push_str(&format!("\"lr\":{},", fjson(self.lr as f64)));
        s.push_str(&format!("\"mean_norm\":{},", fjson(self.mean_norm)));
        s.push_str(&format!("\"best_norm\":{},", fjson(self.best_norm)));
        s.push_str("\"flags\":[");
        for (i, f) in self.flags.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(f);
            s.push('"');
        }
        s.push_str("]}");
        s
    }
}
