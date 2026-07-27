// PRNG deterministik (mulberry32) + Gaussian via Box-Muller.
// Nol dependency baru. Seed -> hasil identik di host & wasm (penting untuk repro & test).
use core::f32::consts::PI;

pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self { state: if seed == 0 { 0x9E37_79B9 } else { seed } }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    /// Uniform di [0, 1).
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) // / 2^24
    }

    /// Gaussian N(0,1).
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // hindari ln(0)
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }
}
