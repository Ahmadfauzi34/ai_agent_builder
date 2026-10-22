// PRNG deterministik (mulberry32) + Gaussian via Box-Muller.
// Nol dependency baru. Seed -> hasil identik di host & wasm (penting untuk repro & test).
use core::f32::consts::PI;

pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9 } else { seed },
        }
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

// ============================================================
// #14 — varian PRNG tambahan. `Rng` (mulberry32) di atas TIDAK
// diubah: aliran deterministik yang sudah ada harus tetap
// byte-identik. Varian baru bersifat aditif.
// ============================================================

/// PCG-XSH-RR 64/32 (O'Neill 2014): 64-bit LCG + permutasi
/// xorshift/high-random-rotation. State kecil, cepat, kualitas
/// statistik jauh di atas mulberry32 (lolos BigCrush).
///
/// Referensi: https://www.pcg-random.org/
/// Vektor uji: seed (initstate=42, initseq=54) -> output pertama
/// 0xa15c02b7 (dari implementasi referensi pcg-c-basic).
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULT: u64 = 6364136223846793005;

    /// Seeding penuh gaya pcg-c-basic (`pcg32_srandom_r`):
    /// state=0, inc=(seq<<1)|1, majukan sekali, tambah seed, majukan lagi.
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut r = Self {
            state: 0,
            inc: (seq << 1) | 1,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// Ergonomis: satu seed u32 seperti RNG lain di modul ini.
    /// `seq` dikunci ke konstanta yang didokumentasikan.
    pub fn from_seed_u32(seed: u32) -> Self {
        Self::new(seed as u64, 0x853c_49e6_748f_ea9b)
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(Self::MULT)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform di [0, 1). Semantik identik dengan `Rng::uniform`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) // / 2^24
    }

    /// Gaussian N(0,1) via Box-Muller. Semantik identik dengan `Rng::gaussian`.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // hindari ln(0)
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }
}

/// Philox4x32-10 (Salmon–Moraes–Dror–Shaw 2011, Random123):
/// PRNG *counter-based*. Tidak ada state yang dimajukan berurutan;
/// output = F(key, counter). Sifat kuncinya: dua counter berbeda
/// memberi dua aliran independen yang reprodusibel tanpa koordinasi
/// — fondasi untuk ES paralel di masa depan.
///
/// Satu pemanggilan menghasilkan 4×u32; `next_u32` menyangga
/// (buffer) ketiganya. `seek` melompat ke counter arbitrer.
///
/// Referensi: https://www.thesalmons.org/john/random123/
/// Vektor uji (10 round): key=[0,0], ctr=[0,0,0,0] ->
/// [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8].
pub struct Philox4x32 {
    key: [u32; 2],
    counter: [u32; 4],
    buf: [u32; 4],
    idx: usize, // posisi baca berikutnya di buf; 4 = penyangga kosong
}

impl Philox4x32 {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    fn round(c: &mut [u32; 4], k: &[u32; 2]) {
        // mulhilo 32-bit: p = a*b sebagai u64, lalu split hi/lo.
        let p0 = (c[0] as u64) * (Self::M0 as u64);
        let p1 = (c[2] as u64) * (Self::M1 as u64);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
    }

    fn philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let mut c = counter;
        let mut k = key;
        for _ in 0..10 {
            Self::round(&mut c, &k);
            // bumpkey di antara round (bump terakhir tidak dipakai).
            k[0] = k[0].wrapping_add(Self::W0);
            k[1] = k[1].wrapping_add(Self::W1);
        }
        c
    }

    fn bump_counter(counter: &mut [u32; 4]) {
        let mut carry: u64 = 1;
        for w in counter.iter_mut() {
            let s = (*w as u64) + carry;
            *w = s as u32;
            carry = s >> 32;
            if carry == 0 {
                break;
            }
        }
    }

    /// Key eksplisit — untuk vektor uji publikasi & kontrol penuh.
    pub fn with_key(key: [u32; 2]) -> Self {
        Self::with_key_counter(key, [0; 4])
    }

    /// Key + counter penuh eksplisit. Untuk vektor uji publikasi dan
    /// untuk agen yang butuh kontrol counter 128-bit penuh (di luar
    /// jangkauan `seek` 64-bit).
    pub fn with_key_counter(key: [u32; 2], counter: [u32; 4]) -> Self {
        Self {
            key,
            counter,
            buf: [0; 4],
            idx: 4,
        }
    }

    /// Satu seed u32: key = splitmix32(seed) ×2, counter = 0.
    /// Deterministik dan didokumentasikan; bukan vektor publikasi.
    pub fn new(seed: u32) -> Self {
        fn splitmix32(x: u32) -> u32 {
            let mut z = x.wrapping_add(0x9E37_79B9);
            z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
            z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
            z ^ (z >> 15)
        }
        Self::with_key([splitmix32(seed), splitmix32(seed ^ 0x9E37_79B9)])
    }

    /// Lompat ke aliran counter `ctr_lo` (64 bit rendah; 64 bit tinggi = 0).
    /// Dua `seek` berbeda = dua aliran independen yang reprodusibel.
    pub fn seek(&mut self, ctr_lo: u64) {
        self.counter = [(ctr_lo as u32), ((ctr_lo >> 32) as u32), 0, 0];
        self.idx = 4; // buang penyangga lama
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 4 {
            self.buf = Self::philox(self.counter, self.key);
            Self::bump_counter(&mut self.counter);
            self.idx = 0;
        }
        let v = self.buf[self.idx];
        self.idx += 1;
        v
    }

    /// Uniform di [0, 1). Semantik identik dengan `Rng::uniform`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) // / 2^24
    }

    /// Gaussian N(0,1) via Box-Muller. Semantik identik dengan `Rng::gaussian`.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // hindari ln(0)
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }
}

/// Pilihan PRNG untuk `EsOptimizer`. Nilai diskriminan = yang dipakai
/// di permukaan WASM (`withRng(kind)`): 0=mulberry32, 1=pcg32, 2=philox.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RngKind {
    Mulberry32 = 0,
    Pcg32 = 1,
    Philox = 2,
}

impl RngKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Mulberry32),
            1 => Some(Self::Pcg32),
            2 => Some(Self::Philox),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mulberry32 => "mulberry32",
            Self::Pcg32 => "pcg32",
            Self::Philox => "philox4x32",
        }
    }
}

/// Enum dispatch di atas ketiga PRNG: strategi ES tetap menerima SATU
/// tipe konkret (`&mut AnyRng`) tanpa generic. Default = mulberry32
/// agar perilaku lama tidak berubah.
pub enum AnyRng {
    Mulberry32(Rng),
    Pcg32(Pcg32),
    Philox(Philox4x32),
}

impl AnyRng {
    pub fn new(kind: RngKind, seed: u32) -> Self {
        match kind {
            RngKind::Mulberry32 => Self::Mulberry32(Rng::new(seed)),
            RngKind::Pcg32 => Self::Pcg32(Pcg32::from_seed_u32(seed)),
            RngKind::Philox => Self::Philox(Philox4x32::new(seed)),
        }
    }

    pub fn kind(&self) -> RngKind {
        match self {
            Self::Mulberry32(_) => RngKind::Mulberry32,
            Self::Pcg32(_) => RngKind::Pcg32,
            Self::Philox(_) => RngKind::Philox,
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        match self {
            Self::Mulberry32(r) => r.next_u32(),
            Self::Pcg32(r) => r.next_u32(),
            Self::Philox(r) => r.next_u32(),
        }
    }

    pub fn uniform(&mut self) -> f32 {
        match self {
            Self::Mulberry32(r) => r.uniform(),
            Self::Pcg32(r) => r.uniform(),
            Self::Philox(r) => r.uniform(),
        }
    }

    pub fn gaussian(&mut self) -> f32 {
        match self {
            Self::Mulberry32(r) => r.gaussian(),
            Self::Pcg32(r) => r.gaussian(),
            Self::Philox(r) => r.gaussian(),
        }
    }

    /// Khusus Philox: lompat ke aliran counter independen.
    /// Mengembalikan Err untuk varian lain (bukan silent no-op).
    pub fn philox_seek(&mut self, ctr_lo: u64) -> Result<(), String> {
        match self {
            Self::Philox(r) => {
                r.seek(ctr_lo);
                Ok(())
            }
            other => Err(format!(
                "philox_seek only valid for philox, got {}",
                other.kind().name()
            )),
        }
    }
}

#[cfg(test)]
mod rng14_tests {
    use super::*;

    #[test]
    fn pcg32_matches_published_vector() {
        // Vektor dari pcg-c-basic: pcg32_srandom(42, 54).
        let mut r = Pcg32::new(42, 54);
        let expected = [
            0xa15c02b7u32,
            0x7b47f409,
            0xba1d3330,
            0x83d2f293,
            0xbfa4784b,
        ];
        for &e in &expected {
            assert_eq!(r.next_u32(), e, "pcg32 divergen dari vektor publikasi");
        }
    }

    #[test]
    fn philox4x32_matches_published_vectors() {
        // Diverifikasi ganda: implementasi Python independen dari spec
        // + tabel nilai referensi publikasi (key=0/ctr=0 dan all-F cocok).
        let mut r = Philox4x32::with_key([0, 0]);
        assert_eq!(
            [r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()],
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        );
        let mut r = Philox4x32::with_key_counter(
            [0xFFFF_FFFF, 0xFFFF_FFFF],
            [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
        );
        assert_eq!(
            [r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()],
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
        );
        // Vektor ketiga dari oracle Python: key=[1,2], ctr=[3,4,5,6].
        let mut r = Philox4x32::with_key_counter([1, 2], [3, 4, 5, 6]);
        assert_eq!(
            [r.next_u32(), r.next_u32(), r.next_u32(), r.next_u32()],
            [0x8850da9c, 0xc159ff4c, 0x75752dca, 0x7d607cf9]
        );
    }

    #[test]
    fn philox_seek_gives_independent_reproducible_streams() {
        // Sifat inti counter-based: seek(n) lalu baca == jalan sekuensial
        // yang sudah maju 4*n output, terlepas dari urutan eksekusi.
        let mut seq = Philox4x32::new(7);
        let seq_out: Vec<u32> = (0..12).map(|_| seq.next_u32()).collect();

        let mut jumped = Philox4x32::new(7);
        jumped.seek(2); // counter=2 -> output ke-8..11 dari aliran sekuensial
        let jumped_out: Vec<u32> = (0..4).map(|_| jumped.next_u32()).collect();
        assert_eq!(jumped_out, seq_out[8..12]);

        // Reprodusibel: seed sama + seek sama = aliran sama.
        let mut again = Philox4x32::new(7);
        again.seek(2);
        let again_out: Vec<u32> = (0..4).map(|_| again.next_u32()).collect();
        assert_eq!(again_out, jumped_out);
    }

    #[test]
    fn all_variants_deterministic_and_distinct() {
        for kind in [RngKind::Mulberry32, RngKind::Pcg32, RngKind::Philox] {
            let mut a = AnyRng::new(kind, 123);
            let mut b = AnyRng::new(kind, 123);
            let sa: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
            let sb: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
            assert_eq!(sa, sb, "{:?} tidak deterministik", kind);
            assert_eq!(a.kind(), kind);
        }
        // Tiga varian dengan seed sama memberi aliran BERBEDA.
        let outs: Vec<Vec<u32>> = [RngKind::Mulberry32, RngKind::Pcg32, RngKind::Philox]
            .iter()
            .map(|&k| {
                let mut r = AnyRng::new(k, 123);
                (0..8).map(|_| r.next_u32()).collect()
            })
            .collect();
        assert_ne!(outs[0], outs[1]);
        assert_ne!(outs[0], outs[2]);
        assert_ne!(outs[1], outs[2]);
    }

    #[test]
    fn uniform_range_and_gaussian_finite_all_variants() {
        for kind in [RngKind::Mulberry32, RngKind::Pcg32, RngKind::Philox] {
            let mut r = AnyRng::new(kind, 999);
            for _ in 0..1000 {
                let u = r.uniform();
                assert!(u >= 0.0 && u < 1.0, "{:?} uniform keluar [0,1)", kind);
                assert!(r.gaussian().is_finite(), "{:?} gaussian tidak finite", kind);
            }
        }
    }

    #[test]
    fn rng_kind_rejects_unknown() {
        assert_eq!(RngKind::from_u8(0), Some(RngKind::Mulberry32));
        assert_eq!(RngKind::from_u8(1), Some(RngKind::Pcg32));
        assert_eq!(RngKind::from_u8(2), Some(RngKind::Philox));
        assert_eq!(RngKind::from_u8(3), None);
        assert_eq!(RngKind::from_u8(255), None);
    }

    #[test]
    fn philox_seek_rejected_for_other_variants() {
        let mut r = AnyRng::new(RngKind::Pcg32, 1);
        assert!(r.philox_seek(0).is_err());
        let mut r = AnyRng::new(RngKind::Mulberry32, 1);
        assert!(r.philox_seek(0).is_err());
        let mut r = AnyRng::new(RngKind::Philox, 1);
        assert!(r.philox_seek(0).is_ok());
    }

    #[test]
    fn mulberry32_stream_unchanged() {
        // Regresi: Rng lama tidak boleh berubah oleh penambahan varian.
        // Nilai hardcoded dihitung dari implementasi mulberry32 via
        // oracle Python independen (bukan dari kode yang diuji).
        let mut r = Rng::new(7);
        assert_eq!(r.next_u32(), 0x02ff_152c);
        let mut r2 = Rng::new(7);
        assert_eq!(r2.next_u32(), 0x02ff_152c);
    }
}
