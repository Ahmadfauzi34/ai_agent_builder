# Pembobotan Manual — domain pack (2026-10-16)

Engine tidak peduli bobot datang dari mana (training vs tangan: `y = Wx`
tetap sama). Pack ini adalah workbench eksplisit untuk sisi tangan:
menulis weight set dalam notasi matematika, memeriksa sifat yang membuat
bobot AMAN, dan menjalankan adaptasi sebagai transisi yang terbukti —
sistem cerdas & adaptif tanpa neural network.

## Komponen

- `lib/weights.mjs` — builder & cek sifat (f64, host-side):
  `affineSpec(W,b,v)` (satu `linear(n,n)`+`add(n)`; W selalu row-major di
  sini, layout column-major engine hanya diterapkan di batas spec),
  `spectralRadius` (power iteration), `orthonormalityError`,
  `det2`, `blendWeights`/`blendBias` (adaptasi via interpolasi),
  `weightFingerprint` (sha256 atas byte f32 W+b: "bobot mana yang live?"),
  `checkProperties` per kind.
- `bin/weight_lab.mjs` — `run` (eksekusi engine vs ref JS),
  `gate <current> <candidate>` (gerbang adaptasi: ACCEPT iff properti
  kandidat lolos DAN lompatan perilaku ≤ `--jump`; exit 0/2),
  `certify` (baterai penuh → catat ke repertoar),
  `compose <A.json> <B.json> [--out C.json]` (fusi afin B∘A dengan
  SERTIFIKAT komposisi sifat; exit 0=COMPOSED, 2=REJECTED),
  `--selftest`.
- `references/weight_repertoire.v1.json` — himpunan bobot TERSERTIFIKASI:
  satu-satunya yang boleh dipasang live oleh kebijakan adaptif.
  Tiap entri: nama, kind, W, b, fingerprint, properties, probe error.

## Kind & sifat yang diperiksa

- `map` — umum `v_out = W·v + b`: tanpa sifat intrinsik; keamanan dari
  probe + jump bound di gate.
- `dynamics` — bobot loop adaptif/damped: `spectral_radius < 1` (kontraksi).
  Contoh: preset "inverted" HTML (`W=-0.8I`) → rho 0.8, terbukti kontraksi.
- `calibration` — kalibrasi sensor: baris-baris ortonormal (toleransi),
  opsional `det_positive` (tidak membalik orientasi).

## Komposisi sifat (semantic composition)

Fusi afin `B∘A`: `W_C = W_B·W_A`, `b_C = W_B·b_A + b_B` — eksak, satu
layer. Pertanyaannya: sifat KEAMANAN mana yang ikut terkomposisi secara
terbukti (tanpa mengukur ulang produknya)? Tiga aturan, semuanya teorema:

1. **Kontraksi via norm bound** (`dynamics` × `dynamics`):
   `||W||_F < 1` ⇒ kontraksi tegas
   (`||Wx||₂ ≤ ||W||₂||x||₂ ≤ ||W||_F||x||₂ < ||x||₂`),
   dan `||W_B W_A||_F ≤ ||W_A||_F·||W_B||_F`.
   Jadi `||W_A||_F·||W_B||_F < 1` ⇒ produk TERBUKTI kontraksi —
   verdict tanpa power iteration pada produk. Sertifikat juga memuat
   kontrol konsistensi: `rho(W_C) ≤ ||W_C||_F ≤ bound` yang diukur.
2. **Ortonormalitas + orientasi** (`calibration` × `calibration`):
   `(W_B W_A)(W_B W_A)ᵀ = I` bila keduanya ortonormal;
   `det(W_C) = det(W_A)·det(W_B)` sehingga `det_positive` ikut
   terkomposisi. Kedua faktor harus ortonormal dalam tol dulu.
3. **Fallback terukur**: bila tak ada aturan yang menyala (mis.
   `rho<1` tapi `||·||_F ≥ 1`, atau kind campuran), produk DIUKUR
   langsung — bukan diwarisi. Bound yang menolak BUKAN penolakan;
   ia tercatat (`refused: true`) tapi tak menjatuhkan verdict.

Perangkap yang didokumentasikan jujur: `rho(W_B W_A) ≤ rho(W_A)·rho(W_B)`
adalah SALAH secara umum (counterexample:
`A=[[0,2],[0,0]]`, `B=[[0,0],[0.4,0]]` → `rho(AB)=0.8 > 0·0`).
Sertifikat WAJIB memakai norm operator, bukan produk spectral radius.
Inilah beda "referensi semantik" vs "verifier angka": yang disertifikasi
adalah MAKNA (kontraksi, ortogonalitas) beserta teorema komposisinya,
bukan sekadar angka yang cocok.

## weights.json

```json
{"name":"elastic","kind":"map","W":[[1.2,-0.4],[0.5,0.8]],"b":[20,-10],
 "probes":[{"in":[80,80],"want":[84,94]}],
 "requires":{"spectral_radius_lt":1}}
```

## Batas jujur

- Cek sifat (spectral radius, ortonormalitas) adalah f64 host-side —
  heuristik/informatif, BUKAN bukti formal. Spectral radius via power
  iteration: aproksimasi, lambat pada matriks defektif.
- Gate hanya sebaik probe-nya: lompatan di luar probe tidak terdeteksi.
  Tulis probe yang mewakili rezim operasi nyata.
- Engine mengeksekusi bagian linear; hukum adaptasi (kapan ganti W,
  blending) tetap logika host — seperti perspective divide: batas
  eksplisit, bukan celah diam-diam.
- Pack ini untuk SERTIFIKASI, bukan eksekutor real-time: rancang &
  buktikan di sini, pasang W tersertifikasi di jalur cepat, jadikan
  `diff_ref --ref` conformance test.

## Bukti (selftest + demo)

- `--selftest` (9/9): engine preset elastic HTML `[80,80]` → `[84,94]`
  eksak; W tak-stabil (rho=2) ter-flag; gate set identik → ACCEPT
  (jump 0); gate kandidat terusik → REJECT (jump 400);
  compose `0.5I∘0.6I` → COMPOSED via norm bound
  (`W=0.3I`, `b=[3.6,5.2]`, `rho=0.3 ≤ ||·||_F=0.424 ≤ bound=0.6`);
  compose `Rz(30)∘Rz(15)` → COMPOSED ortonormal (`det=+1`, hasil
  `Rz(45)`, satu run fused ≡ dua run sekuensial, diff 6e-8);
  compose `inverted²` → bound menolak (1.28≥1) tapi terukur ACCEPT
  (`rho=0.64` — penolakan bound ≠ penolakan);
  compose `0.5I∘2I` → REJECTED (`rho_produk=1.0` tidak `< 1`).
- Demo: `certify elastic` → CERTIFIED (fingerprint `f6ec16ab…`,
  probe_err 0); `gate elastic→perturbed(W[1][1]+0.05)` → REJECT
  (max_jump 4 > tol 1.0).
