# Geometri 3D di burn-math

Domain pack: rantai transformasi affine 3D sebagai graph engine — dengan bukti,
bukan gambar.

## Representasi kanonis

Matriks homogen **4x4, row-major** di `lib/geom3.mjs`. Rantai affine
menyatu lewat perkalian matriks biasa: `T2(T1(x)) = (T2·T1) x` — satu rantai =
tepat satu layer `linear(4,4)`.

## Konvensi (tetap, teruji)

- Sudut dalam **derajat** di semua builder (agen berpikir dalam derajat).
- Koordinat right-handed, rotasi **aktif** (vektor berputar, frame diam).
- Euler order default `'ZYX'`: `R = Rz(yaw)·Ry(pitch)·Rx(roll)`, intrinsik.
- Quaternion `(w,x,y,z)`, dinormalisasi saat input; handedness sama dengan matriks.
- Bobot engine **column-major**: `mat4ToColumnMajor()` saat emit ke spec.
- Input `[x,y,z]` → homogenisasi `[x,y,z,1]`; output `[x,y,z,1]` → dehomogenisasi host-side.

## Batas jujur

Engine **tidak punya perspective divide** (tak ada op `div`). Matriks
proyeksi bisa *dibangun* di sini, tapi pembagian `x/w` tetap host-side.
Jangan pura-pura bisa.

## Alur kerja agen

1. Tulis `chain.json`: `{"input":[x,y,z], "chain":[{op...}, ...]}`.
   Op: `rotate_x/y/z {degrees}`, `rotate_axis {axis, degrees}`,
   `euler {roll,pitch,yaw,order}`, `quaternion {w,x,y,z}`,
   `translate {by}`, `scale {by}`.
2. `node bin/geom_build.mjs chain.json spec.json [--compose]` — tiap op jadi
   satu layer `linear(4,4)`; `--compose` melebur rantai jadi satu layer.
3. `node bin/verify.mjs chain-spec.json composed-spec.json` — bukti rantai ≡
   matriks tunggal. `node bin/shapecheck.mjs` — gerbang bentuk.
4. `node bin/geom_build.mjs --selftest` — 9 uji builder (rotasi kanonik,
   euler≡quaternion, komposisi).

## Proof yang terekam (2026-10-15, engine 2d8f42d)

- `Rz(45)·Rz(45) ≡ Rz(90)` pada `[1,0,0]` → `[0,1,0,1]` eksak; verify
  chain-vs-composed: equivalent, max_abs 5.96e-8.
- `euler(10,20,30) ≡ quaternion(quatFromEuler(10,20,30))`: equivalent,
  max_abs = 0 eksak. Quaternion salah → verify exit 2 (REJECT).
- Chain `rotate_y(30)→translate→scale` ≡ single composed matrix:
  equivalent, max_abs = 0 eksak; cocok dengan hitungan f64 independen
  (`applyMat4`) sampai 6 desimal.

## Prinsip

Tiap builder murni (pure function), perilaku lokal, teruji independen —
operasi geometri yang dipakai agen harus atomik: tak ada efek global
tak terduga dari satu op.

## Proyeksi ke layar — `bin/geom_project.mjs` (2026-10-15)

Builder kamera baru: `lookAt4(eye, target, up)` (world→camera) dan
`perspective4(fov_deg, aspect, near, far)` (gaya OpenGL, w_clip = −z_cam).
`geom_project.mjs <model.json>` menyusun MVP = P·V·Model, menjalankannya
lewat engine sebagai satu layer `linear(4,4)` f32 (mendapat clip
`[x,y,z,w]`), lalu perspective divide + viewport mapping dikerjakan
host-side — engine tetap tak punya op div, batas ini tidak berubah.

Aturan kejujuran: verteks dengan w ≤ 0 (di belakang kamera) DITANDAI
`behind:true`, tidak pernah diproyeksikan (tanpa NaN/Inf).

`--selftest`: kubus 8 verteks + 1 verteks belakang-kamera. Engine f32 vs
referensi f64 independen (kode skalar terpisah, tanpa import builder
geom3): **max_abs = 4.08e-6 px** (tol 1e-4). Kontrol negatif: kamera
digeser → gambar berubah 798 px (bukti uji tidak vakum). Verteks belakang
kamera terdeteksi konsisten di kedua jalur.

## Batching verteks — `bin/geom_batch.mjs` (2026-10-15)

Menjawab batas jujur `geom_project.mjs` (satu verteks = satu run engine):
satu run engine kini memproses N verteks sekaligus via satu layer
`linear(4N,4N)` yang bobotnya **block-diagonal** — N salinan matriks MVP 4x4
(column-major) di diagonal, nol di luar blok. Tiap blok independen, jadi
satu run = N proyeksi clip yang identik dengan N run satuan.

**Batas engine (ditemukan empiris, bukan asumsi):** `setWeights` menolak
jumlah bobot > 65535 (batas u16 protokol) → (4N)² ≤ 65535 → **N ≤ 63
verteks per run**. Mesh lebih besar diproses per chunk 63 (chunk terakhir
di-padding ke 63, output padding dibuang) dengan SATU setup
(validatePlan+create+setWeights) — hanya slot input yang di-bind ulang
per chunk. Percobaan naif (setup ulang per chunk) terbukti TIDAK lebih
cepat dari loop satuan; yang menang adalah setup-sekali.

`--selftest` (70 verteks → 2 chunk, menguji batas chunk): batched vs loop
satuan **bit-identik (max_abs diff = 0)**; vs referensi f64 independen
1.86e-5 px (tol 1e-4); verteks belakang-kamera konsisten.

**Pengukuran skala nyata (2026-10-16, bukan ekstrapolasi):** mesh grid 500 /
2000 verteks, kamera rotate_y 15°:
- `geom_batch`: 500v → **246 ms** (8 chunk), 2000v → **335 ms** (32 chunk).
- `geom_project` (loop satuan): 500v → 1060 ms, 2000v → 1869 ms wall
  (didominasi startup runner; marginal ~0.5 ms/verteks).
- Kebenaran skala penuh: batch vs loop-satuan pada 2000 verteks →
  **max screen diff = 0** (bit-identik, bukan cuma di 70 verteks).

Tetap bukan real-time per-frame — pipeline ini alat verifikasi, bukan
renderer game; kejujuran ini tidak berubah.

## Normal matrix & clipping segitiga (2026-10-16)

**Normal matrix** — `normalMat3(model4)` di `lib/geom3.mjs`: inverse-transpose
dari blok 3x3 matriks model, untuk mentransformasi normal (lighting). Normal
adalah vektor arah: tidak boleh lewat matriks proyeksi, dan di bawah
non-uniform scale matriks model sendiri memberi jawaban SALAH. Invariansi yang
diuji (`geom_build --selftest`): untuk pasangan tegak lurus t·n = 0,
(N·n)·(M·t) = 0 — lolos; kontrol negatif: cara naif (M3·n) memberi |dot| = 1.5
pada pasangan diagonal terhadap sumbu scale, bukti inverse-transpose memang
diperlukan. Terapkan ke [nx,ny,nz] lalu renormalisasi, host-side.

**Clipping segitiga** — `clipTriangleNear(c0,c1,c2)` di `lib/geom3.mjs`:
Sutherland–Hodgman satu half-space melawan near plane (z >= -w) dalam
koordinat clip homogen, jadi interpolasi eksak dan perspective divide
sesudahnya aman. Verteks tepat di bidang dihitung inside. Output
`{triangles, spansCamera}`: 3-in → 1 segitiga, 2-in → quad → 2 segitiga,
1-in → 1 segitiga, 0-in → 0. `spansCamera=true` bila ada verteks dengan
w <= 0 (edge melintasi singularitas bidang kamera) — TIDAK di-clip diam-diam,
caller wajib flag seperti verteks behind.

`bin/geom_project.mjs` menerima `"triangles": [[a,b,c],...]` opsional di
model.json (indeks ke vertices; tanpa field ini perilaku identik seperti
dulu): tiap segitiga di-clip dalam clip space lalu tiap verteks hasil
clip diproyeksikan. Terbukti end-to-end: segitiga aman → 1, straddle near
plane → 2, seluruhnya di depan near plane → 0 (clipped away).
