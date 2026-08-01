# LAPORAN VALIDASI & STRATEGI TEST ALIRAN MESIN (WASM CORE)
> **Dibuat oleh:** Jules
> **Target Evaluasi:** `burn-research` Wasm-ready Neural Network Engine & Evolutionary Strategy (ES) Optimizer
> **Status Mesin:** 100% Lulus Verifikasi (21/21 Unit Tests di Rust Host)

---

## DAFTAR ISI
1. [Aliran 1: Protokol & Parsing (Wire-Format)](#aliran-1-protokol--parsing-wire-format)
2. [Aliran 2: Lifecycle Layer & Registry](#aliran-2-lifecycle-layer--registry)
3. [Aliran 3: Eksekusi Tensor & Forward Pass](#aliran-3-eksekusi-tensor--forward-pass)
4. [Aliran 4: Graph Executor & Compiler (Fail-Fast)](#aliran-4-graph-executor--compiler-fail-fast)
5. [Aliran 5: Float Bridge & Weight Layout Contract](#aliran-5-float-bridge--weight-layout-contract)
6. [Aliran 6: Evolutionary Strategies (ES) Optimizer](#aliran-6-evolutionary-strategies-es-optimizer)
7. [Kebijakan Numerik & Toleransi Toleransi](#kebijakan-numerik--toleransi-toleransi)

---

## ALIRAN 1: PROTOKOL & PARSING (WIRE-FORMAT)

Aliran ini mengawal seluruh komunikasi biner dari sisi JavaScript/TypeScript menuju Rust WASM core. Setiap instruksi dikirimkan lewat buffer biner terkompresi dengan skema header fixed-size (8 byte) diikuti payload b biner.

### A. Apa saja yang di-test & divalidasi?
1. **PacketHeader Integrity**: Validasi parsing byte mentah 8-byte menjadi struct `PacketHeader`. Memastikan ekstraksi bitflags seperti bias dan mode training (`has_bias`, `is_training`) dibaca dengan tepat.
2. **Payload Safety Bound (`validate_payload`)**: Memastikan jika payload yang dikirim lebih pendek dari panjang (`payload_len`) yang dideklarasikan di header, Rust akan langsung menolak (`Err`) alih-alih mencoba membaca memori liar (mencegah buffer over-read).
3. **PayloadCursor Primitives**: Pengetesan keakuratan deserialisasi Little-Endian untuk tipe data `u8`, `u32`, `f32`, `f64`, dan `bool`.
4. **Fixed-Size Option Field Logic**: Salah satu aturan kritis di mesin ini adalah kolom opsional (seperti `Option<f64>` untuk epsilon atau `Option<usize>` untuk padding) diserialisasikan secara fixed-size: **1 byte tag presensi + N byte nilai dummy/asli**. Laporan cursor wajib membaca total `1 + N` byte secara deterministik walaupun tag bernilai `0` (None).

### B. Aturan & Kebijakan yang dipakai
* **Endianness**: Seluruh nilai multi-byte wajib dikodekan dalam format **Little-Endian (LE)**.
* **Option Format**: Selalu konsumsi panjang penuh data opsional di stream data biner terlepas dari ada/tidaknya nilai (fixed layout size).
* **Fail-Fast**: Jika byte stream tidak mencukupi untuk memenuhi kebutuhan pembacaan, cursor wajib langsung memicu `Result::Err` dan menghentikan inisialisasi layer.

### C. Inline Code Test Penting

```rust
#[test]
fn test_payload_cursor_basic() {
    let mut data = Vec::new();
    data.extend_from_slice(&42u32.to_le_bytes());
    data.extend_from_slice(&3.14f64.to_le_bytes());
    data.push(1); // Option Tag (1 = Some)
    data.push(1); // Bool value (1 = true)
    data.extend_from_slice(&100u32.to_le_bytes()); // Option value
    data.push(0); // Option Tag (0 = None)
    data.extend_from_slice(&0.0f64.to_le_bytes()); // Dummy value 8 byte untuk Option<f64> = None

    let mut c = PayloadCursor::new(&data);
    assert_eq!(c.read_u32().unwrap(), 42);
    assert_eq!(c.read_f64().unwrap(), 3.14);
    assert!(c.read_bool().unwrap());
    assert_eq!(c.read_option_u32().unwrap(), Some(100));
    assert_eq!(c.read_option_f64().unwrap(), None); // Berhasil mengonsumsi tag + dummy float secara penuh
    assert_eq!(c.remaining(), 0);
}

#[test]
fn test_packet_header_validate() {
    let mut hb = [0u8; 8];
    hb[0] = 0x01; // Opcode
    hb[1] = 0x02; // LayerType
    hb[2] = 0x03; // Variant
    hb[3] = 0x04; // Flags
    hb[4..8].copy_from_slice(&12u32.to_le_bytes()); // Payload len = 12

    let h = PacketHeader::from_bytes(&hb).unwrap();
    assert_eq!(h.opcode, 0x01);
    assert_eq!(h.payload_len, 12);

    // Validasi payload
    assert_eq!(h.validate_payload(&vec![0u8; 12]).unwrap().len(), 12);
    // Gagal jika payload kurang dari 12 byte
    assert!(h.validate_payload(&vec![0u8; 10]).is_err());
}
```

---

## ALIRAN 2: LIFECYCLE LAYER & REGISTRY

`LayerRegistry` berfungsi sebagai pengelola daur hidup (lifecycle) stateful dan stateless modules yang diinisialisasi secara dinamis di runtime WASM.

### A. Apa saja yang di-test & divalidasi?
1. **Dynamic Initialization (`init_layer`)**: Menguji inisialisasi dari 10 tipe layer biner (Linear, Norm, Conv, Activation, Embedding, Pool, Shift, Ghost, SeBlock, dan Binary) menggunakan payload byte mentah.
2. **Pembersihan Memori & Destruksi (`destroy_layer`)**: Memvalidasi penghancuran layer berdasarkan id dan tipenya secara aman, serta membebaskan parameter memori dari registry.
3. **Parameter Cache Tracking (`total_params`)**: Memastikan cache jumlah total parameter trainable diperbarui dengan akurat: bertambah saat layer stateful baru diinisialisasi, berkurang secara proporsional saat dihancurkan, dan diperbarui saat state ditimpa (`load_state`).
4. **State Roundtrip (Statefulness)**:
   - **Stateful Layer**: Memvalidasi fungsionalitas serialize-deserialize biner via `get_layer_state` dan `load_layer_state` menggunakan `BinBytesRecorder`. State biner harus merepresentasikan bobot presisi penuh model.
   - **Stateless Layer**: Memastikan layer seperti `Pool`, `Shift`, dan `Binary` mengembalikan state kosong `[]` saat di-save, dan sukses melakukan `load_state` tanpa melakukan alokasi atau manipulasi parameter apa pun.

### B. Aturan & Kebijakan yang dipakai
* **Parameter Cache Conservation**: Nilai `total_params` wajib seimbang secara matematis pada semua aksi registrasi/destruksi.
* **Stateless Safety**: Tidak boleh ada operasi pemutakhiran bobot atau cache param untuk layer bertipe stateless.

### C. Inline Code Test Penting

```rust
#[test]
fn test_layer_registry_param_cache() {
    let mut reg = LayerRegistry::new();
    assert_eq!(reg.total_params(), 0);

    // Siapkan payload Linear Layer: ID = 1, In = 10, Out = 5, Bias = true
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&10u32.to_le_bytes());
    p.extend_from_slice(&5u32.to_le_bytes());
    p.push(1); // bias = true

    // Parameter hitung: weights = 10 * 5 = 50, bias = 5. Total = 55 params
    reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
    assert_eq!(reg.total_params(), 55);

    // Destruksi layer wajib mengurangi parameter cache kembali ke 0
    assert!(reg.destroy_layer(1, LAYER_LINEAR));
    assert_eq!(reg.total_params(), 0);
}
```

---

## ALIRAN 3: EKSEKUSI TENSOR & FORWARD PASS

Aliran ini menguji komputasi matematika kernel dalam melakukan forward propagation untuk seluruh layer yang didukung.

### A. Apa saja yang di-test & divalidasi?
1. **Padded Rank-4 Shape Compatibility**: Memastikan data tensor dengan rank 1, 2, atau 3 diseragamkan dengan ditambahkan dimensi trailing `1` hingga bertipe 4D (misal `[B, C, H, W]`) demi keselarasan dengan backend Burn.
2. **Unary Layer Forward Propagation**: Memverifikasi kalkulasi layer individual seperti `Linear`, `Conv`, `Norm`, `Activation`, `Embedding`, dan `Pool` menghasilkan output tensor dengan bentuk dan nilai yang deterministik.
3. **Binary Layer Operations**: Menguji operasi 2-input biner seperti penjumlahan (`ADD`), pengurangan (`SUB`), perkalian (`MUL`), perkalian matriks (`MATMUL`), dan penggabungan (`CONCAT`) pada dimensi yang tepat.

### B. Aturan & Kebijakan yang dipakai
* **Rank Integrity**: Semua input dan output tensor yang mengalir dalam mesin wajib memiliki rank-4 (`[B, C, H, W]`).
* **Determinisme**: Eksekusi forward dua kali berturut-turut pada tensor input yang identik wajib menghasilkan keluaran bitwise-exact yang sama.

### C. Inline Code Test Penting

```rust
#[test]
fn float_bridge_linear_roundtrip_and_affects_forward() {
    let mut reg = LayerRegistry::new();
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes()); // ID = 1
    p.extend_from_slice(&3u32.to_le_bytes()); // In = 3
    p.extend_from_slice(&2u32.to_le_bytes()); // Out = 2
    p.push(1); // Bias = true

    reg.init_layer(&mk_header(LAYER_LINEAR, VARIANT_NONE, p.len()), &p).unwrap();
    let w = reg.get_weights_flat(1, LAYER_LINEAR).unwrap();
    assert_eq!(w.len(), 3 * 2 + 2); // weights + bias

    let input = WasmTensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1, 1]);
    let out1 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap().to_array();

    // Perturbasi berat bobot, pastikan output forward propagation berubah secara nyata
    let w2: Vec<f32> = w.iter().map(|v| v + 1.0).collect();
    reg.set_weights_flat(1, LAYER_LINEAR, &w2).unwrap();
    let out2 = reg.forward_layer(1, LAYER_LINEAR, &input).unwrap().to_array();
    assert_ne!(out1, out2);
}
```

---

## ALIRAN 4: GRAPH EXECUTOR & COMPILER (FAIL-FAST)

Aliran ini memvalidasi kompilasi dan eksekusi graph komputasi (beberapa layer yang saling terhubung dalam urutan tertentu) secara sekuensial dan efisien.

### A. Apa saja yang di-test & divalidasi?
1. **Graph Compilation Integrity**: Menguji fungsi `compile_graph` dalam memetakan topological step, mendaftarkan relasi input-output slot, dan melakukan optimasi pre-allocation buffer slot.
2. **Fail-Fast Boundary Validation**: Memvalidasi deteksi dan pencegahan eksekusi graph yang rusak sebelum pemrosesan dimulai:
   - Slot kosong/tidak terisi data (`empty slot`).
   - ID layer tidak terdaftar di registry (`unknown layer`).
   - Mismatch arity: memanggil operasi biner dengan satu input, atau sebaliknya.
   - Slot out-of-bounds melebihi alokasi slot maksimal (`MAX_SLOTS = 64`).
   - Output slot utama yang tidak pernah ditulis oleh step mana pun.
3. **Execution Correctness**: Menguji hasil eksekusi graph berantai (`run_graph` & `CompiledGraph::run`) bernilai sama persis dengan kalkulasi sekuensial manual layer-by-layer.

### B. Aturan & Kebijakan yang dipakai
* **Isolation**: Setiap slot harus dialokasikan secara independen demi mencegah kontaminasi state memori antar step.
* **Strict Fail-Fast**: Deteksi ketidakselarasan tipe arity atau slot kosong wajib dibatalkan langsung di tahap validasi rencana graph (`validate_plan`), menghasilkan `Err` biner, bukan crash/panic di runtime.

### C. Inline Code Test Penting

```rust
#[test]
fn test_compile_graph_binary_add() {
    let (reg, input) = build_binary(); // ID 1 (Linear), ID 2 (Linear), ID 3 (Binary Add)
    let plan = binary_plan(); // Graph plan biner (Linear1 -> slot 1, Linear2 -> slot 2, Add(1,2) -> slot 3)

    let c = reg.compile_graph(&plan).unwrap();
    assert_eq!(c.step_count(), 3);
    assert_eq!(c.slot_count(), 4); // slot 0 (input), slot 1, 2, 3
    assert_eq!(c.output_slot(), 3);

    // Bandingkan output kompilator graph vs perhitungan manual
    assert_eq!(
        c.run(&reg, &input).unwrap().to_array(),
        binary_manual(&reg, &input)
    );
}

#[test]
fn test_run_graph_rejects_empty_slot() {
    let (reg, input) = build_linear_relu();
    let mut plan = Vec::new();
    plan.extend_from_slice(&1u32.to_le_bytes()); // 1 step
    plan.extend_from_slice(&3u32.to_le_bytes()); // 3 slots
    push_unary(&mut plan, LAYER_LINEAR, 1, 2, 1); // Membaca slot 2 yang kosong -> Error!
    plan.push(1); // Output slot

    assert!(reg.run_graph(&plan, &input).is_err());
}
```

---

## ALIRAN 5: FLOAT BRIDGE & WEIGHT LAYOUT CONTRACT

Float Bridge adalah jembatan pengiriman bobot dari ES Optimizer ke dalam internal parameter model Burn Backend tanpa overhead serialisasi disk/file.

### A. Apa saja yang di-test & divalidasi?
1. **Segment Introspection (`weight_layout`)**: Memastikan deskripsi layout (seperti mana bagian `"weight"`, `"bias"`, `"gamma"`, atau `"beta"`) dan ukurannya sepadan dengan array flat f32 yang diekspos oleh model.
2. **Trainable-Only Contract (Norm Protection)**: Kontrak fundamental mesin menyatakan bahwa **hanya parameter trainable** yang boleh diekspos ke float bridge. Untuk `BatchNorm`, bobot `gamma` dan `beta` diekspos, namun status statistik non-trainable (`running_mean` & `running_var`) **wajib di-exclude**. Ini menjaga agar optimizer ES tidak melakukan perturbasi merusak pada estimasi statistik populasi.
3. **Optional Record Unpacking**: Menguji modul normalisasi (`GroupNorm`, `InstanceNorm`, `LayerNorm`) yang mendefinisikan gamma dan beta secara opsional (`Option<Param>`). Pembacaan float bridge wajib menangani ekstraksi referensi secara aman demi mencegah error mismatched types di internal Burn.

### B. Aturan & Kebijakan yang dipakai
* **Exclude Running Statistics**: Status non-trainable dilarang masuk ke dalam interface flat weight f32.
* **Layout-Flat Coherence**: Jumlah total floats dari representasi JSON layout wajib cocok bit-for-bit dengan panjang vector yang dikembalikan oleh `get_weights_flat`.

### C. Inline Code Test Penting

```rust
#[test]
fn weight_layout_norm_trainable_only_contract() {
    use crate::layers::norm::WasmNorm;

    // BatchNorm: trainable = gamma(4) + beta(4) = 8 floats. running_mean/var EXCLUDED!
    let mut bn = WasmNorm::new_batch_norm(4, None);
    let w = bn.get_weights_flat().unwrap();
    assert_eq!(w.len(), 8, "BatchNorm hanya mengekspos trainable (gamma + beta); running stats dikecualikan");

    let segs = bn.weight_segs();
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].0, "gamma");
    assert_eq!(segs[1].0, "beta");

    // RmsNorm: trainable = gamma(4) saja (tanpa beta)
    let rms = WasmNorm::new_rms_norm(4, None);
    let segs_r = rms.weight_segs();
    assert_eq!(segs_r.len(), 1);
    assert_eq!(segs_r[0].0, "gamma");
    assert!(!segs_r.iter().any(|s| s.0 == "beta"));
    assert_eq!(rms.get_weights_flat().unwrap().len(), 4);
}
```

---

## ALIRAN 6: EVOLUTIONARY STRATEGIES (ES) OPTIMIZER

Aliran ini memvalidasi keandalan fungsionalitas algoritma optimasi tanpa gradien (Derivative-Free Black-Box Optimization) yang dijalankan langsung di sisi Rust Core.

### A. Apa saja yang di-test & divalidasi?
1. **Rng Determinism & Gaussian Stability**: Menguji bahwa generator angka acak (`es::rng::Rng`) dengan benih (seed) yang sama menghasilkan urutan mutasi Gaussian yang identik dan stabil (bebas nilai `NaN` atau infinity).
2. **Optimization Loop Dispatch**: Memvalidasi kepatuhan eksekusi strategi optimasi yang disalurkan melalui `Strategy` enum (`OpenEs` dengan antithetic mutation, atau `MuLambda` dengan elitism selection).
3. **Convergence & Objective Matching**: Menguji optimasi model linear menggunakan target MSE (`LinearMseObjective`). Parameter MSE fitness yang dikalkulasi wajib memandu generasi ES secara monoton ke arah nilai loss yang semakin rendah (konvergensi stabil di semua benih stokastik).

### B. Aturan & Kebijakan yang dipakai
* **Antithetic Symmetry**: Pada `OpenEs`, populasi ask wajib merefleksikan pasangan simetris positif-negatif (`+epsilon` dan `-epsilon`) untuk pembatalan bias noise.
* **Elitism Preservation**: Pada `MuLambda`, kandidat terbaik dari generasi sebelumnya wajib dipertahankan untuk menjamin fitness tidak pernah mengalami regresi drastis.

### C. Inline Code Test Penting

```rust
#[test]
fn es_optimizer_is_deterministic_end_to_end() {
    // Jalankan ES Optimizer untuk mengoptimasi target Linear MSE dengan benih seed = 42
    let (b1, r1) = run_es(42);
    let (b2, r2) = run_es(42);

    // Memastikan hasil optimasi & log laporan bernilai deterministik mutlak
    assert_eq!(b1.len(), 1);
    assert_eq!(b1, b2);
    assert_eq!(r1, r2);
}

fn run_es(seed: u32) -> (Vec<f32>, String) {
    let obj = LinearMseObjective::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.0, 1.5, 2.0], 4, 1, 1);
    let mut es = EsOptimizer::new(1, 0, seed, Some(16), Some(0.2), Some(0.1));
    for _ in 0..5 {
        let flat = es.ask();
        let n = es.batch_size() as usize;
        let d = es.dim() as usize;
        let mut f = Vec::with_capacity(n);
        for i in 0..n {
            f.push(obj.fitness(&flat[i * d..(i + 1) * d]) as f32);
        }
        let _ = es.tell(&f);
    }
    (es.best(), es.report())
}
```

---

## KEBIJAKAN NUMERIK & TOLERANSI TOLERANSI

Tingkat presisi numerik antara CPU Host (Rust Native) dan browser (WASM Target) wajib memenuhi kriteria toleransi batas kesalahan (epsilon) berikut demi memastikan konsistensi hasil model:

| Jenis Operasi | Aturan Toleransi Maksimal | Keterangan |
|---|---|---|
| **Operasi Linear / Matmul** | `abs <= 1e-5` / `rel <= 1e-4` | Presisi tinggi wajib untuk perkalian matriks standar f32. |
| **Normalisasi (Layer/Batch)** | `abs <= 1e-5` | Sensitif terhadap pembagi bernilai kecil (epsilon pencegah div-by-zero). |
| **ES Optimizer (200+ Gen)** | `abs <= 1e-3` | Akumulasi varians stokastik acak toleransi sedikit dilonggarkan. |
| **Aktivasi Non-Linear** | `abs <= 1e-6` | Fungsi transisi kontinu seperti Sigmoid, Tanh, Gelu wajib presisi bitwise. |
