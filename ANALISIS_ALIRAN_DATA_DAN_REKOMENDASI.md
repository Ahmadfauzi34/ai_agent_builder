# Analisis Aliran Data (Data Flow) & Rekomendasi Pertanyaan Testing Suite

Document ID: `ANALISIS_ALIRAN_DATA_DAN_REKOMENDASI.md`
Target System: `burn-research` WASM Engine & JS Orchestrator
Tanggal: 28 Juli 2026

---

## 1. Pemetaan Arsitektur Aliran Data (Data Flow Architecture)

Sistem `burn-research` mengeksekusi pipeline machine learning berbasis WebAssembly dengan memisahkan instansiasi layer, eksekusi grafik komputasi, dan optimasi bobot dari runtime JS.

```
+-----------------------------------------------------------------------------------+
| 1. WIRE-FORMAT / PROTOCOL INGESTION                                               |
|    JS Payload Binary Byte Stream -> PacketHeader (8B) + PayloadCursor             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. LAYER REGISTRY & STATE MANAGMENT                                              |
|    LayerRegistry::init_layer -> Inisialisasi HashMap<LayerId, WasmLayer>          |
|    Aturan: Stateful (cached_params tracking) vs Stateless (0 params)             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. TENSOR & MEMORY BRIDGE                                                         |
|    WasmTensor (4D Burn Tensor) <---> TensorView (JS SharedArrayBuffer / SAB)      |
|    Pad-4D rule: Shape len [1..4] selalu di-pad trailing ke 4D [B, C, H, W]       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. GRAPH EXECUTION (RUN / COMPILED)                                               |
|    CompiledGraph::build / LayerRegistry::run_graph                                |
|    Plan Stride: 9 Byte per Step (arity: u8, layer_type: u8, layer_id: u32,        |
|                                   in_slot: u8, in_slot2: u8, out_slot: u8)        |
|    Slot Array: max 64 slots, Bitmask tracking input-readiness & write-validation   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. FLOAT-BRIDGE & EVOLUTION STRATEGY (ES OPTIMIZATION)                            |
|    get_weights_flat / set_weights_flat <-> OpenES / MuLambda / JS Optimizer       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Analisis Jalur Aliran Data & Potensi Titik Patah (Breakage Points)

### Jalur 1: Ingesti Protocol & Wire-Format Ingestion
* **Mekanisme**: JS mengirim byte array berstruktur `PacketHeader` (8 byte: `opcode`, `layer_type`, `variant`, `flags`, `payload_len`) diikuti oleh payload data yang dibaca menggunakan `PayloadCursor` (little-endian).
* **Potensi Titik Patah**:
  1. *Out-of-Bounds Payload*: Apabila `payload_len` di header tidak cocok dengan ukuran buffer sebenarnya.
  2. *Dual Payload Reading Dialect*: Ada dua pendekatan baca payload di codebase (`PayloadCursor` vs free-functions `read_u32`, `read_f64`). Adanya ketidakseimbangan offset atau handling option tagging dapat menyebabkan data corrupt.
  3. *Unchecked Cast/Enum Out of Range*: Pilihan `variant` di luar konstanta yang didefinisikan (misal `0xFE` pada activation/norm/conv) harus mengembalikan `Err` terstruktur, bukan panic/trap.

### Jalur 2: Tensor Bridge & SharedArrayBuffer (Zero-Copy)
* **Mekanisme**: `WasmTensor` dibentuk dari slice `f32` dan `shape`. Dimensi kurang dari 4 dipad otomatis dengan nilai `1` di bagian trailing (misal `[32, 64]` menjadi `[32, 64, 1, 1]`). `TensorView` memungut memory via `SharedArrayBuffer` (SAB).
* **Potensi Titik Patah**:
  1. *Dimension Mismatch pada Reshape*: Transfer dari `TensorView` ke `WasmTensor` mensyaratkan total perkalian elemen `shape` sama persis dengan byte length SAB.
  2. *Direct Buffer Mutation Risk*: Penulisan JS ke SAB saat WASM sedang melakukan inferensi tanpa sinkronisasi mutex/barrier dapat menyebabkan race condition data.

### Jalur 3: Graph Plan Execution & Layout 9-Byte Stride
* **Mekanisme**: `run_graph` dan `compile_graph` menerima biner plan.
  * Stride: **9 Byte / Step** (`arity` [1B], `layer_type` [1B], `layer_id` [4B LE], `in_slot` [1B], `in_slot2` [1B], `out_slot` [1B]).
  * Validasi *Fail-Fast*: Bitmask 64-bit `filled` memeriksa apakah slot input sudah pernah ditulis sebelum dibaca, serta memastikan `arity == 2` mengikat `LAYER_BINARY` dan `arity == 1` melarang `LAYER_BINARY`.
* **Potensi Titik Patah**:
  1. *Unwritten Output Slot*: Jika plan menunjuk `out_slot` akhir yang tidak pernah diisi oleh step manapun.
  2. *Slot Boundary Overflow*: Slot index $\ge$ `num_slots` (atau `num_slots > 64`).
  3. *Incompatible Tensor Shapes between Graph Branches*: Misal pada `BINARY_ADD` atau `BINARY_MATMUL` dua cabang slot yang bentuk dimensinya tidak kompatibel.

### Jalur 4: State Management & Float-Bridge
* **Mekanisme**: `LayerRegistry` menyimpan layer dalam HashMap dan memperbarui cache parameter `cached_params` via macro (`insert_layer!`, `remove_layer!`, `load_layer_state!`). Layer linear, conv, embedding, dan norm mendukung `getWeightsFlat` / `setWeightsFlat`.
* **Potensi Titik Patah**:
  1. *Desynchronization of Param Cache*: Inisialisasi ulang layer dengan ID sama atau kegagalan pada `load_state` jika tidak mengembalikan state parameter lama dengan benar.
  2. *Partial Float-Bridge Support*: Layer di luar Linear/Conv/Embedding/Norm (misal Custom Layer atau Binary) akan melempar `Err` saat `setWeightsFlat` dipanggil. Testing harus mengonfirmasi ini sebagai perilaku terdesain.
  3. *Unsound Panic Surfaces*: Layer seperti `SeBlockConfig::init` (`assert!(channels >= reduction)`) dan `GhostModuleConfig::init` (`assert!(out_channels % ratio == 0)`) menggunakan `assert!` internal Rust alih-alih mengembalikan `Result::Err`. Ini berpotensi mematikan instans WASM (panic trap).

---

## 3. Rekomendasi Bagian & Pertanyaan Strategis yang Harus Dipertanyakan

Untuk memastikan suite pengujian 657 test case tepat sasaran dan mampu mendiagnosis kesehatan sistem secara obyektif, berikut rekomendasi bagian-bagian yang **WAJIB dipertanyakan** kepada arsitek/tim pengembang:

### A. Kontrak Wire Protocol & Parsing
1. **Aturan Migrasi Protocol**:
   > *"Apakah ada legacy plan 7-byte/step yang masih beredar di modul JS lama, atau dapat dipastikan seluruh JS orchestrator telah dialihkan 100% ke format plan 9-byte/step?"*
2. **Standardisasi Utilitas Payload Cursor**:
   > *"Apakah ada rencana refactoring untuk menghapus fungsi-fungsi free-fn (`read_u32`, `read_f64` di `protocol.rs`) dan memandatkan 100% pembacaan payload melalui `PayloadCursor` guna menghindari bug batas buffer?"*

### B. Batas Panic Surface vs Error Handling (Result Trap)
3. **Konversi Assert ke Result pada Custom Layer**:
   > *"Di `SeBlock` (`channels < reduction`) dan `GhostModule` (`out_channels % ratio != 0`), pemicu kesalahan saat ini memicu Rust `panic!`. Apakah ini akan tetap dipelihara sebagai panic surface (yang diuji via `should_panic` / JS `expect().toThrow()`), ataukah direncanakan untuk di-refactor menjadi `Result::Err` agar WASM runtime tidak crash?"*
4. **Perilaku Inisialisasi Tensor Dimensi Nol**:
   > *"Bagaimana sistem harus merespons jika JS mengirimkan shape dengan dimensi nol (misal `[0, 10]`)? Apakah wajib ditangkap di tingkat `WasmTensor::new` atau dibiarkan panic di backend Burn?"*

### C. Graph Execution & Semantik Slot Memory
5. **Keamanan Slot Isolation pada Graph**:
   > *"Pada `CompiledGraph::run`, apakah slot memori bekas langkah sebelumnya perlu dibersihkan/direset setelah dibaca untuk menghemat memori pada grafik berukuran besar, ataukah retensi slot diizinkan untuk kebutuhan re-use intermediate output?"*
6. **Penanganan Validasi Shape pada Binary Node (Matmul/Concat/Add)**:
   > *"Validasi plan `compile_graph` saat ini memvalidasi keterhubungan slot dan tipe layer secara static. Apakah validasi dimensi tensor antar slot binary node perlu diperluas secara dynamic-dry-run sebelum eksekusi sejati dilakukan?"*

### D. Float-Bridge & Cakupan Layer Optimization
7. **Perluasan Float-Bridge Dispatch**:
   > *"Saat ini `get_weights_flat` dan `set_weights_flat` mendukung Linear, Conv, Embedding, dan Norm. Apakah layer custom seperti GhostModule dan SeBlock akan dibuka jalur float-bridge-nya untuk optimasi ES di masa mendatang?"*
8. **Matmul vs Linear Weight Layout Convention**:
   > *"Bagaimana konvensi transposisi bobot dipastikan konsisten antara bobot `WasmLinear` dengan operand `BINARY_MATMUL` dalam graph ketika dioptimasi oleh Evolution Strategy?"*

---

## 4. Matriks Suite Pengujian yang Direkomendasikan (657 Test Breakdown)

| Dunia / Kelompok | Domain | Jumlah Test | Target Verifikasi & Kontrak |
|---|---|---|---|
| **Rust Host (H)** | Logic Pure, Cursor, Protocol, ES Algo | 156 | Wire-format validity, Endianness, RNG determinism, Convergence |
| **Rust WASM (A)** | Tensor, Layer Registry, Graph Plan, Bridge | 278 | Pad-4D rule, Fail-fast 9-byte stride validation, State roundtrip, Panic surfaces |
| **JS Unit (U)** | Matrix.ts, MathHandler, MathNode | 84 | Zero-allocation out-params, Reshape fallback, UI numerical parsing |
| **JS Bridge (I)** | Integration tests (WASM built pkg + Vitest) | 39 | E2E plan build, SharedArrayBuffer sync, ES graph training loop |
| **Cross-Cutting (X)**| System Invariants & Reference Machine | 100 | Stateful/Stateless invariant, Determinism across seeds, Boundary trash-byte testing |
| **TOTAL** | | **657** | **Pass Rate Requirement: 100%** |

---

## 5. Kesimpulan & Langkah Selanjutnya

1. **Gunakan Dokumen Ini Sebagai Acuan Diskusi**: Ajukan 8 pertanyaan strategis di atas kepada tim/arsitek sistem sebelum melakukan locking pada rilis produksi.
2. **Jalankan Verification Pipeline**: Pastikan generator test (`generate_tests.py`) mengekspansi seluruh 657 test case dengan nama eksplisit `test_<modul>_<perilaku>_<kasus>` dan menghasilkan laporan gating `TEST_REPORT.md` serta `test_report.json`.
