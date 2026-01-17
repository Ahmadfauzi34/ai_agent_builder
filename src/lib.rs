use burn::prelude::*;
use wasm_bindgen::prelude::*;
use burn_ndarray::NdArray;

// 1. KUNCI BACKEND
// Kita putuskan: Untuk WASM ini, kita pakai CPU (NdArray) dengan float 32-bit.
type WasmBackend = NdArray<f32>;
