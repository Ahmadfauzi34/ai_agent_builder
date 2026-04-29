use burn::prelude::*;
use burn::tensor::Shape;
use wasm_bindgen::prelude::*;
use crate::WasmTensor;

// --- SHIFT DIRECTION ---
#[derive(Debug, Clone, Copy)]
pub enum ShiftDirection {
    Up,
    Down,
    Left,
    Right,
}

// --- SHIFT LAYER (Parameter-Free) ---
#[derive(Debug)]
pub struct Shift {
    shift_size: usize,
    direction: ShiftDirection,
}

impl Shift {
    pub fn new(shift_size: usize, direction: ShiftDirection) -> Self {
        Self { shift_size, direction }
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = input.dims();
        let s = self.shift_size;
        let device = input.device();

        // Kalau shift_size >= dimensi target, return zeros
        match self.direction {
            ShiftDirection::Left => {
                if s >= w { return Tensor::zeros(Shape::from([b, c, h, w]), &device); }
                let main = input.slice([0..b, 0..c, 0..h, s..w]);
                let pad = Tensor::zeros(Shape::from([b, c, h, s]), &device);
                Tensor::cat(vec![main, pad], 3)
            }
            ShiftDirection::Right => {
                if s >= w { return Tensor::zeros(Shape::from([b, c, h, w]), &device); }
                let pad = Tensor::zeros(Shape::from([b, c, h, s]), &device);
                let main = input.slice([0..b, 0..c, 0..h, 0..(w - s)]);
                Tensor::cat(vec![pad, main], 3)
            }
            ShiftDirection::Up => {
                if s >= h { return Tensor::zeros(Shape::from([b, c, h, w]), &device); }
                let main = input.slice([0..b, 0..c, s..h, 0..w]);
                let pad = Tensor::zeros(Shape::from([b, c, s, w]), &device);
                Tensor::cat(vec![main, pad], 2)
            }
            ShiftDirection::Down => {
                if s >= h { return Tensor::zeros(Shape::from([b, c, h, w]), &device); }
                let pad = Tensor::zeros(Shape::from([b, c, s, w]), &device);
                let main = input.slice([0..b, 0..c, 0..(h - s), 0..w]);
                Tensor::cat(vec![pad, main], 2)
            }
        }
    }
}

// --- WASM WRAPPER ---
#[wasm_bindgen]
pub struct WasmShift {
    inner: Shift,
}

#[wasm_bindgen]
impl WasmShift {
    #[wasm_bindgen(js_name = newShiftUp)]
    pub fn new_shift_up(shift_size: usize) -> WasmShift {
        WasmShift { inner: Shift::new(shift_size, ShiftDirection::Up) }
    }

    #[wasm_bindgen(js_name = newShiftDown)]
    pub fn new_shift_down(shift_size: usize) -> WasmShift {
        WasmShift { inner: Shift::new(shift_size, ShiftDirection::Down) }
    }

    #[wasm_bindgen(js_name = newShiftLeft)]
    pub fn new_shift_left(shift_size: usize) -> WasmShift {
        WasmShift { inner: Shift::new(shift_size, ShiftDirection::Left) }
    }

    #[wasm_bindgen(js_name = newShiftRight)]
    pub fn new_shift_right(shift_size: usize) -> WasmShift {
        WasmShift { inner: Shift::new(shift_size, ShiftDirection::Right) }
    }

    pub fn forward(&self, input: &WasmTensor) -> WasmTensor {
        let x = input.inner.clone();
        let out = self.inner.forward(x);
        WasmTensor { inner: out }
    }

    // Parameter-free: selalu 0
    pub fn num_params(&self) -> usize {
        0
    }
}

