use wasm_bindgen::prelude::*;

use crate::math::{math_program_v9_capabilities, MathProgramV9, MathProgramV9Builder};
use crate::WasmTensor;

#[wasm_bindgen(js_name = mathProgramV9Capabilities)]
pub fn wasm_math_program_v9_capabilities() -> String {
    math_program_v9_capabilities()
}

#[wasm_bindgen]
pub struct WasmMathProgramV9Builder {
    inner: MathProgramV9Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV9Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV9Builder, String> {
        Ok(WasmMathProgramV9Builder {
            inner: MathProgramV9Builder::new(num_inputs, num_slots)?,
        })
    }

    #[wasm_bindgen(js_name = addUnary)]
    pub fn add_unary(&mut self, op: u8, input: u8, output: u8) -> Result<(), String> {
        self.inner.add_unary(op, input, output)
    }

    #[wasm_bindgen(js_name = addBinary)]
    pub fn add_binary(
        &mut self,
        op: u8,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        self.inner.add_binary(op, lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = addClamp)]
    pub fn add_clamp(&mut self, input: u8, output: u8, min: f32, max: f32) -> Result<(), String> {
        self.inner.add_clamp(input, output, min, max)
    }

    #[wasm_bindgen(js_name = addCosineSimilarity)]
    pub fn add_cosine_similarity(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
        epsilon: f32,
    ) -> Result<(), String> {
        self.inner.add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(&mut self, input: u8, output: u8, shape: &[u32]) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(&mut self, input: u8, output: u8, axes: &[u32]) -> Result<(), String> {
        self.inner.add_permute(input, output, axes)
    }

    #[wasm_bindgen(js_name = addSlice)]
    pub fn add_slice(
        &mut self,
        input: u8,
        output: u8,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<(), String> {
        self.inner.add_slice(input, output, starts, ends)
    }

    #[wasm_bindgen(js_name = addSelectAxis)]
    pub fn add_select_axis(
        &mut self,
        input: u8,
        output: u8,
        axis: u32,
        indices: &[u32],
    ) -> Result<(), String> {
        self.inner.add_select_axis(input, output, axis, indices)
    }

    #[wasm_bindgen(js_name = addFillLike)]
    pub fn add_fill_like(&mut self, reference: u8, output: u8, scalar: f32) -> Result<(), String> {
        self.inner.add_fill_like(reference, output, scalar)
    }

    #[wasm_bindgen(js_name = addExpandLike)]
    pub fn add_expand_like(&mut self, source: u8, reference: u8, output: u8) -> Result<(), String> {
        self.inner.add_expand_like(source, reference, output)
    }

    #[wasm_bindgen(js_name = addSumAxis)]
    pub fn add_sum_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_sum_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMeanAxis)]
    pub fn add_mean_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_mean_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMinAxis)]
    pub fn add_min_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_min_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addMaxAxis)]
    pub fn add_max_axis(&mut self, input: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_max_axis(input, output, axis)
    }

    #[wasm_bindgen(js_name = addIndicesLike)]
    pub fn add_indices_like(&mut self, reference: u8, output: u8, axis: u32) -> Result<(), String> {
        self.inner.add_indices_like(reference, output, axis)
    }

    #[wasm_bindgen(js_name = addLessEqual01)]
    pub fn add_less_equal_01(
        &mut self,
        lhs: u8,
        rhs: u8,
        output: u8,
    ) -> Result<(), String> {
        self.inner.add_less_equal_01(lhs, rhs, output)
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV9, String> {
        Ok(WasmMathProgramV9 {
            inner: self.inner.compile()?,
        })
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }
}

#[wasm_bindgen]
pub struct WasmMathProgramV9 {
    inner: MathProgramV9,
}

impl WasmMathProgramV9 {
    fn run_exact<const N: usize>(&self, inputs: [&WasmTensor; N]) -> Result<WasmTensor, String> {
        let owned: Vec<WasmTensor> = inputs.into_iter().cloned().collect();
        self.inner.run_inputs(&owned)
    }
}

#[wasm_bindgen]
impl WasmMathProgramV9 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV9, String> {
        Ok(WasmMathProgramV9 {
            inner: MathProgramV9::from_plan(plan)?,
        })
    }

    pub fn run1(&self, a: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a])
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b])
    }

    pub fn run3(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c])
    }

    pub fn run4(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor, d: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d])
    }

    pub fn run5(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor, d: &WasmTensor, e: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e])
    }

    pub fn run6(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor, d: &WasmTensor, e: &WasmTensor, f: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f])
    }

    pub fn run7(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor, d: &WasmTensor, e: &WasmTensor, f: &WasmTensor, g: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g])
    }

    pub fn run8(&self, a: &WasmTensor, b: &WasmTensor, c: &WasmTensor, d: &WasmTensor, e: &WasmTensor, f: &WasmTensor, g: &WasmTensor, h: &WasmTensor) -> Result<WasmTensor, String> {
        self.run_exact([a, b, c, d, e, f, g, h])
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.inner.program_plan()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.inner.program_identity()
    }

    #[wasm_bindgen(js_name = numInputs)]
    pub fn num_inputs(&self) -> u8 {
        self.inner.num_inputs()
    }

    #[wasm_bindgen(js_name = numSlots)]
    pub fn num_slots(&self) -> u8 {
        self.inner.num_slots()
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn num_steps(&self) -> usize {
        self.inner.num_steps()
    }

    #[wasm_bindgen(js_name = outSlot)]
    pub fn out_slot(&self) -> u8 {
        self.inner.out_slot()
    }
}
