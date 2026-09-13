use wasm_bindgen::prelude::*;

use crate::math::{
    math_program_capabilities, math_program_v4_capabilities, MathProgram, MathProgramBuilder,
    MathProgramV4, MathProgramV4Builder,
};
use crate::WasmTensor;

#[wasm_bindgen(js_name = mathProgramCapabilities)]
pub fn wasm_math_program_capabilities() -> String {
    math_program_capabilities()
}

#[wasm_bindgen(js_name = mathProgramV4Capabilities)]
pub fn wasm_math_program_v4_capabilities() -> String {
    math_program_v4_capabilities()
}

#[wasm_bindgen]
pub struct WasmMathProgramBuilder {
    inner: MathProgramBuilder,
}

#[wasm_bindgen]
impl WasmMathProgramBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramBuilder, String> {
        Ok(WasmMathProgramBuilder {
            inner: MathProgramBuilder::new(num_inputs, num_slots)?,
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
    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
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
        self.inner
            .add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
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

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgram, String> {
        Ok(WasmMathProgram {
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
pub struct WasmMathProgram {
    inner: MathProgram,
}

#[wasm_bindgen]
impl WasmMathProgram {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgram, String> {
        Ok(WasmMathProgram {
            inner: MathProgram::from_plan(plan)?,
        })
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run1(input)
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run2(a, b)
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

#[wasm_bindgen]
pub struct WasmMathProgramV4Builder {
    inner: MathProgramV4Builder,
}

#[wasm_bindgen]
impl WasmMathProgramV4Builder {
    #[wasm_bindgen(constructor)]
    pub fn new(num_inputs: u8, num_slots: u8) -> Result<WasmMathProgramV4Builder, String> {
        Ok(WasmMathProgramV4Builder {
            inner: MathProgramV4Builder::new(num_inputs, num_slots)?,
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
    pub fn add_clamp(
        &mut self,
        input: u8,
        output: u8,
        min: f32,
        max: f32,
    ) -> Result<(), String> {
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
        self.inner
            .add_cosine_similarity(lhs, rhs, output, epsilon)
    }

    #[wasm_bindgen(js_name = addReshape)]
    pub fn add_reshape(
        &mut self,
        input: u8,
        output: u8,
        shape: &[u32],
    ) -> Result<(), String> {
        self.inner.add_reshape(input, output, shape)
    }

    #[wasm_bindgen(js_name = addPermute)]
    pub fn add_permute(
        &mut self,
        input: u8,
        output: u8,
        axes: &[u32],
    ) -> Result<(), String> {
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

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, slot: u8) -> Result<(), String> {
        self.inner.set_output(slot)
    }

    pub fn compile(&self) -> Result<WasmMathProgramV4, String> {
        Ok(WasmMathProgramV4 {
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
pub struct WasmMathProgramV4 {
    inner: MathProgramV4,
}

#[wasm_bindgen]
impl WasmMathProgramV4 {
    #[wasm_bindgen(js_name = fromPlan)]
    pub fn from_plan(plan: &[u8]) -> Result<WasmMathProgramV4, String> {
        Ok(WasmMathProgramV4 {
            inner: MathProgramV4::from_plan(plan)?,
        })
    }

    pub fn run1(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run1(input)
    }

    pub fn run2(&self, a: &WasmTensor, b: &WasmTensor) -> Result<WasmTensor, String> {
        self.inner.run2(a, b)
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
