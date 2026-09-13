use wasm_bindgen::prelude::*;

use crate::math::{math_program_capabilities, MathProgram, MathProgramBuilder};
use crate::WasmTensor;

#[wasm_bindgen(js_name = mathProgramCapabilities)]
pub fn wasm_math_program_capabilities() -> String {
    math_program_capabilities()
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
