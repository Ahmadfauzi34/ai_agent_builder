pub use crate::facade::math::{wasm_math_program_capabilities, wasm_math_program_v4_capabilities};
use wasm_bindgen::prelude::*;

pub use crate::facade::wasm_types::{
    WasmMathProgram, WasmMathProgramBuilder, WasmMathProgramV4, WasmMathProgramV4Builder,
};
use crate::math::{
    math_program_capabilities, math_program_v4_capabilities, MathProgram, MathProgramBuilder,
    MathProgramV4, MathProgramV4Builder,
};
use crate::WasmTensor;
