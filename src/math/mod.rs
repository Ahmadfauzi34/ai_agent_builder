pub mod linalg;
pub mod numeric;
pub mod tensor;

pub use linalg::{linear_algebra_capabilities, WasmLinearAlgebra};
pub use numeric::{numeric_kernel_capabilities, WasmNumericKernel};
pub use tensor::{tensor_transform_capabilities, WasmTensorTransform};
