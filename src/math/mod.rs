pub mod linalg;
pub mod numeric;
pub mod statistics;
pub mod tensor;

pub use linalg::{linear_algebra_capabilities, WasmLinearAlgebra};
pub use numeric::{numeric_kernel_capabilities, WasmNumericKernel};
pub use statistics::{statistics_capabilities, WasmStatistics};
pub use tensor::{tensor_transform_capabilities, WasmTensorTransform};
