use burn::prelude::*;
use burn::tensor::TensorData;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

pub mod es;
pub mod graph;
pub mod layers;
pub mod protocol;
pub mod registry;
#[cfg(test)]
mod tests;

pub type WasmBackend = burn_ndarray::NdArray<f32>;

fn boundary_fail(message: String) -> ! {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
}

fn normalize_rank4_shape(shape: &[usize], context: &str) -> Result<[usize; 4], String> {
    if shape.len() > 4 {
        return Err(format!(
            "{context}: rank {} exceeds the 4D tensor bridge",
            shape.len()
        ));
    }

    let mut dims = [1usize; 4];
    for (index, &dim) in shape.iter().enumerate() {
        dims[index] = dim;
    }

    checked_element_count(dims, context)?;
    Ok(dims)
}

fn checked_element_count(dims: [usize; 4], context: &str) -> Result<usize, String> {
    dims.into_iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| format!("{context}: shape element count overflow for {dims:?}"))
    })
}

fn validate_element_count(
    dims: [usize; 4],
    actual: usize,
    context: &str,
) -> Result<(), String> {
    let expected = checked_element_count(dims, context)?;
    if actual != expected {
        return Err(format!(
            "{context}: shape {dims:?} requires {expected} elements, got {actual}"
        ));
    }
    Ok(())
}

fn checked_sab_byte_length(total_elements: usize) -> Result<u32, String> {
    let bytes = total_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "TensorView: byte length overflow".to_string())?;
    u32::try_from(bytes).map_err(|_| {
        format!(
            "TensorView: {} elements require {} bytes, exceeding SharedArrayBuffer u32 length",
            total_elements, bytes
        )
    })
}

// -------------------------------------------------------------
// WASM TENSOR — 4D tensor bridge
// -------------------------------------------------------------
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmTensor {
    pub(crate) inner: Tensor<WasmBackend, 4>,
}

#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let dims = normalize_rank4_shape(shape, "WasmTensor::new")
            .unwrap_or_else(|err| boundary_fail(err));
        validate_element_count(dims, data.len(), "WasmTensor::new")
            .unwrap_or_else(|err| boundary_fail(err));

        let device = Default::default();
        let tensor_data = TensorData::new(data.to_vec(), dims);
        let tensor = Tensor::from_data(tensor_data, &device);
        WasmTensor { inner: tensor }
    }

    pub fn to_array(&self) -> Vec<f32> {
        let data = self.inner.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }

    pub fn byte_length(&self) -> usize {
        checked_element_count(self.inner.dims(), "WasmTensor::byte_length")
            .and_then(|count| {
                count
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| "WasmTensor::byte_length: byte length overflow".to_string())
            })
            .unwrap_or_else(|err| boundary_fail(err))
    }
}

// -------------------------------------------------------------
// TENSOR VIEW — SharedArrayBuffer bridge (zero-copy JS side)
// -------------------------------------------------------------
#[wasm_bindgen]
pub struct TensorView {
    sab: JsValue,
    shape: Vec<usize>,
}

#[wasm_bindgen]
impl TensorView {
    #[wasm_bindgen(constructor)]
    pub fn new(total_elements: usize) -> Self {
        let byte_length = checked_sab_byte_length(total_elements)
            .unwrap_or_else(|err| boundary_fail(err));
        let sab = js_sys::SharedArrayBuffer::new(byte_length);
        TensorView {
            sab: JsValue::from(sab),
            shape: vec![total_elements, 1, 1, 1],
        }
    }

    pub fn len(&self) -> usize {
        let arr = Float32Array::new(&self.sab);
        arr.length() as usize
    }

    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape(&mut self, shape: Vec<usize>) {
        let dims = normalize_rank4_shape(&shape, "TensorView::setShape")
            .unwrap_or_else(|err| boundary_fail(err));
        validate_element_count(dims, self.len(), "TensorView::setShape")
            .unwrap_or_else(|err| boundary_fail(err));
        self.shape = dims.to_vec();
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn as_f32_array(&self) -> Float32Array {
        Float32Array::new(&self.sab)
    }

    /// Copy data dari SAB ke slice Rust (Rust baca data JS)
    pub fn read(&self, dst: &mut [f32]) {
        if dst.len() != self.len() {
            boundary_fail(format!(
                "TensorView::read: destination length {} does not match view length {}",
                dst.len(),
                self.len()
            ));
        }
        let arr = self.as_f32_array();
        arr.copy_to(dst);
    }

    /// Copy data dari slice Rust ke SAB (Rust tulis data untuk JS)
    pub fn write(&self, src: &[f32]) {
        if src.len() != self.len() {
            boundary_fail(format!(
                "TensorView::write: source length {} does not match view length {}",
                src.len(),
                self.len()
            ));
        }
        let arr = self.as_f32_array();
        arr.copy_from(src);
    }
}

// -------------------------------------------------------------
// WASM TENSOR ↔ TENSOR VIEW
// -------------------------------------------------------------
#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(js_name = fromTensorView)]
    pub fn from_tensor_view(view: &TensorView) -> WasmTensor {
        let dims = normalize_rank4_shape(&view.shape, "WasmTensor::fromTensorView")
            .unwrap_or_else(|err| boundary_fail(err));
        validate_element_count(dims, view.len(), "WasmTensor::fromTensorView")
            .unwrap_or_else(|err| boundary_fail(err));

        let mut buf = vec![0f32; view.len()];
        view.read(&mut buf);

        let device = Default::default();
        let tensor_data = TensorData::new(buf, dims);
        WasmTensor {
            inner: Tensor::from_data(tensor_data, &device),
        }
    }

    #[wasm_bindgen(js_name = toTensorView)]
    pub fn to_tensor_view(&self, view: &mut TensorView) {
        let dims = self.inner.dims();
        validate_element_count(dims, view.len(), "WasmTensor::toTensorView")
            .unwrap_or_else(|err| boundary_fail(err));

        let data = self.inner.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        view.write(slice);
        view.set_shape(dims.into());
    }
}

#[cfg(test)]
mod tensor_boundary_tests {
    use super::{
        checked_element_count, checked_sab_byte_length, normalize_rank4_shape,
        validate_element_count,
    };

    #[test]
    fn rank4_shape_pads_short_shapes_with_singletons() {
        assert_eq!(
            normalize_rank4_shape(&[2, 3], "test").unwrap(),
            [2, 3, 1, 1]
        );
    }

    #[test]
    fn rank4_shape_rejects_hidden_fifth_axis() {
        assert!(normalize_rank4_shape(&[1, 2, 3, 4, 5], "test").is_err());
    }

    #[test]
    fn element_count_rejects_shape_overflow() {
        assert!(checked_element_count([usize::MAX, 2, 1, 1], "test").is_err());
    }

    #[test]
    fn element_count_rejects_buffer_mismatch() {
        let err = validate_element_count([2, 3, 1, 1], 5, "test").unwrap_err();
        assert!(err.contains("requires 6 elements"));
    }

    #[test]
    fn sab_byte_length_rejects_u32_overflow() {
        let too_many = (u32::MAX as usize / std::mem::size_of::<f32>()) + 1;
        assert!(checked_sab_byte_length(too_many).is_err());
    }

    #[test]
    fn sab_byte_length_accepts_representable_capacity() {
        assert_eq!(checked_sab_byte_length(1024).unwrap(), 4096);
    }
}
