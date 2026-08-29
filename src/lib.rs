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
mod tensor_invariant_tests;
#[cfg(test)]
mod tests;

pub type WasmBackend = burn_ndarray::NdArray<f32>;

pub(crate) fn normalize_4d_shape(
    shape: &[usize],
    total_elements: usize,
) -> Result<[usize; 4], String> {
    if shape.len() > 4 {
        return Err(format!(
            "tensor shape rank must be <= 4, got {}",
            shape.len()
        ));
    }

    let mut dims = [1usize; 4];
    let mut elements = 1usize;
    for (i, &dim) in shape.iter().enumerate() {
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| "tensor shape element count overflow".to_string())?;
        dims[i] = dim;
    }

    if elements != total_elements {
        return Err(format!(
            "tensor shape/data mismatch: shape has {} elements, buffer has {}",
            elements, total_elements
        ));
    }

    Ok(dims)
}

pub(crate) fn tensor_view_byte_len(total_elements: usize) -> Result<u32, String> {
    let bytes = total_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "TensorView byte length overflow".to_string())?;
    u32::try_from(bytes).map_err(|_| {
        format!(
            "TensorView byte length {} exceeds SharedArrayBuffer u32 limit",
            bytes
        )
    })
}

#[inline]
fn bridge_fail<T>(message: String) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen::throw_str(&message)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        panic!("{message}")
    }
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
        let device = Default::default();
        let dims = normalize_4d_shape(shape, data.len()).unwrap_or_else(bridge_fail);
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
        self.inner.dims().iter().product::<usize>() * core::mem::size_of::<f32>()
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
        let byte_len = tensor_view_byte_len(total_elements).unwrap_or_else(bridge_fail);
        let sab = js_sys::SharedArrayBuffer::new(byte_len);
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
        normalize_4d_shape(&shape, self.len()).unwrap_or_else(bridge_fail);
        self.shape = shape;
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
            bridge_fail::<()>(format!(
                "TensorView read length mismatch: view has {}, destination has {}",
                self.len(),
                dst.len()
            ));
        }
        let arr = self.as_f32_array();
        arr.copy_to(dst);
    }

    /// Copy data dari slice Rust ke SAB (Rust tulis data untuk JS)
    pub fn write(&self, src: &[f32]) {
        if src.len() != self.len() {
            bridge_fail::<()>(format!(
                "TensorView write length mismatch: view has {}, source has {}",
                self.len(),
                src.len()
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
        let device = Default::default();
        let shape = view.shape();
        let dims = normalize_4d_shape(&shape, view.len()).unwrap_or_else(bridge_fail);

        let mut buf = vec![0f32; view.len()];
        view.read(&mut buf);

        let tensor_data = TensorData::new(buf, dims);
        WasmTensor {
            inner: Tensor::from_data(tensor_data, &device),
        }
    }

    #[wasm_bindgen(js_name = toTensorView)]
    pub fn to_tensor_view(&self, view: &mut TensorView) {
        let data = self.inner.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        view.write(slice);
        view.set_shape(self.inner.dims().into());
    }
}
