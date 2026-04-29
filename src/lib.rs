use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;
use js_sys::{Float32Array, Object, Reflect};

pub mod layers;
pub type WasmBackend = burn_ndarray::NdArray<f32>;

#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmTensor {
    pub(crate) inner: Tensor<WasmBackend, 4>, 
}

// -------------------------------------------------------------
// SHARED MEMORY BRIDGE
// -------------------------------------------------------------
#[wasm_bindgen]
pub struct TensorView {
    // Simpan sebagai JsValue, bukan extern type
    sab: JsValue,
    shape: Vec<usize>,
}

#[wasm_bindgen]
impl TensorView {
    #[wasm_bindgen(constructor)]
    pub fn new(total_elements: usize) -> Self {
        // Buat SharedArrayBuffer via JS Reflect
        let sab = js_sys::SharedArrayBuffer::new((total_elements * 4) as u32);
        TensorView { 
            sab: JsValue::from(sab), 
            shape: vec![1, 1, 1, 1] 
        }
    }

    pub fn ptr(&self) -> *mut f32 {
        let f32_array = Float32Array::new(&self.sab);
        f32_array.as_mut_ptr()
    }

    pub fn len(&self) -> usize {
        let sab = js_sys::SharedArrayBuffer::from(self.sab.clone());
        (sab.byte_length() / 4) as usize
    }

    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape(&mut self, shape: Vec<usize>) {
        self.shape = shape;
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    
    // Helper untuk akses internal
    fn as_f32_array(&self) -> Float32Array {
        Float32Array::new(&self.sab)
    }
}

// -------------------------------------------------------------
// WASM TENSOR
// -------------------------------------------------------------
#[wasm_bindgen]
impl WasmTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[f32], shape: &[usize]) -> WasmTensor {
        let device = Default::default();
        let mut dims = [1usize, 1, 1, 1];
        for (i, &d) in shape.iter().enumerate().take(4) {
            dims[i] = d;
        }
        let tensor_data = TensorData::new(data.to_vec(), dims);
        let tensor = Tensor::from_data(tensor_data, &device);
        WasmTensor { inner: tensor }
    }

    #[wasm_bindgen(js_name = fromTensorView)]
    pub fn from_tensor_view(view: &TensorView) -> WasmTensor {
        let device = Default::default();
        let mut dims = [1usize; 4];
        for (i, &d) in view.shape().iter().enumerate().take(4) {
            dims[i] = d;
        }
        
        let f32_array = view.as_f32_array();
        let slice = unsafe { 
            std::slice::from_raw_parts(f32_array.as_ptr(), view.len()) 
        };
        
        let tensor_data = TensorData::new(slice.to_vec(), dims);
        WasmTensor { inner: Tensor::from_data(tensor_data, &device) }
    }

    #[wasm_bindgen(js_name = toTensorView)]
    pub fn to_tensor_view(&self, view: &mut TensorView) {
        let data = self.inner.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        
        let f32_array = view.as_f32_array();
        f32_array.copy_from(slice);
        
        view.set_shape(self.inner.dims().into());
    }

    pub fn to_array(&self) -> Vec<f32> {
        let data = self.inner.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        self.inner.dims().into()
    }
    
    pub fn byte_length(&self) -> usize {
        self.inner.dims().iter().product::<usize>() * 4 
    }
}
