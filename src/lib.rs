use wasm_bindgen::prelude::*;
use burn::prelude::*;
use burn::tensor::TensorData;
use js_sys::Float32Array;

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
    sab: JsValue,
    shape: Vec<usize>,
}

#[wasm_bindgen]
impl TensorView {
    #[wasm_bindgen(constructor)]
    pub fn new(total_elements: usize) -> Self {
        let sab = js_sys::SharedArrayBuffer::new((total_elements * 4) as u32);
        TensorView { 
            sab: JsValue::from(sab), 
            shape: vec![1, 1, 1, 1] 
        }
    }

    pub fn len(&self) -> usize {
        let arr = Float32Array::new(&self.sab);
        arr.length() as usize
    }

    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape(&mut self, shape: Vec<usize>) {
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
        let arr = self.as_f32_array();
        arr.copy_to(dst);
    }
    
    /// Copy data dari slice Rust ke SAB (Rust tulis data untuk JS)
    pub fn write(&self, src: &[f32]) {
        let arr = self.as_f32_array();
        arr.copy_from(src);
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

    /// Rust baca dari SAB (JS tulis dulu ke SAB, lalu panggil ini)
    #[wasm_bindgen(js_name = fromTensorView)]
    pub fn from_tensor_view(view: &TensorView) -> WasmTensor {
        let device = Default::default();
        let mut dims = [1usize; 4];
        for (i, &d) in view.shape().iter().enumerate().take(4) {
            dims[i] = d;
        }
        
        let mut buf = vec![0f32; view.len()];
        view.read(&mut buf);
        
        let tensor_data = TensorData::new(buf, dims);
        WasmTensor { inner: Tensor::from_data(tensor_data, &device) }
    }

    /// Rust tulis ke SAB (JS baca hasil dari SAB setelah ini)
    #[wasm_bindgen(js_name = toTensorView)]
    pub fn to_tensor_view(&self, view: &mut TensorView) {
        let data = self.inner.to_data();
        let slice = data.as_slice::<f32>().unwrap();
        view.write(slice);
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
