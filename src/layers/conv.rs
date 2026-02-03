use burn::prelude::*;
use burn::nn::conv::{
    Conv1d, Conv1dConfig,
    Conv2d, Conv2dConfig,
    ConvTranspose2d, ConvTranspose2dConfig
};
use burn::nn::PaddingConfig2d;

// --- CONFIGURATION ENUM ---
#[derive(Config, Debug)]
pub enum ConvolutionConfig {
    Conv1d(Conv1dConfig),
    Conv2d(Conv2dConfig),
    ConvTranspose2d(ConvTranspose2dConfig),
}

impl ConvolutionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Convolution<B> {
        match self {
            ConvolutionConfig::Conv1d(c) => Convolution::Conv1d(c.init(device)),
            ConvolutionConfig::Conv2d(c) => Convolution::Conv2d(c.init(device)),
            ConvolutionConfig::ConvTranspose2d(c) => Convolution::ConvTranspose2d(c.init(device)),
        }
    }
}

// --- MODULE ENUM ---
#[derive(Module, Debug)]
pub enum Convolution<B: Backend> {
    Conv1d(Conv1d<B>),
    Conv2d(Conv2d<B>),
    ConvTranspose2d(ConvTranspose2d<B>),
}

impl<B: Backend> Convolution<B> {
    // Input kita selalu 4D [Batch, Channel, H, W] dari WasmTensor
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            // Conv2d & Transpose2d native support 4D
            Convolution::Conv2d(layer) => layer.forward(input),
            Convolution::ConvTranspose2d(layer) => layer.forward(input),
            
            // Conv1d butuh 3D [Batch, Channel, Length]
            // Kita lakukan Reshape otomatis (Squeeze dimensi terakhir)
            Convolution::Conv1d(layer) => {
                let [b, c, h, _w] = input.dims();
                // Anggap Height sebagai Length, Width diabaikan (biasanya 1)
                let x_3d = input.reshape([b, c, h]);
                
                let out = layer.forward(x_3d);
                
                // Kembalikan ke 4D [Batch, Channel, Length, 1]
                let [b_out, c_out, l_out] = out.dims();
                out.reshape([b_out, c_out, l_out, 1])
            }
        }
    }
}
