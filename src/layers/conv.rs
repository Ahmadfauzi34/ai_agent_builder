use burn::prelude::*;

// --- CONFIG ---
#[derive(Config, Debug)]
pub struct StrictConv2dConfig {
    pub channels_in: usize,
    pub channels_out: usize,
    pub kernel_size: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 0)]
    pub padding: usize,
}

impl StrictConv2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictConv2d<B> {
        let conv = nn::conv::Conv2dConfig::new(
            [self.channels_in, self.channels_out],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(nn::PaddingConfig2d::Explicit(self.padding, self.padding))
        .init(device);

        StrictConv2d { inner: conv }
    }
}

// --- MODULE ---
#[derive(Module, Debug)]
pub struct StrictConv2d<B: Backend> {
    inner: nn::conv::Conv2d<B>,
}

impl<B: Backend> StrictConv2d<B> {
    // Input Gambar biasanya 4 Dimensi: [Batch, Channel, Height, Width]
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.inner.forward(input)
    }
}
