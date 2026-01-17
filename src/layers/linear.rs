use burn::prelude::*;

// 1. CONFIG
#[derive(Config, Debug)]
pub struct StrictLinearConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl StrictLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StrictLinear<B> {
        let linear = nn::LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .init(device);

        StrictLinear {
            inner: linear,
            expected_input_dim: self.d_input,
        }
    }
}

// 2. MODULE
#[derive(Module, Debug)]
pub struct StrictLinear<B: Backend> {
    inner: nn::Linear<B>,
    expected_input_dim: usize,
}

impl<B: Backend> StrictLinear<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_batch, dim] = input.dims();
        
        if dim != self.expected_input_dim {
            panic!("STRICT ERROR: Input dim {} != Layer dim {}", dim, self.expected_input_dim);
        }

        self.inner.forward(input)
    }
}
