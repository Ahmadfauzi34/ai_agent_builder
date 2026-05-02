use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use crate::WasmTensor;
use crate::protocol::*;
use crate::layers::linear::WasmLinear;
use crate::layers::norm::WasmNorm;
use crate::layers::conv::WasmConv;
use crate::layers::activation::WasmActivation;
use crate::layers::embedding::WasmEmbedding;
use crate::layers::pool::WasmPool;
use crate::layers::custom::shift::WasmShift;
use crate::layers::custom::ghost::WasmGhostModule;
use crate::layers::custom::seblock::WasmSeBlock;

type LayerId = u32;

#[wasm_bindgen]
pub struct LayerRegistry {
    linears:     HashMap<LayerId, WasmLinear>,
    norms:       HashMap<LayerId, WasmNorm>,
    convs:       HashMap<LayerId, WasmConv>,
    activations: HashMap<LayerId, WasmActivation>,
    embeddings:  HashMap<LayerId, WasmEmbedding>,
    pools:       HashMap<LayerId, WasmPool>,
    shifts:      HashMap<LayerId, WasmShift>,
    ghosts:      HashMap<LayerId, WasmGhostModule>,
    seblocks:    HashMap<LayerId, WasmSeBlock>,
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        LayerRegistry {
            linears:     HashMap::new(),
            norms:       HashMap::new(),
            convs:       HashMap::new(),
            activations: HashMap::new(),
            embeddings:  HashMap::new(),
            pools:       HashMap::new(),
            shifts:      HashMap::new(),
            ghosts:      HashMap::new(),
            seblocks:    HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = initLayer)]
    pub fn init_layer(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        match header.layer_type {
            LAYER_LINEAR      => self.init_linear(header, payload),
            LAYER_NORM        => self.init_norm(header, payload),
            LAYER_CONV        => self.init_conv(header, payload),
            LAYER_ACTIVATION  => self.init_activation(header, payload),
            LAYER_EMBEDDING   => self.init_embedding(header, payload),
            LAYER_POOL        => self.init_pool(header, payload),
            LAYER_SHIFT       => self.init_shift(header, payload),
            LAYER_GHOST       => self.init_ghost(header, payload),
            LAYER_SEBLOCK     => self.init_seblock(header, payload),
            _ => Err(format!("Unknown layer type: 0x{:02X}", header.layer_type)),
        }
    }

    #[wasm_bindgen(js_name = forwardLayer)]
    pub fn forward_layer(&self, layer_id: LayerId, layer_type: u8, input: &WasmTensor) -> Result<WasmTensor, String> {
        match layer_type {
            LAYER_LINEAR      => self.linears.get(&layer_id).map(|l| l.forward(input)).ok_or("Linear not found".into()),
            LAYER_NORM        => self.norms.get(&layer_id).map(|l| l.forward(input)).ok_or("Norm not found".into()),
            LAYER_CONV        => self.convs.get(&layer_id).map(|l| l.forward(input)).ok_or("Conv not found".into()),
            LAYER_ACTIVATION  => self.activations.get(&layer_id).map(|l| l.forward(input)).ok_or("Activation not found".into()),
            LAYER_EMBEDDING   => self.embeddings.get(&layer_id).map(|l| l.forward(input)).ok_or("Embedding not found".into()),
            LAYER_POOL        => self.pools.get(&layer_id).map(|l| l.forward(input)).ok_or("Pool not found".into()),
            LAYER_SHIFT       => self.shifts.get(&layer_id).map(|l| l.forward(input)).ok_or("Shift not found".into()),
            LAYER_GHOST       => self.ghosts.get(&layer_id).map(|l| l.forward(input)).ok_or("Ghost not found".into()),
            LAYER_SEBLOCK     => self.seblocks.get(&layer_id).map(|l| l.forward(input)).ok_or("SEBlock not found".into()),
            _ => Err(format!("Unknown layer type for forward: 0x{:02X}", layer_type)),
        }
    }

    #[wasm_bindgen(js_name = getLayerState)]
    pub fn get_layer_state(&self, layer_id: LayerId, layer_type: u8) -> Result<Vec<u8>, String> {
        match layer_type {
            LAYER_LINEAR      => self.linears.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_NORM        => self.norms.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_CONV        => self.convs.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_ACTIVATION  => self.activations.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_EMBEDDING   => self.embeddings.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_GHOST       => self.ghosts.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_SEBLOCK     => self.seblocks.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_POOL | LAYER_SHIFT => Ok(vec![]),
            _ => Err(format!("Unknown layer type for get_state: 0x{:02X}", layer_type)),
        }
    }

    #[wasm_bindgen(js_name = loadLayerState)]
    pub fn load_layer_state(&mut self, layer_id: LayerId, layer_type: u8, data: &[u8]) -> Result<(), String> {
        match layer_type {
            LAYER_LINEAR      => self.linears.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_NORM        => self.norms.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_CONV        => self.convs.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_ACTIVATION  => self.activations.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_EMBEDDING   => self.embeddings.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_GHOST       => self.ghosts.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_SEBLOCK     => self.seblocks.get_mut(&layer_id).ok_or("Not found")?.load_state(data),
            LAYER_POOL | LAYER_SHIFT => Ok(()),
            _ => Err(format!("Unknown layer type for load_state: 0x{:02X}", layer_type)),
        }
    }

    #[wasm_bindgen(js_name = destroyLayer)]
    pub fn destroy_layer(&mut self, layer_id: LayerId, layer_type: u8) -> bool {
        match layer_type {
            LAYER_LINEAR      => self.linears.remove(&layer_id).is_some(),
            LAYER_NORM        => self.norms.remove(&layer_id).is_some(),
            LAYER_CONV        => self.convs.remove(&layer_id).is_some(),
            LAYER_ACTIVATION  => self.activations.remove(&layer_id).is_some(),
            LAYER_EMBEDDING   => self.embeddings.remove(&layer_id).is_some(),
            LAYER_POOL        => self.pools.remove(&layer_id).is_some(),
            LAYER_SHIFT       => self.shifts.remove(&layer_id).is_some(),
            LAYER_GHOST       => self.ghosts.remove(&layer_id).is_some(),
            LAYER_SEBLOCK     => self.seblocks.remove(&layer_id).is_some(),
            _ => false,
        }
    }

    #[wasm_bindgen(js_name = totalParams)]
    pub fn total_params(&self) -> usize {
        let mut total = 0;
        total += self.linears.values().map(|l| l.num_params()).sum::<usize>();
        total += self.norms.values().map(|l| l.num_params()).sum::<usize>();
        total += self.convs.values().map(|l| l.num_params()).sum::<usize>();
        total += self.activations.values().map(|l| l.num_params()).sum::<usize>();
        total += self.embeddings.values().map(|l| l.num_params()).sum::<usize>();
        total += self.ghosts.values().map(|l| l.num_params()).sum::<usize>();
        total += self.seblocks.values().map(|l| l.num_params()).sum::<usize>();
        total
    }

    fn init_linear(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let in_dim = read_usize(payload, 4)?;
        let out_dim = read_usize(payload, 8)?;
        let bias = read_bool(payload, 12)?;
        let layer = WasmLinear::new(in_dim, out_dim, bias);
        self.linears.insert(id, layer);
        Ok(())
    }

    fn init_norm(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let size = read_usize(payload, 4)?;
        let eps = read_option_f64(payload, 8)?;

        let layer = match header.variant {
            NORM_BATCH     => WasmNorm::new_batch_norm(size, eps),
            NORM_GROUP     => {
                let num_groups = read_usize(payload, 17)?;
                let num_channels = read_usize(payload, 21)?;
                WasmNorm::new_group_norm(num_groups, num_channels, eps)
            }
            NORM_INSTANCE  => WasmNorm::new_instance_norm(size, eps),
            NORM_LAYER     => WasmNorm::new_layer_norm(size, eps),
            NORM_RMS       => WasmNorm::new_rms_norm(size, eps),
            _ => return Err(format!("Unknown norm variant: 0x{:02X}", header.variant)),
        };
        self.norms.insert(id, layer);
        Ok(())
    }

    fn init_conv(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let in_ch = read_usize(payload, 4)?;
        let out_ch = read_usize(payload, 8)?;
        let kh = read_usize(payload, 12)?;
        let kw = read_usize(payload, 16)?;
        let sh = read_option_u32(payload, 20)?.map(|v| v as usize);
        let sw = read_option_u32(payload, 25)?.map(|v| v as usize);
        let ph = read_option_u32(payload, 30)?.map(|v| v as usize);
        let pw = read_option_u32(payload, 35)?.map(|v| v as usize);

        let layer = match header.variant {
            CONV_CONV1D          => WasmConv::new_conv1d(in_ch, out_ch, kh, sh, ph),
            CONV_CONV2D          => WasmConv::new_conv2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw),
            CONV_CONVTRANSPOSE2D => WasmConv::new_conv_transpose2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw),
            _ => return Err(format!("Unknown conv variant: 0x{:02X}", header.variant)),
        };
        self.convs.insert(id, layer);
        Ok(())
    }

    fn init_activation(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;

        let layer = match header.variant {
            ACT_GELU        => WasmActivation::new_gelu(),
            ACT_RELU        => WasmActivation::new_relu(),
            ACT_SIGMOID     => WasmActivation::new_sigmoid(),
            ACT_TANH        => WasmActivation::new_tanh(),
            ACT_HARDSWISH   => WasmActivation::new_hard_swish(),
            ACT_LEAKYRELU   => {
                let slope = read_option_f64(payload, 4)?;
                WasmActivation::new_leaky_relu(slope)
            }
            ACT_PRELU => {
                let num_params = read_option_u32(payload, 4).map(|v| v as usize);
                let alpha = read_option_f64(payload, 9)?;
                WasmActivation::new_prelu(num_params, alpha)
            }
            ACT_SWIGLU => {
                let d_in = read_usize(payload, 4)?;
                let d_out = read_usize(payload, 8)?;
                let bias = read_option_u32(payload, 12).map(|v| v != 0);
                WasmActivation::new_swiglu(d_in, d_out, bias)
            }
            ACT_HARDSIGMOID => {
                let alpha = read_option_f64(payload, 4)?;
                let beta = read_option_f64(payload, 13)?;
                WasmActivation::new_hard_sigmoid(alpha, beta)
            }
            ACT_SOFTPLUS => {
                let beta = read_option_f64(payload, 4)?;
                WasmActivation::new_softplus(beta)
            }
            ACT_MISH        => WasmActivation::new_mish(),
            ACT_SOFTMAX     => {
                let dim = read_usize(payload, 4)?;
                WasmActivation::new_softmax(dim)
            }
            ACT_LOGSOFTMAX  => {
                let dim = read_usize(payload, 4)?;
                WasmActivation::new_log_softmax(dim)
            }
            ACT_GLU         => {
                let dim = read_usize(payload, 4)?;
                WasmActivation::new_glu(dim)
            }
            _ => return Err(format!("Unknown activation variant: 0x{:02X}", header.variant)),
        };
        self.activations.insert(id, layer);
        Ok(())
    }

    fn init_embedding(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let vocab = read_usize(payload, 4)?;
        let d_model = read_usize(payload, 8)?;
        let layer = WasmEmbedding::new(vocab, d_model);
        self.embeddings.insert(id, layer);
        Ok(())
    }

    fn init_pool(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;

        let layer = match header.variant {
            POOL_MAXPOOL1D => {
                let k = read_usize(payload, 4)?;
                let s = read_option_u32(payload, 8)?.map(|v| v as usize);
                let p = read_option_u32(payload, 13)?.map(|v| v as usize);
                WasmPool::new_max_pool1d(k, s, p)
            }
            POOL_AVGPOOL1D => {
                let k = read_usize(payload, 4)?;
                let s = read_option_u32(payload, 8)?.map(|v| v as usize);
                let p = read_option_u32(payload, 13)?.map(|v| v as usize);
                WasmPool::new_avg_pool1d(k, s, p)
            }
            POOL_MAXPOOL2D => {
                let k = read_usize(payload, 4)?;
                let kw = read_usize(payload, 8)?;
                let sh = read_option_u32(payload, 12)?.map(|v| v as usize);
                let sw = read_option_u32(payload, 17)?.map(|v| v as usize);
                let ph = read_option_u32(payload, 22)?.map(|v| v as usize);
                let pw = read_option_u32(payload, 27)?.map(|v| v as usize);
                WasmPool::new_max_pool2d(k, kw, sh, sw, ph, pw)
            }
            POOL_AVGPOOL2D => {
                let k = read_usize(payload, 4)?;
                let kw = read_usize(payload, 8)?;
                let sh = read_option_u32(payload, 12)?.map(|v| v as usize);
                let sw = read_option_u32(payload, 17)?.map(|v| v as usize);
                let ph = read_option_u32(payload, 22)?.map(|v| v as usize);
                let pw = read_option_u32(payload, 27)?.map(|v| v as usize);
                WasmPool::new_avg_pool2d(k, kw, sh, sw, ph, pw)
            }
            POOL_ADAPTIVEAVGPOOL2D => {
                let oh = read_usize(payload, 4)?;
                let ow = read_usize(payload, 8)?;
                WasmPool::new_adaptive_avg_pool2d(oh, ow)
            }
            _ => return Err(format!("Unknown pool variant: 0x{:02X}", header.variant)),
        };
        self.pools.insert(id, layer);
        Ok(())
    }

    fn init_shift(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let shift_size = read_usize(payload, 4)?;

        let layer = match header.variant {
            SHIFT_UP    => WasmShift::new_shift_up(shift_size),
            SHIFT_DOWN  => WasmShift::new_shift_down(shift_size),
            SHIFT_LEFT  => WasmShift::new_shift_left(shift_size),
            SHIFT_RIGHT => WasmShift::new_shift_right(shift_size),
            _ => return Err(format!("Unknown shift variant: 0x{:02X}", header.variant)),
        };
        self.shifts.insert(id, layer);
        Ok(())
    }

    fn init_ghost(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let in_ch = read_usize(payload, 4)?;
        let out_ch = read_usize(payload, 8)?;
        let kh = read_usize(payload, 12)?;
        let kw = read_usize(payload, 16)?;
        let ratio = read_option_u32(payload, 20)?.map(|v| v as usize);
        let sh = read_option_u32(payload, 25)?.map(|v| v as usize);
        let sw = read_option_u32(payload, 30)?.map(|v| v as usize);
        let ph = read_option_u32(payload, 35)?.map(|v| v as usize);
        let pw = read_option_u32(payload, 40)?.map(|v| v as usize);

        let layer = WasmGhostModule::new(in_ch, out_ch, kh, kw, ratio, sh, sw, ph, pw);
        self.ghosts.insert(id, layer);
        Ok(())
    }

    fn init_seblock(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let id = read_u32(payload, 0)?;
        let channels = read_usize(payload, 4)?;
        let reduction = read_option_u32(payload, 8)?.map(|v| v as usize);

        let layer = WasmSeBlock::new(channels, reduction);
        self.seblocks.insert(id, layer);
        Ok(())
    }
}

