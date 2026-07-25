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

    cached_params: usize,
}

macro_rules! insert_layer {
    ($self:ident, $map:ident, $id:expr, $layer:expr) => {{
        let new_params = $layer.num_params();

        if let Some(old) = $self.$map.insert($id, $layer) {
            $self.cached_params = $self.cached_params.saturating_sub(old.num_params());
        }

        $self.cached_params = $self.cached_params.saturating_add(new_params);
    }};
}

macro_rules! remove_layer {
    ($self:ident, $map:ident, $id:expr) => {{
        if let Some(old) = $self.$map.remove(&$id) {
            $self.cached_params = $self.cached_params.saturating_sub(old.num_params());
            true
        } else {
            false
        }
    }};
}

macro_rules! load_layer_state {
    ($self:ident, $map:ident, $id:expr, $data:expr) => {{
        match $self.$map.get_mut(&$id) {
            Some(layer) => {
                let old_params = layer.num_params();
                let result = layer.load_state($data);

                if result.is_ok() {
                    let new_params = layer.num_params();

                    $self.cached_params = $self
                        .cached_params
                        .saturating_sub(old_params)
                        .saturating_add(new_params);
                }

                result
            }
            None => Err("Not found".into()),
        }
    }};
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

            cached_params: 0,
        }
    }

    #[wasm_bindgen(js_name = initLayer)]
    pub fn init_layer(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let payload = header.validate_payload(payload)?;

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
            LAYER_LINEAR      => load_layer_state!(self, linears, layer_id, data),
            LAYER_NORM        => load_layer_state!(self, norms, layer_id, data),
            LAYER_CONV        => load_layer_state!(self, convs, layer_id, data),
            LAYER_ACTIVATION  => load_layer_state!(self, activations, layer_id, data),
            LAYER_EMBEDDING   => load_layer_state!(self, embeddings, layer_id, data),
            LAYER_GHOST       => load_layer_state!(self, ghosts, layer_id, data),
            LAYER_SEBLOCK     => load_layer_state!(self, seblocks, layer_id, data),
            LAYER_POOL | LAYER_SHIFT => Ok(()),
            _ => Err(format!("Unknown layer type for load_state: 0x{:02X}", layer_type)),
        }
    }

    #[wasm_bindgen(js_name = destroyLayer)]
    pub fn destroy_layer(&mut self, layer_id: LayerId, layer_type: u8) -> bool {
        match layer_type {
            LAYER_LINEAR      => remove_layer!(self, linears, layer_id),
            LAYER_NORM        => remove_layer!(self, norms, layer_id),
            LAYER_CONV        => remove_layer!(self, convs, layer_id),
            LAYER_ACTIVATION  => remove_layer!(self, activations, layer_id),
            LAYER_EMBEDDING   => remove_layer!(self, embeddings, layer_id),
            LAYER_GHOST       => remove_layer!(self, ghosts, layer_id),
            LAYER_SEBLOCK     => remove_layer!(self, seblocks, layer_id),
            LAYER_POOL        => self.pools.remove(&layer_id).is_some(),
            LAYER_SHIFT       => self.shifts.remove(&layer_id).is_some(),
            _ => false,
        }
    }

    #[wasm_bindgen(js_name = totalParams)]
    pub fn total_params(&self) -> usize {
        self.cached_params
    }

    fn init_linear(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_dim = c.read_usize()?;
        let out_dim = c.read_usize()?;
        let bias = c.read_bool()?;
        let layer = WasmLinear::new(in_dim, out_dim, bias);
        insert_layer!(self, linears, id, layer);
        Ok(())
    }

    fn init_norm(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let size = c.read_usize()?;
        let eps = c.read_option_f64()?;

        let layer = match header.variant {
            NORM_BATCH     => WasmNorm::new_batch_norm(size, eps),
            NORM_GROUP     => {
                let num_groups = c.read_usize()?;
                let num_channels = c.read_usize()?;
                WasmNorm::new_group_norm(num_groups, num_channels, eps)
            }
            NORM_INSTANCE  => WasmNorm::new_instance_norm(size, eps),
            NORM_LAYER     => WasmNorm::new_layer_norm(size, eps),
            NORM_RMS       => WasmNorm::new_rms_norm(size, eps),
            _ => return Err(format!("Unknown norm variant: 0x{:02X}", header.variant)),
        };
        insert_layer!(self, norms, id, layer);
        Ok(())
    }

    fn init_conv(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_ch = c.read_usize()?;
        let out_ch = c.read_usize()?;
        let kh = c.read_usize()?;
        let kw = c.read_usize()?;
        let sh = c.read_option_usize()?;
        let sw = c.read_option_usize()?;
        let ph = c.read_option_usize()?;
        let pw = c.read_option_usize()?;

        let layer = match header.variant {
            CONV_CONV1D          => WasmConv::new_conv1d(in_ch, out_ch, kh, sh, ph),
            CONV_CONV2D          => WasmConv::new_conv2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw),
            CONV_CONVTRANSPOSE2D => WasmConv::new_conv_transpose2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw),
            _ => return Err(format!("Unknown conv variant: 0x{:02X}", header.variant)),
        };
        insert_layer!(self, convs, id, layer);
        Ok(())
    }

    fn init_activation(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;

        let layer = match header.variant {
            ACT_GELU        => WasmActivation::new_gelu(),
            ACT_RELU        => WasmActivation::new_relu(),
            ACT_SIGMOID     => WasmActivation::new_sigmoid(),
            ACT_TANH        => WasmActivation::new_tanh(),
            ACT_HARDSWISH   => WasmActivation::new_hard_swish(),
            ACT_LEAKYRELU   => {
                let slope = c.read_option_f64()?;
                WasmActivation::new_leaky_relu(slope)
            }
            ACT_PRELU => {
                let num_params = c.read_option_usize()?;
                let alpha = c.read_option_f64()?;
                WasmActivation::new_prelu(num_params, alpha)
            }
            ACT_SWIGLU => {
                let d_in = c.read_usize()?;
                let d_out = c.read_usize()?;
                let bias = c.read_option_u32()?.map(|v| v != 0);
                WasmActivation::new_swiglu(d_in, d_out, bias)
            }
            ACT_HARDSIGMOID => {
                let alpha = c.read_option_f64()?;
                let beta = c.read_option_f64()?;
                WasmActivation::new_hard_sigmoid(alpha, beta)
            }
            ACT_SOFTPLUS => {
                let beta = c.read_option_f64()?;
                WasmActivation::new_softplus(beta)
            }
            ACT_MISH        => WasmActivation::new_mish(),
            ACT_SOFTMAX     => {
                let dim = c.read_usize()?;
                WasmActivation::new_softmax(dim)
            }
            ACT_LOGSOFTMAX  => {
                let dim = c.read_usize()?;
                WasmActivation::new_log_softmax(dim)
            }
            ACT_GLU         => {
                let dim = c.read_usize()?;
                WasmActivation::new_glu(dim)
            }
            _ => return Err(format!("Unknown activation variant: 0x{:02X}", header.variant)),
        };
        insert_layer!(self, activations, id, layer);
        Ok(())
    }

    fn init_embedding(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let vocab = c.read_usize()?;
        let d_model = c.read_usize()?;
        let layer = WasmEmbedding::new(vocab, d_model);
        insert_layer!(self, embeddings, id, layer);
        Ok(())
    }

    fn init_pool(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;

        let layer = match header.variant {
            POOL_MAXPOOL1D => {
                let k = c.read_usize()?;
                let s = c.read_option_usize()?;
                let p = c.read_option_usize()?;
                WasmPool::new_max_pool1d(k, s, p)
            }
            POOL_AVGPOOL1D => {
                let k = c.read_usize()?;
                let s = c.read_option_usize()?;
                let p = c.read_option_usize()?;
                WasmPool::new_avg_pool1d(k, s, p)
            }
            POOL_MAXPOOL2D => {
                let k = c.read_usize()?;
                let kw = c.read_usize()?;
                let sh = c.read_option_usize()?;
                let sw = c.read_option_usize()?;
                let ph = c.read_option_usize()?;
                let pw = c.read_option_usize()?;
                WasmPool::new_max_pool2d(k, kw, sh, sw, ph, pw)
            }
            POOL_AVGPOOL2D => {
                let k = c.read_usize()?;
                let kw = c.read_usize()?;
                let sh = c.read_option_usize()?;
                let sw = c.read_option_usize()?;
                let ph = c.read_option_usize()?;
                let pw = c.read_option_usize()?;
                WasmPool::new_avg_pool2d(k, kw, sh, sw, ph, pw)
            }
            POOL_ADAPTIVEAVGPOOL2D => {
                let oh = c.read_usize()?;
                let ow = c.read_usize()?;
                WasmPool::new_adaptive_avg_pool2d(oh, ow)
            }
            _ => return Err(format!("Unknown pool variant: 0x{:02X}", header.variant)),
        };
        self.pools.insert(id, layer);
        Ok(())
    }

    fn init_shift(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let shift_size = c.read_usize()?;

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
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_ch = c.read_usize()?;
        let out_ch = c.read_usize()?;
        let kh = c.read_usize()?;
        let kw = c.read_usize()?;
        let ratio = c.read_option_usize()?;
        let sh = c.read_option_usize()?;
        let sw = c.read_option_usize()?;
        let ph = c.read_option_usize()?;
        let pw = c.read_option_usize()?;

        let layer = WasmGhostModule::new(in_ch, out_ch, kh, kw, ratio, sh, sw, ph, pw);
        insert_layer!(self, ghosts, id, layer);
        Ok(())
    }

    fn init_seblock(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let channels = c.read_usize()?;
        let reduction = c.read_option_usize()?;

        let layer = WasmSeBlock::new(channels, reduction);
        insert_layer!(self, seblocks, id, layer);
        Ok(())
    }
}
