//! Fasad WASM tunggal — domain `registry` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::registry::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentLayerSpec;
use crate::registry::runtime_contract::binding::{
    operation_binding_snapshot, BINDING_CAPABILITIES_V1,
};
use crate::registry::runtime_contract::inventory::{inventory_snapshot, INVENTORY_CAPABILITIES_V1};

// Opsi C Fase 2: imports untuk #[wasm_bindgen] impl LayerRegistry yang pindah ke sini.
use crate::layers::activation::WasmActivation;
use crate::layers::binary::WasmBinary;
use crate::layers::conv::WasmConv;
use crate::layers::custom::feature_norm::WasmFeatureNorm;
use crate::layers::custom::ghost::WasmGhostModule;
use crate::layers::custom::seblock::WasmSeBlock;
use crate::layers::custom::shift::WasmShift;
use crate::layers::embedding::WasmEmbedding;
use crate::layers::linear::WasmLinear;
use crate::layers::norm::WasmNorm;
use crate::layers::pool::WasmPool;
use crate::protocol::*;
use crate::registry::{read_run_step, validate_plan, LayerId, LayerInitIdentity, LayerRegistry};
use crate::WasmTensor;
use std::collections::HashMap;

#[wasm_bindgen(js_name = layerRegistryOperationBindingCapabilities)]
pub fn layer_registry_operation_binding_capabilities() -> String {
    BINDING_CAPABILITIES_V1.to_string()
}

#[wasm_bindgen(js_name = layerRegistryInventoryCapabilities)]
pub fn layer_registry_inventory_capabilities() -> String {
    INVENTORY_CAPABILITIES_V1.to_string()
}

// ============================================================
// Opsi C Fase 2 — pindahan murni dari `src/registry.rs`:
// #[wasm_bindgen] impl LayerRegistry (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

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

// ============================================================
// IMPL #1 — lifecycle + init per tipe
// ============================================================
#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        LayerRegistry {
            linears: HashMap::new(),
            norms: HashMap::new(),
            convs: HashMap::new(),
            activations: HashMap::new(),
            embeddings: HashMap::new(),
            pools: HashMap::new(),
            shifts: HashMap::new(),
            ghosts: HashMap::new(),
            seblocks: HashMap::new(),
            binaries: HashMap::new(),
            feature_norms: HashMap::new(),
            init_identities: HashMap::new(),
            cached_params: 0,
        }
    }

    #[wasm_bindgen(js_name = initLayer)]
    pub fn init_layer(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let payload = header.validate_payload(payload)?;
        let mut id_cursor = PayloadCursor::new(payload);
        let layer_id = id_cursor.read_u32()?;
        let result = match header.layer_type {
            LAYER_LINEAR => self.init_linear(header, payload),
            LAYER_NORM => self.init_norm(header, payload),
            LAYER_CONV => self.init_conv(header, payload),
            LAYER_ACTIVATION => self.init_activation(header, payload),
            LAYER_EMBEDDING => self.init_embedding(header, payload),
            LAYER_POOL => self.init_pool(header, payload),
            LAYER_SHIFT => self.init_shift(header, payload),
            LAYER_GHOST => self.init_ghost(header, payload),
            LAYER_SEBLOCK => self.init_seblock(header, payload),
            LAYER_BINARY => self.init_binary(header, payload),
            LAYER_FEATURE_NORM => self.init_feature_norm(header, payload),
            _ => Err(format!("Unknown layer type: 0x{:02X}", header.layer_type)),
        };
        if result.is_ok() {
            self.init_identities.insert(
                (header.layer_type, layer_id),
                LayerInitIdentity::new(header, payload),
            );
        }
        result
    }

    /// Return the canonical init identity for a live layer.
    /// The identity is derived from the validated payload prefix plus variant/flags,
    /// so ignored outer trailing bytes never affect provenance.
    #[wasm_bindgen(js_name = layerInitFingerprint)]
    pub fn layer_init_fingerprint(
        &self,
        layer_type: u8,
        layer_id: LayerId,
    ) -> Result<String, String> {
        if !self.layer_exists(layer_type, layer_id) {
            return Err(format!(
                "layerInitFingerprint: layer type 0x{layer_type:02X} id {layer_id} not found"
            ));
        }
        self.init_identities
            .get(&(layer_type, layer_id))
            .map(|identity| identity.fingerprint(layer_type, layer_id))
            .ok_or_else(|| {
                format!(
                    "layerInitFingerprint: identity missing for live layer type 0x{layer_type:02X} id {layer_id}"
                )
            })
    }

    #[wasm_bindgen(js_name = forwardLayer)]
    pub fn forward_layer(
        &self,
        layer_id: LayerId,
        layer_type: u8,
        input: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        // Complaint #18: validate the input tensor against the allocation
        // budget before dispatch, so an oversized run fails here as a
        // structured per-call error instead of a raw `unreachable` trap
        // deep inside the op.
        check_numel(&input.shape(), "forwardLayer input")?;
        match layer_type {
            LAYER_LINEAR => self
                .linears
                .get(&layer_id)
                .ok_or_else(|| "Linear not found".to_string())?
                .try_forward(input),
            LAYER_NORM => self
                .norms
                .get(&layer_id)
                .ok_or_else(|| "Norm not found".to_string())?
                .try_forward(input),
            LAYER_CONV => self
                .convs
                .get(&layer_id)
                .ok_or_else(|| "Conv not found".to_string())?
                .try_forward(input),
            LAYER_ACTIVATION => self
                .activations
                .get(&layer_id)
                .ok_or_else(|| "Activation not found".to_string())?
                .try_forward(input),
            LAYER_EMBEDDING => self
                .embeddings
                .get(&layer_id)
                .ok_or_else(|| "Embedding not found".to_string())?
                .try_forward(input),
            LAYER_POOL => self
                .pools
                .get(&layer_id)
                .ok_or_else(|| "Pool not found".to_string())?
                .try_forward(input),
            LAYER_SHIFT => Ok(self
                .shifts
                .get(&layer_id)
                .ok_or_else(|| "Shift not found".to_string())?
                .forward(input)),
            LAYER_GHOST => Ok(self
                .ghosts
                .get(&layer_id)
                .ok_or_else(|| "Ghost not found".to_string())?
                .forward(input)),
            LAYER_SEBLOCK => Ok(self
                .seblocks
                .get(&layer_id)
                .ok_or_else(|| "SEBlock not found".to_string())?
                .forward(input)),
            LAYER_FEATURE_NORM => self
                .feature_norms
                .get(&layer_id)
                .ok_or_else(|| "FeatureNorm not found".to_string())?
                .forward(input),
            _ => Err(format!(
                "Unknown layer type for forward: 0x{:02X}",
                layer_type
            )),
        }
    }

    #[wasm_bindgen(js_name = getLayerState)]
    pub fn get_layer_state(&self, layer_id: LayerId, layer_type: u8) -> Result<Vec<u8>, String> {
        match layer_type {
            LAYER_LINEAR => self.linears.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_NORM => self.norms.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_CONV => self.convs.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_ACTIVATION => self
                .activations
                .get(&layer_id)
                .ok_or("Not found")?
                .get_state(),
            LAYER_EMBEDDING => self
                .embeddings
                .get(&layer_id)
                .ok_or("Not found")?
                .get_state(),
            LAYER_GHOST => self.ghosts.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_SEBLOCK => self.seblocks.get(&layer_id).ok_or("Not found")?.get_state(),
            LAYER_POOL | LAYER_SHIFT | LAYER_BINARY | LAYER_FEATURE_NORM => {
                if self.layer_exists(layer_type, layer_id) {
                    Ok(vec![])
                } else {
                    Err("Not found".into())
                }
            }
            _ => Err(format!(
                "Unknown layer type for get_state: 0x{:02X}",
                layer_type
            )),
        }
    }

    #[wasm_bindgen(js_name = loadLayerState)]
    pub fn load_layer_state(
        &mut self,
        layer_id: LayerId,
        layer_type: u8,
        data: &[u8],
    ) -> Result<(), String> {
        match layer_type {
            LAYER_LINEAR => load_layer_state!(self, linears, layer_id, data),
            LAYER_NORM => load_layer_state!(self, norms, layer_id, data),
            LAYER_CONV => load_layer_state!(self, convs, layer_id, data),
            LAYER_ACTIVATION => load_layer_state!(self, activations, layer_id, data),
            LAYER_EMBEDDING => load_layer_state!(self, embeddings, layer_id, data),
            LAYER_GHOST => load_layer_state!(self, ghosts, layer_id, data),
            LAYER_SEBLOCK => load_layer_state!(self, seblocks, layer_id, data),
            LAYER_POOL | LAYER_SHIFT | LAYER_BINARY | LAYER_FEATURE_NORM => {
                if !self.layer_exists(layer_type, layer_id) {
                    return Err("Not found".into());
                }
                if !data.is_empty() {
                    return Err(format!(
                        "loadLayerState: stateless layer type 0x{layer_type:02X} requires empty state"
                    ));
                }
                Ok(())
            }
            _ => Err(format!(
                "Unknown layer type for load_state: 0x{:02X}",
                layer_type
            )),
        }
    }

    #[wasm_bindgen(js_name = destroyLayer)]
    pub fn destroy_layer(&mut self, layer_id: LayerId, layer_type: u8) -> bool {
        let removed = match layer_type {
            LAYER_LINEAR => remove_layer!(self, linears, layer_id),
            LAYER_NORM => remove_layer!(self, norms, layer_id),
            LAYER_CONV => remove_layer!(self, convs, layer_id),
            LAYER_ACTIVATION => remove_layer!(self, activations, layer_id),
            LAYER_EMBEDDING => remove_layer!(self, embeddings, layer_id),
            LAYER_GHOST => remove_layer!(self, ghosts, layer_id),
            LAYER_SEBLOCK => remove_layer!(self, seblocks, layer_id),
            LAYER_POOL => self.pools.remove(&layer_id).is_some(),
            LAYER_SHIFT => self.shifts.remove(&layer_id).is_some(),
            LAYER_BINARY => self.binaries.remove(&layer_id).is_some(),
            LAYER_FEATURE_NORM => remove_layer!(self, feature_norms, layer_id),
            _ => false,
        };
        if removed {
            self.init_identities.remove(&(layer_type, layer_id));
        }
        removed
    }

    #[wasm_bindgen(js_name = totalParams)]
    pub fn total_params(&self) -> usize {
        self.cached_params
    }

    fn init_linear(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_dim = c.read_dim("linear in_dim")?;
        let out_dim = c.read_dim("linear out_dim")?;
        let bias = c.read_bool()?;
        check_numel(&[in_dim, out_dim], "linear weight")?;
        if bias {
            check_numel(&[out_dim], "linear bias")?;
        }
        let layer = WasmLinear::new(in_dim, out_dim, bias);
        insert_layer!(self, linears, id, layer);
        Ok(())
    }

    fn init_norm(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let size = c.read_dim("norm size")?;
        let eps = c.read_option_f64()?;
        check_numel(&[size], "norm weight")?;
        let layer = match header.variant {
            NORM_BATCH => WasmNorm::new_batch_norm(size, eps),
            NORM_GROUP => {
                let num_groups = c.read_dim("group_norm num_groups")?;
                let num_channels = c.read_dim("group_norm num_channels")?;
                check_numel(&[num_channels], "group_norm weight")?;
                WasmNorm::try_new_group_norm(num_groups, num_channels, eps)?
            }
            NORM_INSTANCE => WasmNorm::new_instance_norm(size, eps),
            NORM_LAYER => WasmNorm::new_layer_norm(size, eps),
            NORM_RMS => WasmNorm::try_new_rms_norm(size, eps)?,
            _ => return Err(format!("Unknown norm variant: 0x{:02X}", header.variant)),
        };
        insert_layer!(self, norms, id, layer);
        Ok(())
    }

    fn init_conv(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_ch = c.read_dim("conv in_ch")?;
        let out_ch = c.read_dim("conv out_ch")?;
        let kh = c.read_dim("conv kh")?;
        let kw = c.read_dim("conv kw")?;
        let sh = c.read_bounded_opt_usize("conv sh")?;
        let sw = c.read_bounded_opt_usize("conv sw")?;
        let ph = c.read_bounded_opt_usize("conv ph")?;
        let pw = c.read_bounded_opt_usize("conv pw")?;
        check_numel(&[out_ch, in_ch, kh, kw], "conv weight")?;
        let layer = match header.variant {
            CONV_CONV1D => WasmConv::try_new_conv1d(in_ch, out_ch, kh, sh, ph)?,
            CONV_CONV2D => WasmConv::try_new_conv2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw)?,
            CONV_CONVTRANSPOSE2D => {
                WasmConv::try_new_conv_transpose2d(in_ch, out_ch, kh, kw, sh, sw, ph, pw)?
            }
            _ => return Err(format!("Unknown conv variant: 0x{:02X}", header.variant)),
        };
        insert_layer!(self, convs, id, layer);
        Ok(())
    }

    fn init_activation(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let layer = match header.variant {
            ACT_GELU => WasmActivation::new_gelu(),
            ACT_RELU => WasmActivation::new_relu(),
            ACT_SIGMOID => WasmActivation::new_sigmoid(),
            ACT_TANH => WasmActivation::new_tanh(),
            ACT_HARDSWISH => WasmActivation::new_hard_swish(),
            ACT_LEAKYRELU => {
                let slope = c.read_option_f64()?;
                WasmActivation::new_leaky_relu(slope)
            }
            ACT_PRELU => {
                let num_params = c.read_bounded_opt_usize("prelu num_params")?;
                if let Some(n) = num_params {
                    check_numel(&[n], "prelu weight")?;
                }
                let alpha = c.read_option_f64()?;
                WasmActivation::new_prelu(num_params, alpha)
            }
            ACT_SWIGLU => {
                let d_in = c.read_dim("swiglu d_in")?;
                let d_out = c.read_dim("swiglu d_out")?;
                let bias = c.read_option_u32()?.map(|v| v != 0);
                check_numel(&[d_in, d_out], "swiglu weight")?;
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
            ACT_MISH => WasmActivation::new_mish(),
            ACT_SOFTMAX => {
                let dim = c.read_dim("softmax dim")?;
                WasmActivation::new_softmax(dim)
            }
            ACT_LOGSOFTMAX => {
                let dim = c.read_dim("log_softmax dim")?;
                WasmActivation::new_log_softmax(dim)
            }
            ACT_GLU => {
                let dim = c.read_dim("glu dim")?;
                WasmActivation::new_glu(dim)
            }
            _ => {
                return Err(format!(
                    "Unknown activation variant: 0x{:02X}",
                    header.variant
                ))
            }
        };
        insert_layer!(self, activations, id, layer);
        Ok(())
    }

    fn init_embedding(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let vocab = c.read_dim("embedding vocab")?;
        let d_model = c.read_dim("embedding d_model")?;
        check_numel(&[vocab, d_model], "embedding weight")?;
        let layer = WasmEmbedding::new(vocab, d_model);
        insert_layer!(self, embeddings, id, layer);
        Ok(())
    }

    fn init_pool(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let layer = match header.variant {
            POOL_MAXPOOL1D => {
                let k = c.read_dim("max_pool1d k")?;
                let s = c.read_bounded_opt_usize("max_pool1d s")?;
                let p = c.read_bounded_opt_usize("max_pool1d p")?;
                WasmPool::try_new_max_pool1d(k, s, p)?
            }
            POOL_AVGPOOL1D => {
                let k = c.read_dim("avg_pool1d k")?;
                let s = c.read_bounded_opt_usize("avg_pool1d s")?;
                let p = c.read_bounded_opt_usize("avg_pool1d p")?;
                WasmPool::try_new_avg_pool1d(k, s, p)?
            }
            POOL_MAXPOOL2D => {
                let k = c.read_dim("max_pool2d k")?;
                let kw = c.read_dim("max_pool2d kw")?;
                let sh = c.read_bounded_opt_usize("max_pool2d sh")?;
                let sw = c.read_bounded_opt_usize("max_pool2d sw")?;
                let ph = c.read_bounded_opt_usize("max_pool2d ph")?;
                let pw = c.read_bounded_opt_usize("max_pool2d pw")?;
                WasmPool::try_new_max_pool2d(k, kw, sh, sw, ph, pw)?
            }
            POOL_AVGPOOL2D => {
                let k = c.read_dim("avg_pool2d k")?;
                let kw = c.read_dim("avg_pool2d kw")?;
                let sh = c.read_bounded_opt_usize("avg_pool2d sh")?;
                let sw = c.read_bounded_opt_usize("avg_pool2d sw")?;
                let ph = c.read_bounded_opt_usize("avg_pool2d ph")?;
                let pw = c.read_bounded_opt_usize("avg_pool2d pw")?;
                WasmPool::try_new_avg_pool2d(k, kw, sh, sw, ph, pw)?
            }
            POOL_ADAPTIVEAVGPOOL2D => {
                let oh = c.read_dim("adaptive_avg_pool2d oh")?;
                let ow = c.read_dim("adaptive_avg_pool2d ow")?;
                // Adaptive output dims drive the output allocation at forward time.
                check_numel(&[oh, ow], "adaptive_avg_pool2d output")?;
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
        let shift_size = c.read_dim("shift size")?;
        let layer = match header.variant {
            SHIFT_UP => WasmShift::new_shift_up(shift_size),
            SHIFT_DOWN => WasmShift::new_shift_down(shift_size),
            SHIFT_LEFT => WasmShift::new_shift_left(shift_size),
            SHIFT_RIGHT => WasmShift::new_shift_right(shift_size),
            _ => return Err(format!("Unknown shift variant: 0x{:02X}", header.variant)),
        };
        self.shifts.insert(id, layer);
        Ok(())
    }

    fn init_feature_norm(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let epsilon = c.read_option_f64()?;
        let layer = WasmFeatureNorm::new_feature_norm(epsilon)?;
        insert_layer!(self, feature_norms, id, layer);
        Ok(())
    }

    fn init_ghost(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let in_ch = c.read_dim("ghost in_ch")?;
        let out_ch = c.read_dim("ghost out_ch")?;
        let kh = c.read_dim("ghost kh")?;
        let kw = c.read_dim("ghost kw")?;
        let ratio = c.read_bounded_opt_usize("ghost ratio")?;
        let sh = c.read_bounded_opt_usize("ghost sh")?;
        let sw = c.read_bounded_opt_usize("ghost sw")?;
        let ph = c.read_bounded_opt_usize("ghost ph")?;
        let pw = c.read_bounded_opt_usize("ghost pw")?;
        // Over-approximation of the internal primary+cheap conv weights.
        check_numel(&[out_ch, in_ch, kh, kw], "ghost weight")?;
        let layer = WasmGhostModule::try_new(in_ch, out_ch, kh, kw, ratio, sh, sw, ph, pw)?;
        insert_layer!(self, ghosts, id, layer);
        Ok(())
    }

    fn init_seblock(&mut self, _header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let channels = c.read_dim("seblock channels")?;
        let reduction = c.read_bounded_opt_usize("seblock reduction")?;
        // Over-approximation of the internal squeeze/excitation linear weights.
        check_numel(&[channels, channels], "seblock weight")?;
        let layer = WasmSeBlock::try_new(channels, reduction)?;
        insert_layer!(self, seblocks, id, layer);
        Ok(())
    }
}

// ============================================================
// IMPL #2 — FLOAT-BRIDGE + WEIGHT LAYOUT (LINEAR/CONV/EMBEDDING/NORM)
// ============================================================
#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = getWeightsFlat)]
    pub fn get_weights_flat(&self, layer_id: LayerId, layer_type: u8) -> Result<Vec<f32>, String> {
        match layer_type {
            LAYER_LINEAR => self
                .linears
                .get(&layer_id)
                .ok_or("Linear not found")?
                .get_weights_flat(),
            LAYER_CONV => self
                .convs
                .get(&layer_id)
                .ok_or("Conv not found")?
                .get_weights_flat(),
            LAYER_EMBEDDING => self
                .embeddings
                .get(&layer_id)
                .ok_or("Embedding not found")?
                .get_weights_flat(),
            LAYER_NORM => self
                .norms
                .get(&layer_id)
                .ok_or("Norm not found")?
                .get_weights_flat(),
            _ => Err(format!(
                "getWeightsFlat: not yet supported for type 0x{:02X}",
                layer_type
            )),
        }
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(
        &mut self,
        layer_id: LayerId,
        layer_type: u8,
        data: &[f32],
    ) -> Result<(), String> {
        match layer_type {
            LAYER_LINEAR => self
                .linears
                .get_mut(&layer_id)
                .ok_or("Linear not found")?
                .set_weights_flat(data),
            LAYER_CONV => self
                .convs
                .get_mut(&layer_id)
                .ok_or("Conv not found")?
                .set_weights_flat(data),
            LAYER_EMBEDDING => self
                .embeddings
                .get_mut(&layer_id)
                .ok_or("Embedding not found")?
                .set_weights_flat(data),
            LAYER_NORM => self
                .norms
                .get_mut(&layer_id)
                .ok_or("Norm not found")?
                .set_weights_flat(data),
            _ => Err(format!(
                "setWeightsFlat: not yet supported for type 0x{:02X}",
                layer_type
            )),
        }
    }

    #[wasm_bindgen(js_name = weightLayout)]
    pub fn weight_layout(&self, layer_id: LayerId, layer_type: u8) -> Result<String, String> {
        match layer_type {
            LAYER_LINEAR => Ok(self
                .linears
                .get(&layer_id)
                .ok_or("Linear not found")?
                .weight_layout()),
            LAYER_CONV => Ok(self
                .convs
                .get(&layer_id)
                .ok_or("Conv not found")?
                .weight_layout()),
            LAYER_EMBEDDING => Ok(self
                .embeddings
                .get(&layer_id)
                .ok_or("Embedding not found")?
                .weight_layout()),
            LAYER_NORM => Ok(self
                .norms
                .get(&layer_id)
                .ok_or("Norm not found")?
                .weight_layout()),
            _ => Err(format!(
                "weightLayout: not yet supported for type 0x{:02X}",
                layer_type
            )),
        }
    }
}

// ============================================================
// IMPL #3 — BINARY (stateless 2-input)
// ============================================================
#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = forwardBinaryLayer)]
    pub fn forward_binary_layer(
        &self,
        layer_id: LayerId,
        a: &WasmTensor,
        b: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        self.binaries
            .get(&layer_id)
            .ok_or_else(|| format!("Binary layer {} not found", layer_id))?
            .forward_binary(a, b)
    }

    fn init_binary(&mut self, header: &PacketHeader, payload: &[u8]) -> Result<(), String> {
        let mut c = PayloadCursor::new(payload);
        let id = c.read_u32()?;
        let dim = c.read_usize()?;
        // `dim` is only meaningful for CONCAT; other variants honestly encode
        // it as 0. Bound it in all cases, require positive only for CONCAT.
        if dim > crate::protocol::MAX_DIM {
            return Err(format!(
                "binary dim: parameter {dim} exceeds maximum {}",
                crate::protocol::MAX_DIM
            ));
        }
        let layer = match header.variant {
            BINARY_ADD => WasmBinary::new_add(),
            BINARY_SUB => WasmBinary::new_sub(),
            BINARY_MUL => WasmBinary::new_mul(),
            BINARY_MATMUL => WasmBinary::new_matmul(),
            BINARY_CONCAT => {
                crate::protocol::check_dim(dim, "binary concat dim")?;
                WasmBinary::new_concat(dim)
            }
            _ => return Err(format!("Unknown binary variant: 0x{:02X}", header.variant)),
        };
        self.binaries.insert(id, layer);
        Ok(())
    }
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = runGraph)]
    pub fn run_graph(&self, plan: &[u8], input: &WasmTensor) -> Result<WasmTensor, String> {
        let (num_steps, num_slots, out_slot) = validate_plan(self, plan)?;
        let mut c = PayloadCursor::new(plan);
        let _ = c.read_u32()?;
        let _ = c.read_u32()?;
        let mut slots: Vec<Option<WasmTensor>> = (0..num_slots as usize).map(|_| None).collect();
        slots[0] = Some(input.clone());
        for _ in 0..num_steps {
            let s = read_run_step(&mut c)?;
            let out = if s.arity == crate::graph::ARITY_BINARY {
                let a = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run_graph: runtime empty slot {}", s.in_slot))?;
                let b = slots[s.in_slot2 as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run_graph: runtime empty slot {}", s.in_slot2))?;
                self.forward_binary_layer(s.layer_id, a, b)?
            } else {
                let inp = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run_graph: runtime empty slot {}", s.in_slot))?;
                self.forward_layer(s.layer_id, s.layer_type, inp)?
            };
            slots[s.out_slot as usize] = Some(out);
        }
        slots[out_slot as usize]
            .take()
            .ok_or_else(|| format!("run_graph: runtime empty output slot {}", out_slot))
    }
}

// ============================================================
// GRAPH ENTRY — compile-once + layerExists
// ============================================================
#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = layerExists)]
    pub fn layer_exists(&self, layer_type: u8, layer_id: LayerId) -> bool {
        match layer_type {
            LAYER_LINEAR => self.linears.contains_key(&layer_id),
            LAYER_NORM => self.norms.contains_key(&layer_id),
            LAYER_CONV => self.convs.contains_key(&layer_id),
            LAYER_ACTIVATION => self.activations.contains_key(&layer_id),
            LAYER_EMBEDDING => self.embeddings.contains_key(&layer_id),
            LAYER_POOL => self.pools.contains_key(&layer_id),
            LAYER_SHIFT => self.shifts.contains_key(&layer_id),
            LAYER_GHOST => self.ghosts.contains_key(&layer_id),
            LAYER_SEBLOCK => self.seblocks.contains_key(&layer_id),
            LAYER_BINARY => self.binaries.contains_key(&layer_id),
            LAYER_FEATURE_NORM => self.feature_norms.contains_key(&layer_id),
            _ => false,
        }
    }

    #[wasm_bindgen(js_name = compileGraph)]
    pub fn compile_graph(&self, plan: &[u8]) -> Result<crate::graph::CompiledGraph, String> {
        crate::graph::CompiledGraph::build(self, plan)
    }

    #[wasm_bindgen(js_name = compileMultiInputGraph)]
    pub fn compile_multi_input_graph(
        &self,
        plan: &crate::multi_input_graph::MultiInputGraphPlan,
    ) -> Result<crate::graph::CompiledMultiInputGraph, String> {
        crate::graph::CompiledMultiInputGraph::build(self, plan)
    }
}

// ============================================================
// Opsi C Fase 2 repair — pindahan murni dari modul domain:
// #[wasm_bindgen] impl LayerRegistry (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = operationBindingSnapshot)]
    pub fn operation_binding_snapshot(&self) -> Result<String, String> {
        operation_binding_snapshot(self)
    }
}

// ============================================================
// Opsi C Fase 2 repair — pindahan murni dari modul domain:
// #[wasm_bindgen] impl LayerRegistry (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = inventorySnapshot)]
    pub fn inventory_snapshot(&self) -> Result<String, String> {
        inventory_snapshot(self)
    }
}

// ============================================================
// Opsi C Fase 2 repair — pindahan murni dari modul domain:
// #[wasm_bindgen] impl LayerRegistry (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = initAgentLayer)]
    pub fn init_agent_layer(&mut self, spec: &AgentLayerSpec) -> Result<(), String> {
        let header = spec.header();
        self.init_layer(&header, &spec.payload)
    }
}
