use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

use crate::protocol::{
    LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_FEATURE_NORM,
    LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT,
};

use super::super::LayerRegistry;

const INVENTORY_CAPABILITIES_V1: &str =
    include_str!("../../../docs/layer-registry-inventory.v1.json");
const INVENTORY_SCHEMA: &str = "burn-research.layer-registry-inventory-snapshot.v1";
const INVENTORY_SCOPE: &str =
    "live_structure_plus_validated_init_identity_and_parameter_count_not_numerical_weight_state";

fn sha256_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn live_instance_count(registry: &LayerRegistry) -> usize {
    registry.linears.len()
        + registry.norms.len()
        + registry.convs.len()
        + registry.activations.len()
        + registry.embeddings.len()
        + registry.pools.len()
        + registry.shifts.len()
        + registry.ghosts.len()
        + registry.seblocks.len()
        + registry.binaries.len()
        + registry.feature_norms.len()
}

fn instance_param_count(
    registry: &LayerRegistry,
    layer_type: u8,
    layer_id: u32,
) -> Result<usize, String> {
    match layer_type {
        LAYER_LINEAR => registry
            .linears
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: linear id {layer_id} is not live")),
        LAYER_NORM => registry
            .norms
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: norm id {layer_id} is not live")),
        LAYER_CONV => registry
            .convs
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: conv id {layer_id} is not live")),
        LAYER_ACTIVATION => registry
            .activations
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: activation id {layer_id} is not live")),
        LAYER_EMBEDDING => registry
            .embeddings
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: embedding id {layer_id} is not live")),
        LAYER_POOL => registry
            .pools
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: pool id {layer_id} is not live")),
        LAYER_SHIFT => registry
            .shifts
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: shift id {layer_id} is not live")),
        LAYER_GHOST => registry
            .ghosts
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: ghost id {layer_id} is not live")),
        LAYER_SEBLOCK => registry
            .seblocks
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: seblock id {layer_id} is not live")),
        LAYER_BINARY => registry
            .binaries
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: binary id {layer_id} is not live")),
        LAYER_FEATURE_NORM => registry
            .feature_norms
            .get(&layer_id)
            .map(|layer| layer.num_params())
            .ok_or_else(|| format!("inventorySnapshot: feature norm id {layer_id} is not live")),
        _ => Err(format!(
            "inventorySnapshot: unsupported live layer type 0x{layer_type:02X} id {layer_id}"
        )),
    }
}

fn inventory_snapshot(registry: &LayerRegistry) -> Result<String, String> {
    let live_count = live_instance_count(registry);
    if registry.init_identities.len() != live_count {
        return Err(format!(
            "inventorySnapshot: live instance count {live_count} differs from canonical init identity count {}",
            registry.init_identities.len()
        ));
    }

    let mut keys = registry
        .init_identities
        .keys()
        .copied()
        .collect::<Vec<_>>();
    keys.sort_unstable();

    let mut canonical_records = Vec::with_capacity(keys.len());
    let mut json_records = Vec::with_capacity(keys.len());
    let mut summed_params = 0usize;

    for (layer_type, layer_id) in keys {
        if !registry.layer_exists(layer_type, layer_id) {
            return Err(format!(
                "inventorySnapshot: canonical init identity exists for non-live layer type 0x{layer_type:02X} id {layer_id}"
            ));
        }

        let identity = registry
            .init_identities
            .get(&(layer_type, layer_id))
            .ok_or_else(|| "inventorySnapshot: internal identity lookup drift".to_string())?;
        let init_fingerprint = identity.fingerprint(layer_type, layer_id);
        let parameter_count = instance_param_count(registry, layer_type, layer_id)?;
        summed_params = summed_params
            .checked_add(parameter_count)
            .ok_or_else(|| "inventorySnapshot: parameter count overflow".to_string())?;

        canonical_records.push(format!(
            "type={layer_type:02x}|id={layer_id}|variant={:02x}|flags={:02x}|params={parameter_count}|init={init_fingerprint}",
            identity.variant, identity.flags
        ));
        json_records.push(format!(
            "{{\"layer_type\":{layer_type},\"layer_id\":{layer_id},\"variant\":{},\"flags\":{},\"init_fingerprint\":\"{init_fingerprint}\",\"parameter_count\":{parameter_count}}}",
            identity.variant, identity.flags
        ));
    }

    if summed_params != registry.cached_params {
        return Err(format!(
            "inventorySnapshot: summed live parameter count {summed_params} differs from LayerRegistry totalParams {}",
            registry.cached_params
        ));
    }

    let canonical = format!(
        "schema={INVENTORY_SCHEMA}|scope={INVENTORY_SCOPE}|instances={}|total_params={summed_params}|{}",
        canonical_records.len(),
        canonical_records.join("||")
    );
    let inventory_fingerprint = sha256_text(&canonical);

    Ok(format!(
        "{{\"schema\":\"{INVENTORY_SCHEMA}\",\"instance_count\":{},\"total_params\":{summed_params},\"inventory_fingerprint\":\"{inventory_fingerprint}\",\"state_identity_scope\":\"{INVENTORY_SCOPE}\",\"instances\":[{}],\"execution_authorized\":false,\"mutation\":\"none\"}}",
        json_records.len(),
        json_records.join(",")
    ))
}

#[wasm_bindgen(js_name = layerRegistryInventoryCapabilities)]
pub fn layer_registry_inventory_capabilities() -> String {
    INVENTORY_CAPABILITIES_V1.to_string()
}

#[wasm_bindgen]
impl LayerRegistry {
    #[wasm_bindgen(js_name = inventorySnapshot)]
    pub fn inventory_snapshot(&self) -> Result<String, String> {
        inventory_snapshot(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{inventory_snapshot, INVENTORY_SCHEMA};
    use crate::agent::AgentLayerSpec;
    use crate::registry::LayerRegistry;

    #[test]
    fn empty_inventory_is_deterministic_and_nonexecuting() {
        let registry = LayerRegistry::new();
        let snapshot = inventory_snapshot(&registry).unwrap();
        assert!(snapshot.contains(&format!("\"schema\":\"{INVENTORY_SCHEMA}\"")));
        assert!(snapshot.contains("\"instance_count\":0"));
        assert!(snapshot.contains("\"total_params\":0"));
        assert!(snapshot.contains("\"execution_authorized\":false"));
        assert!(snapshot.contains("\"mutation\":\"none\""));
    }

    #[test]
    fn live_inventory_tracks_agent_initialized_layer_and_destroy() {
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::linear(7, 4, 3, true).unwrap();
        registry.init_agent_layer(&spec).unwrap();

        let snapshot = inventory_snapshot(&registry).unwrap();
        assert!(snapshot.contains("\"instance_count\":1"));
        assert!(snapshot.contains("\"layer_id\":7"));
        assert!(snapshot.contains("\"layer_type\":1"));
        assert!(snapshot.contains("\"parameter_count\":15"));

        assert!(registry.destroy_layer(7, spec.layer_type()));
        let empty = inventory_snapshot(&registry).unwrap();
        assert!(empty.contains("\"instance_count\":0"));
    }
}
