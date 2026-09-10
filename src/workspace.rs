use wasm_bindgen::prelude::*;

use crate::agent::AgentLayerSpec;
use crate::protocol::{
    LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR,
    LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT,
};
use crate::registry::LayerRegistry;

const INTERNAL_PREFIX: &str = "_";
const TABLE_SLOTS: &str = "_slots";
const TABLE_LAYERS: &str = "_layers";
const TABLE_PROOFS: &str = "_proofs";
const TABLE_EVENTS: &str = "_events";

// Working-memory quotas. These bound metadata growth without constraining Burn tensor/math capacity.
const MAX_ROWS: usize = 1024;
const MAX_TABLE_BYTES: usize = 64;
const MAX_KEY_BYTES: usize = 128;
const MAX_KIND_BYTES: usize = 64;
const MAX_STATE_BYTES: usize = 64;
const MAX_VALUE_BYTES: usize = 4096;

const KNOWN_LAYER_TYPES: [u8; 10] = [
    LAYER_LINEAR,
    LAYER_NORM,
    LAYER_CONV,
    LAYER_ACTIVATION,
    LAYER_EMBEDDING,
    LAYER_POOL,
    LAYER_SHIFT,
    LAYER_GHOST,
    LAYER_SEBLOCK,
    LAYER_BINARY,
];

#[derive(Clone, Debug)]
struct WorkspaceRow {
    table: String,
    key: String,
    kind: String,
    state: String,
    value: String,
}

fn validate_text(
    value: &str,
    max_bytes: usize,
    context: &str,
    allow_empty: bool,
) -> Result<(), String> {
    if !allow_empty && value.is_empty() {
        return Err(format!("{context}: value must be non-empty"));
    }
    if value.len() > max_bytes {
        return Err(format!(
            "{context}: {} bytes exceeds limit {max_bytes}",
            value.len()
        ));
    }
    Ok(())
}

fn validate_row_fields(
    table: &str,
    key: &str,
    kind: &str,
    state: &str,
    value: &str,
    context: &str,
) -> Result<(), String> {
    validate_text(table, MAX_TABLE_BYTES, &format!("{context}.table"), false)?;
    validate_text(key, MAX_KEY_BYTES, &format!("{context}.key"), false)?;
    validate_text(kind, MAX_KIND_BYTES, &format!("{context}.kind"), true)?;
    validate_text(state, MAX_STATE_BYTES, &format!("{context}.state"), true)?;
    validate_text(value, MAX_VALUE_BYTES, &format!("{context}.value"), true)?;
    Ok(())
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn row_json(row: &WorkspaceRow) -> String {
    format!(
        "{{\"table\":\"{}\",\"key\":\"{}\",\"kind\":\"{}\",\"state\":\"{}\",\"value\":\"{}\"}}",
        json_escape(&row.table),
        json_escape(&row.key),
        json_escape(&row.kind),
        json_escape(&row.state),
        json_escape(&row.value),
    )
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct AgentWorkspace {
    num_slots: u32,
    next_layer_id: u32,
    next_proof_id: u32,
    next_event_id: u32,
    rows: Vec<WorkspaceRow>,
}

impl AgentWorkspace {
    fn find_row_index(&self, table: &str, key: &str) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| row.table == table && row.key == key)
    }

    fn ensure_insert_capacity(&self, table: &str, key: &str) -> Result<(), String> {
        if self.find_row_index(table, key).is_none() && self.rows.len() >= MAX_ROWS {
            return Err(format!(
                "AgentWorkspace: row limit {MAX_ROWS} reached; remove or reuse rows before inserting"
            ));
        }
        Ok(())
    }

    fn upsert_internal(
        &mut self,
        table: &str,
        key: String,
        kind: String,
        state: String,
        value: String,
    ) -> Result<(), String> {
        validate_row_fields(table, &key, &kind, &state, &value, "AgentWorkspace")?;
        self.ensure_insert_capacity(table, &key)?;

        if let Some(index) = self.find_row_index(table, &key) {
            self.rows[index] = WorkspaceRow {
                table: table.to_string(),
                key,
                kind,
                state,
                value,
            };
        } else {
            self.rows.push(WorkspaceRow {
                table: table.to_string(),
                key,
                kind,
                state,
                value,
            });
        }
        Ok(())
    }

    fn layer_id_in_use(registry: &LayerRegistry, layer_id: u32) -> bool {
        KNOWN_LAYER_TYPES
            .iter()
            .copied()
            .any(|layer_type| registry.layer_exists(layer_type, layer_id))
    }

    fn workspace_layer_id_reserved(&self, layer_id: u32) -> bool {
        let key = layer_id.to_string();
        self.rows.iter().any(|row| {
            row.table == TABLE_LAYERS
                && row.key == key
                && matches!(row.state.as_str(), "reserved" | "initialized")
        })
    }

    pub(crate) fn ensure_slot_readable(&self, slot: u8, context: &str) -> Result<(), String> {
        if u32::from(slot) >= self.num_slots {
            return Err(format!(
                "{context}: slot {slot} is outside workspace num_slots {}",
                self.num_slots
            ));
        }
        let key = slot.to_string();
        let index = self
            .find_row_index(TABLE_SLOTS, &key)
            .ok_or_else(|| format!("{context}: slot {slot} not found"))?;
        let state = self.rows[index].state.as_str();
        if matches!(state, "input" | "reserved") {
            Ok(())
        } else {
            Err(format!(
                "{context}: slot {slot} is not readable while state is {state}"
            ))
        }
    }

    fn query_rows(&self, table: &str, kind: Option<&str>, state: Option<&str>) -> String {
        let rows = self
            .rows
            .iter()
            .filter(|row| row.table == table)
            .filter(|row| kind.is_none_or(|expected| row.kind == expected))
            .filter(|row| state.is_none_or(|expected| row.state == expected))
            .map(row_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"table\":\"{}\",\"rows\":[{}]}}",
            json_escape(table),
            rows
        )
    }
}

#[wasm_bindgen]
impl AgentWorkspace {
    #[wasm_bindgen(constructor)]
    pub fn new(num_slots: u32) -> Result<AgentWorkspace, String> {
        if !(1..=64).contains(&num_slots) {
            return Err(format!(
                "AgentWorkspace: num_slots must be 1..=64, got {num_slots}"
            ));
        }

        let mut workspace = Self {
            num_slots,
            next_layer_id: 1,
            next_proof_id: 1,
            next_event_id: 1,
            rows: Vec::new(),
        };

        for slot in 0..num_slots {
            workspace.rows.push(WorkspaceRow {
                table: TABLE_SLOTS.to_string(),
                key: slot.to_string(),
                kind: "slot".to_string(),
                state: if slot == 0 { "input" } else { "free" }.to_string(),
                value: String::new(),
            });
        }

        Ok(workspace)
    }

    /// Generic user-space row upsert. Tables prefixed with `_` are reserved for workspace internals.
    pub fn put(
        &mut self,
        table: String,
        key: String,
        kind: String,
        state: String,
        value: String,
    ) -> Result<(), String> {
        if table.starts_with(INTERNAL_PREFIX) {
            return Err(format!(
                "AgentWorkspace.put: table {table} is reserved for internal state"
            ));
        }
        validate_row_fields(&table, &key, &kind, &state, &value, "AgentWorkspace.put")?;
        self.ensure_insert_capacity(&table, &key)?;
        self.upsert_internal(&table, key, kind, state, value)
    }

    /// Query any table, including read-only inspection of internal tables.
    pub fn query(
        &self,
        table: String,
        kind: Option<String>,
        state: Option<String>,
    ) -> String {
        self.query_rows(&table, kind.as_deref(), state.as_deref())
    }

    pub fn get(&self, table: String, key: String) -> String {
        self.find_row_index(&table, &key)
            .map(|index| row_json(&self.rows[index]))
            .unwrap_or_else(|| "null".to_string())
    }

    pub fn remove(&mut self, table: String, key: String) -> Result<bool, String> {
        if table.starts_with(INTERNAL_PREFIX) {
            return Err(format!(
                "AgentWorkspace.remove: table {table} is reserved for internal state"
            ));
        }
        if let Some(index) = self.find_row_index(&table, &key) {
            self.rows.remove(index);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[wasm_bindgen(js_name = tableNames)]
    pub fn table_names(&self) -> String {
        let mut names = self.rows.iter().map(|row| row.table.as_str()).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        let body = names
            .into_iter()
            .map(|name| format!("\"{}\"", json_escape(name)))
            .collect::<Vec<_>>()
            .join(",");
        format!("[{}]", body)
    }

    /// Return the explicit metadata quotas for agent planning.
    pub fn limits(&self) -> String {
        format!(
            "{{\"max_rows\":{MAX_ROWS},\"max_table_bytes\":{MAX_TABLE_BYTES},\"max_key_bytes\":{MAX_KEY_BYTES},\"max_kind_bytes\":{MAX_KIND_BYTES},\"max_state_bytes\":{MAX_STATE_BYTES},\"max_value_bytes\":{MAX_VALUE_BYTES}}}"
        )
    }

    #[wasm_bindgen(js_name = reserveSlot)]
    pub fn reserve_slot(&mut self, owner: String) -> Result<u8, String> {
        validate_text(
            &owner,
            MAX_VALUE_BYTES,
            "AgentWorkspace.reserveSlot.owner",
            false,
        )?;
        let index = self
            .rows
            .iter()
            .position(|row| row.table == TABLE_SLOTS && row.state == "free")
            .ok_or_else(|| "AgentWorkspace.reserveSlot: no free slot available".to_string())?;
        let slot = self.rows[index]
            .key
            .parse::<u8>()
            .map_err(|_| "AgentWorkspace.reserveSlot: internal slot key is invalid".to_string())?;
        self.rows[index].state = "reserved".to_string();
        self.rows[index].value = owner;
        Ok(slot)
    }

    #[wasm_bindgen(js_name = releaseSlot)]
    pub fn release_slot(&mut self, slot: u8) -> Result<(), String> {
        if slot == 0 || u32::from(slot) >= self.num_slots {
            return Err(format!(
                "AgentWorkspace.releaseSlot: slot {slot} must be in 1..{}",
                self.num_slots
            ));
        }
        let key = slot.to_string();
        let index = self
            .find_row_index(TABLE_SLOTS, &key)
            .ok_or_else(|| format!("AgentWorkspace.releaseSlot: slot {slot} not found"))?;
        let state = self.rows[index].state.as_str();
        if state != "reserved" {
            return Err(format!(
                "AgentWorkspace.releaseSlot: slot {slot} cannot transition {state}->free; expected reserved->free"
            ));
        }
        self.rows[index].state = "free".to_string();
        self.rows[index].value.clear();
        Ok(())
    }

    #[wasm_bindgen(js_name = reserveLayerId)]
    pub fn reserve_layer_id(
        &mut self,
        registry: &LayerRegistry,
        label: String,
    ) -> Result<u32, String> {
        validate_text(
            &label,
            MAX_VALUE_BYTES,
            "AgentWorkspace.reserveLayerId.label",
            true,
        )?;
        let mut candidate = self.next_layer_id;
        loop {
            if !Self::layer_id_in_use(registry, candidate)
                && !self.workspace_layer_id_reserved(candidate)
            {
                self.upsert_internal(
                    TABLE_LAYERS,
                    candidate.to_string(),
                    "layer".into(),
                    "reserved".into(),
                    label,
                )?;
                self.next_layer_id = candidate
                    .checked_add(1)
                    .ok_or_else(|| "AgentWorkspace.reserveLayerId: allocator exhausted".to_string())?;
                return Ok(candidate);
            }
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| "AgentWorkspace.reserveLayerId: allocator exhausted".to_string())?;
        }
    }

    /// Reconcile a typed layer with the actual registry after caller initializes it manually.
    #[wasm_bindgen(js_name = syncLayer)]
    pub fn sync_layer(
        &mut self,
        registry: &LayerRegistry,
        spec: &AgentLayerSpec,
        label: String,
    ) -> Result<(), String> {
        if !registry.layer_exists(spec.layer_type(), spec.layer_id()) {
            return Err(format!(
                "AgentWorkspace.syncLayer: layer type 0x{:02X} id {} is not initialized in registry",
                spec.layer_type(),
                spec.layer_id()
            ));
        }
        let value = format!(
            "label={};type={};variant={}",
            label,
            spec.layer_type(),
            spec.variant()
        );
        self.upsert_internal(
            TABLE_LAYERS,
            spec.layer_id().to_string(),
            "layer".into(),
            "initialized".into(),
            value,
        )
    }

    /// Forget only workspace metadata. This never mutates LayerRegistry.
    #[wasm_bindgen(js_name = forgetLayer)]
    pub fn forget_layer(&mut self, layer_id: u32) -> bool {
        let key = layer_id.to_string();
        if let Some(index) = self.find_row_index(TABLE_LAYERS, &key) {
            self.rows.remove(index);
            true
        } else {
            false
        }
    }

    #[wasm_bindgen(js_name = recordProof)]
    pub fn record_proof(
        &mut self,
        label: String,
        passed: bool,
        max_error: f64,
        detail: String,
    ) -> Result<u32, String> {
        if !max_error.is_finite() || max_error < 0.0 {
            return Err(format!(
                "AgentWorkspace.recordProof: max_error must be finite and >= 0, got {max_error}"
            ));
        }
        let proof_id = self.next_proof_id;
        let value = format!("max_error={max_error};{detail}");
        self.upsert_internal(
            TABLE_PROOFS,
            proof_id.to_string(),
            label,
            if passed { "passed" } else { "failed" }.into(),
            value,
        )?;
        self.next_proof_id = proof_id
            .checked_add(1)
            .ok_or_else(|| "AgentWorkspace.recordProof: id allocator exhausted".to_string())?;
        Ok(proof_id)
    }

    #[wasm_bindgen(js_name = recordEvent)]
    pub fn record_event(
        &mut self,
        kind: String,
        reference: String,
        detail: String,
    ) -> Result<u32, String> {
        validate_text(
            &kind,
            MAX_KIND_BYTES,
            "AgentWorkspace.recordEvent.kind",
            false,
        )?;
        let event_id = self.next_event_id;
        let value = format!("ref={reference};{detail}");
        self.upsert_internal(
            TABLE_EVENTS,
            event_id.to_string(),
            kind,
            "recorded".into(),
            value,
        )?;
        self.next_event_id = event_id
            .checked_add(1)
            .ok_or_else(|| "AgentWorkspace.recordEvent: id allocator exhausted".to_string())?;
        Ok(event_id)
    }

    pub fn snapshot(&self) -> String {
        let count = |table: &str| self.rows.iter().filter(|row| row.table == table).count();
        let custom_tables = self
            .rows
            .iter()
            .filter(|row| !row.table.starts_with(INTERNAL_PREFIX))
            .map(|row| row.table.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let free_slots = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_SLOTS && row.state == "free")
            .count();
        format!(
            "{{\"num_slots\":{},\"free_slots\":{},\"layers\":{},\"proofs\":{},\"events\":{},\"custom_tables\":{},\"rows\":{},\"max_rows\":{MAX_ROWS}}}",
            self.num_slots,
            free_slots,
            count(TABLE_LAYERS),
            count(TABLE_PROOFS),
            count(TABLE_EVENTS),
            custom_tables,
            self.rows.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{AgentWorkspace, MAX_VALUE_BYTES};
    use crate::agent::AgentLayerSpec;
    use crate::protocol::LAYER_ACTIVATION;
    use crate::registry::LayerRegistry;

    #[test]
    fn custom_tables_are_data_driven_and_queryable() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        workspace
            .put(
                "experiments".into(),
                "candidate-a".into(),
                "python".into(),
                "tested".into(),
                "rmse=0.01".into(),
            )
            .unwrap();
        let query = workspace.query("experiments".into(), None, Some("tested".into()));
        assert!(query.contains("candidate-a"));
        assert!(query.contains("rmse=0.01"));
    }

    #[test]
    fn generic_put_cannot_corrupt_internal_tables() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        assert!(workspace
            .put(
                "_slots".into(),
                "0".into(),
                "slot".into(),
                "free".into(),
                "corrupt".into(),
            )
            .is_err());
        assert!(workspace.get("_slots".into(), "0".into()).contains("input"));
    }

    #[test]
    fn oversized_value_is_rejected_without_mutation() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let oversized = "x".repeat(MAX_VALUE_BYTES + 1);
        assert!(workspace
            .put(
                "memory".into(),
                "large".into(),
                "probe".into(),
                "stored".into(),
                oversized,
            )
            .is_err());
        assert_eq!(workspace.get("memory".into(), "large".into()), "null");
    }

    #[test]
    fn slot_state_is_relational_and_reusable() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let first = workspace.reserve_slot("relu-output".into()).unwrap();
        assert_eq!(first, 1);
        assert!(workspace.get("_slots".into(), "1".into()).contains("reserved"));
        workspace.release_slot(first).unwrap();
        assert!(workspace.get("_slots".into(), "1".into()).contains("free"));
        assert_eq!(workspace.reserve_slot("other".into()).unwrap(), 1);
    }

    #[test]
    fn double_release_is_rejected_without_mutation() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let slot = workspace.reserve_slot("temporary".into()).unwrap();
        workspace.release_slot(slot).unwrap();
        let before = workspace.snapshot();

        let err = workspace.release_slot(slot).unwrap_err();
        assert!(err.contains("free->free"));
        assert_eq!(workspace.snapshot(), before);
        assert_eq!(workspace.reserve_slot("reused".into()).unwrap(), slot);
    }

    #[test]
    fn slot_readability_tracks_lifecycle_state() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        assert!(workspace.ensure_slot_readable(0, "test").is_ok());
        assert!(workspace.ensure_slot_readable(1, "test").is_err());

        let slot = workspace.reserve_slot("producer".into()).unwrap();
        assert_eq!(slot, 1);
        assert!(workspace.ensure_slot_readable(slot, "test").is_ok());

        workspace.release_slot(slot).unwrap();
        let err = workspace.ensure_slot_readable(slot, "test").unwrap_err();
        assert!(err.contains("state is free"));
    }

    #[test]
    fn repeated_reserve_release_has_no_capacity_drift() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        for cycle in 0..1000 {
            let slot = workspace.reserve_slot(format!("cycle-{cycle}")).unwrap();
            assert_eq!(slot, 1);
            workspace.release_slot(slot).unwrap();
        }

        assert!(workspace.snapshot().contains("\"free_slots\":3"));
        assert_eq!(workspace.reserve_slot("a".into()).unwrap(), 1);
        assert_eq!(workspace.reserve_slot("b".into()).unwrap(), 2);
        assert_eq!(workspace.reserve_slot("c".into()).unwrap(), 3);
        assert!(workspace.reserve_slot("overflow".into()).is_err());
    }

    #[test]
    fn layer_allocator_observes_registry_and_workspace_reservations() {
        let mut registry = LayerRegistry::new();
        let existing = AgentLayerSpec::relu(1);
        registry.init_agent_layer(&existing).unwrap();

        let mut workspace = AgentWorkspace::new(4).unwrap();
        let first = workspace.reserve_layer_id(&registry, "candidate".into()).unwrap();
        let second = workspace.reserve_layer_id(&registry, "candidate-2".into()).unwrap();
        assert_eq!(first, 2);
        assert_eq!(second, 3);
    }

    #[test]
    fn sync_layer_requires_external_registry_truth() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(7);
        assert!(workspace.sync_layer(&registry, &spec, "relu".into()).is_err());
        assert_eq!(workspace.get("_layers".into(), "7".into()), "null");

        registry.init_agent_layer(&spec).unwrap();
        workspace.sync_layer(&registry, &spec, "relu".into()).unwrap();
        assert!(workspace.get("_layers".into(), "7".into()).contains("initialized"));
        assert!(registry.layer_exists(LAYER_ACTIVATION, 7));
    }

    #[test]
    fn forgetting_workspace_metadata_does_not_remove_registry_layer() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(9);
        registry.init_agent_layer(&spec).unwrap();
        workspace.sync_layer(&registry, &spec, "relu".into()).unwrap();
        assert!(workspace.forget_layer(9));
        assert!(registry.layer_exists(LAYER_ACTIVATION, 9));
    }

    #[test]
    fn invalid_proof_does_not_consume_proof_id() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        assert!(workspace
            .record_proof("bad".into(), false, f64::NAN, "bad".into())
            .is_err());
        assert_eq!(
            workspace
                .record_proof("good".into(), true, 0.0, "exact".into())
                .unwrap(),
            1
        );
    }

    #[test]
    fn proofs_and_events_are_queryable_state() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        workspace
            .record_proof("python-vs-burn".into(), false, 0.25, "first_failure=3".into())
            .unwrap();
        assert_eq!(
            workspace
                .record_event("candidate".into(), "python-a".into(), "generated".into())
                .unwrap(),
            1
        );
        assert!(workspace.query("_proofs".into(), None, Some("failed".into())).contains("0.25"));
        assert!(workspace.query("_events".into(), Some("candidate".into()), None).contains("python-a"));
    }
}
