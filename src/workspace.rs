use wasm_bindgen::prelude::*;

use crate::protocol::{
    LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR,
    LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT,
};
use crate::registry::LayerRegistry;

pub(crate) const INTERNAL_PREFIX: &str = "_";
pub(crate) const TABLE_SLOTS: &str = "_slots";
pub(crate) const TABLE_LAYERS: &str = "_layers";
pub(crate) const TABLE_PROOFS: &str = "_proofs";
pub(crate) const TABLE_ATTESTATIONS: &str = "_attestations";
pub(crate) const TABLE_VERIFIER_RECEIPTS: &str = "_verifier_receipts";
pub(crate) const TABLE_EVENTS: &str = "_events";

// Working-memory quotas. These bound metadata growth without constraining Burn tensor/math capacity.
pub(crate) const MAX_ROWS: usize = 1024;
pub(crate) const MAX_TABLE_BYTES: usize = 64;
pub(crate) const MAX_KEY_BYTES: usize = 128;
pub(crate) const MAX_KIND_BYTES: usize = 64;
pub(crate) const MAX_STATE_BYTES: usize = 64;
pub(crate) const MAX_VALUE_BYTES: usize = 4096;
pub(crate) const MAX_RUNTIME_PROGRAM_BINDINGS: usize = 32;
pub(crate) const MAX_RUNTIME_PROGRAM_IDENTITY_BYTES: usize = 16_384;

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
pub(crate) struct WorkspaceRow {
    pub(crate) table: String,
    pub(crate) key: String,
    pub(crate) kind: String,
    pub(crate) state: String,
    pub(crate) value: String,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSlotIntrospection {
    pub(crate) slot: u8,
    pub(crate) state: String,
    pub(crate) owner: String,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkspaceLayerIntrospection {
    pub(crate) layer_id: u32,
    pub(crate) state: String,
    pub(crate) label: String,
    pub(crate) layer_type: Option<u8>,
    pub(crate) variant: Option<u8>,
    pub(crate) metadata_valid: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceInputContract {
    pub(crate) shape: [u32; 4],
    pub(crate) layout: String,
    pub(crate) semantics: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceInputPortMetadata {
    pub(crate) role: String,
    pub(crate) source: String,
    pub(crate) revision: u64,
    pub(crate) fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceRuntimeSubjectBinding {
    pub(crate) intent_id: String,
    pub(crate) workflow_revision: u64,
    pub(crate) approval_id: String,
    pub(crate) subject_kind: String,
    pub(crate) subject_identity: String,
    pub(crate) authorization_policy_id: String,
    pub(crate) authorization_policy_revision: u64,
    pub(crate) authorization_is_revision: bool,
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

pub(crate) fn validate_row_fields(
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

pub(crate) fn json_escape(value: &str) -> String {
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

pub(crate) fn row_json(row: &WorkspaceRow) -> String {
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
    pub(crate) num_slots: u32,
    pub(crate) next_layer_id: u32,
    pub(crate) next_proof_id: u32,
    pub(crate) next_attestation_id: u32,
    pub(crate) next_verifier_receipt_id: u32,
    pub(crate) next_event_id: u32,
    pub(crate) input_contract: Option<WorkspaceInputContract>,
    pub(crate) input_port_metadata: Option<WorkspaceInputPortMetadata>,
    pub(crate) runtime_subject_binding: Option<WorkspaceRuntimeSubjectBinding>,
    pub(crate) runtime_program_identities: Vec<String>,
    pub(crate) rows: Vec<WorkspaceRow>,
}

impl AgentWorkspace {
    pub(crate) fn find_row_index(&self, table: &str, key: &str) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| row.table == table && row.key == key)
    }

    pub(crate) fn ensure_insert_capacity(&self, table: &str, key: &str) -> Result<(), String> {
        if self.find_row_index(table, key).is_none() && self.rows.len() >= MAX_ROWS {
            return Err(format!(
                "AgentWorkspace: row limit {MAX_ROWS} reached; remove or reuse rows before inserting"
            ));
        }
        Ok(())
    }

    pub(crate) fn upsert_internal(
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

    pub(crate) fn layer_id_in_use(registry: &LayerRegistry, layer_id: u32) -> bool {
        KNOWN_LAYER_TYPES
            .iter()
            .copied()
            .any(|layer_type| registry.layer_exists(layer_type, layer_id))
    }

    pub(crate) fn workspace_layer_id_reserved(&self, layer_id: u32) -> bool {
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

    pub(crate) fn interaction_num_slots(&self) -> u32 {
        self.num_slots
    }

    pub(crate) fn input_contract(&self) -> Option<&WorkspaceInputContract> {
        self.input_contract.as_ref()
    }

    pub(crate) fn set_input_contract(&mut self, contract: WorkspaceInputContract) {
        self.input_contract = Some(contract);
    }

    pub(crate) fn clear_input_contract_internal(&mut self) -> bool {
        self.input_contract.take().is_some()
    }

    pub(crate) fn input_port_metadata(&self) -> Option<&WorkspaceInputPortMetadata> {
        self.input_port_metadata.as_ref()
    }

    pub(crate) fn set_input_port_metadata(&mut self, metadata: WorkspaceInputPortMetadata) -> bool {
        if self.input_port_metadata.as_ref() == Some(&metadata) {
            return false;
        }
        self.input_port_metadata = Some(metadata);
        true
    }

    pub(crate) fn clear_input_port_metadata_internal(&mut self) -> bool {
        self.input_port_metadata.take().is_some()
    }

    pub(crate) fn runtime_subject_binding(&self) -> Option<&WorkspaceRuntimeSubjectBinding> {
        self.runtime_subject_binding.as_ref()
    }

    pub(crate) fn runtime_subject_bind_available(&self) -> bool {
        if self.runtime_subject_binding.is_some() {
            return false;
        }
        let has_layer_state = self.rows.iter().any(|row| row.table == TABLE_LAYERS);
        let has_reserved_runtime_slot = self
            .rows
            .iter()
            .any(|row| row.table == TABLE_SLOTS && row.key != "0" && row.state != "free");
        let has_proof_state = self.rows.iter().any(|row| {
            matches!(
                row.table.as_str(),
                TABLE_PROOFS | TABLE_ATTESTATIONS | TABLE_VERIFIER_RECEIPTS
            )
        });
        !has_layer_state && !has_reserved_runtime_slot && !has_proof_state
    }

    pub(crate) fn bind_runtime_subject_binding(
        &mut self,
        binding: WorkspaceRuntimeSubjectBinding,
    ) -> Result<bool, String> {
        if let Some(existing) = &self.runtime_subject_binding {
            if existing == &binding {
                return Ok(false);
            }
            return Err(
                "AgentWorkspace: runtime subject binding is immutable for the workspace lifetime"
                    .to_string(),
            );
        }

        if !self.runtime_subject_bind_available() {
            return Err(
                "AgentWorkspace: bind runtime subject before reserving runtime layers or slots or recording proof evidence"
                    .to_string(),
            );
        }

        self.runtime_subject_binding = Some(binding);
        Ok(true)
    }

    pub(crate) fn runtime_program_binding_count(&self) -> usize {
        self.runtime_program_identities.len()
    }

    pub(crate) fn runtime_program_binding_capacity_available(&self) -> bool {
        self.runtime_program_identities.len() < MAX_RUNTIME_PROGRAM_BINDINGS
    }

    pub(crate) fn runtime_program_identity_bound(&self, identity: &str) -> bool {
        self.runtime_program_identities
            .iter()
            .any(|candidate| candidate == identity)
    }

    pub(crate) fn bind_runtime_program_identity(
        &mut self,
        identity: String,
    ) -> Result<bool, String> {
        if self.runtime_subject_binding.is_none() {
            return Err(
                "AgentWorkspace: runtime program binding requires an immutable runtime subject"
                    .to_string(),
            );
        }
        if identity.is_empty() {
            return Err("AgentWorkspace: runtime program identity must be non-empty".to_string());
        }
        if identity.len() > MAX_RUNTIME_PROGRAM_IDENTITY_BYTES {
            return Err(format!(
                "AgentWorkspace: runtime program identity {} bytes exceeds limit {}",
                identity.len(),
                MAX_RUNTIME_PROGRAM_IDENTITY_BYTES
            ));
        }
        if self.runtime_program_identity_bound(&identity) {
            return Ok(false);
        }
        if !self.runtime_program_binding_capacity_available() {
            return Err(format!(
                "AgentWorkspace: runtime program binding limit {} reached",
                MAX_RUNTIME_PROGRAM_BINDINGS
            ));
        }

        self.runtime_program_identities.push(identity);
        Ok(true)
    }

    pub(crate) fn require_runtime_program_identity_if_bound(
        &self,
        identity: &str,
        context: &str,
    ) -> Result<(), String> {
        if self.runtime_subject_binding.is_some() && !self.runtime_program_identity_bound(identity)
        {
            return Err(format!(
                "{context}: graph programIdentity is not bound to this runtime subject; compile it with workspaceCompileForRuntimeSubject first"
            ));
        }
        Ok(())
    }

    pub(crate) fn interaction_row_capacity_available(&self) -> bool {
        self.rows.len() < MAX_ROWS
    }

    pub(crate) fn interaction_free_slots(&self) -> Vec<u8> {
        self.slot_ids_with_state("free")
    }

    pub(crate) fn interaction_readable_slots(&self) -> Vec<u8> {
        let mut slots = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_SLOTS)
            .filter(|row| matches!(row.state.as_str(), "input" | "reserved"))
            .filter_map(|row| row.key.parse::<u8>().ok())
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    pub(crate) fn interaction_releasable_slots(&self) -> Vec<u8> {
        let mut slots = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_SLOTS && row.state == "reserved")
            .filter(|row| !row.value.starts_with("layer:"))
            .filter_map(|row| row.key.parse::<u8>().ok())
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    pub(crate) fn interaction_reserved_layer_ids(&self) -> Vec<u32> {
        self.layer_ids_with_state("reserved")
    }

    pub(crate) fn interaction_initialized_layer_ids(&self) -> Vec<u32> {
        self.layer_ids_with_state("initialized")
    }

    pub(crate) fn interaction_slot_state(&self, slot: u8) -> Option<&str> {
        let key = slot.to_string();
        self.rows
            .iter()
            .find(|row| row.table == TABLE_SLOTS && row.key == key)
            .map(|row| row.state.as_str())
    }

    pub(crate) fn interaction_layer_state(&self, layer_id: u32) -> Option<&str> {
        let key = layer_id.to_string();
        self.rows
            .iter()
            .find(|row| row.table == TABLE_LAYERS && row.key == key)
            .map(|row| row.state.as_str())
    }

    pub(crate) fn introspection_slots(&self) -> Vec<WorkspaceSlotIntrospection> {
        let mut slots = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_SLOTS)
            .filter_map(|row| {
                row.key
                    .parse::<u8>()
                    .ok()
                    .map(|slot| WorkspaceSlotIntrospection {
                        slot,
                        state: row.state.clone(),
                        owner: row.value.clone(),
                    })
            })
            .collect::<Vec<_>>();
        slots.sort_by_key(|row| row.slot);
        slots
    }

    pub(crate) fn introspection_layers(&self) -> Vec<WorkspaceLayerIntrospection> {
        let mut layers =
            self.rows
                .iter()
                .filter(|row| row.table == TABLE_LAYERS)
                .filter_map(|row| {
                    let layer_id = row.key.parse::<u32>().ok()?;
                    if row.state == "initialized" {
                        let parsed = row.value.rsplit_once(";variant=").and_then(
                            |(before_variant, variant)| {
                                before_variant.rsplit_once(";type=").map(
                                    |(before_type, layer_type)| (before_type, layer_type, variant),
                                )
                            },
                        );
                        if let Some((before_type, layer_type, variant)) = parsed {
                            let parsed_type = layer_type.parse::<u8>().ok();
                            let parsed_variant = variant.parse::<u8>().ok();
                            let label = before_type.strip_prefix("label=").unwrap_or(before_type);
                            Some(WorkspaceLayerIntrospection {
                                layer_id,
                                state: row.state.clone(),
                                label: label.to_string(),
                                layer_type: parsed_type,
                                variant: parsed_variant,
                                metadata_valid: parsed_type.is_some() && parsed_variant.is_some(),
                            })
                        } else {
                            Some(WorkspaceLayerIntrospection {
                                layer_id,
                                state: row.state.clone(),
                                label: row.value.clone(),
                                layer_type: None,
                                variant: None,
                                metadata_valid: false,
                            })
                        }
                    } else {
                        Some(WorkspaceLayerIntrospection {
                            layer_id,
                            state: row.state.clone(),
                            label: row.value.clone(),
                            layer_type: None,
                            variant: None,
                            metadata_valid: true,
                        })
                    }
                })
                .collect::<Vec<_>>();
        layers.sort_by_key(|row| row.layer_id);
        layers
    }

    pub(crate) fn introspection_proof_counts(&self) -> (usize, usize, usize) {
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut other = 0usize;
        for row in self.rows.iter().filter(|row| row.table == TABLE_PROOFS) {
            match row.state.as_str() {
                "passed" => passed += 1,
                "failed" => failed += 1,
                _ => other += 1,
            }
        }
        (passed, failed, other)
    }

    pub(crate) fn introspection_attestation_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.table == TABLE_ATTESTATIONS)
            .count()
    }

    pub(crate) fn introspection_verifier_receipt_counts(&self) -> (usize, usize, usize) {
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut other = 0usize;
        for row in self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_VERIFIER_RECEIPTS)
        {
            match row.state.as_str() {
                "passed" => passed += 1,
                "failed" => failed += 1,
                _ => other += 1,
            }
        }
        (passed, failed, other)
    }

    pub(crate) fn next_verifier_receipt_id(&self) -> u32 {
        self.next_verifier_receipt_id
    }

    pub(crate) fn record_attestation_internal(
        &mut self,
        label: String,
        claimed_passed: bool,
        detail: String,
    ) -> Result<u32, String> {
        let attestation_id = self.next_attestation_id;
        let value = format!(
            "authority=caller_attestation;claimed_passed={claimed_passed};label={label};{detail}"
        );
        self.upsert_internal(
            TABLE_ATTESTATIONS,
            attestation_id.to_string(),
            "caller_attestation".into(),
            "recorded".into(),
            value,
        )?;
        self.next_attestation_id = attestation_id
            .checked_add(1)
            .ok_or_else(|| "AgentWorkspace.attestation allocator exhausted".to_string())?;
        Ok(attestation_id)
    }

    pub(crate) fn record_verifier_receipt_internal(
        &mut self,
        verifier: String,
        passed: bool,
        receipt_json: String,
    ) -> Result<u32, String> {
        let receipt_id = self.next_verifier_receipt_id;
        self.upsert_internal(
            TABLE_VERIFIER_RECEIPTS,
            receipt_id.to_string(),
            verifier,
            if passed { "passed" } else { "failed" }.into(),
            receipt_json,
        )?;
        self.next_verifier_receipt_id = receipt_id
            .checked_add(1)
            .ok_or_else(|| "AgentWorkspace.verifier receipt allocator exhausted".to_string())?;
        Ok(receipt_id)
    }

    pub(crate) fn introspection_event_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.table == TABLE_EVENTS)
            .count()
    }

    pub(crate) fn introspection_custom_tables(&self) -> Vec<String> {
        let mut names = self
            .rows
            .iter()
            .filter(|row| !row.table.starts_with(INTERNAL_PREFIX))
            .map(|row| row.table.clone())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        names
    }

    fn slot_ids_with_state(&self, state: &str) -> Vec<u8> {
        let mut slots = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_SLOTS && row.state == state)
            .filter_map(|row| row.key.parse::<u8>().ok())
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    fn layer_ids_with_state(&self, state: &str) -> Vec<u32> {
        let mut layers = self
            .rows
            .iter()
            .filter(|row| row.table == TABLE_LAYERS && row.state == state)
            .filter_map(|row| row.key.parse::<u32>().ok())
            .collect::<Vec<_>>();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    pub(crate) fn query_rows(
        &self,
        table: &str,
        kind: Option<&str>,
        state: Option<&str>,
    ) -> String {
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

// #[wasm_bindgen] impl AgentWorkspace — dipindah ke src/facade/workspace.rs (Opsi C Fase 3).

#[cfg(test)]
mod tests {
    use super::{
        AgentWorkspace, WorkspaceRuntimeSubjectBinding, MAX_RUNTIME_PROGRAM_BINDINGS,
        MAX_RUNTIME_PROGRAM_IDENTITY_BYTES, MAX_VALUE_BYTES,
    };
    use crate::agent::AgentLayerSpec;
    use crate::protocol::LAYER_ACTIVATION;
    use crate::registry::LayerRegistry;

    #[test]
    fn malformed_initialized_layer_metadata_remains_visible_to_introspection() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        workspace.rows.push(super::WorkspaceRow {
            table: super::TABLE_LAYERS.to_string(),
            key: "77".to_string(),
            kind: "layer".to_string(),
            state: "initialized".to_string(),
            value: "broken-provenance".to_string(),
        });

        let layers = workspace.introspection_layers();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].layer_id, 77);
        assert_eq!(layers[0].state, "initialized");
        assert!(!layers[0].metadata_valid);
        assert!(layers[0].layer_type.is_none());
        assert!(layers[0].variant.is_none());
    }

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
        assert!(workspace
            .get("_slots".into(), "1".into())
            .contains("reserved"));
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
        let first = workspace
            .reserve_layer_id(&registry, "candidate".into())
            .unwrap();
        let second = workspace
            .reserve_layer_id(&registry, "candidate-2".into())
            .unwrap();
        assert_eq!(first, 2);
        assert_eq!(second, 3);
    }

    #[test]
    fn sync_layer_requires_external_registry_truth() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(7);
        assert!(workspace
            .sync_layer(&registry, &spec, "relu".into())
            .is_err());
        assert_eq!(workspace.get("_layers".into(), "7".into()), "null");

        registry.init_agent_layer(&spec).unwrap();
        workspace
            .sync_layer(&registry, &spec, "relu".into())
            .unwrap();
        assert!(workspace
            .get("_layers".into(), "7".into())
            .contains("initialized"));
        assert!(registry.layer_exists(LAYER_ACTIVATION, 7));
    }

    #[test]
    fn forgetting_workspace_metadata_does_not_remove_registry_layer() {
        let mut workspace = AgentWorkspace::new(4).unwrap();
        let mut registry = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(9);
        registry.init_agent_layer(&spec).unwrap();
        workspace
            .sync_layer(&registry, &spec, "relu".into())
            .unwrap();
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
            .record_proof(
                "python-vs-burn".into(),
                false,
                0.25,
                "first_failure=3".into(),
            )
            .unwrap();
        assert_eq!(
            workspace
                .record_event("candidate".into(), "python-a".into(), "generated".into())
                .unwrap(),
            1
        );
        assert!(workspace
            .query("_proofs".into(), None, Some("failed".into()))
            .contains("0.25"));
        assert!(workspace
            .query("_events".into(), Some("candidate".into()), None)
            .contains("python-a"));
    }

    fn bind_test_runtime_subject(workspace: &mut AgentWorkspace) {
        workspace
            .bind_runtime_subject_binding(WorkspaceRuntimeSubjectBinding {
                intent_id: "intent".to_string(),
                workflow_revision: 1,
                approval_id: "approval".to_string(),
                subject_kind: "effective-spec".to_string(),
                subject_identity: "spec".to_string(),
                authorization_policy_id: "policy".to_string(),
                authorization_policy_revision: 1,
                authorization_is_revision: false,
            })
            .unwrap();
    }

    #[test]
    fn runtime_program_bindings_are_exact_deduplicated_and_bounded() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        bind_test_runtime_subject(&mut workspace);

        assert!(workspace
            .bind_runtime_program_identity("program-0".to_string())
            .unwrap());
        assert!(!workspace
            .bind_runtime_program_identity("program-0".to_string())
            .unwrap());
        assert_eq!(workspace.runtime_program_binding_count(), 1);

        for index in 1..MAX_RUNTIME_PROGRAM_BINDINGS {
            assert!(workspace
                .bind_runtime_program_identity(format!("program-{index}"))
                .unwrap());
        }
        assert_eq!(
            workspace.runtime_program_binding_count(),
            MAX_RUNTIME_PROGRAM_BINDINGS
        );

        let before = workspace.snapshot();
        let error = workspace
            .bind_runtime_program_identity("overflow".to_string())
            .unwrap_err();
        assert!(error.contains("binding limit"));
        assert_eq!(workspace.snapshot(), before);
    }

    #[test]
    fn oversized_runtime_program_identity_is_rejected_without_mutation() {
        let mut workspace = AgentWorkspace::new(2).unwrap();
        bind_test_runtime_subject(&mut workspace);
        let before = workspace.snapshot();

        let error = workspace
            .bind_runtime_program_identity("x".repeat(MAX_RUNTIME_PROGRAM_IDENTITY_BYTES + 1))
            .unwrap_err();
        assert!(error.contains("exceeds limit"));
        assert_eq!(workspace.snapshot(), before);
    }
}
