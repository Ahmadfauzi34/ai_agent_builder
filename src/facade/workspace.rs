//! Fasad WASM tunggal — domain `workspace` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::workspace::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentLayerSpec;
use crate::graph::CompiledGraph;
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::semantic::semantic_ingress_manifest::validate_text;
use crate::workspace::AgentWorkspace;
use crate::workspace::{
    json_escape, row_json, validate_row_fields, WorkspaceRow, INTERNAL_PREFIX, MAX_KEY_BYTES,
    MAX_KIND_BYTES, MAX_ROWS, MAX_RUNTIME_PROGRAM_BINDINGS, MAX_STATE_BYTES, MAX_TABLE_BYTES,
    MAX_VALUE_BYTES, TABLE_ATTESTATIONS, TABLE_EVENTS, TABLE_LAYERS, TABLE_PROOFS, TABLE_SLOTS,
    TABLE_VERIFIER_RECEIPTS,
};
use crate::workspace_ops::ensure_registry_matches_spec;
use crate::workspace_ops::ensure_workspace_layer_reserved;
use crate::workspace_ops::finalize_initialized_binary;
use crate::workspace_ops::finalize_initialized_unary;
use crate::workspace_ops::reserve_workspace_output_slot;
use crate::workspace_ops::rollback_init_transaction;
use crate::workspace_ops::validate_builder_slot;
use crate::workspace_ops::validate_spec_is_new;
use crate::workspace_ops::validate_workspace_input_slot;
use crate::workspace_ops::validate_workspace_layout_input;
use crate::workspace_ops::validate_workspace_op_label;

/// Discover the canonical agent-facing workspace/control-plane API.
#[wasm_bindgen(js_name = workspaceCapabilities)]
pub fn workspace_capabilities() -> String {
    concat!(
        "{",
        "\"state\":\"AgentWorkspace\",",
        "\"ownership\":\"metadata_only\",",
        "\"execution_truth\":\"LayerRegistry\",",
        "\"graph\":\"AgentGraphBuilder\",",
        "\"provenance\":{\"wire_identity\":\"exact_validated_init_fingerprint\",\"syncLayer\":\"exact_identity_metadata_only_not_canonical_orchestration\",\"layout_preflight\":\"canonical_slot_owner_to_layer_type_variant\"},",
        "\"atomicity\":{\"compile\":\"non_mutating_output_override\",\"subject_bound_compile\":\"bind_exact_program_identity_only_after_successful_compile\",\"workspace_init\":\"transactional_post_registry_rollback\"},",
        "\"slot_lifecycle\":{\"states\":[\"input\",\"free\",\"reserved\"],\"readable\":[\"input\",\"reserved\"],\"reserve\":\"free->reserved\",\"release\":\"reserved->free\",\"invalid_transition\":\"error_no_mutation\"},",
        "\"layout_policy\":{\"known_incompatible\":\"reject_before_mutation\",\"unknown\":\"defer_to_runtime\",\"implicit_relayout\":\"forbidden\"},",
        "\"ops\":[\"workspaceInitUnary\",\"workspaceInitBinary\",\"workspaceWireUnary\",\"workspaceWireBinary\",\"workspaceCompile\",\"workspaceCompileForRuntimeSubject\"],",
        "\"workspace_methods\":[\"reserveLayerId\",\"reserveSlot\",\"releaseSlot\",\"syncLayer\",\"forgetLayer\",\"recordProof\",\"recordEvent\",\"put\",\"get\",\"query\",\"remove\",\"tableNames\",\"snapshot\",\"limits\"],",
        "\"escape_hatches\":[\"workspaceCompile\",\"AgentLayerSpec\",\"AgentGraphBuilder\",\"LayerRegistry\",\"raw_protocol\"],",
        "\"recommended_flow\":[\"bind_runtime_subject_if_used\",\"reserve_layer\",\"construct_spec\",\"init_or_wire\",\"subject_bound_compile_if_bound\",\"run\",\"verify\"]",
        "}"
    )
    .to_string()
}

/// Reconcile an already initialized unary layer into workspace metadata and graph wiring.
/// The supplied spec must exactly match the live registry layer's validated init identity.
#[wasm_bindgen(js_name = workspaceWireUnary)]
pub fn workspace_wire_unary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    input_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() == LAYER_BINARY {
        return Err("workspaceWireUnary: binary spec requires workspaceWireBinary".into());
    }
    validate_workspace_input_slot(workspace, builder, input_slot, "workspaceWireUnary")?;
    validate_workspace_layout_input(workspace, input_slot, spec, "workspaceWireUnary")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireUnary")?;
    validate_workspace_op_label(&label, "workspaceWireUnary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireUnary",
    )?;
    // Metadata reconciliation happens only after identity proof and output reservation succeed.
    if let Err(err) = workspace.sync_layer(registry, spec, label) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    if let Err(err) = builder.add_unary(spec, input_slot, output_slot) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    Ok(output_slot)
}

/// Reconcile an already initialized binary layer into workspace metadata and graph wiring.
/// The supplied spec must exactly match the live registry layer's validated init identity.
#[wasm_bindgen(js_name = workspaceWireBinary)]
pub fn workspace_wire_binary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    left_slot: u8,
    right_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() != LAYER_BINARY {
        return Err("workspaceWireBinary: spec is not binary".into());
    }
    validate_workspace_input_slot(workspace, builder, left_slot, "workspaceWireBinary.left")?;
    validate_workspace_input_slot(workspace, builder, right_slot, "workspaceWireBinary.right")?;
    validate_workspace_layout_input(workspace, left_slot, spec, "workspaceWireBinary.left")?;
    validate_workspace_layout_input(workspace, right_slot, spec, "workspaceWireBinary.right")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireBinary")?;
    validate_workspace_op_label(&label, "workspaceWireBinary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireBinary",
    )?;
    if let Err(err) = workspace.sync_layer(registry, spec, label) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    if let Err(err) = builder.add_binary(spec, left_slot, right_slot, output_slot) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    Ok(output_slot)
}

/// Initialize a reserved unary layer and wire it into the graph.
/// A workspace checkpoint protects the entire control state until graph commit succeeds.
#[wasm_bindgen(js_name = workspaceInitUnary)]
pub fn workspace_init_unary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &mut LayerRegistry,
    spec: &AgentLayerSpec,
    input_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() == LAYER_BINARY {
        return Err("workspaceInitUnary: binary spec requires workspaceInitBinary".into());
    }
    validate_workspace_input_slot(workspace, builder, input_slot, "workspaceInitUnary")?;
    validate_workspace_layout_input(workspace, input_slot, spec, "workspaceInitUnary")?;
    validate_spec_is_new(registry, spec, "workspaceInitUnary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitUnary")?;
    validate_workspace_op_label(&label, "workspaceInitUnary")?;

    let workspace_checkpoint = workspace.clone();
    let output_slot = match reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitUnary",
    ) {
        Ok(slot) => slot,
        Err(err) => {
            *workspace = workspace_checkpoint;
            return Err(err);
        }
    };

    if let Err(err) = registry.init_agent_layer(spec) {
        return Err(rollback_init_transaction(
            workspace,
            &workspace_checkpoint,
            registry,
            spec,
            "workspaceInitUnary",
            err,
        ));
    }

    finalize_initialized_unary(
        workspace,
        &workspace_checkpoint,
        builder,
        registry,
        spec,
        input_slot,
        output_slot,
        label,
    )
}

/// Initialize a reserved binary layer and wire it into the graph.
/// A workspace checkpoint protects the entire control state until graph commit succeeds.
#[wasm_bindgen(js_name = workspaceInitBinary)]
pub fn workspace_init_binary(
    workspace: &mut AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    registry: &mut LayerRegistry,
    spec: &AgentLayerSpec,
    left_slot: u8,
    right_slot: u8,
    label: String,
) -> Result<u8, String> {
    if spec.layer_type() != LAYER_BINARY {
        return Err("workspaceInitBinary: spec is not binary".into());
    }
    validate_workspace_input_slot(workspace, builder, left_slot, "workspaceInitBinary.left")?;
    validate_workspace_input_slot(workspace, builder, right_slot, "workspaceInitBinary.right")?;
    validate_workspace_layout_input(workspace, left_slot, spec, "workspaceInitBinary.left")?;
    validate_workspace_layout_input(workspace, right_slot, spec, "workspaceInitBinary.right")?;
    validate_spec_is_new(registry, spec, "workspaceInitBinary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitBinary")?;
    validate_workspace_op_label(&label, "workspaceInitBinary")?;

    let workspace_checkpoint = workspace.clone();
    let output_slot = match reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitBinary",
    ) {
        Ok(slot) => slot,
        Err(err) => {
            *workspace = workspace_checkpoint;
            return Err(err);
        }
    };

    if let Err(err) = registry.init_agent_layer(spec) {
        return Err(rollback_init_transaction(
            workspace,
            &workspace_checkpoint,
            registry,
            spec,
            "workspaceInitBinary",
            err,
        ));
    }

    finalize_initialized_binary(
        workspace,
        &workspace_checkpoint,
        builder,
        registry,
        spec,
        left_slot,
        right_slot,
        output_slot,
        label,
    )
}

/// Compile using a temporary output selection without mutating builder state.
#[wasm_bindgen(js_name = workspaceCompile)]
pub fn workspace_compile(
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    validate_builder_slot(builder, output_slot, "workspaceCompile")?;
    builder.compile_with_output(registry, output_slot)
}

/// Compile and bind the exact CompiledGraph.programIdentity to the immutable runtime subject.
///
/// This is the canonical compile path for subject-bound verification. The historical
/// workspaceCompile surface remains available as an explicit unbound/legacy escape hatch.
#[wasm_bindgen(js_name = workspaceCompileForRuntimeSubject)]
pub fn workspace_compile_for_runtime_subject(
    workspace: &mut AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    if workspace.runtime_subject_binding().is_none() {
        return Err(
            "workspaceCompileForRuntimeSubject: workspace has no bound runtime subject".to_string(),
        );
    }
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "workspaceCompileForRuntimeSubject: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    validate_builder_slot(builder, output_slot, "workspaceCompileForRuntimeSubject")?;
    let graph = builder.compile_with_output(registry, output_slot)?;
    workspace.bind_runtime_program_identity(graph.program_identity())?;
    Ok(graph)
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
            next_attestation_id: 1,
            next_verifier_receipt_id: 1,
            next_event_id: 1,
            input_contract: None,
            input_port_metadata: None,
            runtime_subject_binding: None,
            runtime_program_identities: Vec::new(),
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
    pub fn query(&self, table: String, kind: Option<String>, state: Option<String>) -> String {
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
        let mut names = self
            .rows
            .iter()
            .map(|row| row.table.as_str())
            .collect::<Vec<_>>();
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
                self.next_layer_id = candidate.checked_add(1).ok_or_else(|| {
                    "AgentWorkspace.reserveLayerId: allocator exhausted".to_string()
                })?;
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

        let actual = registry.layer_init_fingerprint(spec.layer_type(), spec.layer_id())?;
        let mut expected_registry = LayerRegistry::new();
        expected_registry.init_agent_layer(spec)?;
        let expected =
            expected_registry.layer_init_fingerprint(spec.layer_type(), spec.layer_id())?;
        if actual != expected {
            return Err(format!(
                "AgentWorkspace.syncLayer: registry init identity mismatch for layer type 0x{:02X} id {}; supplied AgentLayerSpec does not match the live layer",
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
        let input_contract = self
            .input_contract
            .as_ref()
            .map(|contract| {
                format!(
                    "{{\"shape\":[{},{},{},{}],\"layout\":\"{}\",\"semantics\":\"{}\"}}",
                    contract.shape[0],
                    contract.shape[1],
                    contract.shape[2],
                    contract.shape[3],
                    json_escape(&contract.layout),
                    json_escape(&contract.semantics),
                )
            })
            .unwrap_or_else(|| "null".to_string());
        let input_port_metadata = self
            .input_port_metadata
            .as_ref()
            .map(|metadata| {
                format!(
                    concat!(
                        "{{",
                        "\"role\":\"{}\",",
                        "\"source\":\"{}\",",
                        "\"revision\":{},",
                        "\"fingerprint\":\"{}\"",
                        "}}"
                    ),
                    json_escape(&metadata.role),
                    json_escape(&metadata.source),
                    metadata.revision,
                    json_escape(&metadata.fingerprint),
                )
            })
            .unwrap_or_else(|| "null".to_string());
        let runtime_subject = self
            .runtime_subject_binding
            .as_ref()
            .map(|binding| {
                format!(
                    concat!(
                        "{{",
                        "\"intent_id\":\"{}\",",
                        "\"workflow_revision\":{},",
                        "\"approval_id\":\"{}\",",
                        "\"subject_kind\":\"{}\",",
                        "\"subject_identity\":\"{}\",",
                        "\"authorization_policy_id\":\"{}\",",
                        "\"authorization_policy_revision\":{},",
                        "\"authorization_is_revision\":{}",
                        "}}"
                    ),
                    json_escape(&binding.intent_id),
                    binding.workflow_revision,
                    json_escape(&binding.approval_id),
                    json_escape(&binding.subject_kind),
                    json_escape(&binding.subject_identity),
                    json_escape(&binding.authorization_policy_id),
                    binding.authorization_policy_revision,
                    binding.authorization_is_revision,
                )
            })
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"num_slots\":{},\"free_slots\":{},\"layers\":{},\"proofs\":{},\"attestations\":{},\"verifier_receipts\":{},\"events\":{},\"custom_tables\":{},\"rows\":{},\"input_contract\":{},\"input_port_metadata\":{},\"runtime_subject\":{},\"runtime_program_bindings\":{{\"count\":{},\"max\":{MAX_RUNTIME_PROGRAM_BINDINGS},\"identity_policy\":\"exact_program_identity\"}},\"max_rows\":{MAX_ROWS}}}",
            self.num_slots,
            free_slots,
            count(TABLE_LAYERS),
            count(TABLE_PROOFS),
            count(TABLE_ATTESTATIONS),
            count(TABLE_VERIFIER_RECEIPTS),
            count(TABLE_EVENTS),
            custom_tables,
            self.rows.len(),
            input_contract,
            input_port_metadata,
            runtime_subject,
            self.runtime_program_identities.len(),
        )
    }
}
