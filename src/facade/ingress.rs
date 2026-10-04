//! Fasad WASM tunggal — domain `ingress` (Opsi C, Fase 1).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::ingress::...` di `src/lib.rs`.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentGraphSemanticEdgeBinding;
use crate::agent::AgentLayerSpec;
use crate::contracts::validate_external_input_contract_declaration;
use crate::contracts::validate_external_input_contract_for_spec;
use crate::ingress::input_contract::contract_json;
use crate::ingress::input_contract::json_escape;
use crate::ingress::input_contract::INPUT_CONTRACT_V1;
use crate::ingress::input_contract::MAX_INPUT_SEMANTICS_BYTES;
use crate::ingress::input_port::metadata_json;
use crate::ingress::input_port::role_valid;
use crate::ingress::input_port::validate_bounded;
use crate::ingress::input_port::INPUT_PORT_V1;
use crate::ingress::input_port::MAX_FINGERPRINT_BYTES;
use crate::ingress::input_port::MAX_ROLE_BYTES;
use crate::ingress::input_port::MAX_SOURCE_BYTES;
use crate::ingress::input_port_consumer::bool_json;
use crate::ingress::input_port_consumer::parse_roles;
use crate::ingress::input_port_consumer::string_array_json;
use crate::ingress::input_port_consumer::InputPortConsumerSpec;
use crate::ingress::input_port_consumer::INPUT_PORT_CONSUMER_V1;
use crate::ingress::input_port_consumer::MAX_CONSUMER_ID_BYTES;
use crate::ingress::input_port_edge_binding::binding_fingerprint;
use crate::ingress::input_port_edge_binding::binding_record_json;
use crate::ingress::input_port_edge_binding::external_positions;
use crate::ingress::input_port_edge_binding::semantic_graph_identity_json;
use crate::ingress::input_port_edge_binding::stored_consumer_compatible;
use crate::ingress::input_port_edge_binding::INPUT_PORT_EDGE_BINDING_V1;
use crate::ingress::input_port_routing::positions_json;
use crate::ingress::input_port_routing::INPUT_PORT_ROUTING_V1;
use crate::interaction::valid_actions_json;
use crate::interaction::validate_projection_inputs;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;
use crate::workspace::WorkspaceInputContract;
use crate::workspace::WorkspaceInputPortMetadata;

/// Return the embedded contract for optional external input metadata.
#[wasm_bindgen(js_name = inputContractCapabilities)]
pub fn input_contract_capabilities() -> String {
    INPUT_CONTRACT_V1.to_string()
}

/// Bind optional shape/layout metadata to the canonical external input slot 0.
///
/// This mutates AgentWorkspace metadata only. It does not allocate tensors,
/// initialize layers, alter the graph plan, or mutate LayerRegistry.
#[wasm_bindgen(js_name = workspaceBindInputContract)]
#[allow(clippy::too_many_arguments)]
pub fn workspace_bind_input_contract(
    workspace: &mut AgentWorkspace,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    layout: String,
    semantics: String,
) -> Result<(), String> {
    if semantics.len() > MAX_INPUT_SEMANTICS_BYTES {
        return Err(format!(
            "workspaceBindInputContract: semantics {} bytes exceeds limit {MAX_INPUT_SEMANTICS_BYTES}",
            semantics.len()
        ));
    }

    let shape = [dim0, dim1, dim2, dim3];
    validate_external_input_contract_declaration(shape, &layout)
        .map_err(|err| format!("workspaceBindInputContract: {err}"))?;

    workspace.set_input_contract(WorkspaceInputContract {
        shape,
        layout,
        semantics,
    });
    Ok(())
}

/// Clear optional external input metadata. Runtime execution behavior is unchanged.
#[wasm_bindgen(js_name = workspaceClearInputContract)]
pub fn workspace_clear_input_contract(workspace: &mut AgentWorkspace) -> bool {
    workspace.clear_input_contract_internal()
}

/// Return the current input contract or an explicit unbound marker.
#[wasm_bindgen(js_name = workspaceInputContract)]
pub fn workspace_input_contract(workspace: &AgentWorkspace) -> String {
    match workspace.input_contract() {
        Some(contract) => format!(
            "{{\"status\":\"bound\",\"contract\":{}}}",
            contract_json(contract)
        ),
        None => concat!(
            "{",
            "\"status\":\"unbound\",",
            "\"slot\":0,",
            "\"policy\":\"defer_to_runtime\"",
            "}"
        )
        .to_string(),
    }
}

/// Compare the current optional input contract against one typed consumer spec.
///
/// Unbound and unknown are not failures. Incompatible means the declared
/// shape/layout proves the consumer cannot accept the external input without an
/// explicit transform.
#[wasm_bindgen(js_name = inputContractCompatibility)]
pub fn input_contract_compatibility(
    workspace: &AgentWorkspace,
    consumer: &AgentLayerSpec,
) -> String {
    let Some(contract) = workspace.input_contract() else {
        return concat!(
            "{",
            "\"status\":\"unbound\",",
            "\"compatible\":null,",
            "\"policy\":\"defer_to_runtime\"",
            "}"
        )
        .to_string();
    };

    match validate_external_input_contract_for_spec(contract.shape, &contract.layout, consumer) {
        Ok(result) => {
            let compatible = if result == "compatible" {
                "true"
            } else {
                "null"
            };
            format!(
                concat!(
                    "{{",
                    "\"status\":\"{}\",",
                    "\"compatible\":{},",
                    "\"consumer_layer_type\":{},",
                    "\"consumer_layer_id\":{},",
                    "\"contract\":{}",
                    "}}"
                ),
                json_escape(result),
                compatible,
                consumer.layer_type(),
                consumer.layer_id(),
                contract_json(contract),
            )
        }
        Err(message) => format!(
            concat!(
                "{{",
                "\"status\":\"incompatible\",",
                "\"compatible\":false,",
                "\"consumer_layer_type\":{},",
                "\"consumer_layer_id\":{},",
                "\"message\":\"{}\",",
                "\"contract\":{}",
                "}}"
            ),
            consumer.layer_type(),
            consumer.layer_id(),
            json_escape(&message),
            contract_json(contract),
        ),
    }
}

#[wasm_bindgen(js_name = inputPortCapabilities)]
pub fn input_port_capabilities() -> String {
    INPUT_PORT_V1.to_string()
}

#[wasm_bindgen(js_name = workspaceBindInputPortMetadata)]
pub fn workspace_bind_input_port_metadata(
    workspace: &mut AgentWorkspace,
    role: String,
    source: String,
    revision: u64,
    fingerprint: String,
) -> Result<bool, String> {
    validate_bounded(
        &role,
        MAX_ROLE_BYTES,
        "workspaceBindInputPortMetadata.role",
        false,
    )?;
    if !role_valid(&role) {
        return Err(format!(
            "workspaceBindInputPortMetadata.role: unsupported role {role}; use canonical role or x- extension namespace"
        ));
    }
    validate_bounded(
        &source,
        MAX_SOURCE_BYTES,
        "workspaceBindInputPortMetadata.source",
        false,
    )?;
    validate_bounded(
        &fingerprint,
        MAX_FINGERPRINT_BYTES,
        "workspaceBindInputPortMetadata.fingerprint",
        true,
    )?;

    Ok(
        workspace.set_input_port_metadata(WorkspaceInputPortMetadata {
            role,
            source,
            revision,
            fingerprint,
        }),
    )
}

#[wasm_bindgen(js_name = workspaceClearInputPortMetadata)]
pub fn workspace_clear_input_port_metadata(workspace: &mut AgentWorkspace) -> bool {
    workspace.clear_input_port_metadata_internal()
}

#[wasm_bindgen(js_name = workspaceInputPortMetadata)]
pub fn workspace_input_port_metadata(workspace: &AgentWorkspace) -> String {
    match workspace.input_port_metadata() {
        Some(metadata) => format!(
            "{{\"status\":\"bound\",\"metadata\":{},\"runtime_subject_bound\":{}}}",
            metadata_json(metadata),
            if workspace.runtime_subject_binding().is_some() {
                "true"
            } else {
                "false"
            }
        ),
        None => format!(
            "{{\"status\":\"unbound\",\"slot\":0,\"runtime_subject_bound\":{},\"policy\":\"semantic_role_optional\"}}",
            if workspace.runtime_subject_binding().is_some() {
                "true"
            } else {
                "false"
            }
        ),
    }
}

#[wasm_bindgen(js_name = inputPortConsumerCapabilities)]
pub fn input_port_consumer_capabilities() -> String {
    INPUT_PORT_CONSUMER_V1.to_string()
}

#[wasm_bindgen(js_name = inputPortConsumerCompatibility)]
pub fn input_port_consumer_compatibility(
    workspace: &AgentWorkspace,
    consumer: &InputPortConsumerSpec,
) -> String {
    let Some(metadata) = workspace.input_port_metadata() else {
        return format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.input-port-consumer-result.v1\",",
                "\"status\":\"unknown\",",
                "\"compatible\":null,",
                "\"execution_authorized\":false,",
                "\"decision_authority\":\"agent\",",
                "\"consumer\":{},",
                "\"reason\":\"semantic_port_unbound\"",
                "}}"
            ),
            consumer.json(),
        );
    };

    let role_match = consumer.role_matches(&metadata.role);
    let fingerprint_present = !metadata.fingerprint.is_empty();
    let fingerprint_ok = !consumer.require_fingerprint || fingerprint_present;
    let revision_ok =
        consumer.minimum_revision == 0 || metadata.revision >= consumer.minimum_revision;
    let compatible = role_match && fingerprint_ok && revision_ok;

    let mut reasons = Vec::<String>::new();
    if !role_match {
        reasons.push("role_not_accepted".to_string());
    }
    if !fingerprint_ok {
        reasons.push("fingerprint_required".to_string());
    }
    if !revision_ok {
        reasons.push("revision_too_old".to_string());
    }

    format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-consumer-result.v1\",",
            "\"status\":\"{}\",",
            "\"compatible\":{},",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\",",
            "\"consumer\":{},",
            "\"input_port\":{{",
            "\"slot\":0,",
            "\"role\":\"{}\",",
            "\"provenance\":{{",
            "\"source\":\"{}\",",
            "\"revision\":{},",
            "\"fingerprint_present\":{}",
            "}}",
            "}},",
            "\"predicates\":{{",
            "\"role_match\":{},",
            "\"fingerprint_ok\":{},",
            "\"revision_ok\":{}",
            "}},",
            "\"reasons\":{}",
            "}}"
        ),
        if compatible {
            "compatible"
        } else {
            "incompatible"
        },
        bool_json(compatible),
        consumer.json(),
        json_escape(&metadata.role),
        json_escape(&metadata.source),
        metadata.revision,
        bool_json(fingerprint_present),
        bool_json(role_match),
        bool_json(fingerprint_ok),
        bool_json(revision_ok),
        string_array_json(&reasons),
    )
}

#[wasm_bindgen(js_name = inputPortEdgeBindingCapabilities)]
pub fn input_port_edge_binding_capabilities() -> String {
    INPUT_PORT_EDGE_BINDING_V1.to_string()
}

#[wasm_bindgen(js_name = bindInputPortConsumerEdge)]
pub fn bind_input_port_consumer_edge(
    workspace: &AgentWorkspace,
    builder: &mut AgentGraphBuilder,
    consumer: &InputPortConsumerSpec,
    step_index: u32,
) -> Result<bool, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "bindInputPortConsumerEdge: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let positions = external_positions(builder, step_index, "bindInputPortConsumerEdge")?;
    if positions.is_empty() {
        return Err(format!(
            "bindInputPortConsumerEdge: step {step_index} does not consume external slot 0"
        ));
    }

    if consumer.is_compatible_with_workspace(workspace).is_none() {
        return Err(
            "bindInputPortConsumerEdge: semantic input port is unbound; compatibility is unknown"
                .to_string(),
        );
    }

    let metadata = workspace.input_port_metadata().ok_or_else(|| {
        "bindInputPortConsumerEdge: semantic input metadata disappeared".to_string()
    })?;
    let fingerprint = binding_fingerprint(
        step_index,
        &positions,
        consumer,
        &metadata.role,
        &metadata.source,
        metadata.revision,
        &metadata.fingerprint,
    );

    builder.bind_semantic_edge(AgentGraphSemanticEdgeBinding {
        step_index,
        external_input_positions: positions,
        consumer_id: consumer.consumer_id_ref().to_string(),
        accepted_roles: consumer.accepted_roles_slice().to_vec(),
        allow_extension_roles: consumer.allow_extension_roles_value(),
        require_fingerprint: consumer.require_fingerprint_value(),
        minimum_revision: consumer.minimum_revision_value(),
        bound_role: metadata.role.clone(),
        bound_source: metadata.source.clone(),
        bound_revision: metadata.revision,
        bound_fingerprint: metadata.fingerprint.clone(),
        binding_fingerprint: fingerprint,
    })
}

#[wasm_bindgen(js_name = inputPortConsumerEdgeBinding)]
pub fn input_port_consumer_edge_binding(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    step_index: u32,
) -> Result<String, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "inputPortConsumerEdgeBinding: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }
    let positions = external_positions(builder, step_index, "inputPortConsumerEdgeBinding")?;
    if positions.is_empty() {
        return Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",\"status\":\"not_applicable\",\"step_index\":{step_index},\"reason\":\"step_does_not_consume_external_slot_0\"}}"
        ));
    }

    let Some(binding) = builder.semantic_edge_binding(step_index) else {
        return Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",\"status\":\"unbound\",\"step_index\":{step_index}}}"
        ));
    };

    let (state, current_compatible, snapshot_match) = match workspace.input_port_metadata() {
        None => ("input_unbound", None, false),
        Some(metadata) => {
            let compatible = stored_consumer_compatible(binding, workspace).unwrap_or(false);
            let snapshot_match = metadata.role == binding.bound_role
                && metadata.source == binding.bound_source
                && metadata.revision == binding.bound_revision
                && metadata.fingerprint == binding.bound_fingerprint;
            let state = if snapshot_match {
                "current"
            } else if compatible {
                "drifted_compatible"
            } else {
                "drifted_incompatible"
            };
            (state, Some(compatible), snapshot_match)
        }
    };

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-edge-binding-status.v1\",",
            "\"status\":\"bound\",",
            "\"binding_state\":\"{}\",",
            "\"current_compatible\":{},",
            "\"input_snapshot_match\":{},",
            "\"binding\":{},",
            "\"semantic_graph_identity\":{}",
            "}}"
        ),
        state,
        current_compatible.map(bool_json).unwrap_or("null"),
        bool_json(snapshot_match),
        binding_record_json(binding),
        semantic_graph_identity_json(builder),
    ))
}

#[wasm_bindgen(js_name = semanticGraphIdentity)]
pub fn semantic_graph_identity(builder: &AgentGraphBuilder) -> String {
    semantic_graph_identity_json(builder)
}

#[wasm_bindgen(js_name = inputPortRoutingCapabilities)]
pub fn input_port_routing_capabilities() -> String {
    INPUT_PORT_ROUTING_V1.to_string()
}

#[wasm_bindgen(js_name = interactionValidActionsForInputConsumer)]
pub fn interaction_valid_actions_for_input_consumer(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    registry: &LayerRegistry,
    consumer: &InputPortConsumerSpec,
) -> Result<String, String> {
    validate_projection_inputs(workspace, builder)?;

    let slot_zero_is_canonical_candidate = workspace.interaction_readable_slots().contains(&0);
    let compatibility = input_port_consumer_compatibility(workspace, consumer);

    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-routing-actions.v1\",",
            "\"projection_only\":true,",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\",",
            "\"canonical_valid_actions\":{},",
            "\"semantic_overlay\":{{",
                "\"input_slot\":0,",
                "\"candidate_present\":{},",
                "\"candidate_retained\":true,",
                "\"consumer\":{},",
                "\"compatibility\":{},",
                "\"affected_operations\":[",
                    "{{\"operation\":\"workspaceInitUnary\",\"positions\":[\"input\"],\"effect\":\"advisory_only\"}},",
                    "{{\"operation\":\"workspaceInitBinary\",\"positions\":[\"left\",\"right\"],\"effect\":\"advisory_only\"}}",
                "],",
                "\"policy\":\"semantic_status_never_rewrites_canonical_action_availability\"",
            "}}",
            "}}"
        ),
        valid_actions_json(workspace, builder, registry),
        bool_json(slot_zero_is_canonical_candidate),
        consumer.describe(),
        compatibility,
    ))
}

#[wasm_bindgen(js_name = inputPortConsumerEdgeCompatibility)]
pub fn input_port_consumer_edge_compatibility(
    workspace: &AgentWorkspace,
    builder: &AgentGraphBuilder,
    consumer: &InputPortConsumerSpec,
    step_index: u32,
) -> Result<String, String> {
    if workspace.interaction_num_slots() != builder.num_slots() {
        return Err(format!(
            "inputPortConsumerEdgeCompatibility: workspace num_slots {} does not match builder num_slots {}",
            workspace.interaction_num_slots(),
            builder.num_slots()
        ));
    }

    let steps = builder.introspection_steps();
    let index = usize::try_from(step_index).map_err(|_| {
        "inputPortConsumerEdgeCompatibility: step index conversion failed".to_string()
    })?;
    let Some((arity, layer_type, layer_id, in_slot, in_slot2, out_slot)) =
        steps.get(index).copied()
    else {
        return Err(format!(
            "inputPortConsumerEdgeCompatibility: step index {step_index} is outside num_steps {}",
            steps.len()
        ));
    };

    let positions = builder
        .external_input_positions_for_step(step_index)
        .map_err(|error| format!("inputPortConsumerEdgeCompatibility: {error}"))?;

    if positions.is_empty() {
        return Ok(format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.input-port-routing-edge.v1\",",
                "\"projection_only\":true,",
                "\"status\":\"not_applicable\",",
                "\"execution_authorized\":false,",
                "\"decision_authority\":\"agent\",",
                "\"consumer\":{},",
                "\"edge\":{{",
                "\"step_index\":{},",
                "\"arity\":{},",
                "\"layer_type\":{},",
                "\"layer_id\":{},",
                "\"input_slots\":[{},{}],",
                "\"output_slot\":{},",
                "\"external_input_positions\":[]",
                "}},",
                "\"compatibility\":null,",
                "\"reason\":\"step_does_not_consume_external_slot_0\"",
                "}}"
            ),
            consumer.describe(),
            step_index,
            arity,
            layer_type,
            layer_id,
            in_slot,
            in_slot2,
            out_slot,
        ));
    }

    let compatibility = input_port_consumer_compatibility(workspace, consumer);
    Ok(format!(
        concat!(
            "{{",
            "\"schema_version\":1,",
            "\"schema_id\":\"burn-research.input-port-routing-edge.v1\",",
            "\"projection_only\":true,",
            "\"status\":\"applicable\",",
            "\"execution_authorized\":false,",
            "\"decision_authority\":\"agent\",",
            "\"consumer\":{},",
            "\"edge\":{{",
            "\"step_index\":{},",
            "\"arity\":{},",
            "\"layer_type\":{},",
            "\"layer_id\":{},",
            "\"input_slots\":[{},{}],",
            "\"output_slot\":{},",
            "\"external_input_positions\":{}",
            "}},",
            "\"compatibility\":{},",
            "\"policy\":\"edge_semantics_are_caller_declared_not_inferred_from_layer_type\"",
            "}}"
        ),
        consumer.describe(),
        step_index,
        arity,
        layer_type,
        layer_id,
        in_slot,
        in_slot2,
        out_slot,
        positions_json(&positions),
        compatibility,
    ))
}

#[wasm_bindgen]
impl InputPortConsumerSpec {
    #[wasm_bindgen(constructor)]
    pub fn new(
        consumer_id: String,
        accepted_roles_json: String,
        allow_extension_roles: bool,
        require_fingerprint: bool,
        minimum_revision: u64,
    ) -> Result<InputPortConsumerSpec, String> {
        if consumer_id.is_empty() {
            return Err("InputPortConsumerSpec.consumer_id: value must be non-empty".to_string());
        }
        if consumer_id.len() > MAX_CONSUMER_ID_BYTES {
            return Err(format!(
                "InputPortConsumerSpec.consumer_id: {} bytes exceeds limit {MAX_CONSUMER_ID_BYTES}",
                consumer_id.len()
            ));
        }
        let accepted_roles = parse_roles(&accepted_roles_json)?;
        Ok(Self {
            consumer_id,
            accepted_roles,
            allow_extension_roles,
            require_fingerprint,
            minimum_revision,
        })
    }

    #[wasm_bindgen(js_name = consumerId)]
    pub fn consumer_id(&self) -> String {
        self.consumer_id.clone()
    }

    #[wasm_bindgen(js_name = acceptedRoles)]
    pub fn accepted_roles(&self) -> String {
        string_array_json(&self.accepted_roles)
    }

    #[wasm_bindgen(js_name = allowExtensionRoles)]
    pub fn allow_extension_roles(&self) -> bool {
        self.allow_extension_roles
    }

    #[wasm_bindgen(js_name = requireFingerprint)]
    pub fn require_fingerprint(&self) -> bool {
        self.require_fingerprint
    }

    #[wasm_bindgen(js_name = minimumRevision)]
    pub fn minimum_revision(&self) -> u64 {
        self.minimum_revision
    }

    #[wasm_bindgen(js_name = describe)]
    pub fn describe(&self) -> String {
        self.json()
    }
}
