use wasm_bindgen::prelude::*;

use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::graph::CompiledGraph;
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;

const MAX_WORKSPACE_OP_LABEL_BYTES: usize = 4000;

fn validate_builder_slot(
    builder: &AgentGraphBuilder,
    slot: u8,
    context: &str,
) -> Result<(), String> {
    if u32::from(slot) >= builder.num_slots() {
        return Err(format!(
            "{context}: slot {slot} is outside builder num_slots {}",
            builder.num_slots()
        ));
    }
    Ok(())
}

fn validate_spec_is_new(
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    context: &str,
) -> Result<(), String> {
    if registry.layer_exists(spec.layer_type(), spec.layer_id()) {
        return Err(format!(
            "{context}: layer type 0x{:02X} id {} already exists",
            spec.layer_type(),
            spec.layer_id()
        ));
    }
    Ok(())
}

fn validate_workspace_op_label(label: &str, context: &str) -> Result<(), String> {
    if label.len() > MAX_WORKSPACE_OP_LABEL_BYTES {
        return Err(format!(
            "{context}: label {} bytes exceeds helper limit {MAX_WORKSPACE_OP_LABEL_BYTES}",
            label.len()
        ));
    }
    Ok(())
}

fn ensure_workspace_layer_reserved(
    workspace: &AgentWorkspace,
    spec: &AgentLayerSpec,
    context: &str,
) -> Result<(), String> {
    let row = workspace.get("_layers".into(), spec.layer_id().to_string());
    if row == "null" || !row.contains("\"state\":\"reserved\"") {
        return Err(format!(
            "{context}: layer id {} must be reserved through AgentWorkspace.reserveLayerId before initialization",
            spec.layer_id()
        ));
    }
    Ok(())
}

fn ensure_registry_has_spec(
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    context: &str,
) -> Result<(), String> {
    if !registry.layer_exists(spec.layer_type(), spec.layer_id()) {
        return Err(format!(
            "{context}: layer type 0x{:02X} id {} is not initialized in registry",
            spec.layer_type(),
            spec.layer_id()
        ));
    }
    Ok(())
}

fn ensure_registry_matches_spec(
    registry: &LayerRegistry,
    spec: &AgentLayerSpec,
    context: &str,
) -> Result<(), String> {
    ensure_registry_has_spec(registry, spec, context)?;
    let actual = registry.layer_init_fingerprint(spec.layer_type(), spec.layer_id())?;

    // Build the canonical expected identity through the same registry init boundary.
    // This avoids duplicating or exposing AgentLayerSpec's raw payload representation.
    let mut expected_registry = LayerRegistry::new();
    expected_registry.init_agent_layer(spec)?;
    let expected = expected_registry.layer_init_fingerprint(spec.layer_type(), spec.layer_id())?;

    if actual != expected {
        return Err(format!(
            "{context}: registry init identity mismatch for layer type 0x{:02X} id {}; supplied AgentLayerSpec does not match the live layer",
            spec.layer_type(),
            spec.layer_id()
        ));
    }
    Ok(())
}

fn reserve_workspace_output_slot(
    workspace: &mut AgentWorkspace,
    builder: &AgentGraphBuilder,
    owner: String,
    context: &str,
) -> Result<u8, String> {
    let slot = workspace.reserve_slot(owner)?;
    if let Err(err) = validate_builder_slot(builder, slot, context) {
        let _ = workspace.release_slot(slot);
        return Err(err);
    }
    Ok(slot)
}

/// Discover the canonical agent-facing workspace/control-plane API.
#[wasm_bindgen(js_name = workspaceCapabilities)]
pub fn workspace_capabilities() -> String {
    concat!(
        "{",
        "\"state\":\"AgentWorkspace\",",
        "\"ownership\":\"metadata_only\",",
        "\"execution_truth\":\"LayerRegistry\",",
        "\"graph\":\"AgentGraphBuilder\",",
        "\"provenance\":{\"wire_identity\":\"exact_validated_init_fingerprint\",\"syncLayer\":\"metadata_only_not_canonical_orchestration\"},",
        "\"ops\":[\"workspaceInitUnary\",\"workspaceInitBinary\",\"workspaceWireUnary\",\"workspaceWireBinary\",\"workspaceCompile\"],",
        "\"workspace_methods\":[\"reserveLayerId\",\"reserveSlot\",\"releaseSlot\",\"recordProof\",\"recordEvent\",\"put\",\"get\",\"query\",\"remove\",\"snapshot\",\"limits\"],",
        "\"escape_hatches\":[\"AgentLayerSpec\",\"AgentGraphBuilder\",\"LayerRegistry\",\"raw_protocol\"],",
        "\"recommended_flow\":[\"reserve_layer\",\"construct_spec\",\"init_or_wire\",\"compile\",\"run\",\"verify\"]",
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
    validate_builder_slot(builder, input_slot, "workspaceWireUnary")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireUnary")?;
    validate_workspace_op_label(&label, "workspaceWireUnary")?;

    // Metadata reconciliation happens only after exact registry/spec identity proof.
    workspace.sync_layer(registry, spec, label)?;
    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireUnary",
    )?;
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
    validate_builder_slot(builder, left_slot, "workspaceWireBinary")?;
    validate_builder_slot(builder, right_slot, "workspaceWireBinary")?;
    ensure_registry_matches_spec(registry, spec, "workspaceWireBinary")?;
    validate_workspace_op_label(&label, "workspaceWireBinary")?;

    workspace.sync_layer(registry, spec, label)?;
    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceWireBinary",
    )?;
    if let Err(err) = builder.add_binary(spec, left_slot, right_slot, output_slot) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }
    Ok(output_slot)
}

/// Initialize a reserved unary layer and wire it into the graph.
/// All allocation/state truth remains in AgentWorkspace; this function retains no state.
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
    validate_builder_slot(builder, input_slot, "workspaceInitUnary")?;
    validate_spec_is_new(registry, spec, "workspaceInitUnary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitUnary")?;
    validate_workspace_op_label(&label, "workspaceInitUnary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitUnary",
    )?;

    if let Err(err) = registry.init_agent_layer(spec) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }

    // Prove that the registry persisted the exact init identity we just supplied.
    ensure_registry_matches_spec(registry, spec, "workspaceInitUnary")?;
    workspace.sync_layer(registry, spec, label)?;
    builder.add_unary(spec, input_slot, output_slot)?;
    Ok(output_slot)
}

/// Initialize a reserved binary layer and wire it into the graph.
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
    validate_builder_slot(builder, left_slot, "workspaceInitBinary")?;
    validate_builder_slot(builder, right_slot, "workspaceInitBinary")?;
    validate_spec_is_new(registry, spec, "workspaceInitBinary")?;
    ensure_workspace_layer_reserved(workspace, spec, "workspaceInitBinary")?;
    validate_workspace_op_label(&label, "workspaceInitBinary")?;

    let output_slot = reserve_workspace_output_slot(
        workspace,
        builder,
        format!("layer:{}", spec.layer_id()),
        "workspaceInitBinary",
    )?;

    if let Err(err) = registry.init_agent_layer(spec) {
        let _ = workspace.release_slot(output_slot);
        return Err(err);
    }

    ensure_registry_matches_spec(registry, spec, "workspaceInitBinary")?;
    workspace.sync_layer(registry, spec, label)?;
    builder.add_binary(spec, left_slot, right_slot, output_slot)?;
    Ok(output_slot)
}

/// Set the graph output and compile through the existing registry/compiler boundary.
#[wasm_bindgen(js_name = workspaceCompile)]
pub fn workspace_compile(
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    validate_builder_slot(builder, output_slot, "workspaceCompile")?;
    builder.set_output(output_slot)?;
    builder.compile(registry)
}

#[cfg(test)]
mod tests {
    use super::{
        workspace_capabilities, workspace_compile, workspace_init_binary, workspace_init_unary,
        workspace_wire_unary,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::protocol::{LAYER_ACTIVATION, LAYER_BINARY};
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::WasmTensor;

    #[test]
    fn capabilities_expose_workspace_as_state_and_registry_as_execution_truth() {
        let manifest = workspace_capabilities();
        assert!(manifest.contains("\"state\":\"AgentWorkspace\""));
        assert!(manifest.contains("\"execution_truth\":\"LayerRegistry\""));
        assert!(manifest.contains("exact_validated_init_fingerprint"));
        assert!(manifest.contains("workspaceInitUnary"));
        assert!(manifest.contains("raw_protocol"));
    }

    #[test]
    fn workspace_helpers_keep_all_state_in_workspace() {
        let mut workspace = AgentWorkspace::new(5).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(5).unwrap();

        let relu_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let relu = AgentLayerSpec::relu(relu_id);
        let relu_slot = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &relu,
            0,
            "relu".into(),
        )
        .unwrap();

        let add_id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
        let add = AgentLayerSpec::add(add_id);
        let sum_slot = workspace_init_binary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &add,
            relu_slot,
            relu_slot,
            "add".into(),
        )
        .unwrap();

        let graph = workspace_compile(&mut builder, &registry, sum_slot).unwrap();
        let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);
        assert_eq!(graph.run(&registry, &input).unwrap().to_array(), vec![0.0, 6.0]);
        assert!(workspace
            .get("_layers".into(), relu_id.to_string())
            .contains("initialized"));
    }

    #[test]
    fn workspace_init_requires_workspace_reserved_layer_id() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let relu = AgentLayerSpec::relu(77);

        assert!(workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &relu,
            0,
            "relu".into(),
        )
        .is_err());
        assert!(!registry.layer_exists(LAYER_ACTIVATION, 77));
        assert!(workspace.get("_slots".into(), "1".into()).contains("free"));
    }

    #[test]
    fn wrong_arity_does_not_consume_slot_or_initialize_layer() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let add_id = workspace.reserve_layer_id(&registry, "add".into()).unwrap();
        let add = AgentLayerSpec::add(add_id);

        assert!(workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &add,
            0,
            "add".into(),
        )
        .is_err());
        assert!(!registry.layer_exists(LAYER_BINARY, add_id));
        assert!(workspace.get("_slots".into(), "1".into()).contains("free"));
    }

    #[test]
    fn manual_layer_can_reconcile_without_helper_state_when_identity_matches() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let relu = AgentLayerSpec::relu(42);
        registry.init_agent_layer(&relu).unwrap();

        let out = workspace_wire_unary(
            &mut workspace,
            &mut builder,
            &registry,
            &relu,
            0,
            "manual-relu".into(),
        )
        .unwrap();
        let graph = workspace_compile(&mut builder, &registry, out).unwrap();
        let input = WasmTensor::new(&[-1.0, 4.0], &[1, 2, 1, 1]);
        assert_eq!(graph.run(&registry, &input).unwrap().to_array(), vec![0.0, 4.0]);
        assert!(workspace.get("_layers".into(), "42".into()).contains("manual-relu"));
    }

    #[test]
    fn wire_rejects_same_id_and_type_with_different_variant() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let relu = AgentLayerSpec::relu(1);
        let sigmoid = AgentLayerSpec::sigmoid(1);
        registry.init_agent_layer(&relu).unwrap();

        let err = workspace_wire_unary(
            &mut workspace,
            &mut builder,
            &registry,
            &sigmoid,
            0,
            "wrong-spec".into(),
        )
        .unwrap_err();
        assert!(err.contains("init identity mismatch"));
        assert_eq!(workspace.get("_layers".into(), "1".into()), "null");
        assert!(workspace.get("_slots".into(), "1".into()).contains("free"));
        assert_eq!(builder.num_steps(), 0);
    }

    #[test]
    fn wire_rejects_same_variant_with_different_config_payload() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let softmax_dim_1 = AgentLayerSpec::softmax(9, 1);
        let softmax_dim_2 = AgentLayerSpec::softmax(9, 2);
        registry.init_agent_layer(&softmax_dim_1).unwrap();

        let err = workspace_wire_unary(
            &mut workspace,
            &mut builder,
            &registry,
            &softmax_dim_2,
            0,
            "wrong-config".into(),
        )
        .unwrap_err();
        assert!(err.contains("init identity mismatch"));
        assert_eq!(workspace.get("_layers".into(), "9".into()), "null");
        assert!(workspace.get("_slots".into(), "1".into()).contains("free"));
        assert_eq!(builder.num_steps(), 0);
    }
}