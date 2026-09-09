use wasm_bindgen::prelude::*;

use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
use crate::graph::CompiledGraph;
use crate::protocol::{
    LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING, LAYER_GHOST, LAYER_LINEAR,
    LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK, LAYER_SHIFT,
};
use crate::registry::LayerRegistry;
use crate::workspace::AgentWorkspace;

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

const MAX_WORKSPACE_OP_LABEL_BYTES: usize = 4000;

#[wasm_bindgen]
pub struct AgentReferenceSession {
    num_slots: u32,
    next_slot: u32,
    next_layer_id: u32,
}

impl AgentReferenceSession {
    fn ensure_builder_matches(&self, builder: &AgentGraphBuilder) -> Result<(), String> {
        if builder.num_slots() != self.num_slots {
            return Err(format!(
                "AgentReferenceSession: builder num_slots {} does not match session {}",
                builder.num_slots(), self.num_slots
            ));
        }
        Ok(())
    }

    fn validate_input_slot(
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

    fn allocate_output_slot(&self) -> Result<u8, String> {
        if self.next_slot >= self.num_slots {
            return Err(format!(
                "AgentReferenceSession: no free session-managed slots remain (next {}, capacity {})",
                self.next_slot, self.num_slots
            ));
        }
        u8::try_from(self.next_slot)
            .map_err(|_| "AgentReferenceSession: slot index exceeds u8 range".to_string())
    }

    fn commit_output_slot(&mut self) -> Result<(), String> {
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .ok_or_else(|| "AgentReferenceSession: slot allocator overflow".to_string())?;
        Ok(())
    }

    fn layer_id_in_use(registry: &LayerRegistry, layer_id: u32) -> bool {
        KNOWN_LAYER_TYPES
            .iter()
            .copied()
            .any(|layer_type| registry.layer_exists(layer_type, layer_id))
    }

    fn validate_spec_is_new(registry: &LayerRegistry, spec: &AgentLayerSpec) -> Result<(), String> {
        if registry.layer_exists(spec.layer_type(), spec.layer_id()) {
            return Err(format!(
                "AgentReferenceSession: layer type 0x{:02X} id {} already exists",
                spec.layer_type(),
                spec.layer_id()
            ));
        }
        Ok(())
    }
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

fn reserve_workspace_output_slot(
    workspace: &mut AgentWorkspace,
    builder: &AgentGraphBuilder,
    owner: String,
    context: &str,
) -> Result<u8, String> {
    let slot = workspace.reserve_slot(owner)?;
    if let Err(err) = AgentReferenceSession::validate_input_slot(builder, slot, context) {
        let _ = workspace.release_slot(slot);
        return Err(err);
    }
    Ok(slot)
}

/// Stateless convenience path: all allocator/state truth remains in AgentWorkspace.
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
    AgentReferenceSession::validate_input_slot(builder, input_slot, "workspaceWireUnary")?;
    ensure_registry_has_spec(registry, spec, "workspaceWireUnary")?;
    validate_workspace_op_label(&label, "workspaceWireUnary")?;

    // Reconcile metadata first. If quota/metadata validation fails, graph state is untouched.
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
    AgentReferenceSession::validate_input_slot(builder, left_slot, "workspaceWireBinary")?;
    AgentReferenceSession::validate_input_slot(builder, right_slot, "workspaceWireBinary")?;
    ensure_registry_has_spec(registry, spec, "workspaceWireBinary")?;
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
    AgentReferenceSession::validate_input_slot(builder, input_slot, "workspaceInitUnary")?;
    AgentReferenceSession::validate_spec_is_new(registry, spec)?;
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

    // The reserved _layers row already exists, so this is an in-place metadata transition.
    workspace.sync_layer(registry, spec, label)?;
    builder.add_unary(spec, input_slot, output_slot)?;
    Ok(output_slot)
}

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
    AgentReferenceSession::validate_input_slot(builder, left_slot, "workspaceInitBinary")?;
    AgentReferenceSession::validate_input_slot(builder, right_slot, "workspaceInitBinary")?;
    AgentReferenceSession::validate_spec_is_new(registry, spec)?;
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

    workspace.sync_layer(registry, spec, label)?;
    builder.add_binary(spec, left_slot, right_slot, output_slot)?;
    Ok(output_slot)
}

#[wasm_bindgen(js_name = workspaceCompile)]
pub fn workspace_compile(
    builder: &mut AgentGraphBuilder,
    registry: &LayerRegistry,
    output_slot: u8,
) -> Result<CompiledGraph, String> {
    AgentReferenceSession::validate_input_slot(builder, output_slot, "workspaceCompile")?;
    builder.set_output(output_slot)?;
    builder.compile(registry)
}

#[wasm_bindgen]
impl AgentReferenceSession {
    #[wasm_bindgen(constructor)]
    pub fn new(num_slots: u32) -> Result<AgentReferenceSession, String> {
        if !(1..=64).contains(&num_slots) {
            return Err(format!(
                "AgentReferenceSession: num_slots must be 1..=64, got {num_slots}"
            ));
        }
        Ok(Self {
            num_slots,
            next_slot: 1,
            next_layer_id: 1,
        })
    }

    #[wasm_bindgen(js_name = inputSlot)]
    pub fn input_slot(&self) -> u8 {
        0
    }

    #[wasm_bindgen(js_name = nextSlot)]
    pub fn next_slot(&self) -> u32 {
        self.next_slot
    }

    #[wasm_bindgen(js_name = nextLayerId)]
    pub fn next_layer_id(&self) -> u32 {
        self.next_layer_id
    }

    #[wasm_bindgen(js_name = setNextSlot)]
    pub fn set_next_slot(&mut self, next_slot: u32) -> Result<(), String> {
        if next_slot == 0 || next_slot > self.num_slots {
            return Err(format!(
                "AgentReferenceSession.setNextSlot: expected 1..={}, got {next_slot}",
                self.num_slots
            ));
        }
        self.next_slot = next_slot;
        Ok(())
    }

    #[wasm_bindgen(js_name = setNextLayerId)]
    pub fn set_next_layer_id(&mut self, next_layer_id: u32) {
        self.next_layer_id = next_layer_id;
    }

    #[wasm_bindgen(js_name = reserveSlot)]
    pub fn reserve_slot(&mut self) -> Result<u8, String> {
        let slot = self.allocate_output_slot()?;
        self.commit_output_slot()?;
        Ok(slot)
    }

    #[wasm_bindgen(js_name = reserveLayerId)]
    pub fn reserve_layer_id(&mut self, registry: &LayerRegistry) -> Result<u32, String> {
        let mut candidate = self.next_layer_id;
        loop {
            if !Self::layer_id_in_use(registry, candidate) {
                self.next_layer_id = candidate
                    .checked_add(1)
                    .ok_or_else(|| "AgentReferenceSession: layer id allocator exhausted".to_string())?;
                return Ok(candidate);
            }
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| "AgentReferenceSession: layer id allocator exhausted".to_string())?;
        }
    }

    #[wasm_bindgen(js_name = wireUnary)]
    pub fn wire_unary(
        &mut self,
        builder: &mut AgentGraphBuilder,
        spec: &AgentLayerSpec,
        input_slot: u8,
    ) -> Result<u8, String> {
        self.ensure_builder_matches(builder)?;
        let output_slot = self.allocate_output_slot()?;
        builder.add_unary(spec, input_slot, output_slot)?;
        self.commit_output_slot()?;
        Ok(output_slot)
    }

    #[wasm_bindgen(js_name = wireBinary)]
    pub fn wire_binary(
        &mut self,
        builder: &mut AgentGraphBuilder,
        spec: &AgentLayerSpec,
        left_slot: u8,
        right_slot: u8,
    ) -> Result<u8, String> {
        self.ensure_builder_matches(builder)?;
        let output_slot = self.allocate_output_slot()?;
        builder.add_binary(spec, left_slot, right_slot, output_slot)?;
        self.commit_output_slot()?;
        Ok(output_slot)
    }

    #[wasm_bindgen(js_name = initUnary)]
    pub fn init_unary(
        &mut self,
        builder: &mut AgentGraphBuilder,
        registry: &mut LayerRegistry,
        spec: &AgentLayerSpec,
        input_slot: u8,
    ) -> Result<u8, String> {
        self.ensure_builder_matches(builder)?;
        if spec.layer_type() == LAYER_BINARY {
            return Err("AgentReferenceSession.initUnary: binary spec requires initBinary".into());
        }
        Self::validate_input_slot(builder, input_slot, "AgentReferenceSession.initUnary")?;
        Self::validate_spec_is_new(registry, spec)?;
        let output_slot = self.allocate_output_slot()?;
        registry.init_agent_layer(spec)?;
        builder.add_unary(spec, input_slot, output_slot)?;
        self.commit_output_slot()?;
        Ok(output_slot)
    }

    #[wasm_bindgen(js_name = initBinary)]
    pub fn init_binary(
        &mut self,
        builder: &mut AgentGraphBuilder,
        registry: &mut LayerRegistry,
        spec: &AgentLayerSpec,
        left_slot: u8,
        right_slot: u8,
    ) -> Result<u8, String> {
        self.ensure_builder_matches(builder)?;
        if spec.layer_type() != LAYER_BINARY {
            return Err("AgentReferenceSession.initBinary: spec is not binary".into());
        }
        Self::validate_input_slot(builder, left_slot, "AgentReferenceSession.initBinary")?;
        Self::validate_input_slot(builder, right_slot, "AgentReferenceSession.initBinary")?;
        Self::validate_spec_is_new(registry, spec)?;
        let output_slot = self.allocate_output_slot()?;
        registry.init_agent_layer(spec)?;
        builder.add_binary(spec, left_slot, right_slot, output_slot)?;
        self.commit_output_slot()?;
        Ok(output_slot)
    }

    pub fn compile(
        &self,
        builder: &mut AgentGraphBuilder,
        registry: &LayerRegistry,
        output_slot: u8,
    ) -> Result<CompiledGraph, String> {
        self.ensure_builder_matches(builder)?;
        builder.set_output(output_slot)?;
        builder.compile(registry)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        workspace_compile, workspace_init_binary, workspace_init_unary, workspace_wire_unary,
        AgentReferenceSession,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::protocol::{LAYER_ACTIVATION, LAYER_BINARY};
    use crate::registry::LayerRegistry;
    use crate::workspace::AgentWorkspace;
    use crate::WasmTensor;

    #[test]
    fn session_orchestrates_without_owning_registry_or_builder() {
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        let mut session = AgentReferenceSession::new(4).unwrap();

        let input_slot = session.input_slot();
        let relu_id = session.reserve_layer_id(&registry).unwrap();
        let relu = AgentLayerSpec::relu(relu_id);
        let relu_slot = session
            .init_unary(&mut builder, &mut registry, &relu, input_slot)
            .unwrap();

        assert!(registry.layer_exists(LAYER_ACTIVATION, relu_id));
        assert_eq!(relu_slot, 1);

        let graph = session.compile(&mut builder, &registry, relu_slot).unwrap();
        let input = WasmTensor::new(&[-3.0, 2.0], &[1, 2, 1, 1]);
        assert_eq!(graph.run(&registry, &input).unwrap().to_array(), vec![0.0, 2.0]);
    }

    #[test]
    fn session_supports_mixing_manual_and_convenience_layers() {
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(5).unwrap();
        let mut session = AgentReferenceSession::new(5).unwrap();

        let relu = AgentLayerSpec::relu(77);
        registry.init_agent_layer(&relu).unwrap();
        let relu_slot = session.wire_unary(&mut builder, &relu, 0).unwrap();

        let add_id = session.reserve_layer_id(&registry).unwrap();
        let add = AgentLayerSpec::add(add_id);
        let sum_slot = session
            .init_binary(&mut builder, &mut registry, &add, relu_slot, relu_slot)
            .unwrap();

        assert!(registry.layer_exists(LAYER_BINARY, add_id));
        let graph = session.compile(&mut builder, &registry, sum_slot).unwrap();
        let input = WasmTensor::new(&[-1.0, 3.0], &[1, 2, 1, 1]);
        assert_eq!(graph.run(&registry, &input).unwrap().to_array(), vec![0.0, 6.0]);
    }

    #[test]
    fn failed_init_does_not_consume_session_slot() {
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut session = AgentReferenceSession::new(3).unwrap();
        let before = session.next_slot();

        let add = AgentLayerSpec::add(9);
        assert!(session
            .init_unary(&mut builder, &mut registry, &add, 0)
            .is_err());
        assert_eq!(session.next_slot(), before);
        assert!(!registry.layer_exists(LAYER_BINARY, 9));
    }

    #[test]
    fn invalid_input_slot_does_not_initialize_layer_or_consume_slot() {
        let mut registry = LayerRegistry::new();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut session = AgentReferenceSession::new(3).unwrap();
        let before = session.next_slot();
        let relu = AgentLayerSpec::relu(44);

        assert!(session
            .init_unary(&mut builder, &mut registry, &relu, 3)
            .is_err());
        assert_eq!(session.next_slot(), before);
        assert!(!registry.layer_exists(LAYER_ACTIVATION, 44));
    }

    #[test]
    fn manual_allocator_sync_keeps_escape_hatch_explicit() {
        let mut session = AgentReferenceSession::new(8).unwrap();
        session.set_next_slot(5).unwrap();
        assert_eq!(session.reserve_slot().unwrap(), 5);
        session.set_next_layer_id(900);
        assert_eq!(session.next_layer_id(), 900);
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
        assert!(workspace.get("_layers".into(), relu_id.to_string()).contains("initialized"));
        assert!(workspace.get("_slots".into(), sum_slot.to_string()).contains("reserved"));
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
    fn workspace_failed_arity_does_not_consume_slot_or_initialize_layer() {
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
    fn workspace_wire_manual_layer_reconciles_without_new_helper_state() {
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
}
