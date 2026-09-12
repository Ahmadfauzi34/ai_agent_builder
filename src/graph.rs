use std::fmt::Write as _;
use wasm_bindgen::prelude::*;
use crate::coprocessor::verify_vectors_report;
use crate::protocol::{
    PayloadCursor, LAYER_BINARY, LAYER_CONV, LAYER_GHOST, LAYER_POOL, LAYER_SEBLOCK,
};
use crate::registry::LayerRegistry;
use crate::WasmTensor;

#[path = "registry/runtime_contract.rs"]
mod runtime_contract;

// Satu sumber kebenaran arity untuk graph + registry.
pub(crate) const ARITY_UNARY: u8 = 1;
pub(crate) const ARITY_BINARY: u8 = 2;

const CG_MAX_SLOTS: u32 = 64;
const PLAN_HEADER_BYTES: usize = 8; // num_steps:u32 + num_slots:u32
const PLAN_STEP_BYTES: usize = 9;
const PLAN_OUTPUT_BYTES: usize = 1;

fn expected_plan_len(num_steps: u32) -> Result<usize, String> {
    (num_steps as usize)
        .checked_mul(PLAN_STEP_BYTES)
        .and_then(|steps| PLAN_HEADER_BYTES.checked_add(steps))
        .and_then(|bytes| bytes.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "plan length overflow".to_string())
}

fn bytes_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[derive(Clone, Copy)]
struct CompiledStep {
    arity: u8,
    layer_type: u8,
    layer_id: u32,
    in_slot: u8,
    in_slot2: u8,
    out_slot: u8,
}

#[wasm_bindgen]
pub struct CompiledGraph {
    steps: Vec<CompiledStep>,
    num_slots: u32,
    out_slot: u8,
    canonical_plan: Vec<u8>,
    init_fingerprints: Vec<String>,
}

impl CompiledGraph {
    fn read_step(c: &mut PayloadCursor) -> Result<CompiledStep, String> {
        Ok(CompiledStep {
            arity: c.read_u8()?,
            layer_type: c.read_u8()?,
            layer_id: c.read_u32()?,
            in_slot: c.read_u8()?,
            in_slot2: c.read_u8()?,
            out_slot: c.read_u8()?,
        })
    }

    fn structural_identity_json(&self) -> String {
        let layer_identities = self
            .init_fingerprints
            .iter()
            .map(|fingerprint| format!("\"{fingerprint}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"burn-research.program-identity.v1\",\"plan_hex\":\"{}\",\"layer_init_fingerprints\":[{}]}}",
            bytes_hex(&self.canonical_plan),
            layer_identities
        )
    }

    fn validate_registry_binding_internal(
        &self,
        registry: &LayerRegistry,
        context: &str,
    ) -> Result<(), String> {
        if self.steps.len() != self.init_fingerprints.len() {
            return Err(format!(
                "{context}: internal structural binding cardinality mismatch"
            ));
        }
        for (index, (step, expected)) in self
            .steps
            .iter()
            .zip(self.init_fingerprints.iter())
            .enumerate()
        {
            let actual = registry
                .layer_init_fingerprint(step.layer_type, step.layer_id)
                .map_err(|error| format!("{context}: step {index}: {error}"))?;
            if &actual != expected {
                return Err(format!(
                    "{context}: step {index} structural identity mismatch for layer type 0x{:02X} id {}",
                    step.layer_type, step.layer_id
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn build(reg: &LayerRegistry, plan: &[u8]) -> Result<CompiledGraph, String> {
        let mut c = PayloadCursor::new(plan);
        let num_steps = c.read_u32()?;
        let num_slots = c.read_u32()?;
        if num_steps == 0 {
            return Err("compile_graph: plan has no steps".into());
        }
        if !(1..=CG_MAX_SLOTS).contains(&num_slots) {
            return Err(format!("compile_graph: num_slots must be 1..={}, got {}", CG_MAX_SLOTS, num_slots));
        }

        let expected_len = expected_plan_len(num_steps)
            .map_err(|e| format!("compile_graph: {}", e))?;
        if plan.len() != expected_len {
            return Err(format!(
                "compile_graph: malformed plan length: expected {} bytes for {} steps, got {}",
                expected_len,
                num_steps,
                plan.len()
            ));
        }

        let mut steps: Vec<CompiledStep> = Vec::with_capacity(num_steps as usize);
        let mut init_fingerprints: Vec<String> = Vec::with_capacity(num_steps as usize);
        let mut filled: u64 = 1;
        for _ in 0..num_steps {
            let s = Self::read_step(&mut c)?;
            let in_slot = s.in_slot as u32;
            let in_slot2 = s.in_slot2 as u32;
            let out_slot = s.out_slot as u32;
            if in_slot >= num_slots || in_slot2 >= num_slots || out_slot >= num_slots {
                return Err(format!("compile_graph: slot index out of range (num_slots={})", num_slots));
            }
            if s.arity == ARITY_BINARY {
                if s.layer_type != LAYER_BINARY {
                    return Err(format!("compile_graph: arity 2 requires LAYER_BINARY, got 0x{:02X}", s.layer_type));
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
                if (filled >> in_slot2) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot2));
                }
            } else if s.arity == ARITY_UNARY {
                if s.layer_type == LAYER_BINARY {
                    return Err("compile_graph: arity 1 cannot use LAYER_BINARY (needs 2 inputs)".into());
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
            } else {
                return Err(format!("compile_graph: invalid arity {} (expected 1 or 2)", s.arity));
            }
            if !reg.layer_exists(s.layer_type, s.layer_id) {
                return Err(format!("compile_graph: layer type 0x{:02X} id {} not found", s.layer_type, s.layer_id));
            }
            let fingerprint = reg
                .layer_init_fingerprint(s.layer_type, s.layer_id)
                .map_err(|error| format!("compile_graph: {error}"))?;
            filled |= 1u64 << out_slot;
            steps.push(s);
            init_fingerprints.push(fingerprint);
        }
        let out_slot = c.read_u8()? as u32;
        if out_slot >= num_slots {
            return Err(format!("compile_graph: output slot {} out of range", out_slot));
        }
        if (filled >> out_slot) & 1 == 0 {
            return Err(format!("compile_graph: output slot {} is never written", out_slot));
        }
        Ok(CompiledGraph {
            steps,
            num_slots,
            out_slot: out_slot as u8,
            canonical_plan: plan.to_vec(),
            init_fingerprints,
        })
    }
}

#[wasm_bindgen(js_name = programCapabilities)]
pub fn program_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"identity_schema\":\"burn-research.program-identity.v1\",",
        "\"entry\":\"CompiledGraph\",",
        "\"plan\":\"programPlan\",",
        "\"identity\":\"programIdentity\",",
        "\"binding_validation\":\"validateRegistryBinding\",",
        "\"execution_binding\":\"required\",",
        "\"identity_scope\":\"graph_plan_plus_layer_init_identity\",",
        "\"mutable_state_in_identity\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen]
impl CompiledGraph {
    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        registry: &LayerRegistry,
        input: &WasmTensor,
    ) -> Result<WasmTensor, String> {
        // A compiled graph is structurally bound to the init identities validated at compile time.
        // Mutable weights/state may change under the same init identity, but structural re-init
        // requires recompiling the canonical plan before execution.
        self.validate_registry_binding_internal(registry, "run")?;

        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        slots[0] = Some(input.clone());
        for s in &self.steps {
            let out = if s.arity == ARITY_BINARY {
                let a = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot))?;
                let b = slots[s.in_slot2 as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot2))?;
                registry.forward_binary_layer(s.layer_id, a, b)?
            } else {
                let inp = slots[s.in_slot as usize]
                    .as_ref()
                    .ok_or_else(|| format!("run: empty input slot {}", s.in_slot))?;
                if matches!(
                    s.layer_type,
                    LAYER_CONV | LAYER_POOL | LAYER_GHOST | LAYER_SEBLOCK
                ) {
                    runtime_contract::validate_registry_unary_contract(
                        registry,
                        s.layer_type,
                        s.layer_id,
                        inp.inner.dims(),
                    )?;
                }
                registry.forward_layer(s.layer_id, s.layer_type, inp)?
            };
            slots[s.out_slot as usize] = Some(out);
        }
        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("run: empty output slot {}", self.out_slot))
    }

    #[wasm_bindgen(js_name = verifyFlat)]
    pub fn verify_flat(
        &self,
        registry: &LayerRegistry,
        input: &WasmTensor,
        candidate: &[f32],
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
        let reference = self.run(registry, input)?.to_array();
        verify_vectors_report(&reference, candidate, abs_tol, rel_tol)
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.canonical_plan.clone()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.structural_identity_json()
    }

    #[wasm_bindgen(js_name = validateRegistryBinding)]
    pub fn validate_registry_binding(&self, registry: &LayerRegistry) -> Result<(), String> {
        self.validate_registry_binding_internal(registry, "validateRegistryBinding")
    }

    #[wasm_bindgen(js_name = numSteps)]
    pub fn step_count(&self) -> u32 { self.steps.len() as u32 }
    #[wasm_bindgen(js_name = numSlots)]
    pub fn slot_count(&self) -> u32 { self.num_slots }
    #[wasm_bindgen(js_name = outSlot)]
    pub fn output_slot(&self) -> u8 { self.out_slot }
}

#[cfg(test)]
mod tests {
    use super::{program_capabilities, CompiledGraph};
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::protocol::LAYER_LINEAR;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn compiled_linear(registry: &mut LayerRegistry, out_dim: u32) -> CompiledGraph {
        let spec = AgentLayerSpec::linear(7, 2, out_dim, true).unwrap();
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        builder.compile(registry).unwrap()
    }

    #[test]
    fn compiled_program_plan_replays_exact_structure() {
        let mut registry = LayerRegistry::new();
        let graph = compiled_linear(&mut registry, 2);
        let plan = graph.program_plan();
        let replay = registry.compile_graph(&plan).unwrap();
        assert_eq!(replay.program_plan(), plan);
        assert_eq!(replay.program_identity(), graph.program_identity());
        assert_eq!(replay.step_count(), graph.step_count());
        assert_eq!(replay.slot_count(), graph.slot_count());
        assert_eq!(replay.output_slot(), graph.output_slot());
    }

    #[test]
    fn compatible_mutable_weight_changes_remain_executable_and_visible() {
        let mut registry = LayerRegistry::new();
        let graph = compiled_linear(&mut registry, 2);
        let identity = graph.program_identity();
        let input = WasmTensor::new(&[1.0, 1.0], &[1, 2, 1, 1]);
        let before = graph.run(&registry, &input).unwrap().to_array();

        let mut weights = registry.get_weights_flat(7, LAYER_LINEAR).unwrap();
        for value in &mut weights {
            *value += 1.0;
        }
        registry.set_weights_flat(7, LAYER_LINEAR, &weights).unwrap();

        let after = graph.run(&registry, &input).unwrap().to_array();
        assert_ne!(before, after);
        assert_eq!(graph.program_identity(), identity);
        assert!(graph.validate_registry_binding(&registry).is_ok());
    }

    #[test]
    fn structural_replacement_requires_recompile_before_execution() {
        let mut registry = LayerRegistry::new();
        let graph = compiled_linear(&mut registry, 2);
        let old_identity = graph.program_identity();
        let plan = graph.program_plan();
        let input = WasmTensor::new(&[1.0, -2.0], &[1, 2, 1, 1]);
        assert!(graph.run(&registry, &input).is_ok());

        let replacement = AgentLayerSpec::linear(7, 2, 3, true).unwrap();
        registry.init_agent_layer(&replacement).unwrap();

        assert!(graph.validate_registry_binding(&registry).is_err());
        assert!(graph.run(&registry, &input).is_err());
        assert!(graph.verify_flat(&registry, &input, &[0.0, 0.0], 0.0, 0.0).is_err());

        let rebound = registry.compile_graph(&plan).unwrap();
        assert_ne!(rebound.program_identity(), old_identity);
        assert!(rebound.validate_registry_binding(&registry).is_ok());
        assert!(rebound.run(&registry, &input).is_ok());
    }

    #[test]
    fn missing_referenced_layer_is_rejected_before_execution() {
        let mut registry = LayerRegistry::new();
        let graph = compiled_linear(&mut registry, 2);
        let input = WasmTensor::new(&[1.0, 1.0], &[1, 2, 1, 1]);
        assert!(registry.destroy_layer(7, LAYER_LINEAR));
        assert!(graph.validate_registry_binding(&registry).is_err());
        assert!(graph.run(&registry, &input).is_err());
    }

    #[test]
    fn output_selection_is_part_of_structural_identity() {
        let mut registry = LayerRegistry::new();
        let relu = AgentLayerSpec::relu(11);
        registry.init_agent_layer(&relu).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_unary(&relu, 0, 1).unwrap();
        builder.add_unary(&relu, 1, 2).unwrap();
        let first = builder.compile_with_output(&registry, 1).unwrap();
        let second = builder.compile_with_output(&registry, 2).unwrap();
        assert_ne!(first.program_plan(), second.program_plan());
        assert_ne!(first.program_identity(), second.program_identity());
    }

    #[test]
    fn program_discovery_states_structural_scope_and_execution_binding() {
        let capabilities = program_capabilities();
        assert!(capabilities.contains("burn-research.program-identity.v1"));
        assert!(capabilities.contains("programPlan"));
        assert!(capabilities.contains("programIdentity"));
        assert!(capabilities.contains("validateRegistryBinding"));
        assert!(capabilities.contains("\"execution_binding\":\"required\""));
        assert!(capabilities.contains("\"mutable_state_in_identity\":false"));
    }
}
