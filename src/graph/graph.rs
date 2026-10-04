//! # Kontrak: `graph`
//!
//! ## Tanggung jawab
//! `CompiledGraph`: rencana eksekusi multi-input yang dikompilasi dari
//! `GraphPlanStep`, dengan slot bernomor, rencana kanonis, dan sidik jari
//! init untuk verifikasi.
//!
//! ## Invariant
//! - `CG_MAX_SLOTS = 64`: `num_slots` harus dalam `1..=64`; di luar itu
//!   ditolak dengan pesan yang menyebut batasnya (batas desain).
//! - `ARITY_UNARY` / `ARITY_BINARY` adalah satu-satunya sumber kebenaran
//!   arity untuk graph + registry.
//! - Submodul (`plan_explain`, `execution_trace`, `candidate_verification`,
//!   `mutation_transaction`) di-declare sebagai `pub mod` di
//!   `src/graph/mod.rs`; file ini merujuknya via `super::` agar setiap file
//!   dikompilasi tepat sekali.
//!
//! ## Bukan tanggung jawab modul ini
//! - Engine layer → `layers/*`; kepemilikan instance → `registry`.

use crate::coprocessor::verify_vectors_report;
use crate::graph_plan::{decode_graph_plan, decode_graph_plan_header, GraphPlanStep};
use crate::multi_input_graph::{
    multi_input_graph_capabilities as multi_input_graph_capabilities_json, MultiInputGraphPlan,
    MultiInputInputBundle,
};
use crate::protocol::{LAYER_BINARY, LAYER_CONV, LAYER_GHOST, LAYER_POOL, LAYER_SEBLOCK};
use crate::registry::LayerRegistry;
use crate::WasmTensor;
use std::fmt::Write as _;
use wasm_bindgen::prelude::*;

// Sibling modules are declared as `pub mod` in `super` (src/graph/mod.rs);
// refer to them via `super::` instead of private `#[path]` copies so each
// file compiles exactly once (a `#[path]` copy here would double-compile
// the #[wasm_bindgen] exports).
pub use super::graph_candidate_verification::MultiInputVerificationCases;
pub use super::graph_execution_trace::TracedMultiInputRun;
pub use super::graph_mutation_transaction::{
    checkpoint_branch_capabilities, CheckpointBranchSet, GraphMutationTransaction,
};
use super::graph_plan_explain;
pub use crate::facade::graph::{multi_input_graph_capabilities, program_capabilities};

// Satu sumber kebenaran arity untuk graph + registry.
pub(crate) const ARITY_UNARY: u8 = 1;
pub(crate) const ARITY_BINARY: u8 = 2;

const CG_MAX_SLOTS: u32 = 64;

pub(crate) fn bytes_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[wasm_bindgen]
pub struct CompiledGraph {
    pub(crate) steps: Vec<GraphPlanStep>,
    pub(crate) num_slots: u32,
    pub(crate) out_slot: u8,
    pub(crate) canonical_plan: Vec<u8>,
    pub(crate) init_fingerprints: Vec<String>,
}

#[wasm_bindgen]
pub struct CompiledMultiInputGraph {
    pub(crate) graph: CompiledGraph,
    pub(crate) plan: MultiInputGraphPlan,
    pub(crate) input_plan_bytes: Vec<u8>,
}

impl CompiledGraph {
    pub(crate) fn structural_identity_json(&self) -> String {
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

    pub(crate) fn validate_registry_binding_internal(
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
        Self::build_with_external_slots(reg, plan, &[0])
    }

    pub(crate) fn build_with_external_slots(
        reg: &LayerRegistry,
        plan: &[u8],
        external_slots: &[u8],
    ) -> Result<CompiledGraph, String> {
        // Preserve the historical validation order: header readability and the
        // execution-profile checks happen before exact-envelope decoding.
        let header = decode_graph_plan_header(plan)?;
        let num_steps = header.num_steps;
        let num_slots = header.num_slots;
        if num_steps == 0 {
            return Err("compile_graph: plan has no steps".into());
        }
        if !(1..=CG_MAX_SLOTS).contains(&num_slots) {
            return Err(format!(
                "compile_graph: num_slots must be 1..={}, got {}",
                CG_MAX_SLOTS, num_slots
            ));
        }

        // The shared decoder owns only frozen byte structure. Registry, arity,
        // slot-lifecycle and execution policy remain below in CompiledGraph.
        let decoded = decode_graph_plan(plan).map_err(|error| format!("compile_graph: {error}"))?;
        debug_assert_eq!(decoded.num_steps, num_steps);
        debug_assert_eq!(decoded.num_slots, num_slots);

        let out_slot = u32::from(decoded.output_slot);
        let steps = decoded.steps;
        let mut init_fingerprints: Vec<String> = Vec::with_capacity(num_steps as usize);
        let mut filled: u64 = 0;
        for slot in external_slots {
            let slot = u32::from(*slot);
            if slot >= num_slots {
                return Err(format!(
                    "compile_graph: external input slot {slot} out of range (num_slots={num_slots})"
                ));
            }
            filled |= 1u64 << slot;
        }
        for s in &steps {
            let in_slot = s.in_slot as u32;
            let in_slot2 = s.in_slot2 as u32;
            let step_out_slot = s.out_slot as u32;
            if in_slot >= num_slots || in_slot2 >= num_slots || step_out_slot >= num_slots {
                return Err(format!(
                    "compile_graph: slot index out of range (num_slots={})",
                    num_slots
                ));
            }
            if s.arity == ARITY_BINARY {
                if s.layer_type != LAYER_BINARY {
                    return Err(format!(
                        "compile_graph: arity 2 requires LAYER_BINARY, got 0x{:02X}",
                        s.layer_type
                    ));
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
                if (filled >> in_slot2) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot2));
                }
            } else if s.arity == ARITY_UNARY {
                if s.layer_type == LAYER_BINARY {
                    return Err(
                        "compile_graph: arity 1 cannot use LAYER_BINARY (needs 2 inputs)".into(),
                    );
                }
                if (filled >> in_slot) & 1 == 0 {
                    return Err(format!("compile_graph: input slot {} is empty", in_slot));
                }
            } else {
                return Err(format!(
                    "compile_graph: invalid arity {} (expected 1 or 2)",
                    s.arity
                ));
            }
            if !reg.layer_exists(s.layer_type, s.layer_id) {
                return Err(format!(
                    "compile_graph: layer type 0x{:02X} id {} not found",
                    s.layer_type, s.layer_id
                ));
            }
            let fingerprint = reg
                .layer_init_fingerprint(s.layer_type, s.layer_id)
                .map_err(|error| format!("compile_graph: {error}"))?;
            filled |= 1u64 << step_out_slot;
            init_fingerprints.push(fingerprint);
        }
        if out_slot >= num_slots {
            return Err(format!(
                "compile_graph: output slot {} out of range",
                out_slot
            ));
        }
        if (filled >> out_slot) & 1 == 0 {
            return Err(format!(
                "compile_graph: output slot {} is never written",
                out_slot
            ));
        }
        Ok(CompiledGraph {
            steps,
            num_slots,
            out_slot: out_slot as u8,
            canonical_plan: plan.to_vec(),
            init_fingerprints,
        })
    }

    pub(crate) fn run_with_external_inputs_internal(
        &self,
        registry: &LayerRegistry,
        inputs: &[(u8, WasmTensor)],
    ) -> Result<WasmTensor, String> {
        self.run_with_external_inputs_observed(registry, inputs, |_, _, _| {})
    }

    pub(crate) fn run_with_external_inputs_observed<F>(
        &self,
        registry: &LayerRegistry,
        inputs: &[(u8, WasmTensor)],
        mut observe: F,
    ) -> Result<WasmTensor, String>
    where
        F: FnMut(usize, &GraphPlanStep, &Result<WasmTensor, String>),
    {
        self.validate_registry_binding_internal(registry, "run")?;
        let mut slots: Vec<Option<WasmTensor>> = vec![None; self.num_slots as usize];
        for (slot, tensor) in inputs {
            let slot_index = usize::from(*slot);
            if slot_index >= slots.len() {
                return Err(format!("run: external input slot {slot} out of range"));
            }
            if slots[slot_index].is_some() {
                return Err(format!(
                    "run: external input slot {slot} is bound more than once"
                ));
            }
            slots[slot_index] = Some(tensor.clone());
        }
        for (index, s) in self.steps.iter().enumerate() {
            let out = (|| -> Result<WasmTensor, String> {
                Ok(if s.arity == ARITY_BINARY {
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
                        crate::registry::runtime_contract::validate_registry_unary_contract(
                            registry,
                            s.layer_type,
                            s.layer_id,
                            inp.inner.dims(),
                        )?;
                    }
                    registry.forward_layer(s.layer_id, s.layer_type, inp)?
                })
            })();
            observe(index, s, &out);
            let out = out?;
            slots[s.out_slot as usize] = Some(out);
        }
        slots[self.out_slot as usize]
            .take()
            .ok_or_else(|| format!("run: empty output slot {}", self.out_slot))
    }
}

impl CompiledMultiInputGraph {
    pub(crate) fn build(
        registry: &LayerRegistry,
        plan: &MultiInputGraphPlan,
    ) -> Result<Self, String> {
        let input_plan_bytes = plan.validate_for_compile()?;
        let required_slots = plan
            .ports()
            .iter()
            .map(|port| port.slot)
            .collect::<Vec<_>>();
        let graph =
            CompiledGraph::build_with_external_slots(registry, plan.graph_plan(), &required_slots)?;
        Ok(Self {
            graph,
            plan: plan.clone(),
            input_plan_bytes,
        })
    }

    pub(crate) fn preflight_state(
        &self,
        registry: &LayerRegistry,
        bundle: &MultiInputInputBundle,
    ) -> (bool, String) {
        let inputs = bundle.input_preflight(&self.plan);
        let registry_ok = self
            .graph
            .validate_registry_binding_internal(registry, "preflight")
            .is_ok();
        let slots = self
            .plan
            .ports()
            .iter()
            .map(|port| port.slot.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let bundle_plan_matches = bundle.matches_plan_internal(&self.plan);
        let ready = inputs.ready && registry_ok && bundle_plan_matches;
        let report = format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.multi-input-preflight-report.v1\",\"required_input_slots\":[{}],\"bundle_plan_matches\":{},\"registry_binding_current\":{},\"ready\":{},\"execution_authorized\":false,\"inputs\":{}}}",
            slots,
            if bundle_plan_matches { "true" } else { "false" },
            if registry_ok { "true" } else { "false" },
            if ready { "true" } else { "false" },
            inputs.json,
        );
        (ready, report)
    }

    pub(crate) fn program_identity_json(&self) -> String {
        let layer_identities = self
            .graph
            .init_fingerprints
            .iter()
            .map(|fingerprint| format!("\"{fingerprint}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"burn-research.multi-input-program-identity.v1\",\"input_plan_hex\":\"{}\",\"layer_init_fingerprints\":[{}]}}",
            bytes_hex(&self.input_plan_bytes),
            layer_identities,
        )
    }
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
        registry
            .set_weights_flat(7, LAYER_LINEAR, &weights)
            .unwrap();

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
        assert!(graph
            .verify_flat(&registry, &input, &[0.0, 0.0], 0.0, 0.0)
            .is_err());

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
