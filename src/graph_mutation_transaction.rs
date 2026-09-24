//! Isolated graph edits with an explicit, receipt-bound promotion gate.

use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

use super::{bytes_hex, CompiledMultiInputGraph, MultiInputVerificationCases};
use crate::agent::AgentLayerSpec;
use crate::graph_plan::{decode_graph_plan, DecodedGraphPlan, GraphPlanStep};
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::program_bundle::{export_multi_input_program_bundle, import_multi_input_program_bundle};
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;

const MAX_BUNDLE_BYTES: usize = 16 * 1024 * 1024;

fn digest(bytes: &[u8]) -> String {
    format!("sha256:{}", bytes_hex(&Sha256::digest(bytes)))
}

fn encode(plan: &DecodedGraphPlan) -> Result<Vec<u8>, String> {
    let steps = u32::try_from(plan.steps.len()).map_err(|_| "mutation: too many steps")?;
    let mut bytes = Vec::with_capacity(9 + plan.steps.len().saturating_mul(9));
    bytes.extend_from_slice(&steps.to_le_bytes());
    bytes.extend_from_slice(&plan.num_slots.to_le_bytes());
    for step in &plan.steps {
        bytes.push(step.arity);
        bytes.push(step.layer_type);
        bytes.extend_from_slice(&step.layer_id.to_le_bytes());
        bytes.extend_from_slice(&[step.in_slot, step.in_slot2, step.out_slot]);
    }
    bytes.push(plan.output_slot);
    Ok(bytes)
}

struct StagedCandidate {
    registry: LayerRegistry,
    graph: CompiledMultiInputGraph,
    bundle: Vec<u8>,
    receipt_digest: Option<String>,
}

#[wasm_bindgen]
pub struct GraphMutationTransaction {
    baseline_bundle: Vec<u8>,
    baseline_identity: String,
    baseline_digest: String,
    baseline_registry: LayerRegistry,
    baseline_graph: CompiledMultiInputGraph,
    baseline_plan: MultiInputGraphPlan,
    proposal: DecodedGraphPlan,
    init_specs: Vec<AgentLayerSpec>,
    weight_patches: Vec<(u8, u32, Vec<f32>)>,
    candidate: Option<StagedCandidate>,
    committed: bool,
}

impl GraphMutationTransaction {
    fn editable(&self) -> Result<(), String> {
        if self.committed { Err("mutation: transaction already committed".into()) } else { Ok(()) }
    }

    fn invalidate(&mut self) {
        self.candidate = None;
    }

    fn step(spec: &AgentLayerSpec, first: u8, second: u8, output: u8) -> GraphPlanStep {
        let layer_type = spec.layer_type();
        GraphPlanStep {
            arity: if layer_type == LAYER_BINARY { 2 } else { 1 },
            layer_type,
            layer_id: spec.layer_id(),
            in_slot: first,
            in_slot2: if layer_type == LAYER_BINARY { second } else { first },
            out_slot: output,
        }
    }

    fn check_slots(&self, slots: &[u8]) -> Result<(), String> {
        if slots.iter().any(|slot| u32::from(*slot) >= self.proposal.num_slots) {
            return Err("mutation: slot outside the fixed graph slot count".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentGraphBuilder;
    use crate::multi_input_graph::MultiInputInputBundle;
    use crate::WasmTensor;

    fn setup() -> (LayerRegistry, CompiledMultiInputGraph, MultiInputGraphPlan) {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(21);
        registry.init_agent_layer(&add).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let mut plan = builder.multi_input_plan_v1().unwrap();
        for (slot, role) in [(0, "observation"), (1, "state")] {
            plan.add_input_port(slot, role.into(), 1, 2, 1, 1,
                "feature_axis1_singleton".into(), false, 0).unwrap();
        }
        let graph = registry.compile_multi_input_graph(&plan).unwrap();
        (registry, graph, plan)
    }

    fn input(plan: &MultiInputGraphPlan) -> MultiInputInputBundle {
        let mut bundle = MultiInputInputBundle::new(plan).unwrap();
        for (slot, role, values) in [(0, "observation", [1.0, 2.0]), (1, "state", [3.0, 4.0])] {
            bundle.bind_input(slot, &WasmTensor::new(&values, &[1, 2, 1, 1]), role.into(),
                "feature_axis1_singleton".into(), "test".into(), 0, String::new()).unwrap();
        }
        bundle
    }

    fn tx(registry: &LayerRegistry, graph: &CompiledMultiInputGraph) -> GraphMutationTransaction {
        let state = digest(&export_multi_input_program_bundle(graph, registry, true).unwrap());
        GraphMutationTransaction::new(registry, graph, &graph.program_identity(), &state).unwrap()
    }

    #[test]
    fn edit_stage_verify_and_commit_requires_same_receipt_and_preimage() {
        let (mut live, baseline, baseline_plan) = setup();
        let original = export_multi_input_program_bundle(&baseline, &live, true).unwrap();
        let mut transaction = tx(&live, &baseline);
        transaction.replace_step(0, &AgentLayerSpec::add(22), 0, 1, 3).unwrap();
        transaction.set_output(3).unwrap();
        assert!(transaction.stage_candidate().is_ok());
        assert_eq!(export_multi_input_program_bundle(&baseline, &live, true).unwrap(), original);
        let mut candidate_registry = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(&mut candidate_registry, &transaction.candidate_bundle().unwrap()).unwrap();
        let candidate_plan = MultiInputGraphPlan::from_bytes(&candidate.input_plan_v1()).unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases.add_case(&input(&baseline_plan), &input(&candidate_plan)).unwrap();
        let receipt: serde_json::Value = serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        let identity = baseline.program_identity();
        let state = transaction.baseline_state_digest();
        let hash = receipt["receipt_digest"].as_str().unwrap();
        assert!(transaction.commit_by_receipt(&mut live, &baseline, &identity, &state, "sha256:fake", true).is_err());
        assert!(transaction.commit_by_receipt(&mut live, &baseline, &identity, &state, hash, false).is_err());
        assert_eq!(export_multi_input_program_bundle(&baseline, &live, true).unwrap(), original);
        let promoted = transaction.commit_by_receipt(&mut live, &baseline, &identity, &state, hash, true).unwrap();
        assert_eq!(promoted.program_identity(), candidate.program_identity());
        assert!(transaction.commit_by_receipt(&mut live, &baseline, &identity, &state, hash, true).is_err());
    }

    #[test]
    fn invalid_candidate_and_negative_receipt_leave_baseline_intact() {
        let (mut live, baseline, plan) = setup();
        let original = export_multi_input_program_bundle(&baseline, &live, true).unwrap();
        let mut transaction = tx(&live, &baseline);
        transaction.remove_step(0).unwrap();
        assert!(transaction.stage_candidate().is_err());
        assert!(transaction.candidate_bundle().is_err());
        transaction.insert_step(0, &AgentLayerSpec::sub(21), 0, 1, 2).unwrap();
        transaction.stage_candidate().unwrap();
        let mut isolated = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(&mut isolated, &transaction.candidate_bundle().unwrap()).unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases.add_case(&input(&plan), &input(&plan)).unwrap();
        let receipt: serde_json::Value = serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], false);
        assert!(transaction.commit_by_receipt(&mut live, &baseline, &baseline.program_identity(),
            &transaction.baseline_state_digest(), receipt["receipt_digest"].as_str().unwrap(), true).is_err());
        assert_eq!(export_multi_input_program_bundle(&baseline, &live, true).unwrap(), original);
    }

    #[test]
    fn stale_state_rejects_commit_after_successful_verification() {
        let (mut live, baseline, plan) = setup();
        let mut transaction = tx(&live, &baseline);
        transaction.insert_step(1, &AgentLayerSpec::relu(22), 2, 2, 3).unwrap();
        transaction.set_output(3).unwrap();
        transaction.stage_candidate().unwrap();
        let mut isolated = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(&mut isolated, &transaction.candidate_bundle().unwrap()).unwrap();
        let candidate_plan = MultiInputGraphPlan::from_bytes(&candidate.input_plan_v1()).unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases.add_case(&input(&plan), &input(&candidate_plan)).unwrap();
        let receipt: serde_json::Value = serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], true);
        live.destroy_layer(21, LAYER_BINARY);
        assert!(transaction.commit_by_receipt(&mut live, &baseline, &baseline.program_identity(),
            &transaction.baseline_state_digest(), receipt["receipt_digest"].as_str().unwrap(), true).is_err());
    }
}

#[wasm_bindgen]
impl GraphMutationTransaction {
    #[wasm_bindgen(constructor)]
    pub fn new(registry: &LayerRegistry, graph: &CompiledMultiInputGraph,
        expected_program_identity: &str, expected_state_digest: &str) -> Result<Self, String> {
        if graph.program_identity() != expected_program_identity {
            return Err("mutation: expected baseline program identity mismatch".into());
        }
        let baseline_bundle = export_multi_input_program_bundle(graph, registry, true)?;
        if baseline_bundle.len() > MAX_BUNDLE_BYTES { return Err("mutation: baseline bundle exceeds 16 MiB".into()); }
        let baseline_digest = digest(&baseline_bundle);
        if baseline_digest != expected_state_digest {
            return Err("mutation: expected baseline state digest mismatch".into());
        }
        let mut baseline_registry = LayerRegistry::new();
        let baseline_graph = import_multi_input_program_bundle(&mut baseline_registry, &baseline_bundle)?;
        let baseline_plan = graph.plan.clone();
        let proposal = decode_graph_plan(baseline_plan.graph_plan())?;
        Ok(Self { baseline_bundle, baseline_identity: expected_program_identity.into(), baseline_digest,
            baseline_registry, baseline_graph, baseline_plan, proposal, init_specs: Vec::new(),
            weight_patches: Vec::new(), candidate: None, committed: false })
    }

    #[wasm_bindgen(js_name = replaceStep)]
    pub fn replace_step(&mut self, index: u32, spec: &AgentLayerSpec,
        first: u8, second: u8, output: u8) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let index = index as usize;
        if index >= self.proposal.steps.len() { return Err("mutation.replaceStep: index out of range".into()); }
        self.proposal.steps[index] = Self::step(spec, first, second, output);
        self.init_specs.push(spec.clone());
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = insertStep)]
    pub fn insert_step(&mut self, index: u32, spec: &AgentLayerSpec,
        first: u8, second: u8, output: u8) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let index = index as usize;
        if index > self.proposal.steps.len() { return Err("mutation.insertStep: index out of range".into()); }
        if self.proposal.steps.len() >= 4096 { return Err("mutation.insertStep: 4096 step limit".into()); }
        self.proposal.steps.insert(index, Self::step(spec, first, second, output));
        self.init_specs.push(spec.clone());
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = removeStep)]
    pub fn remove_step(&mut self, index: u32) -> Result<(), String> {
        self.editable()?;
        let index = index as usize;
        if index >= self.proposal.steps.len() { return Err("mutation.removeStep: index out of range".into()); }
        self.proposal.steps.remove(index);
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = reconnectStep)]
    pub fn reconnect_step(&mut self, index: u32, first: u8, second: u8, output: u8) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let step = self.proposal.steps.get_mut(index as usize)
            .ok_or("mutation.reconnectStep: index out of range")?;
        step.in_slot = first;
        step.in_slot2 = if step.arity == 2 { second } else { first };
        step.out_slot = output;
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, output: u8) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[output])?;
        self.proposal.output_slot = output;
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(&mut self, layer_type: u8, layer_id: u32, values: &[f32]) -> Result<(), String> {
        self.editable()?;
        if values.iter().any(|value| !value.is_finite()) { return Err("mutation: weights must be finite".into()); }
        if !self.proposal.steps.iter().any(|step| (step.layer_type, step.layer_id) == (layer_type, layer_id)) {
            return Err("mutation: weight target is not referenced by proposal".into());
        }
        self.weight_patches.retain(|(kind, id, _)| (*kind, *id) != (layer_type, layer_id));
        self.weight_patches.push((layer_type, layer_id, values.to_vec()));
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = baselineBundle)]
    pub fn baseline_bundle(&self) -> Vec<u8> { self.baseline_bundle.clone() }

    #[wasm_bindgen(js_name = baselineStateDigest)]
    pub fn baseline_state_digest(&self) -> String { self.baseline_digest.clone() }

    #[wasm_bindgen(js_name = stageCandidate)]
    pub fn stage_candidate(&mut self) -> Result<String, String> {
        self.editable()?;
        self.invalidate();
        let plan = self.baseline_plan.with_graph_plan(encode(&self.proposal)?)?;
        let mut registry = LayerRegistry::new();
        import_multi_input_program_bundle(&mut registry, &self.baseline_bundle)?;
        for spec in &self.init_specs {
            if self.proposal.steps.iter().any(|step| (step.layer_type, step.layer_id) == (spec.layer_type(), spec.layer_id())) {
                registry.init_agent_layer(spec)?;
            }
        }
        for (layer_type, layer_id, values) in &self.weight_patches {
            if !self.proposal.steps.iter().any(|step| (step.layer_type, step.layer_id) == (*layer_type, *layer_id)) {
                return Err("mutation: weight target was removed".into());
            }
            registry.set_weights_flat(*layer_id, *layer_type, values)?;
        }
        let graph = registry.compile_multi_input_graph(&plan)?;
        let bundle = export_multi_input_program_bundle(&graph, &registry, true)?;
        if bundle.len() > MAX_BUNDLE_BYTES { return Err("mutation: candidate bundle exceeds 16 MiB".into()); }
        let report = format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.graph-mutation-stage.v1\",\"baseline_program_identity\":{},\"candidate_program_identity\":{},\"baseline_state_checkpoint_bytes_sha256\":\"{}\",\"candidate_state_checkpoint_bytes_sha256\":\"{}\",\"promotion_authorized\":false}}",
            self.baseline_identity, graph.program_identity(), self.baseline_digest, digest(&bundle));
        self.candidate = Some(StagedCandidate { registry, graph, bundle, receipt_digest: None });
        Ok(report)
    }

    #[wasm_bindgen(js_name = candidateBundle)]
    pub fn candidate_bundle(&self) -> Result<Vec<u8>, String> {
        Ok(self.candidate.as_ref().ok_or("mutation: no staged candidate")?.bundle.clone())
    }

    #[wasm_bindgen(js_name = verifyCases)]
    pub fn verify_cases(&mut self, cases: &MultiInputVerificationCases,
        abs_tol: f64, rel_tol: f64) -> Result<String, String> {
        self.editable()?;
        let candidate = self.candidate.as_mut().ok_or("mutation: no staged candidate")?;
        candidate.receipt_digest = None;
        let outcome = cases.verify_outcome(&self.baseline_registry, &self.baseline_graph,
            &candidate.registry, &candidate.graph, abs_tol, rel_tol)?;
        if outcome.baseline_state != self.baseline_digest || outcome.candidate_state != digest(&candidate.bundle) {
            return Err("mutation: verification checkpoint differs from staged snapshot".into());
        }
        if outcome.equivalent { candidate.receipt_digest = Some(outcome.digest); }
        Ok(outcome.json)
    }

    #[wasm_bindgen(js_name = commitByReceipt)]
    pub fn commit_by_receipt(&mut self, live_registry: &mut LayerRegistry,
        live_graph: &CompiledMultiInputGraph, expected_program_identity: &str,
        expected_state_digest: &str, receipt_digest: &str, authorize: bool) -> Result<CompiledMultiInputGraph, String> {
        self.editable()?;
        if !authorize { return Err("mutation: explicit promotion authorization required".into()); }
        if expected_program_identity != self.baseline_identity || live_graph.program_identity() != self.baseline_identity {
            return Err("mutation: baseline program identity changed".into());
        }
        if expected_state_digest != self.baseline_digest {
            return Err("mutation: expected baseline state digest mismatch".into());
        }
        let candidate = self.candidate.as_ref().ok_or("mutation: no staged candidate")?;
        if candidate.receipt_digest.as_deref() != Some(receipt_digest) {
            return Err("mutation: passing verification receipt from this transaction required".into());
        }
        let live = export_multi_input_program_bundle(live_graph, live_registry, true)?;
        if live != self.baseline_bundle { return Err("mutation: baseline state changed; no commit".into()); }
        if digest(&candidate.bundle) != digest(&export_multi_input_program_bundle(&candidate.graph, &candidate.registry, true)?) {
            return Err("mutation: staged candidate state changed; no commit".into());
        }
        let graph = import_multi_input_program_bundle(live_registry, &candidate.bundle)?;
        self.committed = true;
        Ok(graph)
    }
}
