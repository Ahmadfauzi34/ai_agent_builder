//! Fasad WASM tunggal — domain `graph` (Opsi C, Fase 1 + Fase 2).
//!
//! Pindahan murni dari modul sumber: free function `#[wasm_bindgen]`
//! dipindah byte-identik (nama export JS tidak berubah).
//! Kompatibilitas path lama dijaga via `pub use` re-export di modul sumber
//! dan `pub use facade::graph::...` di `src/lib.rs`.
//!
//! Fase 2: blok `#[wasm_bindgen] impl` untuk struct domain dipindah ke sini
//! byte-identik; struct tetap di domain dengan `#[wasm_bindgen]` sebagai
//! marker ABI (atribut ini men-generate trait impl WasmDescribe/ABI yang
//! dibutuhkan semua signature wasm — menghapusnya = E0277). Item privat
//! yang dipakai blok impl dibuka secukupnya sebagai `pub(crate)` di modul
//! domain.

use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::agent::AgentLayerSpec;
use crate::contracts::validate_external_input_contract_declaration;
use crate::coprocessor::verify_vectors_report;
use crate::graph::graph_candidate_verification::{
    case_digest, MultiInputVerificationCases, VerificationCase, MAX_CASES, MAX_INPUT_BYTES,
};
use crate::graph::graph_execution_trace::{self, TracedMultiInputRun};
use crate::graph::graph_mutation_transaction::MAX_BRANCH_CANDIDATE_BYTES;
use crate::graph::graph_mutation_transaction::MAX_BRANCH_SNAPSHOT_BYTES;
use crate::graph::graph_mutation_transaction::MAX_CHECKPOINT_BRANCHES;
use crate::graph::graph_mutation_transaction::MAX_STATE_DIFF_RANGES;
use crate::graph::graph_mutation_transaction::{
    digest, encode, state_diff_json, valid_branch_id, BranchVerification, CheckpointBranch,
    CheckpointBranchSet, GraphMutationTransaction, StagedCandidate, MAX_BUNDLE_BYTES,
};
use crate::graph::graph_parameters_wasm::apply_fresh_binding;
use crate::graph::graph_parameters_wasm::read_fresh_binding;
use crate::graph::graph_plan_explain;
use crate::graph::multi_input_graph::multi_input_graph_capabilities as multi_input_graph_capabilities_json;
use crate::graph::multi_input_graph::{
    fnv1a64, port_contract_json, tensor_shape, tensor_value_fingerprint, validate_port_contract,
};
use crate::graph::multi_input_graph::{
    BoundInput, InputPortBinding, MultiInputGraphPlan, MultiInputInputBundle,
    MultiInputPortContract, PlanCursor, MAX_INPUT_PORTS, PLAN_MAGIC, PLAN_SCHEMA_ID,
    PLAN_SCHEMA_VERSION,
};
use crate::graph::{CompiledGraph, CompiledMultiInputGraph, ARITY_BINARY};
use crate::graph_parameters::GraphParameterBinding;
use crate::graph_plan::decode_graph_plan;
use crate::ingress::input_port::{
    role_valid, MAX_FINGERPRINT_BYTES, MAX_ROLE_BYTES, MAX_SOURCE_BYTES,
};
use crate::program_bundle::{export_multi_input_program_bundle, import_multi_input_program_bundle};
use crate::protocol::{LAYER_CONV, LAYER_GHOST, LAYER_POOL, LAYER_SEBLOCK};
use crate::registry::LayerRegistry;
use crate::WasmTensor;

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

#[wasm_bindgen(js_name = multiInputGraphCapabilities)]
pub fn multi_input_graph_capabilities() -> String {
    multi_input_graph_capabilities_json()
}

#[wasm_bindgen(js_name = checkpointBranchCapabilities)]
pub fn checkpoint_branch_capabilities() -> String {
    format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.checkpoint-branch-capabilities.v1\",\"implementation\":\"wasm_burn_reference_machine\",\"contract\":\"transactional-graph-checkpoint-branch.v1\",\"checkpoint_format\":\"burn-research.multi-input-program-bundle.v1\",\"fork_source\":\"one validated exact stateful checkpoint snapshot\",\"branch_limit\":{MAX_CHECKPOINT_BRANCHES},\"baseline_snapshot_budget_bytes\":{MAX_BRANCH_SNAPSHOT_BYTES},\"candidate_bundle_budget_bytes\":{MAX_BRANCH_CANDIDATE_BYTES},\"state_diff_ranges_limit\":{MAX_STATE_DIFF_RANGES},\"promotion\":\"one branch per set; explicit caller authorization and that branch's passing receipt required\",\"state_diff_semantics\":\"exact checkpoint bundle byte difference with digests; does not assert semantic state equivalence\",\"output_comparison\":\"MultiInputVerificationCases Burn receipt included in branch receipt\",\"durable_host_ledger\":false}}")
}

#[wasm_bindgen(js_name = graphParameterCapabilities)]
pub fn graph_parameter_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"schema\":\"burn-research.graph-parameter-binding.v1\",",
        "\"layout_schema\":\"burn-research.graph-parameter-layout.v1\",",
        "\"ordering\":\"unique_first_use_graph_plan\",",
        "\"supported_flat_owner_types\":[\"linear\",\"conv\",\"embedding\",\"norm\"],",
        "\"parameterized_owner_without_flat_bridge\":\"fail_closed\",",
        "\"stateless_owner_coordinates\":0,",
        "\"apply_atomicity\":\"full_prevalidation_before_first_mutation\",",
        "\"mutable_parameter_values_in_identity\":false,",
        "\"program_bundle_is_candidate_format\":false,",
        "\"optimizer_state_in_binding\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = graphParameterLayout)]
pub fn graph_parameter_layout(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    Ok(GraphParameterBinding::build(graph, registry)?.layout_json())
}

#[wasm_bindgen(js_name = graphParameterIdentity)]
pub fn graph_parameter_identity(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    Ok(GraphParameterBinding::build(graph, registry)?.identity_json())
}

#[wasm_bindgen(js_name = getGraphParametersFlat)]
pub fn get_graph_parameters_flat(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<Vec<f32>, String> {
    let binding = GraphParameterBinding::build(graph, registry)?;
    read_fresh_binding(binding, registry)
}

#[wasm_bindgen(js_name = setGraphParametersFlat)]
pub fn set_graph_parameters_flat(
    graph: &CompiledGraph,
    registry: &mut LayerRegistry,
    candidate: &[f32],
) -> Result<(), String> {
    let binding = GraphParameterBinding::build(graph, registry)?;
    apply_fresh_binding(binding, registry, candidate)
}

// -------------------------------------------------------------
// FASE 2 — #[wasm_bindgen] impl blocks (pindahan murni dari domain).
// Struct domain tetap di modul asalnya dengan #[wasm_bindgen] sebagai marker
// ABI; hanya blok impl binding WASM yang pindah ke sini, byte-identik (nama
// export JS tidak berubah).
// -------------------------------------------------------------

#[wasm_bindgen]
impl CompiledMultiInputGraph {
    #[wasm_bindgen(js_name = explainPlan)]
    pub fn explain_plan(&self, registry: &LayerRegistry) -> String {
        graph_plan_explain::report(self, registry)
    }

    #[wasm_bindgen(js_name = runWithTrace)]
    pub fn run_with_trace(
        &self,
        registry: &LayerRegistry,
        bundle: &MultiInputInputBundle,
        start_step: u32,
        max_steps: u32,
        max_tensor_bytes: u32,
    ) -> Result<TracedMultiInputRun, String> {
        let (ready, _) = self.preflight_state(registry, bundle);
        if !ready {
            return Err("CompiledMultiInputGraph.runWithTrace: input or registry preflight failed; execution was not started".into());
        }
        graph_execution_trace::run(
            self,
            registry,
            bundle,
            start_step,
            max_steps,
            max_tensor_bytes,
        )
    }

    #[wasm_bindgen(js_name = preflight)]
    pub fn preflight(&self, registry: &LayerRegistry, bundle: &MultiInputInputBundle) -> String {
        self.preflight_state(registry, bundle).1
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(
        &self,
        registry: &LayerRegistry,
        bundle: &MultiInputInputBundle,
    ) -> Result<WasmTensor, String> {
        let (ready, _) = self.preflight_state(registry, bundle);
        if !ready {
            return Err(
                "CompiledMultiInputGraph.run: input or registry preflight failed; execution was not started".into(),
            );
        }
        self.graph
            .run_with_external_inputs_internal(registry, &bundle.bound_inputs())
    }

    #[wasm_bindgen(js_name = verifyFlat)]
    pub fn verify_flat(
        &self,
        registry: &LayerRegistry,
        bundle: &MultiInputInputBundle,
        candidate: &[f32],
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
        let (ready, preflight) = self.preflight_state(registry, bundle);
        if !ready {
            return Err(
                "CompiledMultiInputGraph.verifyFlat: input or registry preflight failed; execution was not started".into(),
            );
        }
        let reference = self
            .graph
            .run_with_external_inputs_internal(registry, &bundle.bound_inputs())?
            .to_array();
        let verification = verify_vectors_report(&reference, candidate, abs_tol, rel_tol)?;
        Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.multi-input-verification.v1\",\"preflight\":{},\"verification\":{}}}",
            preflight,
            verification,
        ))
    }

    #[wasm_bindgen(js_name = inputPlanV1)]
    pub fn input_plan_v1(&self) -> Vec<u8> {
        self.input_plan_bytes.clone()
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.graph.canonical_plan.clone()
    }

    #[wasm_bindgen(js_name = programIdentity)]
    pub fn program_identity(&self) -> String {
        self.program_identity_json()
    }

    #[wasm_bindgen(js_name = requiredInputSlots)]
    pub fn required_input_slots(&self) -> Vec<u8> {
        self.plan.ports().iter().map(|port| port.slot).collect()
    }

    #[wasm_bindgen(js_name = validateRegistryBinding)]
    pub fn validate_registry_binding(&self, registry: &LayerRegistry) -> Result<(), String> {
        self.graph
            .validate_registry_binding_internal(registry, "validateRegistryBinding")
    }
}

#[wasm_bindgen]
impl CompiledGraph {
    #[wasm_bindgen(js_name = run)]
    pub fn run(&self, registry: &LayerRegistry, input: &WasmTensor) -> Result<WasmTensor, String> {
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
                    crate::registry::runtime_contract::validate_registry_unary_contract(
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
    pub fn step_count(&self) -> u32 {
        self.steps.len() as u32
    }
    #[wasm_bindgen(js_name = numSlots)]
    pub fn slot_count(&self) -> u32 {
        self.num_slots
    }
    #[wasm_bindgen(js_name = outSlot)]
    pub fn output_slot(&self) -> u8 {
        self.out_slot
    }
}

#[wasm_bindgen]
impl MultiInputVerificationCases {
    #[wasm_bindgen(constructor)]
    pub fn new(
        baseline: &CompiledMultiInputGraph,
        candidate: &CompiledMultiInputGraph,
    ) -> Result<Self, String> {
        if baseline.plan.ports() != candidate.plan.ports() {
            return Err("verification: baseline and candidate input port contracts differ".into());
        }
        Ok(Self {
            baseline_plan: baseline.plan.clone(),
            candidate_plan: candidate.plan.clone(),
            baseline_identity: baseline.program_identity(),
            candidate_identity: candidate.program_identity(),
            cases: Vec::new(),
            input_bytes: 0,
        })
    }

    #[wasm_bindgen(js_name = addCase)]
    pub fn add_case(
        &mut self,
        baseline: &MultiInputInputBundle,
        candidate: &MultiInputInputBundle,
    ) -> Result<u32, String> {
        if self.cases.len() == MAX_CASES {
            return Err("verification.addCase: at most 128 test vectors".into());
        }
        if !baseline.input_preflight(&self.baseline_plan).ready
            || !candidate.input_preflight(&self.candidate_plan).ready
        {
            return Err("verification.addCase: input preflight failed".into());
        }
        let (input_digest, input_bytes) = case_digest(baseline, candidate)?;
        let total = self
            .input_bytes
            .checked_add(input_bytes)
            .ok_or("verification.addCase: input budget overflow")?;
        if total > MAX_INPUT_BYTES {
            return Err("verification.addCase: total test vector bytes exceed 64 MiB".into());
        }
        self.cases.push(VerificationCase {
            baseline: baseline.clone(),
            candidate: candidate.clone(),
            input_digest,
            input_bytes,
        });
        self.input_bytes = total;
        Ok(self.cases.len() as u32)
    }

    #[wasm_bindgen(js_name = caseCount)]
    pub fn case_count(&self) -> u32 {
        self.cases.len() as u32
    }

    pub fn verify(
        &self,
        baseline_registry: &LayerRegistry,
        baseline: &CompiledMultiInputGraph,
        candidate_registry: &LayerRegistry,
        candidate: &CompiledMultiInputGraph,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
        Ok(self
            .verify_outcome(
                baseline_registry,
                baseline,
                candidate_registry,
                candidate,
                abs_tol,
                rel_tol,
            )?
            .json)
    }
}

#[wasm_bindgen]
impl TracedMultiInputRun {
    /// A failed numerical step has no output and never issues an execution receipt.
    pub fn output(&self) -> Result<WasmTensor, String> {
        self.output
            .clone()
            .ok_or_else(|| "TracedMultiInputRun: execution did not complete".into())
    }

    pub fn report(&self) -> String {
        self.report.clone()
    }
}

#[wasm_bindgen]
impl GraphMutationTransaction {
    #[wasm_bindgen(constructor)]
    pub fn new(
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        expected_program_identity: &str,
        expected_state_digest: &str,
    ) -> Result<Self, String> {
        if graph.program_identity() != expected_program_identity {
            return Err("mutation: expected baseline program identity mismatch".into());
        }
        let baseline_bundle = export_multi_input_program_bundle(graph, registry, true)?;
        if baseline_bundle.len() > MAX_BUNDLE_BYTES {
            return Err("mutation: baseline bundle exceeds 16 MiB".into());
        }
        let baseline_digest = digest(&baseline_bundle);
        if baseline_digest != expected_state_digest {
            return Err("mutation: expected baseline state digest mismatch".into());
        }
        let mut baseline_registry = LayerRegistry::new();
        let baseline_graph =
            import_multi_input_program_bundle(&mut baseline_registry, &baseline_bundle)?;
        let baseline_plan = graph.plan.clone();
        let proposal = decode_graph_plan(baseline_plan.graph_plan())?;
        Ok(Self {
            baseline_bundle,
            baseline_identity: expected_program_identity.into(),
            baseline_digest,
            baseline_registry,
            baseline_graph,
            baseline_plan,
            proposal,
            init_specs: Vec::new(),
            weight_patches: Vec::new(),
            candidate: None,
            committed: false,
        })
    }

    #[wasm_bindgen(js_name = replaceStep)]
    pub fn replace_step(
        &mut self,
        index: u32,
        spec: &AgentLayerSpec,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let index = index as usize;
        if index >= self.proposal.steps.len() {
            return Err("mutation.replaceStep: index out of range".into());
        }
        self.proposal.steps[index] = Self::step(spec, first, second, output);
        self.init_specs.push(spec.clone());
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = insertStep)]
    pub fn insert_step(
        &mut self,
        index: u32,
        spec: &AgentLayerSpec,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let index = index as usize;
        if index > self.proposal.steps.len() {
            return Err("mutation.insertStep: index out of range".into());
        }
        if self.proposal.steps.len() >= 4096 {
            return Err("mutation.insertStep: 4096 step limit".into());
        }
        self.proposal
            .steps
            .insert(index, Self::step(spec, first, second, output));
        self.init_specs.push(spec.clone());
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = removeStep)]
    pub fn remove_step(&mut self, index: u32) -> Result<(), String> {
        self.editable()?;
        let index = index as usize;
        if index >= self.proposal.steps.len() {
            return Err("mutation.removeStep: index out of range".into());
        }
        self.proposal.steps.remove(index);
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = reconnectStep)]
    pub fn reconnect_step(
        &mut self,
        index: u32,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        self.check_slots(&[first, second, output])?;
        let step = self
            .proposal
            .steps
            .get_mut(index as usize)
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
    pub fn set_weights_flat(
        &mut self,
        layer_type: u8,
        layer_id: u32,
        values: &[f32],
    ) -> Result<(), String> {
        self.editable()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err("mutation: weights must be finite".into());
        }
        if !self
            .proposal
            .steps
            .iter()
            .any(|step| (step.layer_type, step.layer_id) == (layer_type, layer_id))
        {
            return Err("mutation: weight target is not referenced by proposal".into());
        }
        self.weight_patches
            .retain(|(kind, id, _)| (*kind, *id) != (layer_type, layer_id));
        self.weight_patches
            .push((layer_type, layer_id, values.to_vec()));
        self.invalidate();
        Ok(())
    }

    #[wasm_bindgen(js_name = baselineBundle)]
    pub fn baseline_bundle(&self) -> Vec<u8> {
        self.baseline_bundle.clone()
    }

    #[wasm_bindgen(js_name = baselineStateDigest)]
    pub fn baseline_state_digest(&self) -> String {
        self.baseline_digest.clone()
    }

    #[wasm_bindgen(js_name = stageCandidate)]
    pub fn stage_candidate(&mut self) -> Result<String, String> {
        self.editable()?;
        self.invalidate();
        let plan = self
            .baseline_plan
            .with_graph_plan(encode(&self.proposal)?)?;
        let mut registry = LayerRegistry::new();
        import_multi_input_program_bundle(&mut registry, &self.baseline_bundle)?;
        for spec in &self.init_specs {
            if self.proposal.steps.iter().any(|step| {
                (step.layer_type, step.layer_id) == (spec.layer_type(), spec.layer_id())
            }) {
                registry.init_agent_layer(spec)?;
            }
        }
        for (layer_type, layer_id, values) in &self.weight_patches {
            if !self
                .proposal
                .steps
                .iter()
                .any(|step| (step.layer_type, step.layer_id) == (*layer_type, *layer_id))
            {
                return Err("mutation: weight target was removed".into());
            }
            registry.set_weights_flat(*layer_id, *layer_type, values)?;
        }
        let graph = registry.compile_multi_input_graph(&plan)?;
        let bundle = export_multi_input_program_bundle(&graph, &registry, true)?;
        if bundle.len() > MAX_BUNDLE_BYTES {
            return Err("mutation: candidate bundle exceeds 16 MiB".into());
        }
        let report = format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.graph-mutation-stage.v1\",\"baseline_program_identity\":{},\"candidate_program_identity\":{},\"baseline_state_checkpoint_bytes_sha256\":\"{}\",\"candidate_state_checkpoint_bytes_sha256\":\"{}\",\"promotion_authorized\":false}}",
            self.baseline_identity, graph.program_identity(), self.baseline_digest, digest(&bundle));
        self.candidate = Some(StagedCandidate {
            registry,
            graph,
            bundle,
            receipt_digest: None,
        });
        Ok(report)
    }

    #[wasm_bindgen(js_name = candidateBundle)]
    pub fn candidate_bundle(&self) -> Result<Vec<u8>, String> {
        Ok(self
            .candidate
            .as_ref()
            .ok_or("mutation: no staged candidate")?
            .bundle
            .clone())
    }

    #[wasm_bindgen(js_name = verifyCases)]
    pub fn verify_cases(
        &mut self,
        cases: &MultiInputVerificationCases,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
        self.editable()?;
        let candidate = self
            .candidate
            .as_mut()
            .ok_or("mutation: no staged candidate")?;
        candidate.receipt_digest = None;
        let outcome = cases.verify_outcome(
            &self.baseline_registry,
            &self.baseline_graph,
            &candidate.registry,
            &candidate.graph,
            abs_tol,
            rel_tol,
        )?;
        if outcome.baseline_state != self.baseline_digest
            || outcome.candidate_state != digest(&candidate.bundle)
        {
            return Err("mutation: verification checkpoint differs from staged snapshot".into());
        }
        if outcome.equivalent {
            candidate.receipt_digest = Some(outcome.digest);
        }
        Ok(outcome.json)
    }

    #[wasm_bindgen(js_name = commitByReceipt)]
    pub fn commit_by_receipt(
        &mut self,
        live_registry: &mut LayerRegistry,
        live_graph: &CompiledMultiInputGraph,
        expected_program_identity: &str,
        expected_state_digest: &str,
        receipt_digest: &str,
        authorize: bool,
    ) -> Result<CompiledMultiInputGraph, String> {
        self.editable()?;
        if !authorize {
            return Err("mutation: explicit promotion authorization required".into());
        }
        if expected_program_identity != self.baseline_identity
            || live_graph.program_identity() != self.baseline_identity
        {
            return Err("mutation: baseline program identity changed".into());
        }
        if expected_state_digest != self.baseline_digest {
            return Err("mutation: expected baseline state digest mismatch".into());
        }
        let candidate = self
            .candidate
            .as_ref()
            .ok_or("mutation: no staged candidate")?;
        if candidate.receipt_digest.as_deref() != Some(receipt_digest) {
            return Err(
                "mutation: passing verification receipt from this transaction required".into(),
            );
        }
        let live = export_multi_input_program_bundle(live_graph, live_registry, true)?;
        if live != self.baseline_bundle {
            return Err("mutation: baseline state changed; no commit".into());
        }
        if digest(&candidate.bundle)
            != digest(&export_multi_input_program_bundle(
                &candidate.graph,
                &candidate.registry,
                true,
            )?)
        {
            return Err("mutation: staged candidate state changed; no commit".into());
        }
        let graph = import_multi_input_program_bundle(live_registry, &candidate.bundle)?;
        self.committed = true;
        Ok(graph)
    }
}

#[wasm_bindgen]
impl CheckpointBranchSet {
    #[wasm_bindgen(constructor)]
    pub fn new(
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        expected_program_identity: &str,
        expected_state_digest: &str,
    ) -> Result<Self, String> {
        let snapshot = GraphMutationTransaction::new(
            registry,
            graph,
            expected_program_identity,
            expected_state_digest,
        )?;
        Ok(Self {
            baseline_bundle: snapshot.baseline_bundle.clone(),
            baseline_identity: snapshot.baseline_identity,
            baseline_digest: snapshot.baseline_digest,
            baseline_registry: snapshot.baseline_registry,
            baseline_graph: snapshot.baseline_graph,
            branches: Vec::new(),
            promoted_branch: None,
        })
    }

    #[wasm_bindgen(js_name = branchCount)]
    pub fn branch_count(&self) -> u32 {
        self.branches.len() as u32
    }

    #[wasm_bindgen(js_name = promotedBranch)]
    pub fn promoted_branch(&self) -> Option<String> {
        self.promoted_branch.clone()
    }

    #[wasm_bindgen(js_name = baselineStateDigest)]
    pub fn baseline_state_digest(&self) -> String {
        self.baseline_digest.clone()
    }

    #[wasm_bindgen(js_name = baselineBundle)]
    pub fn baseline_bundle(&self) -> Vec<u8> {
        self.baseline_bundle.clone()
    }

    #[wasm_bindgen(js_name = fork)]
    pub fn fork(&mut self, branch_id: &str) -> Result<u32, String> {
        self.editable()?;
        if !valid_branch_id(branch_id) {
            return Err("checkpoint branches.fork: id must be 1..=64 ASCII letters, digits, '.', '_' or '-'".into());
        }
        if self.branches.iter().any(|branch| branch.id == branch_id) {
            return Err("checkpoint branches.fork: branch id already exists".into());
        }
        if self.branches.len() >= MAX_CHECKPOINT_BRANCHES {
            return Err("checkpoint branches.fork: at most 8 branches".into());
        }
        let snapshots = self
            .baseline_bundle
            .len()
            .checked_mul(self.branches.len() + 2)
            .ok_or("checkpoint branches.fork: snapshot budget overflow")?;
        if snapshots > MAX_BRANCH_SNAPSHOT_BYTES {
            return Err(
                "checkpoint branches.fork: aggregate baseline snapshots exceed 64 MiB".into(),
            );
        }
        let transaction = GraphMutationTransaction::new(
            &self.baseline_registry,
            &self.baseline_graph,
            &self.baseline_identity,
            &self.baseline_digest,
        )?;
        self.branches.push(CheckpointBranch {
            id: branch_id.into(),
            transaction,
            verification: None,
        });
        Ok(self.branches.len() as u32)
    }

    #[wasm_bindgen(js_name = replaceStep)]
    pub fn replace_step(
        &mut self,
        branch_id: &str,
        index: u32,
        spec: &AgentLayerSpec,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch
            .transaction
            .replace_step(index, spec, first, second, output)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = insertStep)]
    pub fn insert_step(
        &mut self,
        branch_id: &str,
        index: u32,
        spec: &AgentLayerSpec,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch
            .transaction
            .insert_step(index, spec, first, second, output)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = removeStep)]
    pub fn remove_step(&mut self, branch_id: &str, index: u32) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch.transaction.remove_step(index)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = reconnectStep)]
    pub fn reconnect_step(
        &mut self,
        branch_id: &str,
        index: u32,
        first: u8,
        second: u8,
        output: u8,
    ) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch
            .transaction
            .reconnect_step(index, first, second, output)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = setOutput)]
    pub fn set_output(&mut self, branch_id: &str, output: u8) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch.transaction.set_output(output)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = setWeightsFlat)]
    pub fn set_weights_flat(
        &mut self,
        branch_id: &str,
        layer_type: u8,
        layer_id: u32,
        values: &[f32],
    ) -> Result<(), String> {
        self.editable()?;
        let branch = self.branch_mut(branch_id)?;
        branch
            .transaction
            .set_weights_flat(layer_type, layer_id, values)?;
        branch.verification = None;
        Ok(())
    }

    #[wasm_bindgen(js_name = stageBranch)]
    pub fn stage_branch(&mut self, branch_id: &str) -> Result<String, String> {
        self.editable()?;
        let other_candidate_bytes = self
            .branches
            .iter()
            .filter(|branch| branch.id != branch_id)
            .filter_map(|branch| branch.transaction.candidate.as_ref())
            .map(|candidate| candidate.bundle.len())
            .sum::<usize>();
        let branch = self.branch_mut(branch_id)?;
        branch.verification = None;
        let stage = branch.transaction.stage_candidate()?;
        let candidate_len = branch.transaction.candidate.as_ref().unwrap().bundle.len();
        if other_candidate_bytes.saturating_add(candidate_len) > MAX_BRANCH_CANDIDATE_BYTES {
            branch.transaction.invalidate();
            return Err(
                "checkpoint branches.stageBranch: aggregate candidate bundles exceed 64 MiB".into(),
            );
        }
        Ok(format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.checkpoint-branch-stage.v1\",\"branch_id\":\"{}\",\"stage\":{stage}}}", branch.id))
    }

    #[wasm_bindgen(js_name = candidateBundle)]
    pub fn candidate_bundle(&self, branch_id: &str) -> Result<Vec<u8>, String> {
        let branch = self
            .branches
            .iter()
            .find(|branch| branch.id == branch_id)
            .ok_or_else(|| format!("checkpoint branches: unknown branch {branch_id:?}"))?;
        branch.transaction.candidate_bundle()
    }

    #[wasm_bindgen(js_name = stateDiff)]
    pub fn state_diff(&self, branch_id: &str) -> Result<String, String> {
        let branch = self
            .branches
            .iter()
            .find(|branch| branch.id == branch_id)
            .ok_or_else(|| format!("checkpoint branches: unknown branch {branch_id:?}"))?;
        let candidate = branch
            .transaction
            .candidate
            .as_ref()
            .ok_or("checkpoint branches: stage branch before diff")?;
        Ok(format!(
            "{{\"branch_id\":\"{}\",\"state_diff\":{}}}",
            branch.id,
            state_diff_json(&self.baseline_bundle, &candidate.bundle)
        ))
    }

    #[wasm_bindgen(js_name = verifyBranch)]
    pub fn verify_branch(
        &mut self,
        branch_id: &str,
        cases: &MultiInputVerificationCases,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Result<String, String> {
        self.editable()?;
        let baseline_identity = self.baseline_identity.clone();
        let baseline_digest = self.baseline_digest.clone();
        let baseline_bundle = self.baseline_bundle.clone();
        let branch = self.branch_mut(branch_id)?;
        branch.verification = None;
        let verification = branch.transaction.verify_cases(cases, abs_tol, rel_tol)?;
        let candidate = branch
            .transaction
            .candidate
            .as_ref()
            .ok_or("checkpoint branches: candidate disappeared")?;
        let verifier_receipt_digest = candidate.receipt_digest.clone().unwrap_or_default();
        let equivalent = !verifier_receipt_digest.is_empty();
        let state_diff = state_diff_json(&baseline_bundle, &candidate.bundle);
        let body = format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.checkpoint-branch-verification.v1\",\"authority\":\"wasm_burn_reference_observation\",\"branch_id\":\"{}\",\"baseline_program_identity\":{},\"candidate_program_identity\":{},\"baseline_state_checkpoint_bytes_sha256\":\"{}\",\"candidate_state_checkpoint_bytes_sha256\":\"{}\",\"equivalent\":{equivalent},\"promotion_authorized\":false,\"verifier_receipt\":{},\"state_diff\":{}}}",
            branch.id, baseline_identity, candidate.graph.program_identity(), baseline_digest,
            digest(&candidate.bundle), verification, state_diff);
        let receipt_digest = digest(body.as_bytes());
        let receipt = format!(
            "{},\"receipt_digest\":\"{receipt_digest}\"}}",
            &body[..body.len() - 1]
        );
        branch.verification = Some(BranchVerification {
            receipt_digest,
            verifier_receipt_digest,
            equivalent,
        });
        Ok(receipt)
    }

    #[wasm_bindgen(js_name = commitBranchByReceipt)]
    pub fn commit_branch_by_receipt(
        &mut self,
        branch_id: &str,
        live_registry: &mut LayerRegistry,
        live_graph: &CompiledMultiInputGraph,
        expected_program_identity: &str,
        expected_state_digest: &str,
        receipt_digest: &str,
        authorize: bool,
    ) -> Result<CompiledMultiInputGraph, String> {
        self.editable()?;
        if !authorize {
            return Err("checkpoint branches: explicit promotion authorization required".into());
        }
        let branch = self.branch_mut(branch_id)?;
        let proof = branch
            .verification
            .as_ref()
            .ok_or("checkpoint branches: verified branch receipt required")?;
        if !proof.equivalent || proof.receipt_digest != receipt_digest {
            return Err("checkpoint branches: equivalent receipt for this branch required".into());
        }
        let promoted = branch.transaction.commit_by_receipt(
            live_registry,
            live_graph,
            expected_program_identity,
            expected_state_digest,
            &proof.verifier_receipt_digest,
            true,
        )?;
        self.promoted_branch = Some(branch_id.into());
        Ok(promoted)
    }
}

#[wasm_bindgen]
impl MultiInputGraphPlan {
    #[wasm_bindgen(constructor)]
    pub fn new(builder: &AgentGraphBuilder) -> Result<MultiInputGraphPlan, String> {
        let graph_plan = builder.plan_bytes()?;
        let decoded = decode_graph_plan(&graph_plan)
            .map_err(|error| format!("MultiInputGraphPlan.new: {error}"))?;
        Ok(Self {
            graph_plan,
            num_slots: decoded.num_slots,
            ports: Vec::new(),
        })
    }

    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<MultiInputGraphPlan, String> {
        let mut cursor = PlanCursor::new(bytes);
        if cursor.take(PLAN_MAGIC.len(), "magic")? != PLAN_MAGIC {
            return Err("MultiInputGraphPlan.fromBytes: invalid magic".into());
        }
        let version = cursor.read_u32("schema version")?;
        if version != PLAN_SCHEMA_VERSION {
            return Err(format!(
                "MultiInputGraphPlan.fromBytes: unsupported schema version {version}"
            ));
        }
        let graph_len = cursor.read_u32("graph plan length")? as usize;
        let port_count = usize::from(cursor.read_u8("port count")?);
        if port_count > MAX_INPUT_PORTS {
            return Err(format!(
                "MultiInputGraphPlan.fromBytes: port count {port_count} exceeds {MAX_INPUT_PORTS}"
            ));
        }
        let graph_plan = cursor.take(graph_len, "graph plan")?.to_vec();
        let decoded = decode_graph_plan(&graph_plan)
            .map_err(|error| format!("MultiInputGraphPlan.fromBytes: {error}"))?;
        let mut ports = Vec::with_capacity(port_count);
        let mut prior_slot = None;
        for _ in 0..port_count {
            let slot = cursor.read_u8("input slot")?;
            if prior_slot.is_some_and(|prior| prior >= slot) {
                return Err(
                    "MultiInputGraphPlan.fromBytes: input ports must be unique and sorted by slot"
                        .into(),
                );
            }
            prior_slot = Some(slot);
            let role_len = usize::from(cursor.read_u8("role length")?);
            let role = std::str::from_utf8(cursor.take(role_len, "role")?)
                .map_err(|_| "MultiInputGraphPlan.fromBytes: role is not valid UTF-8".to_string())?
                .to_string();
            let shape = [
                cursor.read_u32("shape dim0")?,
                cursor.read_u32("shape dim1")?,
                cursor.read_u32("shape dim2")?,
                cursor.read_u32("shape dim3")?,
            ];
            let layout_len = usize::from(cursor.read_u16("layout length")?);
            let layout = std::str::from_utf8(cursor.take(layout_len, "layout")?)
                .map_err(|_| {
                    "MultiInputGraphPlan.fromBytes: layout is not valid UTF-8".to_string()
                })?
                .to_string();
            let require_fingerprint = match cursor.read_u8("fingerprint policy")? {
                0 => false,
                1 => true,
                value => {
                    return Err(format!(
                        "MultiInputGraphPlan.fromBytes: invalid fingerprint policy {value}"
                    ))
                }
            };
            let minimum_revision = cursor.read_u64("minimum revision")?;
            let port = MultiInputPortContract {
                slot,
                role,
                shape,
                layout,
                require_fingerprint,
                minimum_revision,
            };
            validate_port_contract(&port, decoded.num_slots)
                .map_err(|error| format!("MultiInputGraphPlan.fromBytes: {error}"))?;
            ports.push(port);
        }
        if !cursor.is_finished() {
            return Err("MultiInputGraphPlan.fromBytes: trailing bytes after final port".into());
        }
        Ok(Self {
            graph_plan,
            num_slots: decoded.num_slots,
            ports,
        })
    }

    #[wasm_bindgen(js_name = addInputPort)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_input_port(
        &mut self,
        slot: u8,
        role: String,
        dim0: u32,
        dim1: u32,
        dim2: u32,
        dim3: u32,
        layout: String,
        require_fingerprint: bool,
        minimum_revision: u64,
    ) -> Result<bool, String> {
        let port = MultiInputPortContract {
            slot,
            role,
            shape: [dim0, dim1, dim2, dim3],
            layout,
            require_fingerprint,
            minimum_revision,
        };
        validate_port_contract(&port, self.num_slots)?;
        match self.ports.iter().find(|existing| existing.slot == slot) {
            Some(existing) if existing == &port => return Ok(false),
            Some(_) => {
                return Err(format!(
                    "MultiInputGraphPlan.addInputPort: conflicting definition for slot {slot}"
                ))
            }
            None => {}
        }
        if self.ports.len() >= MAX_INPUT_PORTS {
            return Err(format!(
                "MultiInputGraphPlan.addInputPort: maximum {MAX_INPUT_PORTS} ports reached"
            ));
        }
        self.ports.push(port);
        self.ports.sort_by_key(|port| port.slot);
        Ok(true)
    }

    #[wasm_bindgen(js_name = portCount)]
    pub fn port_count(&self) -> u32 {
        self.ports.len() as u32
    }

    #[wasm_bindgen(js_name = inputSlots)]
    pub fn input_slots(&self) -> Vec<u8> {
        self.ports.iter().map(|port| port.slot).collect()
    }

    #[wasm_bindgen(js_name = programPlan)]
    pub fn program_plan(&self) -> Vec<u8> {
        self.graph_plan.clone()
    }

    #[wasm_bindgen(js_name = planFingerprint)]
    pub fn plan_fingerprint(&self) -> Result<String, String> {
        self.fingerprint_internal()
    }

    #[wasm_bindgen(js_name = toBytes)]
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.encode()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, String> {
        let ports = self
            .ports
            .iter()
            .map(port_contract_json)
            .collect::<Vec<_>>()
            .join(",");
        let fingerprint = self.fingerprint_internal()?;
        Ok(format!(
            "{{\"schema_version\":1,\"schema_id\":\"{}\",\"plan_fingerprint\":\"{}\",\"topology_plan_bytes\":{},\"input_ports\":[{}],\"runtime_policy\":\"all_declared_inputs_must_be_bound_before_execution\",\"execution_authorized\":false}}",
            PLAN_SCHEMA_ID,
            fingerprint,
            self.graph_plan.len(),
            ports,
        ))
    }
}

#[wasm_bindgen]
impl MultiInputInputBundle {
    #[wasm_bindgen(constructor)]
    pub fn new(plan: &MultiInputGraphPlan) -> Result<MultiInputInputBundle, String> {
        let plan_bytes = plan.validate_for_compile()?;
        let plan_fingerprint = fnv1a64(plan_bytes.iter().copied());
        Ok(Self {
            plan_bytes,
            plan_fingerprint,
            ports: plan
                .ports
                .iter()
                .cloned()
                .map(|contract| InputPortBinding {
                    contract,
                    bound: None,
                })
                .collect(),
        })
    }

    #[wasm_bindgen(js_name = bindInput)]
    pub fn bind_input(
        &mut self,
        slot: u8,
        tensor: &WasmTensor,
        role: String,
        layout: String,
        source: String,
        revision: u64,
        fingerprint: String,
    ) -> Result<bool, String> {
        let index = self
            .ports
            .iter()
            .position(|port| port.contract.slot == slot)
            .ok_or_else(|| {
                format!("MultiInputInputBundle.bindInput: slot {slot} is not declared in the plan")
            })?;
        if role.is_empty() || role.len() > MAX_ROLE_BYTES || !role_valid(&role) {
            return Err(format!(
                "MultiInputInputBundle.bindInput: unsupported role {role:?}"
            ));
        }
        if source.is_empty() || source.len() > MAX_SOURCE_BYTES {
            return Err(format!(
                "MultiInputInputBundle.bindInput: source must be 1..={MAX_SOURCE_BYTES} bytes"
            ));
        }
        if fingerprint.len() > MAX_FINGERPRINT_BYTES {
            return Err(format!("MultiInputInputBundle.bindInput: fingerprint exceeds {MAX_FINGERPRINT_BYTES} bytes"));
        }
        let actual_shape = tensor_shape(tensor)?;
        validate_external_input_contract_declaration(actual_shape, &layout)
            .map_err(|error| format!("MultiInputInputBundle.bindInput: {error}"))?;
        let value_fingerprint = tensor_value_fingerprint(tensor)?;
        if let Some(existing) = self.ports[index].bound.as_ref() {
            if existing.role == role
                && existing.layout == layout
                && existing.source == source
                && existing.revision == revision
                && existing.fingerprint == fingerprint
                && existing.value_fingerprint == value_fingerprint
            {
                return Ok(false);
            }
            return Err(format!("MultiInputInputBundle.bindInput: slot {slot} is already bound; clear it before replacement"));
        }
        self.ports[index].bound = Some(BoundInput {
            tensor: tensor.clone(),
            role,
            layout,
            source,
            revision,
            fingerprint,
            value_fingerprint,
        });
        Ok(true)
    }

    #[wasm_bindgen(js_name = clearInput)]
    pub fn clear_input(&mut self, slot: u8) -> bool {
        let Some(port) = self
            .ports
            .iter_mut()
            .find(|port| port.contract.slot == slot)
        else {
            return false;
        };
        port.bound.take().is_some()
    }

    #[wasm_bindgen(js_name = boundPortCount)]
    pub fn bound_port_count(&self) -> u32 {
        self.ports
            .iter()
            .filter(|port| port.bound.is_some())
            .count() as u32
    }

    #[wasm_bindgen(js_name = planFingerprint)]
    pub fn plan_fingerprint(&self) -> String {
        self.plan_fingerprint.clone()
    }
}
