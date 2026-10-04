//! Isolated graph edits with an explicit, receipt-bound promotion gate.

pub use crate::facade::graph::checkpoint_branch_capabilities;
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

use super::{bytes_hex, CompiledMultiInputGraph, MultiInputVerificationCases};
use crate::agent::AgentLayerSpec;
use crate::graph_plan::{decode_graph_plan, DecodedGraphPlan, GraphPlanStep};
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::program_bundle::{export_multi_input_program_bundle, import_multi_input_program_bundle};
use crate::protocol::LAYER_BINARY;
use crate::registry::LayerRegistry;

pub(crate) const MAX_BUNDLE_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const MAX_CHECKPOINT_BRANCHES: usize = 8;
pub(crate) const MAX_BRANCH_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_BRANCH_CANDIDATE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_STATE_DIFF_RANGES: usize = 64;

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("sha256:{}", bytes_hex(&Sha256::digest(bytes)))
}

pub(crate) fn valid_branch_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

pub(crate) fn state_diff_json(before: &[u8], after: &[u8]) -> String {
    let common = before.len().min(after.len());
    let mut changed = 0usize;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut range_start = None;
    let mut truncated = false;
    for offset in 0..common {
        if before[offset] != after[offset] {
            changed += 1;
            if range_start.is_none() {
                range_start = Some(offset);
            }
        } else if let Some(start) = range_start.take() {
            if ranges.len() < MAX_STATE_DIFF_RANGES {
                ranges.push((start, offset));
            } else {
                truncated = true;
            }
        }
    }
    if let Some(start) = range_start.take() {
        if ranges.len() < MAX_STATE_DIFF_RANGES {
            ranges.push((start, common));
        } else {
            truncated = true;
        }
    }
    if after.len() > common {
        changed += after.len() - common;
        if ranges.len() < MAX_STATE_DIFF_RANGES {
            ranges.push((common, after.len()));
        } else {
            truncated = true;
        }
    } else if before.len() > common {
        changed += before.len() - common;
        if ranges.len() < MAX_STATE_DIFF_RANGES {
            ranges.push((common, before.len()));
        } else {
            truncated = true;
        }
    }
    let encoded = ranges.iter().map(|(start, end)| {
        let old_end = (*end).min(before.len());
        let new_end = (*end).min(after.len());
        format!("{{\"offset\":{start},\"length\":{},\"baseline_sha256\":\"{}\",\"candidate_sha256\":\"{}\"}}",
            end - start, digest(&before[*start..old_end]), digest(&after[*start..new_end]))
    }).collect::<Vec<_>>().join(",");
    let delta = after.len() as i64 - before.len() as i64;
    format!("{{\"schema_version\":1,\"schema_id\":\"burn-research.checkpoint-state-diff.v1\",\"baseline_bundle_sha256\":\"{}\",\"candidate_bundle_sha256\":\"{}\",\"baseline_bytes\":{},\"candidate_bytes\":{},\"byte_length_delta\":{delta},\"changed_byte_count\":{changed},\"same_length\":{},\"ranges\":[{encoded}],\"ranges_truncated\":{truncated},\"semantic_equivalence_asserted\":false}}",
        digest(before), digest(after), before.len(), after.len(), before.len() == after.len())
}

pub(crate) fn encode(plan: &DecodedGraphPlan) -> Result<Vec<u8>, String> {
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

pub(crate) struct StagedCandidate {
    pub(crate) registry: LayerRegistry,
    pub(crate) graph: CompiledMultiInputGraph,
    pub(crate) bundle: Vec<u8>,
    pub(crate) receipt_digest: Option<String>,
}

pub(crate) struct BranchVerification {
    pub(crate) receipt_digest: String,
    pub(crate) verifier_receipt_digest: String,
    pub(crate) equivalent: bool,
}

pub(crate) struct CheckpointBranch {
    pub(crate) id: String,
    pub(crate) transaction: GraphMutationTransaction,
    pub(crate) verification: Option<BranchVerification>,
}

#[wasm_bindgen]
pub struct GraphMutationTransaction {
    pub(crate) baseline_bundle: Vec<u8>,
    pub(crate) baseline_identity: String,
    pub(crate) baseline_digest: String,
    pub(crate) baseline_registry: LayerRegistry,
    pub(crate) baseline_graph: CompiledMultiInputGraph,
    pub(crate) baseline_plan: MultiInputGraphPlan,
    pub(crate) proposal: DecodedGraphPlan,
    pub(crate) init_specs: Vec<AgentLayerSpec>,
    pub(crate) weight_patches: Vec<(u8, u32, Vec<f32>)>,
    pub(crate) candidate: Option<StagedCandidate>,
    pub(crate) committed: bool,
}

#[wasm_bindgen]
pub struct CheckpointBranchSet {
    pub(crate) baseline_bundle: Vec<u8>,
    pub(crate) baseline_identity: String,
    pub(crate) baseline_digest: String,
    pub(crate) baseline_registry: LayerRegistry,
    pub(crate) baseline_graph: CompiledMultiInputGraph,
    pub(crate) branches: Vec<CheckpointBranch>,
    pub(crate) promoted_branch: Option<String>,
}

impl GraphMutationTransaction {
    pub(crate) fn editable(&self) -> Result<(), String> {
        if self.committed {
            Err("mutation: transaction already committed".into())
        } else {
            Ok(())
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.candidate = None;
    }

    pub(crate) fn step(spec: &AgentLayerSpec, first: u8, second: u8, output: u8) -> GraphPlanStep {
        let layer_type = spec.layer_type();
        GraphPlanStep {
            arity: if layer_type == LAYER_BINARY { 2 } else { 1 },
            layer_type,
            layer_id: spec.layer_id(),
            in_slot: first,
            in_slot2: if layer_type == LAYER_BINARY {
                second
            } else {
                first
            },
            out_slot: output,
        }
    }

    pub(crate) fn check_slots(&self, slots: &[u8]) -> Result<(), String> {
        if slots
            .iter()
            .any(|slot| u32::from(*slot) >= self.proposal.num_slots)
        {
            return Err("mutation: slot outside the fixed graph slot count".into());
        }
        Ok(())
    }
}

impl CheckpointBranchSet {
    pub(crate) fn editable(&self) -> Result<(), String> {
        if self.promoted_branch.is_some() {
            Err("checkpoint branches: a branch has already been promoted".into())
        } else {
            Ok(())
        }
    }

    pub(crate) fn branch_mut(&mut self, id: &str) -> Result<&mut CheckpointBranch, String> {
        self.branches
            .iter_mut()
            .find(|branch| branch.id == id)
            .ok_or_else(|| format!("checkpoint branches: unknown branch {id:?}"))
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
            plan.add_input_port(
                slot,
                role.into(),
                1,
                2,
                1,
                1,
                "feature_axis1_singleton".into(),
                false,
                0,
            )
            .unwrap();
        }
        let graph = registry.compile_multi_input_graph(&plan).unwrap();
        (registry, graph, plan)
    }

    fn input(plan: &MultiInputGraphPlan) -> MultiInputInputBundle {
        let mut bundle = MultiInputInputBundle::new(plan).unwrap();
        for (slot, role, values) in [(0, "observation", [1.0, 2.0]), (1, "state", [3.0, 4.0])] {
            bundle
                .bind_input(
                    slot,
                    &WasmTensor::new(&values, &[1, 2, 1, 1]),
                    role.into(),
                    "feature_axis1_singleton".into(),
                    "test".into(),
                    0,
                    String::new(),
                )
                .unwrap();
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
        transaction
            .replace_step(0, &AgentLayerSpec::add(22), 0, 1, 3)
            .unwrap();
        transaction.set_output(3).unwrap();
        assert!(transaction.stage_candidate().is_ok());
        assert_eq!(
            export_multi_input_program_bundle(&baseline, &live, true).unwrap(),
            original
        );
        let mut candidate_registry = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(
            &mut candidate_registry,
            &transaction.candidate_bundle().unwrap(),
        )
        .unwrap();
        let candidate_plan = MultiInputGraphPlan::from_bytes(&candidate.input_plan_v1()).unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases
            .add_case(&input(&baseline_plan), &input(&candidate_plan))
            .unwrap();
        let receipt: serde_json::Value =
            serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        let identity = baseline.program_identity();
        let state = transaction.baseline_state_digest();
        let hash = receipt["receipt_digest"].as_str().unwrap();
        assert!(transaction
            .commit_by_receipt(&mut live, &baseline, &identity, &state, "sha256:fake", true)
            .is_err());
        assert!(transaction
            .commit_by_receipt(&mut live, &baseline, &identity, &state, hash, false)
            .is_err());
        assert_eq!(
            export_multi_input_program_bundle(&baseline, &live, true).unwrap(),
            original
        );
        let promoted = transaction
            .commit_by_receipt(&mut live, &baseline, &identity, &state, hash, true)
            .unwrap();
        assert_eq!(promoted.program_identity(), candidate.program_identity());
        assert!(transaction
            .commit_by_receipt(&mut live, &baseline, &identity, &state, hash, true)
            .is_err());
    }

    #[test]
    fn invalid_candidate_and_negative_receipt_leave_baseline_intact() {
        let (mut live, baseline, plan) = setup();
        let original = export_multi_input_program_bundle(&baseline, &live, true).unwrap();
        let mut transaction = tx(&live, &baseline);
        transaction.remove_step(0).unwrap();
        assert!(transaction.stage_candidate().is_err());
        assert!(transaction.candidate_bundle().is_err());
        transaction
            .insert_step(0, &AgentLayerSpec::sub(21), 0, 1, 2)
            .unwrap();
        transaction.stage_candidate().unwrap();
        let mut isolated = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(
            &mut isolated,
            &transaction.candidate_bundle().unwrap(),
        )
        .unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases.add_case(&input(&plan), &input(&plan)).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], false);
        assert!(transaction
            .commit_by_receipt(
                &mut live,
                &baseline,
                &baseline.program_identity(),
                &transaction.baseline_state_digest(),
                receipt["receipt_digest"].as_str().unwrap(),
                true
            )
            .is_err());
        assert_eq!(
            export_multi_input_program_bundle(&baseline, &live, true).unwrap(),
            original
        );
    }

    #[test]
    fn stale_state_rejects_commit_after_successful_verification() {
        let (mut live, baseline, plan) = setup();
        let mut transaction = tx(&live, &baseline);
        transaction
            .insert_step(1, &AgentLayerSpec::relu(22), 2, 2, 3)
            .unwrap();
        transaction.set_output(3).unwrap();
        transaction.stage_candidate().unwrap();
        let mut isolated = LayerRegistry::new();
        let candidate = import_multi_input_program_bundle(
            &mut isolated,
            &transaction.candidate_bundle().unwrap(),
        )
        .unwrap();
        let candidate_plan = MultiInputGraphPlan::from_bytes(&candidate.input_plan_v1()).unwrap();
        let mut cases = MultiInputVerificationCases::new(&baseline, &candidate).unwrap();
        cases
            .add_case(&input(&plan), &input(&candidate_plan))
            .unwrap();
        let receipt: serde_json::Value =
            serde_json::from_str(&transaction.verify_cases(&cases, 0.0, 0.0).unwrap()).unwrap();
        assert_eq!(receipt["equivalent"], true);
        live.destroy_layer(21, LAYER_BINARY);
        assert!(transaction
            .commit_by_receipt(
                &mut live,
                &baseline,
                &baseline.program_identity(),
                &transaction.baseline_state_digest(),
                receipt["receipt_digest"].as_str().unwrap(),
                true
            )
            .is_err());
    }

    #[test]
    fn checkpoint_branches_diff_compare_and_promote_one_exact_branch() {
        let (mut live, baseline, plan) = setup();
        let original = export_multi_input_program_bundle(&baseline, &live, true).unwrap();
        let identity = baseline.program_identity();
        let baseline_digest = digest(&original);
        let mut branches =
            CheckpointBranchSet::new(&live, &baseline, &identity, &baseline_digest).unwrap();
        assert_eq!(branches.fork("relu-path").unwrap(), 1);
        branches.fork("sub-path").unwrap();
        assert!(branches.fork("../bad").is_err());
        branches
            .insert_step("relu-path", 1, &AgentLayerSpec::relu(22), 2, 2, 3)
            .unwrap();
        branches.set_output("relu-path", 3).unwrap();
        branches
            .replace_step("sub-path", 0, &AgentLayerSpec::sub(23), 0, 1, 2)
            .unwrap();
        branches.stage_branch("relu-path").unwrap();
        branches.stage_branch("sub-path").unwrap();
        let diff: serde_json::Value =
            serde_json::from_str(&branches.state_diff("relu-path").unwrap()).unwrap();
        assert!(diff["state_diff"]["changed_byte_count"].as_u64().unwrap() > 0);
        assert_eq!(diff["state_diff"]["semantic_equivalence_asserted"], false);
        assert_eq!(
            export_multi_input_program_bundle(&baseline, &live, true).unwrap(),
            original
        );

        let mut relu_registry = LayerRegistry::new();
        let relu_graph = import_multi_input_program_bundle(
            &mut relu_registry,
            &branches.candidate_bundle("relu-path").unwrap(),
        )
        .unwrap();
        let relu_plan = MultiInputGraphPlan::from_bytes(&relu_graph.input_plan_v1()).unwrap();
        let mut sub_registry = LayerRegistry::new();
        let sub_graph = import_multi_input_program_bundle(
            &mut sub_registry,
            &branches.candidate_bundle("sub-path").unwrap(),
        )
        .unwrap();
        let sub_plan = MultiInputGraphPlan::from_bytes(&sub_graph.input_plan_v1()).unwrap();
        let mut relu_cases = MultiInputVerificationCases::new(&baseline, &relu_graph).unwrap();
        relu_cases
            .add_case(&input(&plan), &input(&relu_plan))
            .unwrap();
        let mut sub_cases = MultiInputVerificationCases::new(&baseline, &sub_graph).unwrap();
        sub_cases
            .add_case(&input(&plan), &input(&sub_plan))
            .unwrap();
        let relu_receipt: serde_json::Value = serde_json::from_str(
            &branches
                .verify_branch("relu-path", &relu_cases, 0.0, 0.0)
                .unwrap(),
        )
        .unwrap();
        let sub_receipt: serde_json::Value = serde_json::from_str(
            &branches
                .verify_branch("sub-path", &sub_cases, 0.0, 0.0)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(relu_receipt["equivalent"], true);
        assert_eq!(relu_receipt["branch_id"], "relu-path");
        assert_eq!(sub_receipt["equivalent"], false);
        assert!(branches
            .commit_branch_by_receipt(
                "sub-path",
                &mut live,
                &baseline,
                &identity,
                &baseline_digest,
                relu_receipt["receipt_digest"].as_str().unwrap(),
                true
            )
            .is_err());
        assert!(branches
            .commit_branch_by_receipt(
                "relu-path",
                &mut live,
                &baseline,
                &identity,
                &baseline_digest,
                relu_receipt["receipt_digest"].as_str().unwrap(),
                false
            )
            .is_err());
        assert_eq!(
            export_multi_input_program_bundle(&baseline, &live, true).unwrap(),
            original
        );
        let promoted = branches
            .commit_branch_by_receipt(
                "relu-path",
                &mut live,
                &baseline,
                &identity,
                &baseline_digest,
                relu_receipt["receipt_digest"].as_str().unwrap(),
                true,
            )
            .unwrap();
        assert_eq!(branches.promoted_branch().as_deref(), Some("relu-path"));
        assert_eq!(promoted.program_identity(), relu_graph.program_identity());
        assert!(branches.fork("after-promotion").is_err());
    }
}
