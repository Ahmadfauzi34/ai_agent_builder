use wasm_bindgen::prelude::*;

use crate::contracts::validate_external_input_contract_declaration;
use crate::graph_plan::decode_graph_plan;
use crate::input_port::role_valid;
use crate::WasmTensor;

pub(crate) const PLAN_MAGIC: &[u8; 8] = b"BRMIP001";
pub(crate) const PLAN_SCHEMA_VERSION: u32 = 1;
pub(crate) const PLAN_SCHEMA_ID: &str = "burn-research.multi-input-graph-plan.v1";
pub(crate) const MAX_INPUT_PORTS: usize = 64;
const MAX_ROLE_BYTES: usize = 64;
const MAX_LAYOUT_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MultiInputPortContract {
    pub(crate) slot: u8,
    pub(crate) role: String,
    pub(crate) shape: [u32; 4],
    pub(crate) layout: String,
    pub(crate) require_fingerprint: bool,
    pub(crate) minimum_revision: u64,
}

#[derive(Clone)]
pub(crate) struct BoundInput {
    pub(crate) tensor: WasmTensor,
    pub(crate) role: String,
    pub(crate) layout: String,
    pub(crate) source: String,
    pub(crate) revision: u64,
    pub(crate) fingerprint: String,
    pub(crate) value_fingerprint: String,
}

#[derive(Clone)]
pub(crate) struct InputPortBinding {
    pub(crate) contract: MultiInputPortContract,
    pub(crate) bound: Option<BoundInput>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct MultiInputGraphPlan {
    pub(crate) graph_plan: Vec<u8>,
    pub(crate) num_slots: u32,
    pub(crate) ports: Vec<MultiInputPortContract>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct MultiInputInputBundle {
    pub(crate) plan_bytes: Vec<u8>,
    pub(crate) plan_fingerprint: String,
    pub(crate) ports: Vec<InputPortBinding>,
}

pub(crate) struct InputPreflight {
    pub(crate) ready: bool,
    pub(crate) json: String,
    pub(crate) port_checks: Vec<(u8, bool)>,
}

fn json_escape(value: &str) -> String {
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

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Incremental FNV-1a 64 hasher. `fnv1a64` is the one-shot form; use the
/// hasher directly to feed multiple byte slices without concatenating them.
pub(crate) struct Fnv1a64(u64);

impl Fnv1a64 {
    pub(crate) fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub(crate) fn finish(self) -> String {
        format!("fnv1a64:{:016x}", self.0)
    }
}

pub(crate) fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hasher = Fnv1a64::new();
    for byte in bytes {
        hasher.update(&[byte]);
    }
    hasher.finish()
}

pub(crate) fn tensor_shape(tensor: &WasmTensor) -> Result<[u32; 4], String> {
    let dims = tensor.inner.dims();
    let mut shape = [0u32; 4];
    for (index, dim) in dims.into_iter().enumerate() {
        shape[index] = u32::try_from(dim)
            .map_err(|_| format!("multi-input: shape dimension {dim} exceeds u32"))?;
    }
    Ok(shape)
}

pub(crate) fn tensor_value_fingerprint(tensor: &WasmTensor) -> Result<String, String> {
    let shape = tensor_shape(tensor)?;
    let data = tensor.inner.to_data();
    // Fail closed on non-f32 (the old path validated f32 via as_slice::<f32>()).
    // This probe materializes nothing.
    data.as_slice::<f32>()
        .map_err(|_| "multi-input: tensor value fingerprint requires an f32 tensor".to_string())?;
    // Single pass, zero intermediate allocation: feed the shape bytes and the
    // raw value bytes straight into the hasher. The preimage is byte-identical
    // to the old 3-pass serialization on little-endian targets (wasm32 is
    // LE-only): into_data()/to_data() normalizes to logical row-major order,
    // and f32 in-memory bytes equal value.to_bits().to_le_bytes().
    // WARNING: the fingerprint string must stay stable — it feeds bindInput
    // idempotency and JSON evidence. Guarded by fingerprint_one_pass_matches_legacy.
    let mut hasher = Fnv1a64::new();
    for dim in shape {
        hasher.update(&dim.to_le_bytes());
    }
    hasher.update(data.as_bytes());
    Ok(hasher.finish())
}

pub(crate) fn validate_port_contract(
    port: &MultiInputPortContract,
    num_slots: u32,
) -> Result<(), String> {
    if u32::from(port.slot) >= num_slots {
        return Err(format!(
            "MultiInputGraphPlan: input slot {} is outside graph num_slots {num_slots}",
            port.slot
        ));
    }
    if port.role.is_empty() || port.role.len() > MAX_ROLE_BYTES || !role_valid(&port.role) {
        return Err(format!(
            "MultiInputGraphPlan: unsupported input role {:?}; use a canonical role or x- extension",
            port.role
        ));
    }
    if port.layout.is_empty() || port.layout.len() > MAX_LAYOUT_BYTES {
        return Err(format!(
            "MultiInputGraphPlan: layout must be 1..={MAX_LAYOUT_BYTES} bytes"
        ));
    }
    validate_external_input_contract_declaration(port.shape, &port.layout)
        .map_err(|error| format!("MultiInputGraphPlan.slot{}: {error}", port.slot))
}

pub(crate) fn port_contract_json(port: &MultiInputPortContract) -> String {
    format!(
        concat!(
            "{{",
            "\"slot\":{},",
            "\"role\":\"{}\",",
            "\"dtype\":\"f32\",",
            "\"shape\":[{},{},{},{}],",
            "\"layout\":\"{}\",",
            "\"required\":true,",
            "\"require_fingerprint\":{},",
            "\"minimum_revision\":{}",
            "}}"
        ),
        port.slot,
        json_escape(&port.role),
        port.shape[0],
        port.shape[1],
        port.shape[2],
        port.shape[3],
        json_escape(&port.layout),
        bool_json(port.require_fingerprint),
        port.minimum_revision,
    )
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) struct PlanCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PlanCursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn take(&mut self, len: usize, context: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| format!("MultiInputGraphPlan: {context} offset overflow"))?;
        let value = self.data.get(self.pos..end).ok_or_else(|| {
            format!(
                "MultiInputGraphPlan: truncated {context} at byte {}",
                self.pos
            )
        })?;
        self.pos = end;
        Ok(value)
    }

    pub(crate) fn read_u8(&mut self, context: &str) -> Result<u8, String> {
        Ok(self.take(1, context)?[0])
    }

    pub(crate) fn read_u16(&mut self, context: &str) -> Result<u16, String> {
        let bytes = self.take(2, context)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn read_u32(&mut self, context: &str) -> Result<u32, String> {
        let bytes = self.take(4, context)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn read_u64(&mut self, context: &str) -> Result<u64, String> {
        let bytes = self.take(8, context)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.pos == self.data.len()
    }
}

impl MultiInputGraphPlan {
    pub(crate) fn with_graph_plan(&self, graph_plan: Vec<u8>) -> Result<Self, String> {
        let decoded = decode_graph_plan(&graph_plan)?;
        if decoded.num_slots != self.num_slots {
            return Err("mutation: slot count must preserve the input port contract".into());
        }
        let candidate = Self {
            graph_plan,
            num_slots: self.num_slots,
            ports: self.ports.clone(),
        };
        candidate.validate_for_compile()?;
        Ok(candidate)
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, String> {
        if self.ports.len() > MAX_INPUT_PORTS {
            return Err(format!(
                "MultiInputGraphPlan: port count {} exceeds {MAX_INPUT_PORTS}",
                self.ports.len()
            ));
        }
        let graph_len = u32::try_from(self.graph_plan.len())
            .map_err(|_| "MultiInputGraphPlan: graph plan exceeds u32 bytes".to_string())?;
        let port_count = u8::try_from(self.ports.len())
            .map_err(|_| "MultiInputGraphPlan: port count exceeds u8".to_string())?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PLAN_MAGIC);
        push_u32(&mut bytes, PLAN_SCHEMA_VERSION);
        push_u32(&mut bytes, graph_len);
        bytes.push(port_count);
        bytes.extend_from_slice(&self.graph_plan);
        let mut prior_slot = None;
        for port in &self.ports {
            validate_port_contract(port, self.num_slots)?;
            if prior_slot.is_some_and(|prior| prior >= port.slot) {
                return Err(
                    "MultiInputGraphPlan: input ports must be unique and sorted by slot".into(),
                );
            }
            prior_slot = Some(port.slot);
            let role_len = u8::try_from(port.role.len())
                .map_err(|_| "MultiInputGraphPlan: role exceeds u8 bytes".to_string())?;
            let layout_len = u16::try_from(port.layout.len())
                .map_err(|_| "MultiInputGraphPlan: layout exceeds u16 bytes".to_string())?;
            bytes.push(port.slot);
            bytes.push(role_len);
            bytes.extend_from_slice(port.role.as_bytes());
            for dim in port.shape {
                push_u32(&mut bytes, dim);
            }
            push_u16(&mut bytes, layout_len);
            bytes.extend_from_slice(port.layout.as_bytes());
            bytes.push(u8::from(port.require_fingerprint));
            push_u64(&mut bytes, port.minimum_revision);
        }
        Ok(bytes)
    }

    pub(crate) fn validate_for_compile(&self) -> Result<Vec<u8>, String> {
        if self.ports.len() < 2 {
            return Err(
                "MultiInputGraphPlan: at least two external input ports are required".into(),
            );
        }
        let bytes = self.encode()?;
        let slots = self.ports.iter().map(|port| port.slot).collect::<Vec<_>>();
        if !slots.contains(&0) {
            return Err("MultiInputGraphPlan: slot 0 is required for compatibility with the canonical external input".into());
        }
        let decoded = decode_graph_plan(&self.graph_plan)
            .map_err(|error| format!("MultiInputGraphPlan: {error}"))?;
        if !(1..=64).contains(&decoded.num_slots) {
            return Err(format!(
                "MultiInputGraphPlan: graph num_slots must be 1..=64, got {}",
                decoded.num_slots
            ));
        }
        for (index, step) in decoded.steps.iter().enumerate() {
            let second_input = if step.arity == 2 {
                step.in_slot2
            } else {
                step.in_slot
            };
            if [step.in_slot, second_input, step.out_slot]
                .iter()
                .any(|slot| u32::from(*slot) >= decoded.num_slots)
            {
                return Err(format!(
                    "MultiInputGraphPlan: step {index} contains a slot outside graph num_slots {}",
                    decoded.num_slots
                ));
            }
        }
        if u32::from(decoded.output_slot) >= decoded.num_slots {
            return Err(format!(
                "MultiInputGraphPlan: output slot {} is outside graph num_slots {}",
                decoded.output_slot, decoded.num_slots
            ));
        }
        let declared_mask = slots.iter().fold(0u64, |mask, slot| mask | (1u64 << slot));
        let mut written = 0u64;
        let mut external_used = 0u64;
        for (index, step) in decoded.steps.iter().enumerate() {
            let input_slots = if step.arity == 2 {
                [Some(step.in_slot), Some(step.in_slot2)]
            } else {
                [Some(step.in_slot), None]
            };
            for slot in input_slots.into_iter().flatten() {
                let mask = 1u64 << slot;
                if written & mask == 0 {
                    if declared_mask & mask == 0 {
                        return Err(format!(
                            "MultiInputGraphPlan: step {index} reads slot {slot} before any writer; declare it as an external input"
                        ));
                    }
                    external_used |= mask;
                }
            }
            written |= 1u64 << step.out_slot;
        }
        let output_mask = 1u64 << decoded.output_slot;
        if written & output_mask == 0 {
            if declared_mask & output_mask == 0 {
                return Err(format!(
                    "MultiInputGraphPlan: output slot {} is never written and is not declared as an external input",
                    decoded.output_slot
                ));
            }
            external_used |= output_mask;
        }
        if external_used != declared_mask {
            let unused = slots
                .iter()
                .copied()
                .filter(|slot| external_used & (1u64 << slot) == 0)
                .collect::<Vec<_>>();
            return Err(format!(
                "MultiInputGraphPlan: declared external input slots are not read before a graph write or selected as output: {unused:?}"
            ));
        }
        Ok(bytes)
    }

    pub(crate) fn graph_plan(&self) -> &[u8] {
        &self.graph_plan
    }

    pub(crate) fn ports(&self) -> &[MultiInputPortContract] {
        &self.ports
    }

    pub(crate) fn fingerprint_internal(&self) -> Result<String, String> {
        Ok(fnv1a64(self.encode()?))
    }
}

// #[wasm_bindgen] impl MultiInputGraphPlan — dipindah ke src/facade/graph.rs (Opsi C Fase 3).

// #[wasm_bindgen] impl MultiInputInputBundle — dipindah ke src/facade/graph.rs (Opsi C Fase 3).

impl MultiInputInputBundle {
    pub(crate) fn matches_plan_internal(&self, plan: &MultiInputGraphPlan) -> bool {
        let Ok(current_bytes) = plan.encode() else {
            return false;
        };
        self.plan_bytes == current_bytes
            && self.plan_fingerprint == fnv1a64(current_bytes.iter().copied())
            && self.ports.len() == plan.ports.len()
            && self
                .ports
                .iter()
                .zip(plan.ports.iter())
                .all(|(bound, declared)| bound.contract == *declared)
    }

    pub(crate) fn bound_input(&self, slot: u8) -> Option<&BoundInput> {
        self.ports
            .iter()
            .find(|port| port.contract.slot == slot)?
            .bound
            .as_ref()
    }

    pub(crate) fn input_preflight(&self, plan: &MultiInputGraphPlan) -> InputPreflight {
        // Compare exact canonical bytes; the short FNV value is correlation only.
        let plan_matches = self.matches_plan_internal(plan);
        let mut ready = plan_matches && self.ports.len() == plan.ports.len();
        let mut missing_count = 0usize;
        let mut mismatch_count = 0usize;
        let mut port_json = Vec::with_capacity(self.ports.len());
        let mut port_checks = Vec::with_capacity(self.ports.len());
        for binding in &self.ports {
            let contract = &binding.contract;
            let bound = binding.bound.as_ref();
            let present = bound.is_some();
            if !present {
                missing_count += 1;
                ready = false;
                port_checks.push((contract.slot, false));
                port_json.push(format!(
                    "{{\"slot\":{},\"status\":\"missing\",\"required\":true}}",
                    contract.slot
                ));
                continue;
            }
            let Some(actual) = bound else {
                continue;
            };
            let actual_shape = tensor_shape(&actual.tensor).unwrap_or([0; 4]);
            let role_matches = actual.role == contract.role;
            let shape_matches = actual_shape == contract.shape;
            let layout_matches = actual.layout == contract.layout;
            let revision_satisfies = actual.revision >= contract.minimum_revision;
            let fingerprint_satisfies =
                !contract.require_fingerprint || !actual.fingerprint.is_empty();
            let finite_values = actual
                .tensor
                .to_array()
                .iter()
                .all(|value| value.is_finite());
            let port_ready = role_matches
                && shape_matches
                && layout_matches
                && revision_satisfies
                && fingerprint_satisfies
                && finite_values;
            if !port_ready {
                mismatch_count += 1;
                ready = false;
            }
            port_checks.push((contract.slot, port_ready));
            port_json.push(format!(
                concat!(
                    "{{",
                    "\"slot\":{},",
                    "\"status\":\"{}\",",
                    "\"role_matches\":{},",
                    "\"shape_matches\":{},",
                    "\"layout_matches\":{},",
                    "\"revision_satisfies\":{},",
                    "\"fingerprint_satisfies\":{},",
                    "\"finite_values\":{},",
                    "\"actual_shape\":[{},{},{},{}],",
                    "\"input_value_fingerprint\":\"{}\",",
                    "\"source\":\"{}\",",
                    "\"revision\":{},",
                    "\"fingerprint\":{}",
                    "}}"
                ),
                contract.slot,
                if port_ready {
                    "ready"
                } else {
                    "contract_mismatch"
                },
                bool_json(role_matches),
                bool_json(shape_matches),
                bool_json(layout_matches),
                bool_json(revision_satisfies),
                bool_json(fingerprint_satisfies),
                bool_json(finite_values),
                actual_shape[0],
                actual_shape[1],
                actual_shape[2],
                actual_shape[3],
                actual.value_fingerprint,
                json_escape(&actual.source),
                actual.revision,
                if actual.fingerprint.is_empty() {
                    "null".to_string()
                } else {
                    format!("\"{}\"", json_escape(&actual.fingerprint))
                },
            ));
        }
        let slots = self
            .ports
            .iter()
            .map(|port| port.contract.slot)
            .collect::<Vec<_>>();
        let slots_json = slots
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.multi-input-preflight.v1\",",
                "\"plan_fingerprint\":\"{}\",",
                "\"bundle_plan_matches\":{},",
                "\"declared_input_slots\":[{}],",
                "\"bound_port_count\":{},",
                "\"missing_count\":{},",
                "\"mismatch_count\":{},",
                "\"ready\":{},",
                "\"execution_authorized\":false,",
                "\"mutation\":\"none\",",
                "\"decision_authority\":\"agent\",",
                "\"ports\":[{}]",
                "}}"
            ),
            self.plan_fingerprint,
            bool_json(plan_matches),
            slots_json,
            self.ports
                .iter()
                .filter(|port| port.bound.is_some())
                .count(),
            missing_count,
            mismatch_count,
            bool_json(ready),
            port_json.join(","),
        );
        InputPreflight {
            ready,
            json,
            port_checks,
        }
    }

    pub(crate) fn bound_inputs(&self) -> Vec<(u8, WasmTensor)> {
        self.ports
            .iter()
            .filter_map(|port| {
                port.bound
                    .as_ref()
                    .map(|bound| (port.contract.slot, bound.tensor.clone()))
            })
            .collect()
    }
}

pub(crate) fn multi_input_graph_capabilities() -> String {
    concat!(
        "{",
        "\"schema_version\":1,",
        "\"schema_id\":\"burn-research.multi-input-graph.v1\",",
        "\"plan\":\"MultiInputGraphPlan.v1\",",
        "\"compile\":\"LayerRegistry.compileMultiInputGraph\",",
        "\"input_bundle\":\"MultiInputInputBundle\",",
        "\"preflight\":\"CompiledMultiInputGraph.preflight\",",
        "\"plan_explain\":\"CompiledMultiInputGraph.explainPlan\",",
        "\"plan_explain_scope\":\"compiled_topology_and_declared_input_shapes_with_partial_static_shape_inference\",",
        "\"execution_trace\":\"CompiledMultiInputGraph.runWithTrace\",",
        "\"execution_trace_scope\":\"bounded_observation_of_one_graph_execution\",",
        "\"baseline_candidate_verification\":\"MultiInputVerificationCases\",",
        "\"baseline_candidate_verification_scope\":\"exact_test_vectors_program_and_state_bound_burn_comparison\",",
        "\"execution\":\"CompiledMultiInputGraph.run\",",
        "\"runtime_inputs\":\"all declared external slots must be bound and contract-valid\",",
        "\"dtype\":\"f32\",",
        "\"maximum_external_slots\":64,",
        "\"program_bundle_support\":\"burn-research.multi-input-program-bundle.v1\",",
        "\"execution_authorized_by_preflight\":false",
        "}"
    )
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{MultiInputGraphPlan, MultiInputInputBundle};
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::graph::CompiledMultiInputGraph;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn two_input_add() -> (LayerRegistry, AgentGraphBuilder, MultiInputGraphPlan) {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(21);
        registry.init_agent_layer(&add).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
        plan.add_input_port(
            0,
            "observation".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            2,
        )
        .unwrap();
        plan.add_input_port(
            1,
            "state".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            3,
        )
        .unwrap();
        (registry, builder, plan)
    }

    fn tensor(values: &[f32]) -> WasmTensor {
        WasmTensor::new(values, &[1, 2, 1, 1])
    }

    /// Legacy 3-pass serialization, kept as the equivalence oracle for
    /// tensor_value_fingerprint. Must stay byte-identical to the
    /// pre-optimization implementation (to_array + per-element to_bits LE).
    fn legacy_fingerprint(tensor: &WasmTensor) -> String {
        let shape = super::tensor_shape(tensor).unwrap();
        let values = tensor.to_array();
        let mut bytes = Vec::with_capacity(16 + values.len().saturating_mul(4));
        for dim in shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        super::fnv1a64(bytes)
    }

    #[test]
    fn fingerprint_one_pass_matches_legacy() {
        // Contiguous tensor.
        let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let contiguous = WasmTensor::new(&values, &[2, 3, 2, 2]);
        assert_eq!(
            super::tensor_value_fingerprint(&contiguous).unwrap(),
            legacy_fingerprint(&contiguous)
        );

        // Non-contiguous permuted view: to_data() must normalize to logical
        // row-major order so the raw bytes match the logical serialization.
        let permuted = WasmTensor {
            inner: contiguous.inner.clone().permute([0, 2, 1, 3]),
        };
        assert_eq!(
            super::tensor_value_fingerprint(&permuted).unwrap(),
            legacy_fingerprint(&permuted)
        );
        let reversed = WasmTensor {
            inner: contiguous.inner.clone().permute([3, 2, 1, 0]),
        };
        assert_eq!(
            super::tensor_value_fingerprint(&reversed).unwrap(),
            legacy_fingerprint(&reversed)
        );

        // Special values are bit-preserved by both paths (-0.0 != 0.0 in bits,
        // NaN/infinities keep their payloads).
        let special = [
            0.0,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-30,
            std::f32::consts::PI,
            -2.5,
        ];
        let special_tensor = WasmTensor::new(&special, &[2, 2, 2, 1]);
        assert_eq!(
            super::tensor_value_fingerprint(&special_tensor).unwrap(),
            legacy_fingerprint(&special_tensor)
        );
    }

    fn bind_valid_inputs(bundle: &mut MultiInputInputBundle) {
        let left = tensor(&[1.0, 2.0]);
        let right = tensor(&[3.0, 4.0]);
        bundle
            .bind_input(
                0,
                &left,
                "observation".into(),
                "feature_axis1_singleton".into(),
                "sensor-a".into(),
                2,
                "sha256:obs".into(),
            )
            .unwrap();
        bundle
            .bind_input(
                1,
                &right,
                "state".into(),
                "feature_axis1_singleton".into(),
                "memory-b".into(),
                3,
                "sha256:state".into(),
            )
            .unwrap();
    }

    #[test]
    fn trace_observes_execution_with_exact_digest_and_budget() {
        use sha2::{Digest, Sha256};
        let (registry, _, plan) = two_input_add();
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        bind_valid_inputs(&mut bundle);
        let traced = graph.run_with_trace(&registry, &bundle, 0, 1, 64).unwrap();
        let report: serde_json::Value = serde_json::from_str(&traced.report()).unwrap();
        let expected = format!(
            "sha256:{:x}",
            Sha256::digest([4.0f32.to_le_bytes(), 6.0f32.to_le_bytes()].concat())
        );
        assert_eq!(
            report["program_identity"],
            serde_json::from_str::<serde_json::Value>(&graph.program_identity()).unwrap()
        );
        assert_eq!(report["execution_status"], "completed");
        assert_eq!(report["execution_authorized"], false);
        assert_eq!(report["trace_complete"], true);
        assert_eq!(report["steps"][0]["index"], 0);
        assert_eq!(report["steps"][0]["observation"]["value_sha256"], expected);
        assert_eq!(report["terminal_output"]["value_sha256"], expected);
        assert_eq!(
            traced.output().unwrap().to_array(),
            graph.run(&registry, &bundle).unwrap().to_array()
        );
        let capped: serde_json::Value = serde_json::from_str(
            &graph
                .run_with_trace(&registry, &bundle, 0, 1, 4)
                .unwrap()
                .report(),
        )
        .unwrap();
        assert_eq!(
            capped["steps"][0]["observation"]["capture_status"],
            "tensor_budget_exceeded"
        );
        assert!(capped["steps"][0]["observation"]["value_sha256"].is_null());
        assert!(graph.run_with_trace(&registry, &bundle, 1, 1, 64).is_err());
        assert!(graph.run_with_trace(&registry, &bundle, 0, 0, 64).is_err());
    }

    #[test]
    fn trace_returns_fault_without_output_for_numerical_shape_mismatch() {
        let (registry, builder, _) = two_input_add();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
        for (slot, role, features) in [(0, "observation", 2), (1, "state", 3)] {
            plan.add_input_port(
                slot,
                role.into(),
                1,
                features,
                1,
                1,
                "feature_axis1_singleton".into(),
                false,
                0,
            )
            .unwrap();
        }
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        bundle
            .bind_input(
                0,
                &tensor(&[1.0, 2.0]),
                "observation".into(),
                "feature_axis1_singleton".into(),
                "sensor-a".into(),
                1,
                String::new(),
            )
            .unwrap();
        let right = WasmTensor::new(&[3.0, 4.0, 5.0], &[1, 3, 1, 1]);
        bundle
            .bind_input(
                1,
                &right,
                "state".into(),
                "feature_axis1_singleton".into(),
                "memory-b".into(),
                1,
                String::new(),
            )
            .unwrap();
        let traced = graph.run_with_trace(&registry, &bundle, 0, 1, 64).unwrap();
        let report: serde_json::Value = serde_json::from_str(&traced.report()).unwrap();
        assert_eq!(report["execution_status"], "failed");
        assert_eq!(report["fault_step_index"], 0);
        assert_eq!(
            report["steps"][0]["observation"]["capture_status"],
            "step_failed"
        );
        assert!(report["terminal_output"].is_null());
        assert!(traced.output().is_err());
    }

    #[test]
    fn trace_window_observes_only_selected_step_but_executes_full_graph() {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(23);
        let relu = AgentLayerSpec::relu(24);
        registry.init_agent_layer(&add).unwrap();
        registry.init_agent_layer(&relu).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.add_unary(&relu, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
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
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        for (slot, role, values) in [(0, "observation", [-5.0, 2.0]), (1, "state", [3.0, 4.0])] {
            bundle
                .bind_input(
                    slot,
                    &tensor(&values),
                    role.into(),
                    "feature_axis1_singleton".into(),
                    "test".into(),
                    0,
                    String::new(),
                )
                .unwrap();
        }
        let traced = graph.run_with_trace(&registry, &bundle, 1, 1, 64).unwrap();
        let report: serde_json::Value = serde_json::from_str(&traced.report()).unwrap();
        assert_eq!(report["completed_steps"], 2);
        assert_eq!(report["observed_steps"], 1);
        assert_eq!(report["steps"][0]["index"], 1);
        assert_eq!(report["trace_complete"], false);
        assert_eq!(traced.output().unwrap().to_array(), vec![0.0, 6.0]);
    }

    #[test]
    fn explain_plan_projects_declared_shapes_without_running_burn() {
        let (mut registry, _builder, plan) = two_input_add();
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let empty_bundle = MultiInputInputBundle::new(&plan).unwrap();
        let report: serde_json::Value =
            serde_json::from_str(&graph.explain_plan(&registry)).unwrap();

        assert_eq!(
            report["schema_id"],
            "burn-research.multi-input-plan-explain.v1"
        );
        assert_eq!(
            report["program_identity"],
            serde_json::from_str::<serde_json::Value>(&graph.program_identity()).unwrap()
        );
        assert_eq!(
            report["declared_input_ports"][1]["declared_shape"],
            serde_json::json!([1, 2, 1, 1])
        );
        assert_eq!(report["steps"][0]["input_slots"], serde_json::json!([0, 1]));
        assert_eq!(
            report["steps"][0]["output_shape"],
            serde_json::json!([1, 2, 1, 1])
        );
        assert_eq!(report["steps"][0]["output_f32_payload_bytes"], 8);
        assert_eq!(report["static_shape_status"], "complete");
        assert_eq!(report["registry_binding_current"], true);
        assert_eq!(report["burn_executed"], false);
        assert_eq!(report["execution_authorized"], false);
        assert_eq!(
            report["estimated_runtime_allocation_bytes"],
            serde_json::Value::Null
        );
        assert!(graph.run(&registry, &empty_bundle).is_err());

        assert!(registry.destroy_layer(21, crate::protocol::LAYER_BINARY));
        let stale: serde_json::Value =
            serde_json::from_str(&graph.explain_plan(&registry)).unwrap();
        assert_eq!(stale["registry_binding_current"], false);
        assert_eq!(stale["program_identity"], report["program_identity"]);
    }

    #[test]
    fn explain_plan_detects_known_shape_mismatch_that_input_preflight_cannot() {
        let (registry, builder, _) = two_input_add();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
        for (slot, role, features) in [(0, "observation", 2), (1, "state", 3)] {
            plan.add_input_port(
                slot,
                role.into(),
                1,
                features,
                1,
                1,
                "feature_axis1_singleton".into(),
                false,
                0,
            )
            .unwrap();
        }
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let report: serde_json::Value =
            serde_json::from_str(&graph.explain_plan(&registry)).unwrap();
        assert_eq!(report["static_shape_status"], "incompatible");
        assert_eq!(report["steps"][0]["reason"], "elementwise_shape_mismatch");
        assert!(report["steps"][0]["output_shape"].is_null());

        let mut bundle = MultiInputInputBundle::new(&plan).unwrap();
        bundle
            .bind_input(
                0,
                &tensor(&[1.0, 2.0]),
                "observation".into(),
                "feature_axis1_singleton".into(),
                "sensor-a".into(),
                1,
                String::new(),
            )
            .unwrap();
        let right = WasmTensor::new(&[3.0, 4.0, 5.0], &[1, 3, 1, 1]);
        bundle
            .bind_input(
                1,
                &right,
                "state".into(),
                "feature_axis1_singleton".into(),
                "memory-b".into(),
                1,
                String::new(),
            )
            .unwrap();
        assert!(graph
            .preflight(&registry, &bundle)
            .contains("\"ready\":true"));
        assert!(graph
            .run(&registry, &bundle)
            .err()
            .unwrap()
            .contains("shape mismatch"));
    }

    #[test]
    fn explain_plan_marks_unsupported_operator_as_unknown() {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(21);
        let glu = AgentLayerSpec::glu(22, 1);
        registry.init_agent_layer(&add).unwrap();
        registry.init_agent_layer(&glu).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.add_unary(&glu, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let mut plan = MultiInputGraphPlan::new(&builder).unwrap();
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
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let report: serde_json::Value =
            serde_json::from_str(&graph.explain_plan(&registry)).unwrap();
        assert_eq!(report["known_steps"], 1);
        assert_eq!(report["unknown_steps"], 1);
        assert_eq!(report["static_shape_status"], "partial");
        assert_eq!(report["steps"][1]["shape_status"], "unknown");
        assert!(report["output_shape"].is_null());
    }

    #[test]
    fn multi_input_plan_round_trip_and_add_execution() {
        let (registry, _builder, plan) = two_input_add();
        let encoded = plan.to_bytes().unwrap();
        let replay = MultiInputGraphPlan::from_bytes(&encoded).unwrap();
        assert_eq!(replay.to_bytes().unwrap(), encoded);
        assert_eq!(replay.input_slots(), vec![0, 1]);
        assert_eq!(
            replay.plan_fingerprint().unwrap(),
            plan.plan_fingerprint().unwrap()
        );

        let graph = CompiledMultiInputGraph::build(&registry, &replay).unwrap();
        let mut bundle = MultiInputInputBundle::new(&replay).unwrap();
        bind_valid_inputs(&mut bundle);
        assert!(graph
            .preflight(&registry, &bundle)
            .contains("\"ready\":true"));
        let result = graph.run(&registry, &bundle).unwrap().to_array();
        assert_eq!(result, vec![4.0, 6.0]);
        let proof = graph
            .verify_flat(&registry, &bundle, &[4.0, 6.0], 1e-6, 1e-6)
            .unwrap();
        assert!(proof.contains("\"passed\":true"));
    }

    #[test]
    fn missing_or_mismatched_external_input_fails_before_run() {
        let (registry, _builder, plan) = two_input_add();
        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        let empty = MultiInputInputBundle::new(&plan).unwrap();
        assert!(graph
            .preflight(&registry, &empty)
            .contains("\"ready\":false"));
        assert!(graph.run(&registry, &empty).is_err());

        let mut mismatch = MultiInputInputBundle::new(&plan).unwrap();
        let left = tensor(&[1.0, 2.0]);
        let right = tensor(&[3.0, 4.0]);
        mismatch
            .bind_input(
                0,
                &left,
                "state".into(),
                "feature_axis1_singleton".into(),
                "sensor-a".into(),
                1,
                String::new(),
            )
            .unwrap();
        mismatch
            .bind_input(
                1,
                &right,
                "state".into(),
                "feature_axis1_singleton".into(),
                "memory-b".into(),
                3,
                "sha256:state".into(),
            )
            .unwrap();
        let report = graph.preflight(&registry, &mismatch);
        assert!(report.contains("\"ready\":false"));
        assert!(report.contains("\"revision_satisfies\":false"));
        assert!(report.contains("\"fingerprint_satisfies\":false"));
        assert!(graph.run(&registry, &mismatch).is_err());
    }

    #[test]
    fn every_declared_port_must_correspond_to_a_read_before_write_slot() {
        let (registry, builder, mut plan) = two_input_add();
        assert!(plan.validate_for_compile().is_ok());

        let mut unused = MultiInputGraphPlan::new(&builder).unwrap();
        for slot in [0, 1, 2] {
            unused
                .add_input_port(
                    slot,
                    (if slot == 0 { "observation" } else { "state" }).into(),
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
        assert!(unused
            .validate_for_compile()
            .unwrap_err()
            .contains("not read before a graph write"));

        let graph = CompiledMultiInputGraph::build(&registry, &plan).unwrap();
        assert_eq!(graph.required_input_slots(), vec![0, 1]);
        plan.ports.clear();
        assert!(plan.validate_for_compile().is_err());
    }
}
