//! Bounded observations of the same interpreter used by CompiledGraph.run.
//! A trace records tensor facts after each selected Burn step, never tensor values.

use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;

use super::{bytes_hex, CompiledMultiInputGraph};
use crate::graph_plan::GraphPlanStep;
use crate::multi_input_graph::MultiInputInputBundle;
use crate::registry::LayerRegistry;
use crate::WasmTensor;

const MAX_TRACE_STEPS: u32 = 256;
const MAX_TENSOR_BYTES: u32 = 16 * 1024 * 1024;
const MAX_TOTAL_HASHED_BYTES: u64 = 64 * 1024 * 1024;

#[wasm_bindgen]
pub struct TracedMultiInputRun {
    output: Option<WasmTensor>,
    report: String,
}

#[wasm_bindgen]
impl TracedMultiInputRun {
    /// A failed numerical step has no output and never issues an execution receipt.
    pub fn output(&self) -> Result<WasmTensor, String> {
        self.output.clone().ok_or_else(|| "TracedMultiInputRun: execution did not complete".into())
    }

    pub fn report(&self) -> String {
        self.report.clone()
    }
}

fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control.is_control() => out.push_str(&format!("\\u{:04x}", control as u32)),
            ordinary => out.push(ordinary),
        }
    }
    out
}

fn shape_json(shape: [usize; 4]) -> String {
    format!("[{},{},{},{}]", shape[0], shape[1], shape[2], shape[3])
}

struct TensorObservation {
    json: String,
}

struct TraceCollector {
    start: usize,
    end: usize,
    max_tensor_bytes: u64,
    remaining_bytes: u64,
    observations: Vec<String>,
    completed_steps: usize,
    fault: Option<(usize, String)>,
}

impl TraceCollector {
    fn new(total_steps: usize, start: u32, count: u32, max_tensor_bytes: u32) -> Result<Self, String> {
        let start = start as usize;
        if start >= total_steps || count == 0 || count > MAX_TRACE_STEPS || max_tensor_bytes > MAX_TENSOR_BYTES {
            return Err(format!(
                "runWithTrace: require start < {total_steps}, steps in 1..={MAX_TRACE_STEPS}, and max tensor bytes <= {MAX_TENSOR_BYTES}"
            ));
        }
        Ok(Self {
            start,
            end: start.saturating_add(count as usize).min(total_steps),
            max_tensor_bytes: u64::from(max_tensor_bytes),
            remaining_bytes: MAX_TOTAL_HASHED_BYTES,
            observations: Vec::with_capacity(count as usize),
            completed_steps: 0,
            fault: None,
        })
    }

    fn observe_tensor(&mut self, tensor: &WasmTensor) -> TensorObservation {
        let dims = tensor.inner.dims();
        let shape = shape_json(dims);
        let byte_length = dims.into_iter().try_fold(4u64, |total, dim| {
            u64::try_from(dim).ok().and_then(|n| total.checked_mul(n))
        });
        let Some(bytes) = byte_length else {
            return TensorObservation { json: format!(
                "{{\"shape\":{shape},\"byte_length\":null,\"capture_status\":\"size_overflow\",\"finite_values\":null,\"value_sha256\":null}}"
            ) };
        };
        let reason = if bytes > self.max_tensor_bytes { Some("tensor_budget_exceeded") }
            else if bytes > self.remaining_bytes { Some("total_budget_exceeded") }
            else { None };
        if let Some(reason) = reason {
            return TensorObservation { json: format!(
                "{{\"shape\":{shape},\"byte_length\":{bytes},\"capture_status\":\"{reason}\",\"finite_values\":null,\"value_sha256\":null}}"
            ) };
        }
        let values = tensor.to_array();
        let mut hash = Sha256::new();
        let mut finite = true;
        for value in values {
            finite &= value.is_finite();
            hash.update(value.to_bits().to_le_bytes());
        }
        self.remaining_bytes -= bytes;
        TensorObservation { json: format!(
            "{{\"shape\":{shape},\"byte_length\":{bytes},\"capture_status\":\"captured\",\"finite_values\":{finite},\"value_sha256\":\"sha256:{}\"}}",
            bytes_hex(&hash.finalize())
        ) }
    }

    fn observe_step(&mut self, index: usize, step: &GraphPlanStep, result: &Result<WasmTensor, String>) {
        if let Err(error) = result {
            self.fault = Some((index, error.clone()));
        } else {
            self.completed_steps += 1;
        }
        if !(self.start..self.end).contains(&index) { return; }
        let result_json = match result {
            Ok(output) => self.observe_tensor(output).json,
            Err(error) => format!("{{\"capture_status\":\"step_failed\",\"error\":\"{}\"}}", escape_json(error)),
        };
        self.observations.push(format!(
            "{{\"index\":{index},\"layer_type\":{},\"layer_id\":{},\"arity\":{},\"input_slots\":[{}],\"output_slot\":{},\"observation\":{result_json}}}",
            step.layer_type, step.layer_id, step.arity,
            if step.arity == 2 { format!("{},{}", step.in_slot, step.in_slot2) } else { step.in_slot.to_string() },
            step.out_slot,
        ));
    }

    fn report(&mut self, graph: &CompiledMultiInputGraph, output: Option<&WasmTensor>, inputs: String) -> String {
        let terminal = output.map_or_else(|| "null".to_string(), |tensor| self.observe_tensor(tensor).json);
        let fault_index = self.fault.as_ref().map_or_else(|| "null".to_string(), |(step, _)| step.to_string());
        let error = self.fault.as_ref().map_or_else(|| "null".to_string(), |(_, message)| format!("\"{}\"", escape_json(message)));
        let completed = output.is_some();
        let total_steps = graph.graph.steps.len();
        let trace_complete = completed && self.start == 0 && self.end == total_steps
            && self.observations.len() == total_steps;
        let used_bytes = MAX_TOTAL_HASHED_BYTES - self.remaining_bytes;
        format!(concat!(
            "{{\"schema_version\":1,\"schema_id\":\"burn-research.multi-input-execution-trace.v1\",",
            "\"program_identity\":{},\"execution_status\":\"{}\",\"execution_authorized\":false,",
            "\"input_observations\":[{}],\"start_step\":{},\"end_step_exclusive\":{},",
            "\"total_steps\":{},\"completed_steps\":{},\"observed_steps\":{},\"trace_complete\":{},",
            "\"max_tensor_bytes\":{},\"max_total_hashed_bytes\":{},\"hashed_bytes\":{},",
            "\"steps\":[{}],\"terminal_output\":{},\"fault_step_index\":{},\"error\":{}}}"
        ), graph.program_identity_json(), if completed { "completed" } else { "failed" },
            inputs, self.start, self.end, total_steps, self.completed_steps, self.observations.len(),
            trace_complete, self.max_tensor_bytes, MAX_TOTAL_HASHED_BYTES, used_bytes,
            self.observations.join(","), terminal, fault_index, error)
    }
}

pub(super) fn run(
    graph: &CompiledMultiInputGraph,
    registry: &LayerRegistry,
    bundle: &MultiInputInputBundle,
    start_step: u32,
    max_steps: u32,
    max_tensor_bytes: u32,
) -> Result<TracedMultiInputRun, String> {
    // Validate observation bounds before any numerical call.
    let mut collector = TraceCollector::new(graph.graph.steps.len(), start_step, max_steps, max_tensor_bytes)?;
    let bound_inputs = bundle.bound_inputs();
    let mut inputs = Vec::with_capacity(bound_inputs.len());
    for (slot, tensor) in &bound_inputs {
        let observation = collector.observe_tensor(tensor);
        inputs.push(format!("{{\"slot\":{slot},\"observation\":{}}}", observation.json));
    }
    let outcome = graph.graph.run_with_external_inputs_observed(registry, &bound_inputs,
        |index, step, result| collector.observe_step(index, step, result));
    if collector.fault.is_none() {
        if let Err(error) = &outcome {
            // A terminal slot fault is possible even when all steps succeeded.
            collector.fault = Some((graph.graph.steps.len(), error.clone()));
        }
    }
    let report = collector.report(graph, outcome.as_ref().ok(), inputs.join(","));
    Ok(TracedMultiInputRun { output: outcome.ok(), report })
}
