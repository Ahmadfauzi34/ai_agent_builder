//! Read-only explanation of a compiled multi-input plan. Shape projections are
//! deliberately partial: execution and operator validity still belong to Burn.

use super::{runtime_contract, CompiledMultiInputGraph};
use crate::graph_plan::GraphPlanStep;
use crate::protocol::{
    PayloadCursor, ACT_GLU, ACT_SWIGLU, BINARY_ADD, BINARY_CONCAT, BINARY_MATMUL,
    BINARY_MUL, BINARY_SUB, LAYER_ACTIVATION, LAYER_BINARY, LAYER_FEATURE_NORM,
    LAYER_LINEAR, LAYER_NORM, LAYER_SHIFT,
};
use crate::registry::LayerRegistry;

type Shape = [u32; 4];

struct Projection {
    shape: Option<Shape>,
    status: &'static str,
    reason: &'static str,
}

impl Projection {
    fn known(shape: Shape) -> Self {
        Self { shape: Some(shape), status: "known", reason: "none" }
    }

    fn unknown(reason: &'static str) -> Self {
        Self { shape: None, status: "unknown", reason }
    }

    fn incompatible(reason: &'static str) -> Self {
        Self { shape: None, status: "incompatible", reason }
    }
}

fn shape_json(shape: Option<Shape>) -> String {
    match shape {
        Some([b, c, h, w]) => format!("[{b},{c},{h},{w}]"),
        None => "null".to_string(),
    }
}

fn payload_bytes(shape: Option<Shape>) -> Option<u64> {
    shape?.into_iter().try_fold(4u64, |total, dimension| {
        total.checked_mul(u64::from(dimension))
    })
}

fn infer_shape(step: &GraphPlanStep, identity: &str, inputs: &[Option<Shape>]) -> Projection {
    let Some(input) = inputs.first().copied().flatten() else {
        return Projection::unknown("upstream_shape_unknown");
    };
    let Ok((variant, payload)) = runtime_contract::parse_init_fingerprint(identity) else {
        return Projection::unknown("init_identity_unreadable");
    };

    match step.layer_type {
        LAYER_BINARY => {
            let Some(other) = inputs.get(1).copied().flatten() else {
                return Projection::unknown("upstream_shape_unknown");
            };
            match variant {
                BINARY_ADD | BINARY_SUB | BINARY_MUL => {
                    if input == other { Projection::known(input) }
                    else { Projection::incompatible("elementwise_shape_mismatch") }
                }
                BINARY_MATMUL => {
                    if input[0] == other[0] && input[1] == other[1] && input[3] == other[2] {
                        Projection::known([input[0], input[1], input[2], other[3]])
                    } else {
                        Projection::incompatible("matmul_shape_mismatch")
                    }
                }
                BINARY_CONCAT => {
                    let mut cursor = PayloadCursor::new(&payload);
                    let dim = cursor.read_u32().and_then(|_| cursor.read_u32());
                    let Ok(dim) = dim else {
                        return Projection::unknown("concat_axis_unreadable");
                    };
                    let Ok(dim) = usize::try_from(dim) else {
                        return Projection::unknown("concat_axis_unreadable");
                    };
                    if dim >= 4 {
                        return Projection::incompatible("concat_axis_out_of_range");
                    }
                    if (0..4).any(|axis| axis != dim && input[axis] != other[axis]) {
                        return Projection::incompatible("concat_shape_mismatch");
                    }
                    let Some(extent) = input[dim].checked_add(other[dim]) else {
                        return Projection::unknown("concat_extent_exceeds_u32");
                    };
                    let mut output = input;
                    output[dim] = extent;
                    Projection::known(output)
                }
                _ => Projection::unknown("operator_shape_rule_unavailable"),
            }
        }
        LAYER_LINEAR => {
            let mut cursor = PayloadCursor::new(&payload);
            let dims = cursor.read_u32().and_then(|_| {
                Ok((cursor.read_u32()?, cursor.read_u32()?))
            });
            let Ok((in_dim, out_dim)) = dims else {
                return Projection::unknown("linear_dimensions_unreadable");
            };
            if input[1] != in_dim || input[2] != 1 || input[3] != 1 {
                Projection::incompatible("linear_input_shape_mismatch")
            } else {
                Projection::known([input[0], out_dim, 1, 1])
            }
        }
        LAYER_ACTIVATION if variant != ACT_GLU && variant != ACT_SWIGLU => {
            Projection::known(input)
        }
        LAYER_SHIFT | LAYER_NORM => Projection::known(input),
        LAYER_FEATURE_NORM => {
            if input[1] == 0 || input[2] != 1 || input[3] != 1 {
                Projection::incompatible("feature_norm_input_shape_mismatch")
            } else {
                Projection::known(input)
            }
        }
        _ => Projection::unknown("operator_shape_rule_unavailable"),
    }
}

pub(super) fn report(compiled: &CompiledMultiInputGraph, registry: &LayerRegistry) -> String {
    let mut slots = vec![None; compiled.graph.num_slots as usize];
    let input_ports = compiled.plan.ports().iter().map(|port| {
        slots[port.slot as usize] = Some(port.shape);
        format!("{{\"slot\":{},\"declared_shape\":{}}}", port.slot, shape_json(Some(port.shape)))
    }).collect::<Vec<_>>().join(",");

    let mut known_steps = 0usize;
    let mut unknown_steps = 0usize;
    let mut incompatible_steps = 0usize;
    let steps = compiled.graph.steps.iter().zip(&compiled.graph.init_fingerprints)
        .enumerate().map(|(index, (step, identity))| {
            let input_slots = if step.arity == 2 {
                vec![step.in_slot, step.in_slot2]
            } else {
                vec![step.in_slot]
            };
            let shapes = input_slots.iter().map(|slot| slots[*slot as usize]).collect::<Vec<_>>();
            let projection = infer_shape(step, identity, &shapes);
            match projection.status {
                "known" => known_steps += 1,
                "incompatible" => incompatible_steps += 1,
                _ => unknown_steps += 1,
            }
            slots[step.out_slot as usize] = projection.shape;
            let input_slots = input_slots.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            let input_shapes = shapes.into_iter().map(shape_json).collect::<Vec<_>>().join(",");
            let output_bytes = payload_bytes(projection.shape).map_or_else(|| "null".to_string(), |n| n.to_string());
            format!(concat!(
                "{{\"index\":{},\"layer_type\":{},\"layer_id\":{},\"arity\":{},",
                "\"input_slots\":[{}],\"output_slot\":{},\"input_shapes\":[{}],",
                "\"output_shape\":{},\"shape_status\":\"{}\",\"reason\":\"{}\",",
                "\"output_f32_payload_bytes\":{}}}"
            ), index, step.layer_type, step.layer_id, step.arity, input_slots,
                step.out_slot, input_shapes, shape_json(projection.shape),
                projection.status, projection.reason, output_bytes)
        }).collect::<Vec<_>>().join(",");

    let status = if incompatible_steps > 0 { "incompatible" }
        else if unknown_steps > 0 { "partial" } else { "complete" };
    let binding_current = compiled.graph
        .validate_registry_binding_internal(registry, "explainPlan").is_ok();
    format!(concat!(
        "{{\"schema_version\":1,\"schema_id\":\"burn-research.multi-input-plan-explain.v1\",",
        "\"program_identity\":{},\"registry_binding_current\":{},",
        "\"declared_input_ports\":[{}],\"steps\":[{}],\"output_slot\":{},",
        "\"output_shape\":{},\"static_shape_status\":\"{}\",",
        "\"known_steps\":{},\"unknown_steps\":{},\"incompatible_steps\":{},",
        "\"estimated_runtime_allocation_bytes\":null,\"burn_executed\":false,",
        "\"execution_authorized\":false}}"
    ), compiled.program_identity_json(), if binding_current { "true" } else { "false" },
        input_ports, steps, compiled.graph.out_slot,
        shape_json(slots[compiled.graph.out_slot as usize]), status,
        known_steps, unknown_steps, incompatible_steps)
}
