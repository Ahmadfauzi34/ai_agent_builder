use std::collections::HashSet;

use crate::protocol::PayloadCursor;

const PLAN_HEADER_BYTES: usize = 8; // num_steps:u32 + num_slots:u32
const PLAN_STEP_BYTES: usize = 9;
const PLAN_OUTPUT_BYTES: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphPlanHeader {
    pub(crate) num_steps: u32,
    pub(crate) num_slots: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphPlanStep {
    pub(crate) arity: u8,
    pub(crate) layer_type: u8,
    pub(crate) layer_id: u32,
    pub(crate) in_slot: u8,
    pub(crate) in_slot2: u8,
    pub(crate) out_slot: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecodedGraphPlan {
    pub(crate) num_steps: u32,
    pub(crate) num_slots: u32,
    pub(crate) steps: Vec<GraphPlanStep>,
    pub(crate) output_slot: u8,
}

impl DecodedGraphPlan {
    pub(crate) fn unique_first_use_layer_keys(&self) -> Vec<(u8, u32)> {
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        for step in &self.steps {
            let key = (step.layer_type, step.layer_id);
            if seen.insert(key) {
                keys.push(key);
            }
        }
        keys
    }
}

fn expected_plan_len(num_steps: u32) -> Result<usize, String> {
    (num_steps as usize)
        .checked_mul(PLAN_STEP_BYTES)
        .and_then(|steps| PLAN_HEADER_BYTES.checked_add(steps))
        .and_then(|bytes| bytes.checked_add(PLAN_OUTPUT_BYTES))
        .ok_or_else(|| "plan length overflow".to_string())
}

pub(crate) fn decode_graph_plan_header(plan: &[u8]) -> Result<GraphPlanHeader, String> {
    let mut cursor = PayloadCursor::new(plan);
    Ok(GraphPlanHeader {
        num_steps: cursor.read_u32()?,
        num_slots: cursor.read_u32()?,
    })
}

/// Decode only the frozen CompiledGraph byte representation.
///
/// This intentionally does not validate execution semantics such as:
/// - whether a graph may contain zero steps;
/// - the supported slot-count profile;
/// - unary/binary layer compatibility;
/// - slot readiness / write order;
/// - referenced layer existence or registry identity.
///
/// Those policies remain owned by their existing consumers.
pub(crate) fn decode_graph_plan(plan: &[u8]) -> Result<DecodedGraphPlan, String> {
    let header = decode_graph_plan_header(plan)?;
    let expected_len = expected_plan_len(header.num_steps)?;
    if plan.len() != expected_len {
        return Err(format!(
            "malformed plan length: expected {} bytes for {} steps, got {}",
            expected_len,
            header.num_steps,
            plan.len()
        ));
    }

    let mut cursor = PayloadCursor::new(plan);
    let num_steps = cursor.read_u32()?;
    let num_slots = cursor.read_u32()?;
    debug_assert_eq!(num_steps, header.num_steps);
    debug_assert_eq!(num_slots, header.num_slots);

    let mut steps = Vec::with_capacity(num_steps as usize);
    for _ in 0..num_steps {
        steps.push(GraphPlanStep {
            arity: cursor.read_u8()?,
            layer_type: cursor.read_u8()?,
            layer_id: cursor.read_u32()?,
            in_slot: cursor.read_u8()?,
            in_slot2: cursor.read_u8()?,
            out_slot: cursor.read_u8()?,
        });
    }
    let output_slot = cursor.read_u8()?;

    Ok(DecodedGraphPlan {
        num_steps,
        num_slots,
        steps,
        output_slot,
    })
}

#[cfg(test)]
mod tests {
    use super::{decode_graph_plan, decode_graph_plan_header};

    #[test]
    fn decodes_frozen_graph_plan_fields_without_semantic_policy() {
        let plan = vec![
            1, 0, 0, 0, // num_steps = 1
            2, 0, 0, 0, // num_slots = 2
            1, // arity
            4, // layer_type
            7, 0, 0, 0, // layer_id
            0, // in_slot
            0, // in_slot2
            1, // out_slot
            1, // output_slot
        ];

        let header = decode_graph_plan_header(&plan).unwrap();
        assert_eq!(header.num_steps, 1);
        assert_eq!(header.num_slots, 2);

        let decoded = decode_graph_plan(&plan).unwrap();
        assert_eq!(decoded.num_steps, 1);
        assert_eq!(decoded.num_slots, 2);
        assert_eq!(decoded.output_slot, 1);
        assert_eq!(decoded.steps.len(), 1);
        assert_eq!(decoded.steps[0].arity, 1);
        assert_eq!(decoded.steps[0].layer_type, 4);
        assert_eq!(decoded.steps[0].layer_id, 7);
        assert_eq!(decoded.steps[0].in_slot, 0);
        assert_eq!(decoded.steps[0].in_slot2, 0);
        assert_eq!(decoded.steps[0].out_slot, 1);
        assert_eq!(decoded.unique_first_use_layer_keys(), vec![(4, 7)]);
    }

    #[test]
    fn exact_envelope_rejects_trailing_and_truncated_bytes() {
        let valid = vec![
            1, 0, 0, 0,
            2, 0, 0, 0,
            1, 4, 7, 0, 0, 0, 0, 0, 1,
            1,
        ];

        let mut trailing = valid.clone();
        trailing.push(0xFF);
        assert!(
            decode_graph_plan(&trailing)
                .unwrap_err()
                .contains("malformed plan length")
        );

        let mut truncated = valid;
        truncated.pop();
        assert!(
            decode_graph_plan(&truncated)
                .unwrap_err()
                .contains("malformed plan length")
        );
    }
}
