//! Canonical self-delimiting Math Program v4 step records.
//!
//! This module is internal foundation for variable-length Math Program plans. It does not alter
//! existing v1/v2/v3 plan bytes or the public WASM surface.

#![allow(dead_code)]

use crate::math::program_select_params::{
    SelectAxisParams, MAX_SELECT_INDICES, PARAM_SELECT_AXIS, SELECT_HEADER_BYTES,
};
use crate::math::program_shape_params::{
    FixedShapeParams, PARAM_PERMUTE_RANK4, PARAM_RESHAPE_RANK4, PARAM_SLICE_RANK4,
    SHAPE_PARAM_BYTES,
};

pub(crate) const PARAM_NONE: u8 = 0;
pub(crate) const PARAM_CLAMP: u8 = 1;
pub(crate) const PARAM_EPSILON: u8 = 2;
pub(crate) const V4_STEP_HEADER_BYTES: usize = 10;
pub(crate) const MAX_V4_PARAM_BYTES: usize = SELECT_HEADER_BYTES + MAX_SELECT_INDICES * 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct V4StepRecord {
    pub(crate) op: u8,
    pub(crate) arity: u8,
    pub(crate) in_a: u8,
    pub(crate) in_b: u8,
    pub(crate) out: u8,
    pub(crate) param_kind: u8,
    pub(crate) payload: Vec<u8>,
}

impl V4StepRecord {
    pub(crate) fn new(
        op: u8,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
        param_kind: u8,
        payload: Vec<u8>,
    ) -> Result<Self, String> {
        validate_payload(param_kind, &payload)?;
        Ok(Self {
            op,
            arity,
            in_a,
            in_b,
            out,
            param_kind,
            payload,
        })
    }

    pub(crate) fn encoded_len(&self) -> usize {
        V4_STEP_HEADER_BYTES + self.payload.len()
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.encoded_len());
        bytes.push(self.op);
        bytes.push(self.arity);
        bytes.push(self.in_a);
        bytes.push(self.in_b);
        bytes.push(self.out);
        bytes.push(self.param_kind);
        bytes.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    pub(crate) fn decode_prefix(bytes: &[u8]) -> Result<(Self, usize), String> {
        if bytes.len() < V4_STEP_HEADER_BYTES {
            return Err(format!(
                "MathProgram v4 step: truncated header: expected at least {V4_STEP_HEADER_BYTES} bytes, got {}",
                bytes.len()
            ));
        }

        let payload_len = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if payload_len > MAX_V4_PARAM_BYTES {
            return Err(format!(
                "MathProgram v4 step: payload length {payload_len} exceeds maximum {MAX_V4_PARAM_BYTES}"
            ));
        }
        let record_len = V4_STEP_HEADER_BYTES
            .checked_add(payload_len)
            .ok_or_else(|| "MathProgram v4 step: record length overflow".to_string())?;
        if bytes.len() < record_len {
            return Err(format!(
                "MathProgram v4 step: truncated payload: record needs {record_len} bytes, got {}",
                bytes.len()
            ));
        }

        let record = Self::new(
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3],
            bytes[4],
            bytes[5],
            bytes[V4_STEP_HEADER_BYTES..record_len].to_vec(),
        )?;
        if record.encode().as_slice() != &bytes[..record_len] {
            return Err("MathProgram v4 step: noncanonical record encoding".into());
        }
        Ok((record, record_len))
    }

    pub(crate) fn decode_exact(bytes: &[u8]) -> Result<Self, String> {
        let (record, consumed) = Self::decode_prefix(bytes)?;
        if consumed != bytes.len() {
            return Err(format!(
                "MathProgram v4 step: trailing bytes after record: consumed {consumed}, got {}",
                bytes.len()
            ));
        }
        Ok(record)
    }
}

fn validate_payload(kind: u8, payload: &[u8]) -> Result<(), String> {
    if payload.len() > MAX_V4_PARAM_BYTES {
        return Err(format!(
            "MathProgram v4 step: payload length {} exceeds maximum {MAX_V4_PARAM_BYTES}",
            payload.len()
        ));
    }
    match kind {
        PARAM_NONE => {
            if !payload.is_empty() {
                return Err("MathProgram v4 step: plain parameter kind requires empty payload".into());
            }
        }
        PARAM_CLAMP | PARAM_EPSILON => {
            if payload.len() != 8 {
                return Err(format!(
                    "MathProgram v4 step: scalar parameter kind {kind} requires exactly 8 bytes, got {}",
                    payload.len()
                ));
            }
        }
        PARAM_RESHAPE_RANK4 | PARAM_PERMUTE_RANK4 | PARAM_SLICE_RANK4 => {
            if payload.len() != SHAPE_PARAM_BYTES {
                return Err(format!(
                    "MathProgram v4 step: fixed rank-4 kind {kind} requires {SHAPE_PARAM_BYTES} bytes, got {}",
                    payload.len()
                ));
            }
            FixedShapeParams::decode(kind, payload)?;
        }
        PARAM_SELECT_AXIS => {
            SelectAxisParams::decode(payload)?;
        }
        _ => {
            return Err(format!(
                "MathProgram v4 step: unknown parameter kind {kind}; fail-closed"
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: u8, payload: Vec<u8>) -> V4StepRecord {
        V4StepRecord::new(0x20, 1, 0, 0, 1, kind, payload).unwrap()
    }

    #[test]
    fn plain_record_is_self_delimiting_and_exact() {
        let value = record(PARAM_NONE, vec![]);
        let encoded = value.encode();
        assert_eq!(encoded.len(), V4_STEP_HEADER_BYTES);
        assert_eq!(V4StepRecord::decode_exact(&encoded).unwrap(), value);

        let mut with_next = encoded.clone();
        with_next.extend_from_slice(&[9, 8, 7]);
        let (decoded, consumed) = V4StepRecord::decode_prefix(&with_next).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(consumed, encoded.len());
        assert!(V4StepRecord::decode_exact(&with_next).is_err());
    }

    #[test]
    fn scalar_payload_requires_exact_eight_bytes() {
        let payload = [1.0f32.to_bits().to_le_bytes(), 2.0f32.to_bits().to_le_bytes()].concat();
        let value = record(PARAM_CLAMP, payload);
        assert_eq!(V4StepRecord::decode_exact(&value.encode()).unwrap(), value);
        assert!(V4StepRecord::new(1, 1, 0, 0, 1, PARAM_CLAMP, vec![0; 4]).is_err());
        assert!(V4StepRecord::new(1, 1, 0, 0, 1, PARAM_EPSILON, vec![0; 9]).is_err());
    }

    #[test]
    fn fixed_shape_payload_reuses_proven_codec() {
        let reshape = FixedShapeParams::reshape(&[1, 2, 3, 4]).unwrap();
        let value = record(PARAM_RESHAPE_RANK4, reshape.encode().to_vec());
        assert_eq!(V4StepRecord::decode_exact(&value.encode()).unwrap(), value);

        let mut noncanonical = reshape.encode().to_vec();
        noncanonical[16] = 1;
        assert!(V4StepRecord::new(1, 1, 0, 0, 1, PARAM_RESHAPE_RANK4, noncanonical).is_err());
    }

    #[test]
    fn select_payload_reuses_bounded_variable_length_codec() {
        let select = SelectAxisParams::new(1, &[3, 1, 3, 0]).unwrap();
        let value = record(PARAM_SELECT_AXIS, select.encode());
        let encoded = value.encode();
        assert_eq!(V4StepRecord::decode_exact(&encoded).unwrap(), value);
        assert_eq!(u32::from_le_bytes([encoded[6], encoded[7], encoded[8], encoded[9]]) as usize, value.payload.len());
    }

    #[test]
    fn malformed_lengths_unknown_kinds_and_trailing_bytes_fail_closed() {
        let value = record(PARAM_NONE, vec![]);
        let mut truncated = value.encode();
        truncated.truncate(V4_STEP_HEADER_BYTES - 1);
        assert!(V4StepRecord::decode_exact(&truncated).is_err());

        let select = SelectAxisParams::new(0, &[0, 1]).unwrap();
        let mut malformed = record(PARAM_SELECT_AXIS, select.encode()).encode();
        malformed[6..10].copy_from_slice(&100u32.to_le_bytes());
        assert!(V4StepRecord::decode_exact(&malformed).is_err());

        assert!(V4StepRecord::new(1, 1, 0, 0, 1, 0xFF, vec![]).is_err());
    }

    #[test]
    fn semantic_payload_change_changes_record_bytes() {
        let a = record(PARAM_SELECT_AXIS, SelectAxisParams::new(1, &[0, 2, 1]).unwrap().encode());
        let b = record(PARAM_SELECT_AXIS, SelectAxisParams::new(1, &[0, 2, 1]).unwrap().encode());
        let c = record(PARAM_SELECT_AXIS, SelectAxisParams::new(1, &[2, 0, 1]).unwrap().encode());
        assert_eq!(a.encode(), b.encode());
        assert_ne!(a.encode(), c.encode());
    }
}
