//! Canonical bounded variable-length metadata codec for Math Program `selectAxis`.
//!
//! This is intentionally internal until a variable-length Math Program plan version is wired.
//! The payload uses only fixed-width integer fields so replay does not depend on host `usize`.

#![allow(dead_code)]

pub(crate) const PARAM_SELECT_AXIS: u8 = 6;
pub(crate) const SELECT_HEADER_BYTES: usize = 8;
pub(crate) const MAX_SELECT_INDICES: usize = u16::MAX as usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SelectAxisParams {
    axis: u8,
    indices: Vec<u32>,
}

impl SelectAxisParams {
    pub(crate) fn new(axis: u32, indices: &[u32]) -> Result<Self, String> {
        if axis >= 4 {
            return Err(format!(
                "MathProgram.selectAxis: axis must be in 0..4, got {axis}"
            ));
        }
        if indices.is_empty() {
            return Err("MathProgram.selectAxis: indices must not be empty".into());
        }
        if indices.len() > MAX_SELECT_INDICES {
            return Err(format!(
                "MathProgram.selectAxis: index count {} exceeds maximum {MAX_SELECT_INDICES}",
                indices.len()
            ));
        }
        Ok(Self {
            axis: axis as u8,
            indices: indices.to_vec(),
        })
    }

    pub(crate) fn kind(&self) -> u8 {
        PARAM_SELECT_AXIS
    }

    pub(crate) fn axis(&self) -> u8 {
        self.axis
    }

    pub(crate) fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub(crate) fn axis_usize(&self) -> usize {
        self.axis as usize
    }

    pub(crate) fn indices_usize(&self) -> Vec<usize> {
        self.indices.iter().map(|&index| index as usize).collect()
    }

    pub(crate) fn encoded_len(&self) -> usize {
        SELECT_HEADER_BYTES + self.indices.len() * 4
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(self.encoded_len());
        payload.push(self.axis);
        payload.extend_from_slice(&[0, 0, 0]);
        payload.extend_from_slice(&(self.indices.len() as u32).to_le_bytes());
        for index in &self.indices {
            payload.extend_from_slice(&index.to_le_bytes());
        }
        payload
    }

    pub(crate) fn decode(payload: &[u8]) -> Result<Self, String> {
        if payload.len() < SELECT_HEADER_BYTES {
            return Err(format!(
                "MathProgram selectAxis parameters: payload is truncated: expected at least {SELECT_HEADER_BYTES} bytes, got {}",
                payload.len()
            ));
        }
        if payload[1..4].iter().any(|byte| *byte != 0) {
            return Err(
                "MathProgram selectAxis parameters: reserved header bytes must be zero".into(),
            );
        }

        let axis = payload[0] as u32;
        let count = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]])
            as usize;
        if count == 0 {
            return Err("MathProgram selectAxis parameters: index count must be non-zero".into());
        }
        if count > MAX_SELECT_INDICES {
            return Err(format!(
                "MathProgram selectAxis parameters: index count {count} exceeds maximum {MAX_SELECT_INDICES}"
            ));
        }
        let expected_len = SELECT_HEADER_BYTES
            .checked_add(
                count
                    .checked_mul(4)
                    .ok_or_else(|| {
                        "MathProgram selectAxis parameters: payload length overflow".to_string()
                    })?,
            )
            .ok_or_else(|| {
                "MathProgram selectAxis parameters: payload length overflow".to_string()
            })?;
        if payload.len() != expected_len {
            return Err(format!(
                "MathProgram selectAxis parameters: malformed payload length: expected {expected_len}, got {}",
                payload.len()
            ));
        }

        let mut indices = Vec::with_capacity(count);
        let mut offset = SELECT_HEADER_BYTES;
        for _ in 0..count {
            indices.push(u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]));
            offset += 4;
        }

        let decoded = Self::new(axis, &indices)?;
        if decoded.encode() != payload {
            return Err("MathProgram selectAxis parameters: noncanonical payload".into());
        }
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_order_and_duplicates() {
        let params = SelectAxisParams::new(1, &[3, 1, 3, 0]).unwrap();
        let encoded = params.encode();
        assert_eq!(params.kind(), PARAM_SELECT_AXIS);
        assert_eq!(params.axis(), 1);
        assert_eq!(params.indices(), &[3, 1, 3, 0]);
        assert_eq!(params.encoded_len(), SELECT_HEADER_BYTES + 16);
        assert_eq!(&encoded[4..8], &4u32.to_le_bytes());
        assert_eq!(&encoded[8..12], &3u32.to_le_bytes());
        assert_eq!(SelectAxisParams::decode(&encoded).unwrap(), params);
        assert_eq!(params.axis_usize(), 1);
        assert_eq!(params.indices_usize(), vec![3, 1, 3, 0]);
    }

    #[test]
    fn constructor_rejects_invalid_axis_empty_or_unbounded_count() {
        assert!(SelectAxisParams::new(4, &[0]).is_err());
        assert!(SelectAxisParams::new(0, &[]).is_err());
        let too_many = vec![0u32; MAX_SELECT_INDICES + 1];
        assert!(SelectAxisParams::new(0, &too_many).is_err());
    }

    #[test]
    fn decoder_rejects_reserved_trailing_truncated_and_count_mismatch() {
        let params = SelectAxisParams::new(2, &[0, 4]).unwrap();
        let encoded = params.encode();

        let mut reserved = encoded.clone();
        reserved[1] = 1;
        assert!(SelectAxisParams::decode(&reserved).is_err());

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(SelectAxisParams::decode(&trailing).is_err());

        assert!(SelectAxisParams::decode(&encoded[..7]).is_err());

        let mut wrong_count = encoded.clone();
        wrong_count[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(SelectAxisParams::decode(&wrong_count).is_err());
    }

    #[test]
    fn decoder_rejects_invalid_axis_and_zero_count() {
        let mut invalid_axis = SelectAxisParams::new(3, &[0]).unwrap().encode();
        invalid_axis[0] = 4;
        assert!(SelectAxisParams::decode(&invalid_axis).is_err());

        let mut zero_count = SelectAxisParams::new(0, &[0]).unwrap().encode();
        zero_count[4..8].copy_from_slice(&0u32.to_le_bytes());
        zero_count.truncate(SELECT_HEADER_BYTES);
        assert!(SelectAxisParams::decode(&zero_count).is_err());
    }

    #[test]
    fn semantic_changes_change_canonical_bytes() {
        let base = SelectAxisParams::new(1, &[0, 2, 1]).unwrap();
        let same = SelectAxisParams::new(1, &[0, 2, 1]).unwrap();
        let different_axis = SelectAxisParams::new(2, &[0, 2, 1]).unwrap();
        let different_order = SelectAxisParams::new(1, &[2, 0, 1]).unwrap();
        assert_eq!(base.encode(), same.encode());
        assert_ne!(base.encode(), different_axis.encode());
        assert_ne!(base.encode(), different_order.encode());
    }
}
