//! Shared structural codec for self-delimiting Math Program step records.
//!
//! This module owns only the byte envelope used by Math Program v7-v9. It does
//! not know which opcodes/parameter kinds are legal for a version and does not
//! perform topology or execution validation.

pub(crate) const STEP_HEADER_BYTES: usize = 10;
pub(crate) const MAX_PARAM_BYTES: usize = 8 + (u16::MAX as usize) * 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProgramStepRecord {
    pub(crate) op: u8,
    pub(crate) arity: u8,
    pub(crate) in_a: u8,
    pub(crate) in_b: u8,
    pub(crate) out: u8,
    pub(crate) param_kind: u8,
    pub(crate) payload: Vec<u8>,
}

impl ProgramStepRecord {
    pub(crate) fn new(
        op: u8,
        arity: u8,
        in_a: u8,
        in_b: u8,
        out: u8,
        param_kind: u8,
        payload: Vec<u8>,
        context: &str,
    ) -> Result<Self, String> {
        if payload.len() > MAX_PARAM_BYTES {
            return Err(format!(
                "{context}: payload length {} exceeds maximum {MAX_PARAM_BYTES}",
                payload.len()
            ));
        }
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
        STEP_HEADER_BYTES + self.payload.len()
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

    pub(crate) fn decode_prefix(
        bytes: &[u8],
        context: &str,
    ) -> Result<(Self, usize), String> {
        if bytes.len() < STEP_HEADER_BYTES {
            return Err(format!(
                "{context}: truncated header: expected at least {STEP_HEADER_BYTES} bytes, got {}",
                bytes.len()
            ));
        }
        let payload_len =
            u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
        if payload_len > MAX_PARAM_BYTES {
            return Err(format!(
                "{context}: payload length {payload_len} exceeds maximum {MAX_PARAM_BYTES}"
            ));
        }
        let record_len = STEP_HEADER_BYTES
            .checked_add(payload_len)
            .ok_or_else(|| format!("{context}: record length overflow"))?;
        if bytes.len() < record_len {
            return Err(format!(
                "{context}: truncated payload: record needs {record_len} bytes, got {}",
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
            bytes[STEP_HEADER_BYTES..record_len].to_vec(),
            context,
        )?;
        if record.encode().as_slice() != &bytes[..record_len] {
            return Err(format!("{context}: noncanonical record encoding"));
        }
        Ok((record, record_len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_record_roundtrips_exactly() {
        let record =
            ProgramStepRecord::new(0x26, 2, 0, 1, 2, 0, vec![1, 2, 3], "test step")
                .unwrap();
        let encoded = record.encode();
        let (decoded, consumed) =
            ProgramStepRecord::decode_prefix(&encoded, "test step").unwrap();
        assert_eq!(decoded, record);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn diagnostic_context_is_preserved() {
        let error = ProgramStepRecord::decode_prefix(&[0; 9], "MathProgramV7 step")
            .unwrap_err();
        assert_eq!(
            error,
            "MathProgramV7 step: truncated header: expected at least 10 bytes, got 9"
        );
    }
}
