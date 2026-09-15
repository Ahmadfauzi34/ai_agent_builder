//! Canonical identity-bound positional-index parameters for Math Program plans.
//!
//! The index axis is immutable plan metadata. It is deliberately distinct from reduction-axis
//! metadata even though both currently encode one rank-4 axis byte: the semantic identity is not
//! interchangeable and replay must commit to which primitive owns the parameter.

pub(crate) const PARAM_INDEX_AXIS: u8 = 9;
pub(crate) const INDEX_AXIS_PARAM_BYTES: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexAxisParams {
    axis: u8,
}

impl IndexAxisParams {
    pub(crate) fn new(axis: u32) -> Result<Self, String> {
        if axis >= 4 {
            return Err(format!(
                "MathProgram.indexAxis: axis must be in 0..4, got {axis}"
            ));
        }
        Ok(Self { axis: axis as u8 })
    }

    pub(crate) fn decode(payload: &[u8]) -> Result<Self, String> {
        if payload.len() != INDEX_AXIS_PARAM_BYTES {
            return Err(format!(
                "MathProgram.indexAxis: payload must contain exactly {INDEX_AXIS_PARAM_BYTES} byte, got {}",
                payload.len()
            ));
        }
        Self::new(payload[0] as u32)
    }

    pub(crate) fn encode(self) -> [u8; INDEX_AXIS_PARAM_BYTES] {
        [self.axis]
    }

    pub(crate) fn axis(self) -> u32 {
        self.axis as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_codec_is_exact_bounded_and_semantically_distinct() {
        assert_eq!(PARAM_INDEX_AXIS, 9);
        for axis in 0..4 {
            let params = IndexAxisParams::new(axis).unwrap();
            assert_eq!(params.encode(), [axis as u8]);
            assert_eq!(IndexAxisParams::decode(&params.encode()).unwrap(), params);
            assert_eq!(params.axis(), axis);
        }
        assert!(IndexAxisParams::new(4).is_err());
        assert!(IndexAxisParams::decode(&[]).is_err());
        assert!(IndexAxisParams::decode(&[0, 1]).is_err());
        assert!(IndexAxisParams::decode(&[4]).is_err());
    }
}
