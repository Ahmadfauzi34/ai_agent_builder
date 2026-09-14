//! Canonical identity-bound generic-reduction parameters for Math Program plans.
//!
//! The reduction axis is immutable plan metadata. It is deliberately not inferred from tensor
//! shapes and is not a runtime side channel, so replay identity commits to the exact axis choice.

pub(crate) const PARAM_REDUCTION_AXIS: u8 = 8;
pub(crate) const REDUCTION_AXIS_PARAM_BYTES: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReductionAxisParams {
    axis: u8,
}

impl ReductionAxisParams {
    pub(crate) fn new(axis: u32) -> Result<Self, String> {
        if axis >= 4 {
            return Err(format!(
                "MathProgram.reductionAxis: axis must be in 0..4, got {axis}"
            ));
        }
        Ok(Self { axis: axis as u8 })
    }

    pub(crate) fn decode(payload: &[u8]) -> Result<Self, String> {
        if payload.len() != REDUCTION_AXIS_PARAM_BYTES {
            return Err(format!(
                "MathProgram.reductionAxis: payload must contain exactly {REDUCTION_AXIS_PARAM_BYTES} byte, got {}",
                payload.len()
            ));
        }
        Self::new(payload[0] as u32)
    }

    pub(crate) fn encode(self) -> [u8; REDUCTION_AXIS_PARAM_BYTES] {
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
    fn axis_codec_is_exact_and_bounded() {
        for axis in 0..4 {
            let params = ReductionAxisParams::new(axis).unwrap();
            assert_eq!(params.encode(), [axis as u8]);
            assert_eq!(ReductionAxisParams::decode(&params.encode()).unwrap(), params);
            assert_eq!(params.axis(), axis);
        }
        assert!(ReductionAxisParams::new(4).is_err());
        assert!(ReductionAxisParams::decode(&[]).is_err());
        assert!(ReductionAxisParams::decode(&[0, 1]).is_err());
        assert!(ReductionAxisParams::decode(&[4]).is_err());
    }
}
