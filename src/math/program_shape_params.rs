//! Canonical fixed-size rank-4 metadata codec for Math Program plan v3.
//!
//! This module is intentionally internal until the codec is wired into `MathProgram`.
//! The fixed-width representation keeps program identity deterministic and makes replay
//! validation independent of host pointer width.

#![allow(dead_code)]

pub(crate) const SHAPE_PARAM_WORDS: usize = 8;
pub(crate) const SHAPE_PARAM_BYTES: usize = SHAPE_PARAM_WORDS * 4;

// Scalar Math Program v2 currently reserves kinds 1 and 2.
pub(crate) const PARAM_RESHAPE_RANK4: u8 = 3;
pub(crate) const PARAM_PERMUTE_RANK4: u8 = 4;
pub(crate) const PARAM_SLICE_RANK4: u8 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FixedShapeParams {
    Reshape([u32; 4]),
    Permute([u32; 4]),
    Slice {
        starts: [u32; 4],
        ends: [u32; 4],
    },
}

impl FixedShapeParams {
    pub(crate) fn reshape(shape: &[u32]) -> Result<Self, String> {
        let shape = rank4(shape, "MathProgram.reshape")?;
        if let Some((axis, _)) = shape.iter().enumerate().find(|(_, dim)| **dim == 0) {
            return Err(format!(
                "MathProgram.reshape: dimension at axis {axis} must be non-zero"
            ));
        }
        Ok(Self::Reshape(shape))
    }

    pub(crate) fn permute(axes: &[u32]) -> Result<Self, String> {
        let axes = rank4(axes, "MathProgram.permute")?;
        let mut seen = [false; 4];
        for (position, axis) in axes.iter().copied().enumerate() {
            if axis >= 4 {
                return Err(format!(
                    "MathProgram.permute: axis at position {position} is out of range: {axis}"
                ));
            }
            if seen[axis as usize] {
                return Err(format!(
                    "MathProgram.permute: duplicate axis {axis} at position {position}"
                ));
            }
            seen[axis as usize] = true;
        }
        Ok(Self::Permute(axes))
    }

    pub(crate) fn slice(starts: &[u32], ends: &[u32]) -> Result<Self, String> {
        let starts = rank4(starts, "MathProgram.slice starts")?;
        let ends = rank4(ends, "MathProgram.slice ends")?;
        for axis in 0..4 {
            if starts[axis] >= ends[axis] {
                return Err(format!(
                    "MathProgram.slice: axis {axis} requires start < end, got {}..{}",
                    starts[axis], ends[axis]
                ));
            }
        }
        Ok(Self::Slice { starts, ends })
    }

    pub(crate) fn kind(self) -> u8 {
        match self {
            Self::Reshape(_) => PARAM_RESHAPE_RANK4,
            Self::Permute(_) => PARAM_PERMUTE_RANK4,
            Self::Slice { .. } => PARAM_SLICE_RANK4,
        }
    }

    pub(crate) fn words(self) -> [u32; SHAPE_PARAM_WORDS] {
        match self {
            Self::Reshape(shape) | Self::Permute(shape) => [
                shape[0], shape[1], shape[2], shape[3], 0, 0, 0, 0,
            ],
            Self::Slice { starts, ends } => [
                starts[0], starts[1], starts[2], starts[3],
                ends[0], ends[1], ends[2], ends[3],
            ],
        }
    }

    pub(crate) fn encode(self) -> [u8; SHAPE_PARAM_BYTES] {
        let mut bytes = [0u8; SHAPE_PARAM_BYTES];
        for (index, word) in self.words().into_iter().enumerate() {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    pub(crate) fn decode(kind: u8, payload: &[u8]) -> Result<Self, String> {
        if payload.len() != SHAPE_PARAM_BYTES {
            return Err(format!(
                "MathProgram shape parameters: expected {SHAPE_PARAM_BYTES} bytes, got {}",
                payload.len()
            ));
        }

        let mut words = [0u32; SHAPE_PARAM_WORDS];
        for (index, word) in words.iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]);
        }

        let decoded = match kind {
            PARAM_RESHAPE_RANK4 => {
                require_zero_padding(&words, "reshape")?;
                Self::reshape(&words[..4])?
            }
            PARAM_PERMUTE_RANK4 => {
                require_zero_padding(&words, "permute")?;
                Self::permute(&words[..4])?
            }
            PARAM_SLICE_RANK4 => Self::slice(&words[..4], &words[4..])?,
            _ => {
                return Err(format!(
                    "MathProgram shape parameters: unknown parameter kind {kind}"
                ))
            }
        };

        // Canonical replay requires a unique byte representation for the same metadata.
        if decoded.encode().as_slice() != payload {
            return Err("MathProgram shape parameters: noncanonical payload".into());
        }
        Ok(decoded)
    }

    pub(crate) fn reshape_usize(self) -> Option<[usize; 4]> {
        match self {
            Self::Reshape(shape) => Some(shape.map(|value| value as usize)),
            _ => None,
        }
    }

    pub(crate) fn permute_usize(self) -> Option<[usize; 4]> {
        match self {
            Self::Permute(axes) => Some(axes.map(|value| value as usize)),
            _ => None,
        }
    }

    pub(crate) fn slice_usize(self) -> Option<([usize; 4], [usize; 4])> {
        match self {
            Self::Slice { starts, ends } => Some((
                starts.map(|value| value as usize),
                ends.map(|value| value as usize),
            )),
            _ => None,
        }
    }
}

fn rank4(values: &[u32], context: &str) -> Result<[u32; 4], String> {
    if values.len() != 4 {
        return Err(format!(
            "{context}: expected exactly 4 values, got {}",
            values.len()
        ));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

fn require_zero_padding(words: &[u32; SHAPE_PARAM_WORDS], op: &str) -> Result<(), String> {
    if words[4..].iter().any(|word| *word != 0) {
        return Err(format!(
            "MathProgram {op}: unused rank-4 parameter words must be zero"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_round_trip_is_little_endian_and_canonical() {
        let params = FixedShapeParams::reshape(&[1, 2, 3, 4]).unwrap();
        let encoded = params.encode();
        assert_eq!(params.kind(), PARAM_RESHAPE_RANK4);
        assert_eq!(&encoded[0..4], &1u32.to_le_bytes());
        assert_eq!(&encoded[4..8], &2u32.to_le_bytes());
        assert!(encoded[16..].iter().all(|byte| *byte == 0));
        assert_eq!(
            FixedShapeParams::decode(params.kind(), &encoded).unwrap(),
            params
        );
        assert_eq!(params.reshape_usize(), Some([1, 2, 3, 4]));
    }

    #[test]
    fn permute_requires_complete_unique_rank4_axes() {
        let params = FixedShapeParams::permute(&[0, 2, 3, 1]).unwrap();
        let encoded = params.encode();
        assert_eq!(params.kind(), PARAM_PERMUTE_RANK4);
        assert_eq!(
            FixedShapeParams::decode(params.kind(), &encoded).unwrap(),
            params
        );
        assert_eq!(params.permute_usize(), Some([0, 2, 3, 1]));
        assert!(FixedShapeParams::permute(&[0, 1, 1, 3]).is_err());
        assert!(FixedShapeParams::permute(&[0, 1, 2, 4]).is_err());
        assert!(FixedShapeParams::permute(&[0, 1, 2]).is_err());
    }

    #[test]
    fn slice_uses_all_eight_words_and_validates_nonempty_ranges() {
        let params = FixedShapeParams::slice(&[0, 1, 2, 3], &[1, 2, 4, 8]).unwrap();
        let encoded = params.encode();
        assert_eq!(params.kind(), PARAM_SLICE_RANK4);
        assert_eq!(
            FixedShapeParams::decode(params.kind(), &encoded).unwrap(),
            params
        );
        assert_eq!(
            params.slice_usize(),
            Some(([0, 1, 2, 3], [1, 2, 4, 8]))
        );
        assert!(FixedShapeParams::slice(&[0, 0, 2, 0], &[1, 1, 2, 1]).is_err());
    }

    #[test]
    fn invalid_or_noncanonical_payloads_fail_closed() {
        let reshape = FixedShapeParams::reshape(&[1, 2, 3, 4]).unwrap();
        let mut padded = reshape.encode();
        padded[16] = 1;
        assert!(FixedShapeParams::decode(PARAM_RESHAPE_RANK4, &padded).is_err());

        let mut zero_dim = reshape.encode();
        zero_dim[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(FixedShapeParams::decode(PARAM_RESHAPE_RANK4, &zero_dim).is_err());

        assert!(FixedShapeParams::decode(0xFF, &reshape.encode()).is_err());
        assert!(FixedShapeParams::decode(PARAM_RESHAPE_RANK4, &[0u8; 4]).is_err());
    }

    #[test]
    fn same_metadata_has_same_bytes_and_semantic_change_changes_bytes() {
        let a = FixedShapeParams::reshape(&[1, 2, 3, 4]).unwrap();
        let b = FixedShapeParams::reshape(&[1, 2, 3, 4]).unwrap();
        let c = FixedShapeParams::reshape(&[1, 2, 4, 3]).unwrap();
        assert_eq!(a.encode(), b.encode());
        assert_ne!(a.encode(), c.encode());
    }
}
