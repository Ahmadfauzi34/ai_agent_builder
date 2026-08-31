use wasm_bindgen::prelude::*;

// ============================================================
// OPCODES — Aksi yang bisa dilakukan pada layer
// ============================================================
pub const OP_INIT:       u8 = 0x01;
pub const OP_FORWARD:    u8 = 0x02;
pub const OP_GET_STATE:  u8 = 0x03;
pub const OP_LOAD_STATE: u8 = 0x04;
pub const OP_DESTROY:    u8 = 0x05;
pub const OP_GET_PARAMS: u8 = 0x06;
pub const OP_RUN_GRAPH:    u8 = 0x07;

// ============================================================
// LAYER TYPES — 1 file.rs = 1 engine = 1 byte
// ============================================================
pub const LAYER_LINEAR:      u8 = 0x01;
pub const LAYER_NORM:        u8 = 0x02;
pub const LAYER_CONV:        u8 = 0x03;
pub const LAYER_ACTIVATION:  u8 = 0x04;
pub const LAYER_EMBEDDING:   u8 = 0x05;
pub const LAYER_POOL:        u8 = 0x06;

// --- Custom layers (0x10+) ---
pub const LAYER_SHIFT:       u8 = 0x10;
pub const LAYER_GHOST:       u8 = 0x11;
pub const LAYER_SEBLOCK:     u8 = 0x12;
pub const LAYER_BINARY:      u8 = 0x13;
// ============================================================
// VARIANTS — Pilihan dalam 1 engine
// ============================================================

// Conv variants
pub const CONV_CONV1D:          u8 = 0x00;
pub const CONV_CONV2D:          u8 = 0x01;
pub const CONV_CONVTRANSPOSE2D: u8 = 0x02;

// Norm variants
pub const NORM_BATCH:     u8 = 0x00;
pub const NORM_GROUP:     u8 = 0x01;
pub const NORM_INSTANCE:  u8 = 0x02;
pub const NORM_LAYER:     u8 = 0x03;
pub const NORM_RMS:       u8 = 0x04;

// Activation variants
pub const ACT_GELU:         u8 = 0x00;
pub const ACT_RELU:         u8 = 0x01;
pub const ACT_SIGMOID:      u8 = 0x02;
pub const ACT_TANH:         u8 = 0x03;
pub const ACT_HARDSWISH:    u8 = 0x04;
pub const ACT_LEAKYRELU:    u8 = 0x05;
pub const ACT_PRELU:        u8 = 0x06;
pub const ACT_SWIGLU:       u8 = 0x07;
pub const ACT_HARDSIGMOID:  u8 = 0x08;
pub const ACT_SOFTPLUS:     u8 = 0x09;
pub const ACT_MISH:         u8 = 0x0A;
pub const ACT_SOFTMAX:      u8 = 0x0B;
pub const ACT_LOGSOFTMAX:   u8 = 0x0C;
pub const ACT_GLU:          u8 = 0x0D;

// Pool variants
pub const POOL_MAXPOOL1D:          u8 = 0x00;
pub const POOL_MAXPOOL2D:          u8 = 0x01;
pub const POOL_AVGPOOL1D:          u8 = 0x02;
pub const POOL_AVGPOOL2D:          u8 = 0x03;
pub const POOL_ADAPTIVEAVGPOOL2D:  u8 = 0x04;

// Shift variants
pub const SHIFT_UP:    u8 = 0x00;
pub const SHIFT_DOWN:  u8 = 0x01;
pub const SHIFT_LEFT:  u8 = 0x02;
pub const SHIFT_RIGHT: u8 = 0x03;
// Binary variants (op 2-input)
pub const BINARY_ADD:    u8 = 0x00;
pub const BINARY_SUB:    u8 = 0x01;
pub const BINARY_MUL:    u8 = 0x02;
pub const BINARY_MATMUL: u8 = 0x03;
pub const BINARY_CONCAT: u8 = 0x04;

// ============================================================
// PACKET HEADER — Fixed 8 bytes
// ============================================================
// [0]     : OpCode
// [1]     : LayerType
// [2]     : Variant (pilihan dalam engine, 0xFF kalau tidak relevan)
// [3]     : Flags (bitmask: bit0=bias, bit1=training, dst.)
// [4..8]  : Payload length (u32, little-endian)
// ============================================================

#[wasm_bindgen]
pub struct PacketHeader {
    pub opcode: u8,
    pub layer_type: u8,
    pub variant: u8,
    pub flags: u8,
    pub payload_len: u32,
}

#[wasm_bindgen]
impl PacketHeader {
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(bytes: &[u8]) -> Result<PacketHeader, String> {
        if bytes.len() < 8 {
            return Err("Header too short, need 8 bytes".into());
        }
        Ok(PacketHeader {
            opcode: bytes[0],
            layer_type: bytes[1],
            variant: bytes[2],
            flags: bytes[3],
            payload_len: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        self.write_to_slice(&mut buf);
        buf
    }

    pub fn has_bias(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    pub fn is_training(&self) -> bool {
        (self.flags & 0x02) != 0
    }
}

// Non-wasm_bindgen implementations of helper functions so we don't hit wasm_bindgen limitations.
impl PacketHeader {
    /// Potong payload sesuai payload_len.
    #[inline]
    pub fn validate_payload<'a>(&self, payload: &'a [u8]) -> Result<&'a [u8], String> {
        let expected = self.payload_len as usize;

        if payload.len() < expected {
            return Err(format!(
                "payload shorter than payload_len: expected {}, got {}",
                expected,
                payload.len()
            ));
        }

        Ok(&payload[..expected])
    }

    /// Tulis header ke buffer fixed 8 byte tanpa alokasi Vec.
    #[inline]
    pub fn write_to(&self, buf: &mut [u8; 8]) {
        buf[0] = self.opcode;
        buf[1] = self.layer_type;
        buf[2] = self.variant;
        buf[3] = self.flags;
        buf[4..8].copy_from_slice(&self.payload_len.to_le_bytes());
    }

    /// Tulis header ke slice (min 8 byte).
    #[inline]
    pub fn write_to_slice(&self, buf: &mut [u8]) {
        buf[0] = self.opcode;
        buf[1] = self.layer_type;
        buf[2] = self.variant;
        buf[3] = self.flags;
        buf[4..8].copy_from_slice(&self.payload_len.to_le_bytes());
    }
}

pub const VARIANT_NONE: u8 = 0xFF;
pub const FLAG_BIAS: u8 = 1 << 0;
pub const FLAG_TRAINING: u8 = 1 << 1;

pub struct PayloadCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PayloadCursor<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn ensure(&self, n: usize) -> Result<(), String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "payload offset overflow".to_string())?;

        if end > self.data.len() {
            return Err(format!(
                "payload out of bounds: need {} bytes at offset {}",
                n, self.pos
            ));
        }

        Ok(())
    }

    #[inline]
    pub fn read_u8(&mut self) -> Result<u8, String> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    #[inline]
    pub fn read_u32(&mut self) -> Result<u32, String> {
        self.ensure(4)?;

        let b = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];

        self.pos += 4;

        Ok(u32::from_le_bytes(b))
    }

    #[inline]
    pub fn read_f32(&mut self) -> Result<f32, String> {
        self.ensure(4)?;

        let mut b = [0u8; 4];
        b.copy_from_slice(&self.data[self.pos..self.pos + 4]);

        self.pos += 4;

        Ok(f32::from_le_bytes(b))
    }

    #[inline]
    pub fn read_f64(&mut self) -> Result<f64, String> {
        self.ensure(8)?;

        let mut b = [0u8; 8];
        b.copy_from_slice(&self.data[self.pos..self.pos + 8]);

        self.pos += 8;

        Ok(f64::from_le_bytes(b))
    }

    #[inline]
    pub fn read_bool(&mut self) -> Result<bool, String> {
        self.read_u8().map(|v| v != 0)
    }

    #[inline]
    pub fn read_usize(&mut self) -> Result<usize, String> {
        self.read_u32().map(|v| v as usize)
    }

    /// Option<u32> fixed-size:
    /// 1 byte tag + 4 byte value
    #[inline]
    pub fn read_option_u32(&mut self) -> Result<Option<u32>, String> {
        let present = self.read_u8()? != 0;
        let value = self.read_u32()?;

        Ok(if present { Some(value) } else { None })
    }

    /// Option<f64> fixed-size:
    /// 1 byte tag + 8 byte value
    #[inline]
    pub fn read_option_f64(&mut self) -> Result<Option<f64>, String> {
        let present = self.read_u8()? != 0;
        let value = self.read_f64()?;

        Ok(if present { Some(value) } else { None })
    }

    #[inline]
    pub fn read_option_usize(&mut self) -> Result<Option<usize>, String> {
        Ok(self.read_option_u32()?.map(|v| v as usize))
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

// ============================================================
// HELPER: read/write multi-byte dari payload
// ============================================================

fn checked_read_range<'a>(
    payload: &'a [u8],
    offset: usize,
    width: usize,
    context: &str,
) -> Result<&'a [u8], String> {
    let end = offset
        .checked_add(width)
        .ok_or_else(|| format!("{context} offset overflow"))?;
    payload
        .get(offset..end)
        .ok_or_else(|| format!("{context} out of bounds"))
}

pub fn read_u32(payload: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = checked_read_range(payload, offset, 4, "read_u32")?;
    Ok(u32::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
    ]))
}

pub fn read_f64(payload: &[u8], offset: usize) -> Result<f64, String> {
    let bytes = checked_read_range(payload, offset, 8, "read_f64")?;
    Ok(f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

pub fn read_usize(payload: &[u8], offset: usize) -> Result<usize, String> {
    read_u32(payload, offset).map(|v| v as usize)
}

pub fn read_bool(payload: &[u8], offset: usize) -> Result<bool, String> {
    payload
        .get(offset)
        .copied()
        .map(|v| v != 0)
        .ok_or_else(|| "read_bool out of bounds".into())
}

pub fn read_option_u32(payload: &[u8], offset: usize) -> Result<Option<u32>, String> {
    let bytes = checked_read_range(payload, offset, 5, "read_option_u32")?;
    let present = bytes[0] != 0;
    let value = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
    Ok(if present { Some(value) } else { None })
}

pub fn read_option_f64(payload: &[u8], offset: usize) -> Result<Option<f64>, String> {
    let bytes = checked_read_range(payload, offset, 9, "read_option_f64")?;
    let present = bytes[0] != 0;
    let value = f64::from_le_bytes([
        bytes[1], bytes[2], bytes[3], bytes[4],
        bytes[5], bytes[6], bytes[7], bytes[8],
    ]);
    Ok(if present { Some(value) } else { None })
}

#[cfg(test)]
mod offset_tests {
    use super::{
        read_f64, read_option_f64, read_option_u32, read_u32, PayloadCursor,
    };

    #[test]
    fn read_u32_rejects_offset_overflow_without_panicking() {
        let err = read_u32(&[0u8; 8], usize::MAX).unwrap_err();
        assert!(err.contains("offset overflow"));
    }

    #[test]
    fn read_f64_rejects_offset_overflow_without_panicking() {
        let err = read_f64(&[0u8; 8], usize::MAX).unwrap_err();
        assert!(err.contains("offset overflow"));
    }

    #[test]
    fn fixed_width_reader_rejects_truncated_range() {
        assert!(read_u32(&[1, 2, 3], 0).is_err());
        assert!(read_f64(&[0u8; 7], 0).is_err());
    }

    #[test]
    fn option_u32_matches_cursor_fixed_width_none() {
        assert!(read_option_u32(&[0], 0).is_err());
        let bytes = [0, 0, 0, 0, 0];
        assert_eq!(read_option_u32(&bytes, 0).unwrap(), None);
        let mut cursor = PayloadCursor::new(&bytes);
        assert_eq!(cursor.read_option_u32().unwrap(), None);
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn option_f64_matches_cursor_fixed_width_none() {
        assert!(read_option_f64(&[0], 0).is_err());
        let bytes = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(read_option_f64(&bytes, 0).unwrap(), None);
        let mut cursor = PayloadCursor::new(&bytes);
        assert_eq!(cursor.read_option_f64().unwrap(), None);
        assert_eq!(cursor.remaining(), 0);
    }

    #[test]
    fn option_readers_decode_present_values() {
        let mut u32_bytes = vec![1];
        u32_bytes.extend_from_slice(&42u32.to_le_bytes());
        assert_eq!(read_option_u32(&u32_bytes, 0).unwrap(), Some(42));

        let mut f64_bytes = vec![1];
        f64_bytes.extend_from_slice(&3.5f64.to_le_bytes());
        assert_eq!(read_option_f64(&f64_bytes, 0).unwrap(), Some(3.5));
    }

    #[test]
    fn option_readers_reject_offset_overflow() {
        assert!(read_option_u32(&[0u8; 5], usize::MAX).is_err());
        assert!(read_option_f64(&[0u8; 9], usize::MAX).is_err());
    }
}
