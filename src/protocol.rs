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
        buf[0] = self.opcode;
        buf[1] = self.layer_type;
        buf[2] = self.variant;
        buf[3] = self.flags;
        let len_bytes = self.payload_len.to_le_bytes();
        buf[4..8].copy_from_slice(&len_bytes);
        buf
    }

    pub fn has_bias(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    pub fn is_training(&self) -> bool {
        (self.flags & 0x02) != 0
    }
}

// ============================================================
// HELPER: read/write multi-byte dari payload
// ============================================================

pub fn read_u32(payload: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > payload.len() {
        return Err("read_u32 out of bounds".into());
    }
    Ok(u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ]))
}

pub fn read_f64(payload: &[u8], offset: usize) -> Result<f64, String> {
    if offset + 8 > payload.len() {
        return Err("read_f64 out of bounds".into());
    }
    Ok(f64::from_le_bytes([
        payload[offset], payload[offset + 1], payload[offset + 2], payload[offset + 3],
        payload[offset + 4], payload[offset + 5], payload[offset + 6], payload[offset + 7],
    ]))
}

pub fn read_usize(payload: &[u8], offset: usize) -> Result<usize, String> {
    read_u32(payload, offset).map(|v| v as usize)
}

pub fn read_bool(payload: &[u8], offset: usize) -> Result<bool, String> {
    if offset >= payload.len() {
        return Err("read_bool out of bounds".into());
    }
    Ok(payload[offset] != 0)
}

pub fn read_option_u32(payload: &[u8], offset: usize) -> Result<Option<u32>, String> {
    if offset >= payload.len() {
        return Err("read_option out of bounds".into());
    }
    if payload[offset] == 0 {
        Ok(None)
    } else {
        read_u32(payload, offset + 1).map(Some)
    }
}

pub fn read_option_f64(payload: &[u8], offset: usize) -> Result<Option<f64>, String> {
    if offset >= payload.len() {
        return Err("read_option out of bounds".into());
    }
    if payload[offset] == 0 {
        Ok(None)
    } else {
        read_f64(payload, offset + 1).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_from_bytes_valid() {
        let bytes = vec![0x01, 0x02, 0x03, 0x01, 0x04, 0x00, 0x00, 0x00];
        let header = PacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header.opcode, 0x01);
        assert_eq!(header.layer_type, 0x02);
        assert_eq!(header.variant, 0x03);
        assert_eq!(header.flags, 0x01);
        assert_eq!(header.payload_len, 4);
        assert_eq!(header.has_bias(), true);
        assert_eq!(header.is_training(), false);
    }

    #[test]
    fn test_packet_header_from_bytes_too_short() {
        let bytes = vec![0x01, 0x02, 0x03, 0x01, 0x04, 0x00, 0x00];
        let result = PacketHeader::from_bytes(&bytes);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Header too short, need 8 bytes");
    }

    #[test]
    fn test_packet_header_from_bytes_exact_length() {
        let bytes = [0xFF, 0xAA, 0xBB, 0xCC, 0x10, 0x20, 0x30, 0x40];
        let header = PacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header.opcode, 0xFF);
        assert_eq!(header.layer_type, 0xAA);
        assert_eq!(header.variant, 0xBB);
        assert_eq!(header.flags, 0xCC);
        assert_eq!(header.payload_len, 0x40302010);
    }

    #[test]
    fn test_read_f64_valid() {
        let val: f64 = 3.14159265359;
        let bytes = val.to_le_bytes();

        let mut payload = vec![0, 0];
        payload.extend_from_slice(&bytes);
        payload.push(0);

        let result = read_f64(&payload, 2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), val);
    }

    #[test]
    fn test_read_f64_out_of_bounds() {
        let payload = vec![1, 2, 3, 4, 5, 6, 7]; // 7 bytes, 8 bytes needed
        let result = read_f64(&payload, 0);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "read_f64 out of bounds");

        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = read_f64(&payload, 1); // 1 + 8 > 8
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "read_f64 out of bounds");
    }
}
