// src/protocol.rs
use wasm_bindgen::prelude::*;

/// Header: 8 byte
/// [0..1]: OpCode (u8)
/// [1..2]: LayerType (u8)  
/// [2..4]: Reserved (u16)
/// [4..8]: Payload length (u32, little-endian)
#[wasm_bindgen]
pub struct PacketHeader {
    pub opcode: u8,
    pub layer_type: u8,
    pub payload_len: u32,
}

#[wasm_bindgen]
impl PacketHeader {
    #[wasm_bindgen(constructor)]
    pub fn from_bytes(bytes: &[u8]) -> Result<PacketHeader, String> {
        if bytes.len() < 8 {
            return Err("Header too short".into());
        }
        Ok(PacketHeader {
            opcode: bytes[0],
            layer_type: bytes[1],
            payload_len: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        buf[0] = self.opcode;
        buf[1] = self.layer_type;
        // bytes[2..4] reserved
        let len_bytes = self.payload_len.to_le_bytes();
        buf[4..8].copy_from_slice(&len_bytes);
        buf
    }
}

/// OpCodes untuk komunikasi
pub const OP_INIT: u8 = 0x01;
pub const OP_FORWARD: u8 = 0x02;
pub const OP_GET_STATE: u8 = 0x03;
pub const OP_LOAD_STATE: u8 = 0x04;
pub const OP_GET_PARAMS: u8 = 0x05;

