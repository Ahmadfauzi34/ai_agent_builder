//! Fasad WASM tunggal — domain `protocol` (Opsi C, Fase 2).
//!
//! Pindahan murni dari `src/protocol.rs`: `#[wasm_bindgen] impl PacketHeader`
//! (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
//! Nama export JS tidak berubah.

use wasm_bindgen::prelude::*;

use crate::protocol::PacketHeader;

// ============================================================
// Opsi C Fase 2 — pindahan murni dari `src/protocol.rs`:
// #[wasm_bindgen] impl PacketHeader (struct tetap di domain dengan #[wasm_bindgen] sebagai marker ABI).
// Method bodies byte-identik; nama export JS tidak berubah.
// ============================================================

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
