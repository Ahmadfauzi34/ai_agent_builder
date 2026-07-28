// ============================================================
// WEIGHT LAYOUT (M2) — mesin mendeskripsikan bentuk bobotnya sendiri.
// Tiap layer stateful expose daftar segmen (name, len) yang URUTANNYA
// PERSIS sama dengan getWeightsFlat(). JS/ES susun vektor kandidat dari
// layout ini (bukan hardcode offset) -> mixing trainable jadi self-describing.
//
// Konvensi: len dihitung dari dims() (== jumlah float yang getWeightsFlat hasilkan),
// sehingga invariant `Σ len == getWeightsFlat().len()` terjaga secara struktural.
// Formatter JSON dipakai bersama oleh M1b/M1c nanti (satu sumber, tidak duplikat).
// ============================================================

/// Serialisasi daftar segmen jadi JSON array: [{"name":"weight","len":6}, ...]
/// Nama segmen alfanumerik + '.' (mis. "fc1.weight") -> tidak butuh escape.
pub fn segs_json(segs: &[(&'static str, usize)]) -> String {
    let mut s = String::from("[");
    for (i, (name, len)) in segs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"name\":\"");
        s.push_str(name);
        s.push_str("\",\"len\":");
        s.push_str(&len.to_string());
        s.push('}');
    }
    s.push(']');
    s
}
