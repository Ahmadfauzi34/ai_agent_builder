pub use crate::facade::evidence::{
    export_multi_input_program_bundle, export_program_bundle, import_multi_input_program_bundle,
    import_program_bundle, multi_input_program_bundle_capabilities, program_bundle_capabilities,
};
use wasm_bindgen::prelude::*;

use crate::graph::{CompiledGraph, CompiledMultiInputGraph};
use crate::graph_plan::decode_graph_plan;
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::protocol::{PacketHeader, OP_INIT};
use crate::registry::LayerRegistry;

pub(crate) const BUNDLE_MAGIC: &[u8; 8] = b"BRPGBNDL";
pub(crate) const MULTI_INPUT_BUNDLE_MAGIC: &[u8; 8] = b"BRMIBNDL";
pub(crate) const BUNDLE_SCHEMA_VERSION: u32 = 1;
pub(crate) const BUNDLE_FLAG_STATE_INCLUDED: u32 = 1 << 0;
const BUNDLE_KNOWN_FLAGS: u32 = BUNDLE_FLAG_STATE_INCLUDED;

pub(crate) fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn checked_u32(value: usize, context: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{context}: length exceeds u32"))
}

fn read_u32_at(bytes: &[u8], offset: usize, context: &str) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| format!("{context}: offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| format!("{context}: truncated u32 at offset {offset}"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

pub(crate) fn referenced_layer_keys(plan: &[u8]) -> Result<Vec<(u8, u32)>, String> {
    decode_graph_plan(plan)
        .map(|decoded| decoded.unique_first_use_layer_keys())
        .map_err(|error| format!("program bundle: {error}"))
}

pub(crate) fn multi_input_referenced_layer_keys(plan: &[u8]) -> Result<Vec<(u8, u32)>, String> {
    let input_plan = MultiInputGraphPlan::from_bytes(plan)
        .map_err(|error| format!("multi-input program bundle: {error}"))?;
    referenced_layer_keys(input_plan.graph_plan())
}

fn parse_hex_u8(value: &str, context: &str) -> Result<u8, String> {
    u8::from_str_radix(value, 16)
        .map_err(|_| format!("program bundle: invalid {context} hex value {value:?}"))
}

fn decode_hex(value: &str, context: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err(format!("program bundle: {context} hex length must be even"));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    for offset in (0..value.len()).step_by(2) {
        out.push(parse_hex_u8(&value[offset..offset + 2], context)?);
    }
    Ok(out)
}

fn field<'a>(part: &'a str, prefix: &str, context: &str) -> Result<&'a str, String> {
    part.strip_prefix(prefix)
        .ok_or_else(|| format!("program bundle: malformed {context} fingerprint field {part:?}"))
}

pub(crate) fn parse_init_fingerprint(
    fingerprint: &str,
    expected_type: u8,
    expected_id: u32,
) -> Result<(u8, u8, Vec<u8>), String> {
    let parts = fingerprint.split(';').collect::<Vec<_>>();
    if parts.len() != 5 {
        return Err(format!(
            "program bundle: unsupported program-identity.v1 layer fingerprint {fingerprint:?}"
        ));
    }
    let layer_type = parse_hex_u8(field(parts[0], "type=", "type")?, "layer type")?;
    let layer_id = field(parts[1], "id=", "id")?
        .parse::<u32>()
        .map_err(|_| "program bundle: invalid layer id in fingerprint".to_string())?;
    let variant = parse_hex_u8(field(parts[2], "variant=", "variant")?, "variant")?;
    let flags = parse_hex_u8(field(parts[3], "flags=", "flags")?, "flags")?;
    let payload = decode_hex(field(parts[4], "payload=", "payload")?, "payload")?;

    if layer_type != expected_type || layer_id != expected_id {
        return Err(format!(
            "program bundle: fingerprint key mismatch: expected type 0x{expected_type:02X} id {expected_id}, got type 0x{layer_type:02X} id {layer_id}"
        ));
    }
    if payload.len() < 4 {
        return Err("program bundle: fingerprint payload is too short to contain layer id".into());
    }
    let payload_id = read_u32_at(&payload, 0, "program bundle fingerprint payload")?;
    if payload_id != layer_id {
        return Err(format!(
            "program bundle: fingerprint payload id {payload_id} does not match layer id {layer_id}"
        ));
    }
    Ok((variant, flags, payload))
}

struct BundleCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BundleCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize, context: &str) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| format!("program bundle: {context} length overflow"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| format!("program bundle: truncated {context}"))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self, context: &str) -> Result<u8, String> {
        Ok(self.take(1, context)?[0])
    }

    fn read_u32(&mut self, context: &str) -> Result<u32, String> {
        let bytes = self.take(4, context)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn is_finished(&self) -> bool {
        self.pos == self.data.len()
    }
}

pub(crate) struct DecodedLayer {
    pub(crate) layer_type: u8,
    pub(crate) variant: u8,
    pub(crate) flags: u8,
    pub(crate) layer_id: u32,
    pub(crate) init_payload: Vec<u8>,
    pub(crate) state: Vec<u8>,
}

pub(crate) struct DecodedBundle {
    pub(crate) state_included: bool,
    pub(crate) plan: Vec<u8>,
    pub(crate) expected_identity: String,
    pub(crate) layers: Vec<DecodedLayer>,
}

fn decode_bundle(
    bundle: &[u8],
    expected_magic: &[u8; 8],
    context: &str,
    decode_keys: fn(&[u8]) -> Result<Vec<(u8, u32)>, String>,
) -> Result<DecodedBundle, String> {
    let mut cursor = BundleCursor::new(bundle);
    if cursor.take(expected_magic.len(), "magic")? != expected_magic {
        return Err(format!("{context}: invalid magic"));
    }
    let schema = cursor.read_u32("schema version")?;
    if schema != BUNDLE_SCHEMA_VERSION {
        return Err(format!("{context}: unsupported schema version {schema}"));
    }
    let flags = cursor.read_u32("flags")?;
    if flags & !BUNDLE_KNOWN_FLAGS != 0 {
        return Err(format!("{context}: unknown flags 0x{flags:08X}"));
    }
    let state_included = flags & BUNDLE_FLAG_STATE_INCLUDED != 0;
    let plan_len = cursor.read_u32("plan length")? as usize;
    let identity_len = cursor.read_u32("identity length")? as usize;
    let layer_count = cursor.read_u32("layer count")? as usize;

    let plan = cursor.take(plan_len, "plan")?.to_vec();
    let identity_bytes = cursor.take(identity_len, "program identity")?;
    let expected_identity = std::str::from_utf8(identity_bytes)
        .map_err(|_| format!("{context}: program identity is not valid UTF-8"))?
        .to_string();

    let expected_keys = decode_keys(&plan)?;
    if layer_count != expected_keys.len() {
        return Err(format!(
            "{context}: layer count mismatch: bundle has {layer_count}, plan references {} unique layers",
            expected_keys.len()
        ));
    }

    let mut layers = Vec::with_capacity(layer_count);
    for (index, expected_key) in expected_keys.iter().copied().enumerate() {
        let layer_type = cursor.read_u8("layer type")?;
        let variant = cursor.read_u8("layer variant")?;
        let layer_flags = cursor.read_u8("layer flags")?;
        let reserved = cursor.read_u8("layer reserved byte")?;
        if reserved != 0 {
            return Err(format!(
                "{context}: layer {index} reserved byte must be zero"
            ));
        }
        let layer_id = cursor.read_u32("layer id")?;
        let init_len = cursor.read_u32("init payload length")? as usize;
        let state_len = cursor.read_u32("state length")? as usize;
        if !state_included && state_len != 0 {
            return Err(format!(
                "{context}: layer {index} carries state without state-included flag"
            ));
        }
        if (layer_type, layer_id) != expected_key {
            return Err(format!(
                "{context}: layer record {index} key mismatch: expected type 0x{:02X} id {}, got type 0x{layer_type:02X} id {layer_id}",
                expected_key.0, expected_key.1
            ));
        }
        let init_payload = cursor.take(init_len, "init payload")?.to_vec();
        if init_payload.len() < 4 {
            return Err(format!(
                "{context}: layer {index} init payload is too short to contain layer id"
            ));
        }
        let payload_id = read_u32_at(&init_payload, 0, "program bundle init payload")?;
        if payload_id != layer_id {
            return Err(format!(
                "{context}: layer {index} init payload id {payload_id} does not match record id {layer_id}"
            ));
        }
        let state = cursor.take(state_len, "layer state")?.to_vec();
        layers.push(DecodedLayer {
            layer_type,
            variant,
            flags: layer_flags,
            layer_id,
            init_payload,
            state,
        });
    }
    if !cursor.is_finished() {
        return Err(format!("{context}: trailing bytes are not allowed"));
    }

    Ok(DecodedBundle {
        state_included,
        plan,
        expected_identity,
        layers,
    })
}

pub(crate) fn decode_program_bundle(bundle: &[u8]) -> Result<DecodedBundle, String> {
    decode_bundle(
        bundle,
        BUNDLE_MAGIC,
        "program bundle",
        referenced_layer_keys,
    )
}

pub(crate) fn decode_multi_input_program_bundle(bundle: &[u8]) -> Result<DecodedBundle, String> {
    decode_bundle(
        bundle,
        MULTI_INPUT_BUNDLE_MAGIC,
        "multi-input program bundle",
        multi_input_referenced_layer_keys,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        decode_multi_input_program_bundle, decode_program_bundle,
        export_multi_input_program_bundle, export_program_bundle,
        import_multi_input_program_bundle, import_program_bundle,
        multi_input_program_bundle_capabilities, parse_init_fingerprint,
        program_bundle_capabilities,
    };
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::graph::CompiledMultiInputGraph;
    use crate::multi_input_graph::{MultiInputGraphPlan, MultiInputInputBundle};
    use crate::protocol::LAYER_LINEAR;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn linear_graph(
        registry: &mut LayerRegistry,
    ) -> (
        AgentLayerSpec,
        AgentGraphBuilder,
        crate::graph::CompiledGraph,
    ) {
        let spec = AgentLayerSpec::linear(7, 2, 2, true).unwrap();
        registry.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph = builder.compile(registry).unwrap();
        (spec, builder, graph)
    }

    fn two_input_linear_graph(
        registry: &mut LayerRegistry,
    ) -> (
        AgentLayerSpec,
        AgentLayerSpec,
        AgentGraphBuilder,
        MultiInputGraphPlan,
        CompiledMultiInputGraph,
    ) {
        let add = AgentLayerSpec::add(21);
        let linear = AgentLayerSpec::linear(22, 2, 2, true).unwrap();
        registry.init_agent_layer(&add).unwrap();
        registry.init_agent_layer(&linear).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.add_unary(&linear, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let mut plan = builder.multi_input_plan_v1().unwrap();
        plan.add_input_port(
            0,
            "observation".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            1,
        )
        .unwrap();
        plan.add_input_port(
            1,
            "state".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            1,
        )
        .unwrap();
        let graph = registry.compile_multi_input_graph(&plan).unwrap();
        (add, linear, builder, plan, graph)
    }

    fn multi_input_output(
        registry: &LayerRegistry,
        graph: &CompiledMultiInputGraph,
        plan: &MultiInputGraphPlan,
    ) -> Vec<f32> {
        let mut bundle = MultiInputInputBundle::new(plan).unwrap();
        let left = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);
        let right = WasmTensor::new(&[3.0, 4.0], &[1, 2, 1, 1]);
        bundle
            .bind_input(
                0,
                &left,
                "observation".into(),
                "feature_axis1_singleton".into(),
                "sensor-a".into(),
                1,
                "obs".into(),
            )
            .unwrap();
        bundle
            .bind_input(
                1,
                &right,
                "state".into(),
                "feature_axis1_singleton".into(),
                "memory-b".into(),
                1,
                "state".into(),
            )
            .unwrap();
        graph.run(registry, &bundle).unwrap().to_array()
    }

    #[test]
    fn fingerprint_parser_is_schema_bound_and_exact() {
        let mut registry = LayerRegistry::new();
        let (spec, _builder, _graph) = linear_graph(&mut registry);
        let fingerprint = registry
            .layer_init_fingerprint(spec.layer_type(), spec.layer_id())
            .unwrap();
        let (_variant, _flags, payload) =
            parse_init_fingerprint(&fingerprint, spec.layer_type(), spec.layer_id()).unwrap();
        assert_eq!(
            u32::from_le_bytes(payload[0..4].try_into().unwrap()),
            spec.layer_id()
        );
        assert!(
            parse_init_fingerprint("future-format", spec.layer_type(), spec.layer_id()).is_err()
        );
    }

    #[test]
    fn state_bundle_round_trip_preserves_identity_plan_and_output() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = linear_graph(&mut source);
        let mut weights = source.get_weights_flat(7, LAYER_LINEAR).unwrap();
        for (index, value) in weights.iter_mut().enumerate() {
            *value = (index + 1) as f32 * 0.25;
        }
        source.set_weights_flat(7, LAYER_LINEAR, &weights).unwrap();
        let input = WasmTensor::new(&[1.5, -0.5], &[1, 2, 1, 1]);
        let expected = graph.run(&source, &input).unwrap().to_array();
        let bundle = export_program_bundle(&graph, &source, true).unwrap();

        let mut target = LayerRegistry::new();
        let imported = import_program_bundle(&mut target, &bundle).unwrap();
        assert_eq!(imported.program_plan(), graph.program_plan());
        assert_eq!(imported.program_identity(), graph.program_identity());
        assert_eq!(imported.run(&target, &input).unwrap().to_array(), expected);
        assert!(imported.validate_registry_binding(&target).is_ok());
    }

    #[test]
    fn structure_only_bundle_preserves_structural_identity() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = linear_graph(&mut source);
        let bundle = export_program_bundle(&graph, &source, false).unwrap();
        let decoded = decode_program_bundle(&bundle).unwrap();
        assert!(!decoded.state_included);
        assert!(decoded.layers.iter().all(|layer| layer.state.is_empty()));

        let mut target = LayerRegistry::new();
        let imported = import_program_bundle(&mut target, &bundle).unwrap();
        assert_eq!(imported.program_identity(), graph.program_identity());
    }

    #[test]
    fn repeated_layer_reference_serializes_one_layer_record() {
        let mut source = LayerRegistry::new();
        let spec = AgentLayerSpec::relu(11);
        source.init_agent_layer(&spec).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_unary(&spec, 0, 1).unwrap();
        builder.add_unary(&spec, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let graph = builder.compile(&source).unwrap();
        let bundle = export_program_bundle(&graph, &source, true).unwrap();
        let decoded = decode_program_bundle(&bundle).unwrap();
        assert_eq!(decoded.layers.len(), 1);
    }

    #[test]
    fn malformed_bundle_is_atomic_and_valid_retry_succeeds() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = linear_graph(&mut source);
        let valid = export_program_bundle(&graph, &source, true).unwrap();
        let mut corrupt = valid.clone();
        corrupt.truncate(corrupt.len() - 1);

        let mut target = LayerRegistry::new();
        let existing = AgentLayerSpec::relu(99);
        target.init_agent_layer(&existing).unwrap();
        assert!(import_program_bundle(&mut target, &corrupt).is_err());
        assert!(target.layer_exists(existing.layer_type(), existing.layer_id()));

        let imported = import_program_bundle(&mut target, &valid).unwrap();
        assert_eq!(imported.program_identity(), graph.program_identity());
        assert!(!target.layer_exists(existing.layer_type(), existing.layer_id()));
        assert!(target.layer_exists(LAYER_LINEAR, 7));
    }

    #[test]
    fn malformed_embedded_graph_plan_is_rejected_without_target_mutation() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = linear_graph(&mut source);
        let mut corrupt = export_program_bundle(&graph, &source, false).unwrap();

        // Bundle v1 header is 28 bytes. The embedded graph plan starts immediately
        // afterward and begins with num_steps:u32. Inflate the declared step count
        // while leaving the exact plan byte envelope unchanged.
        let plan_start = 28usize;
        let original_steps =
            u32::from_le_bytes(corrupt[plan_start..plan_start + 4].try_into().unwrap());
        corrupt[plan_start..plan_start + 4].copy_from_slice(&(original_steps + 1).to_le_bytes());

        let mut target = LayerRegistry::new();
        let existing = AgentLayerSpec::relu(99);
        target.init_agent_layer(&existing).unwrap();

        let error = match import_program_bundle(&mut target, &corrupt) {
            Ok(_) => panic!("expected malformed embedded graph plan to fail"),
            Err(error) => error,
        };
        assert!(
            error.contains("malformed plan length"),
            "unexpected malformed embedded-plan error: {error}"
        );
        assert!(target.layer_exists(existing.layer_type(), existing.layer_id()));
    }

    #[test]
    fn identity_corruption_is_rejected_without_target_mutation() {
        let mut source = LayerRegistry::new();
        let (_spec, _builder, graph) = linear_graph(&mut source);
        let valid = export_program_bundle(&graph, &source, false).unwrap();
        let decoded = decode_program_bundle(&valid).unwrap();
        let needle = decoded.expected_identity.as_bytes();
        let start = valid
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        let mut corrupt = valid.clone();
        corrupt[start] ^= 1;

        let mut target = LayerRegistry::new();
        let existing = AgentLayerSpec::relu(99);
        target.init_agent_layer(&existing).unwrap();
        assert!(import_program_bundle(&mut target, &corrupt).is_err());
        assert!(target.layer_exists(existing.layer_type(), existing.layer_id()));
    }

    #[test]
    fn discovery_declares_atomic_replace_and_state_separation() {
        let caps = program_bundle_capabilities();
        assert!(caps.contains("burn-research.program-bundle.v1"));
        assert!(caps.contains("program-identity.v1_layer_init_fingerprint"));
        assert!(caps.contains("atomic_replace_on_success"));
        assert!(caps.contains("atomic_after_identity_validation"));
        assert!(caps.contains("optional_separate_section"));
    }

    #[test]
    fn multi_input_state_bundle_round_trip_preserves_plan_identity_and_output() {
        let mut source = LayerRegistry::new();
        let (_add, _linear, _builder, plan, graph) = two_input_linear_graph(&mut source);
        let identity = graph.program_identity();
        let mut weights = source.get_weights_flat(22, LAYER_LINEAR).unwrap();
        for (index, value) in weights.iter_mut().enumerate() {
            *value = (index + 1) as f32 * 0.125;
        }
        source.set_weights_flat(22, LAYER_LINEAR, &weights).unwrap();
        assert_eq!(
            graph.program_identity(),
            identity,
            "mutable weights changed structural identity"
        );
        let expected = multi_input_output(&source, &graph, &plan);
        let bundle = export_multi_input_program_bundle(&graph, &source, true).unwrap();
        let decoded = decode_multi_input_program_bundle(&bundle).unwrap();
        assert!(decoded.state_included);
        assert_eq!(decoded.plan, graph.input_plan_v1());

        let mut target = LayerRegistry::new();
        let old = AgentLayerSpec::relu(99);
        target.init_agent_layer(&old).unwrap();
        let imported = import_multi_input_program_bundle(&mut target, &bundle).unwrap();
        assert_eq!(imported.program_identity(), identity);
        assert_eq!(imported.input_plan_v1(), graph.input_plan_v1());
        assert_eq!(multi_input_output(&target, &imported, &plan), expected);
        assert!(!target.layer_exists(old.layer_type(), old.layer_id()));
        assert!(imported.validate_registry_binding(&target).is_ok());

        let stateful = export_multi_input_program_bundle(&graph, &source, true).unwrap();
        let mut changed = weights;
        changed[0] += 1.0;
        source.set_weights_flat(22, LAYER_LINEAR, &changed).unwrap();
        assert_eq!(graph.program_identity(), identity);
        let changed_state = export_multi_input_program_bundle(&graph, &source, true).unwrap();
        assert_ne!(
            stateful, changed_state,
            "state checkpoint bytes did not track changed weights"
        );
    }

    #[test]
    fn multi_input_structure_only_bundle_and_failed_import_are_safe() {
        let mut source = LayerRegistry::new();
        let (_add, _linear, _builder, _plan, graph) = two_input_linear_graph(&mut source);
        let bundle = export_multi_input_program_bundle(&graph, &source, false).unwrap();
        let decoded = decode_multi_input_program_bundle(&bundle).unwrap();
        assert!(!decoded.state_included);
        assert!(decoded.layers.iter().all(|layer| layer.state.is_empty()));

        let mut target = LayerRegistry::new();
        let existing = AgentLayerSpec::relu(99);
        target.init_agent_layer(&existing).unwrap();
        let mut corrupt = bundle.clone();
        corrupt.truncate(corrupt.len() - 1);
        assert!(import_multi_input_program_bundle(&mut target, &corrupt).is_err());
        assert!(target.layer_exists(existing.layer_type(), existing.layer_id()));

        let imported = import_multi_input_program_bundle(&mut target, &bundle).unwrap();
        assert_eq!(imported.program_identity(), graph.program_identity());
        assert!(!target.layer_exists(existing.layer_type(), existing.layer_id()));
    }

    #[test]
    fn multi_input_bundle_capability_separates_state_from_authentication() {
        let caps = multi_input_program_bundle_capabilities();
        assert!(caps.contains("burn-research.multi-input-program-bundle.v1"));
        assert!(caps.contains("optional_separate_layer_state_section"));
        assert!(caps.contains("no_signature_or_authentication"));
        assert!(caps.contains("\"authorization\":false"));
    }

    /// R-18 (complaint #14): corrupt bundle bytes must never panic or abort the
    /// importing process. A full single-byte-flip sweep over an exported
    /// bundle asserts every mutation either imports cleanly or returns a
    /// per-call `Err` — zero panics. On WASM, a panic here would be an
    /// `unreachable` trap that permanently wedges the in-process runtime.
    #[test]
    fn r18_corrupt_bundle_import_never_panics() {
        use std::panic::AssertUnwindSafe;
        let mut source = LayerRegistry::new();
        let lin0 = AgentLayerSpec::linear(0, 2, 2, false).unwrap();
        let lin1 = AgentLayerSpec::linear(1, 2, 2, false).unwrap();
        let add = AgentLayerSpec::add(2);
        let act = AgentLayerSpec::relu(3);
        for s in [&lin0, &lin1, &add, &act] {
            source.init_agent_layer(s).unwrap();
        }
        source
            .set_weights_flat(0, crate::protocol::LAYER_LINEAR, &[1.0, 0.5, -0.25, 2.0])
            .unwrap();
        source
            .set_weights_flat(1, crate::protocol::LAYER_LINEAR, &[2.0, 1.0, 0.5, -1.0])
            .unwrap();
        let mut builder = AgentGraphBuilder::new(6).unwrap();
        builder.add_unary(&lin0, 0, 2).unwrap();
        builder.add_unary(&lin1, 1, 3).unwrap();
        builder.add_binary(&add, 2, 3, 4).unwrap();
        builder.add_unary(&act, 4, 5).unwrap();
        builder.set_output(5).unwrap();
        let mut plan = builder.multi_input_plan_v1().unwrap();
        plan.add_input_port(
            0,
            "observation".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            false,
            0,
        )
        .unwrap();
        plan.add_input_port(
            1,
            "observation".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            false,
            0,
        )
        .unwrap();
        let graph = source.compile_multi_input_graph(&plan).unwrap();
        let bundle = export_multi_input_program_bundle(&graph, &source, true).unwrap();
        // Pinned case from the complaint: 1525-byte bundle, offset 973 set to
        // 0xFF (corrupts linear in_dim to 0xFF000002). Must be a per-call Err,
        // never a panic/abort.
        assert_eq!(
            bundle.len(),
            1525,
            "bundle layout changed; re-pin offset 973"
        );
        let mut pinned = bundle.clone();
        pinned[973] = 0xFF;
        let pinned_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut target = LayerRegistry::new();
            import_multi_input_program_bundle(&mut target, &pinned)
        }));
        assert!(
            matches!(pinned_result, Ok(Err(_))),
            "pinned corrupt import must be a per-call Err"
        );
        let mut panics = Vec::new();
        let mut errs = 0;
        for off in 0..bundle.len() {
            let mut b = bundle.clone();
            b[off] ^= 0xff;
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut target = LayerRegistry::new();
                import_multi_input_program_bundle(&mut target, &b)
            }));
            match r {
                Ok(Ok(_)) => {}
                Ok(Err(_)) => errs += 1,
                Err(_) => panics.push(off),
            }
        }
        assert!(panics.is_empty(), "panicking offsets: {panics:?}");
        assert!(errs > 0, "sweep mutated nothing observable");
    }

    /// R-19 (complaint #15): fresh layers have deterministic initial weights.
    ///
    /// Two fresh registries built from the same specs must produce identical
    /// weights (no implicit RNG): linear/conv/embedding are exactly all-zero,
    /// norms carry Burn's deterministic defaults (gamma=1, beta=0). Layers
    /// without a weight accessor (ghost/seblock/swiglu) are covered through
    /// identical forward outputs from two fresh instances.
    #[test]
    fn r19_fresh_layers_have_deterministic_zero_weights() {
        use crate::layers::activation::WasmActivation;
        use crate::layers::custom::ghost::WasmGhostModule;
        use crate::layers::custom::seblock::WasmSeBlock;
        use crate::protocol::{LAYER_CONV, LAYER_EMBEDDING, LAYER_NORM};

        let specs: Vec<(AgentLayerSpec, u8, u32)> = vec![
            (
                AgentLayerSpec::linear(0, 4, 3, true).unwrap(),
                LAYER_LINEAR,
                0,
            ),
            (
                AgentLayerSpec::conv2d(1, 2, 3, 2, 2, None, None, None, None).unwrap(),
                LAYER_CONV,
                1,
            ),
            (
                AgentLayerSpec::embedding(2, 8, 4).unwrap(),
                LAYER_EMBEDDING,
                2,
            ),
            (
                AgentLayerSpec::layer_norm(3, 4, None).unwrap(),
                LAYER_NORM,
                3,
            ),
        ];

        let mut r1 = LayerRegistry::new();
        let mut r2 = LayerRegistry::new();
        for (spec, _, _) in &specs {
            r1.init_agent_layer(spec).unwrap();
            r2.init_agent_layer(spec).unwrap();
        }

        for (_spec, layer_type, layer_id) in &specs {
            let w1 = r1.get_weights_flat(*layer_id, *layer_type).unwrap();
            let w2 = r2.get_weights_flat(*layer_id, *layer_type).unwrap();
            assert_eq!(
                w1, w2,
                "fresh weights differ between registries: type={layer_type:#04X} id={layer_id}"
            );
            assert!(
                !w1.is_empty(),
                "fresh weights must not be empty: type={layer_type:#04X} id={layer_id}"
            );
            if *layer_type == LAYER_NORM {
                // Burn deterministic defaults: gamma=1, beta=0.
                let half = w1.len() / 2;
                assert!(
                    w1[..half].iter().all(|&v| v == 1.0),
                    "norm gamma != 1: type={layer_type:#04X} id={layer_id}"
                );
                assert!(
                    w1[half..].iter().all(|&v| v == 0.0),
                    "norm beta != 0: type={layer_type:#04X} id={layer_id}"
                );
            } else {
                assert!(
                    w1.iter().all(|&v| v == 0.0),
                    "fresh weights not all-zero: type={layer_type:#04X} id={layer_id}"
                );
            }
        }

        // Layers without a weight accessor: identical forward outputs from
        // two fresh instances prove deterministic initialization.
        let ghost_in = WasmTensor::new(&[0.5; 2 * 4 * 4], &[1, 2, 4, 4]);
        let g1 = WasmGhostModule::try_new(2, 4, 2, 2, Some(2), None, None, None, None).unwrap();
        let g2 = WasmGhostModule::try_new(2, 4, 2, 2, Some(2), None, None, None, None).unwrap();
        assert_eq!(
            g1.forward(&ghost_in).to_array(),
            g2.forward(&ghost_in).to_array()
        );

        let se_in = WasmTensor::new(&[0.25; 8 * 2 * 2], &[1, 8, 2, 2]);
        let s1 = WasmSeBlock::try_new(8, Some(4)).unwrap();
        let s2 = WasmSeBlock::try_new(8, Some(4)).unwrap();
        assert_eq!(s1.forward(&se_in).to_array(), s2.forward(&se_in).to_array());

        let sw_in = WasmTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let a1 = WasmActivation::new_swiglu(4, 4, Some(true));
        let a2 = WasmActivation::new_swiglu(4, 4, Some(true));
        assert_eq!(a1.forward(&sw_in).to_array(), a2.forward(&sw_in).to_array());
    }

    // Regression test (complaint #15 follow-up; CI durable-ingress diagnostic
    // 2026-10-01): two fresh same-structure graphs must export byte-identical
    // state checkpoints. Burn assigns each `Param` a random `ParamId` at
    // creation and serializes it as the tensor's record name, so `get_state()`
    // used to emit different bytes per export even with identical weights.
    // `deterministic_record_bytes` re-keys those ids on the export clone.
    fn fresh_audit_graph() -> (LayerRegistry, CompiledMultiInputGraph) {
        let mut registry = LayerRegistry::new();
        let add = AgentLayerSpec::add(31);
        let linear = AgentLayerSpec::linear(32, 2, 2, true).unwrap();
        registry.init_agent_layer(&add).unwrap();
        registry.init_agent_layer(&linear).unwrap();
        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_binary(&add, 0, 1, 2).unwrap();
        builder.add_unary(&linear, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let mut plan = builder.multi_input_plan_v1().unwrap();
        plan.add_input_port(
            0,
            "observation".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            1,
        )
        .unwrap();
        plan.add_input_port(
            1,
            "state".into(),
            1,
            2,
            1,
            1,
            "feature_axis1_singleton".into(),
            true,
            2,
        )
        .unwrap();
        let graph = registry.compile_multi_input_graph(&plan).unwrap();
        (registry, graph)
    }

    #[test]
    fn two_fresh_graphs_export_identical_state_checkpoints() {
        let (r1, g1) = fresh_audit_graph();
        let (r2, g2) = fresh_audit_graph();
        assert_eq!(
            g1.program_identity(),
            g2.program_identity(),
            "fixture graphs diverged in program identity"
        );
        let b1 = export_multi_input_program_bundle(&g1, &r1, true).unwrap();
        let b2 = export_multi_input_program_bundle(&g2, &r2, true).unwrap();
        assert_eq!(
            b1, b2,
            "two fresh same-structure graphs diverged in checkpoint bytes"
        );
    }

    #[test]
    fn state_checkpoint_bytes_still_track_weight_changes() {
        // Companion to the determinism test above: normalization must not
        // erase real state differences.
        let (mut r1, g1) = fresh_audit_graph();
        let (r2, g2) = fresh_audit_graph();
        let b1 = export_multi_input_program_bundle(&g1, &r1, true).unwrap();
        let b2 = export_multi_input_program_bundle(&g2, &r2, true).unwrap();
        assert_eq!(b1, b2);
        let mut weights = r1.get_weights_flat(32, LAYER_LINEAR).unwrap();
        weights[0] += 1.0;
        r1.set_weights_flat(32, LAYER_LINEAR, &weights).unwrap();
        let b1_changed = export_multi_input_program_bundle(&g1, &r1, true).unwrap();
        assert_ne!(
            b1, b1_changed,
            "state checkpoint bytes did not track changed weights"
        );
    }
}
