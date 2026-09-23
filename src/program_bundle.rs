use wasm_bindgen::prelude::*;

use crate::graph::{CompiledGraph, CompiledMultiInputGraph};
use crate::graph_plan::decode_graph_plan;
use crate::multi_input_graph::MultiInputGraphPlan;
use crate::protocol::{PacketHeader, OP_INIT};
use crate::registry::LayerRegistry;

const BUNDLE_MAGIC: &[u8; 8] = b"BRPGBNDL";
const MULTI_INPUT_BUNDLE_MAGIC: &[u8; 8] = b"BRMIBNDL";
const BUNDLE_SCHEMA_VERSION: u32 = 1;
const BUNDLE_FLAG_STATE_INCLUDED: u32 = 1 << 0;
const BUNDLE_KNOWN_FLAGS: u32 = BUNDLE_FLAG_STATE_INCLUDED;

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn checked_u32(value: usize, context: &str) -> Result<u32, String> {
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

fn referenced_layer_keys(plan: &[u8]) -> Result<Vec<(u8, u32)>, String> {
    decode_graph_plan(plan)
        .map(|decoded| decoded.unique_first_use_layer_keys())
        .map_err(|error| format!("program bundle: {error}"))
}

fn multi_input_referenced_layer_keys(plan: &[u8]) -> Result<Vec<(u8, u32)>, String> {
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

fn parse_init_fingerprint(
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

struct DecodedLayer {
    layer_type: u8,
    variant: u8,
    flags: u8,
    layer_id: u32,
    init_payload: Vec<u8>,
    state: Vec<u8>,
}

struct DecodedBundle {
    state_included: bool,
    plan: Vec<u8>,
    expected_identity: String,
    layers: Vec<DecodedLayer>,
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

fn decode_program_bundle(bundle: &[u8]) -> Result<DecodedBundle, String> {
    decode_bundle(
        bundle,
        BUNDLE_MAGIC,
        "program bundle",
        referenced_layer_keys,
    )
}

fn decode_multi_input_program_bundle(bundle: &[u8]) -> Result<DecodedBundle, String> {
    decode_bundle(
        bundle,
        MULTI_INPUT_BUNDLE_MAGIC,
        "multi-input program bundle",
        multi_input_referenced_layer_keys,
    )
}

#[wasm_bindgen(js_name = programBundleCapabilities)]
pub fn program_bundle_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.program-bundle.v1\",",
        "\"schema_version\":1,",
        "\"export\":\"exportProgramBundle\",",
        "\"import\":\"importProgramBundle\",",
        "\"structural_identity\":\"burn-research.program-identity.v1\",",
        "\"structural_source\":\"program-identity.v1_layer_init_fingerprint\",",
        "\"target_registry\":\"atomic_replace_on_success\",",
        "\"import_commit\":\"atomic_after_identity_validation\",",
        "\"mutable_state\":\"optional_separate_section\"",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = exportProgramBundle)]
pub fn export_program_bundle(
    graph: &CompiledGraph,
    registry: &LayerRegistry,
    include_state: bool,
) -> Result<Vec<u8>, String> {
    graph
        .validate_registry_binding(registry)
        .map_err(|error| format!("exportProgramBundle: {error}"))?;

    let plan = graph.program_plan();
    let identity = graph.program_identity();
    let keys = referenced_layer_keys(&plan)?;
    let mut records = Vec::with_capacity(keys.len());
    for (layer_type, layer_id) in keys {
        let fingerprint = registry
            .layer_init_fingerprint(layer_type, layer_id)
            .map_err(|error| format!("exportProgramBundle: {error}"))?;
        let (variant, flags, init_payload) =
            parse_init_fingerprint(&fingerprint, layer_type, layer_id)?;
        let state = if include_state {
            registry
                .get_layer_state(layer_id, layer_type)
                .map_err(|error| format!("exportProgramBundle: state for type 0x{layer_type:02X} id {layer_id}: {error}"))?
        } else {
            Vec::new()
        };
        records.push((layer_type, variant, flags, layer_id, init_payload, state));
    }

    let mut out = Vec::new();
    out.extend_from_slice(BUNDLE_MAGIC);
    push_u32(&mut out, BUNDLE_SCHEMA_VERSION);
    push_u32(
        &mut out,
        if include_state {
            BUNDLE_FLAG_STATE_INCLUDED
        } else {
            0
        },
    );
    push_u32(&mut out, checked_u32(plan.len(), "exportProgramBundle plan")?);
    push_u32(
        &mut out,
        checked_u32(identity.len(), "exportProgramBundle identity")?,
    );
    push_u32(
        &mut out,
        checked_u32(records.len(), "exportProgramBundle layer count")?,
    );
    out.extend_from_slice(&plan);
    out.extend_from_slice(identity.as_bytes());

    for (layer_type, variant, flags, layer_id, init_payload, state) in records {
        out.push(layer_type);
        out.push(variant);
        out.push(flags);
        out.push(0);
        push_u32(&mut out, layer_id);
        push_u32(
            &mut out,
            checked_u32(init_payload.len(), "exportProgramBundle init payload")?,
        );
        push_u32(
            &mut out,
            checked_u32(state.len(), "exportProgramBundle layer state")?,
        );
        out.extend_from_slice(&init_payload);
        out.extend_from_slice(&state);
    }
    Ok(out)
}

#[wasm_bindgen(js_name = importProgramBundle)]
pub fn import_program_bundle(
    registry: &mut LayerRegistry,
    bundle: &[u8],
) -> Result<CompiledGraph, String> {
    let decoded = decode_program_bundle(bundle)
        .map_err(|error| format!("importProgramBundle: {error}"))?;
    let mut staged = LayerRegistry::new();
    for (index, layer) in decoded.layers.iter().enumerate() {
        let payload_len = checked_u32(
            layer.init_payload.len(),
            "importProgramBundle init payload",
        )?;
        let header = PacketHeader {
            opcode: OP_INIT,
            layer_type: layer.layer_type,
            variant: layer.variant,
            flags: layer.flags,
            payload_len,
        };
        staged
            .init_layer(&header, &layer.init_payload)
            .map_err(|error| format!("importProgramBundle: layer {index} init failed: {error}"))?;
        if decoded.state_included {
            staged
                .load_layer_state(layer.layer_id, layer.layer_type, &layer.state)
                .map_err(|error| format!("importProgramBundle: layer {index} state failed: {error}"))?;
        }
    }

    let graph = staged
        .compile_graph(&decoded.plan)
        .map_err(|error| format!("importProgramBundle: compile failed: {error}"))?;
    let actual_identity = graph.program_identity();
    if actual_identity != decoded.expected_identity {
        return Err(format!(
            "importProgramBundle: structural identity mismatch: expected {}, got {}",
            decoded.expected_identity, actual_identity
        ));
    }
    graph
        .validate_registry_binding(&staged)
        .map_err(|error| format!("importProgramBundle: staged binding invalid: {error}"))?;

    *registry = staged;
    Ok(graph)
}

#[wasm_bindgen(js_name = multiInputProgramBundleCapabilities)]
pub fn multi_input_program_bundle_capabilities() -> String {
    concat!(
        "{",
        "\"schema\":\"burn-research.multi-input-program-bundle.v1\",",
        "\"schema_version\":1,",
        "\"export\":\"exportMultiInputProgramBundle\",",
        "\"import\":\"importMultiInputProgramBundle\",",
        "\"structural_identity\":\"burn-research.multi-input-program-identity.v1\",",
        "\"structural_source\":\"exact_multi_input_plan_and_layer_init_fingerprints\",",
        "\"mutable_state\":\"optional_separate_layer_state_section\",",
        "\"state_integrity\":\"no_signature_or_authentication\",",
        "\"target_registry\":\"atomic_replace_on_success\",",
        "\"import_commit\":\"atomic_after_identity_validation\",",
        "\"authorization\":false",
        "}"
    )
    .to_string()
}

#[wasm_bindgen(js_name = exportMultiInputProgramBundle)]
pub fn export_multi_input_program_bundle(
    graph: &CompiledMultiInputGraph,
    registry: &LayerRegistry,
    include_state: bool,
) -> Result<Vec<u8>, String> {
    graph
        .validate_registry_binding(registry)
        .map_err(|error| format!("exportMultiInputProgramBundle: {error}"))?;

    let plan = graph.input_plan_v1();
    let identity = graph.program_identity();
    let keys = multi_input_referenced_layer_keys(&plan)?;
    let mut records = Vec::with_capacity(keys.len());
    for (layer_type, layer_id) in keys {
        let fingerprint = registry
            .layer_init_fingerprint(layer_type, layer_id)
            .map_err(|error| format!("exportMultiInputProgramBundle: {error}"))?;
        let (variant, flags, init_payload) =
            parse_init_fingerprint(&fingerprint, layer_type, layer_id)?;
        let state = if include_state {
            registry
                .get_layer_state(layer_id, layer_type)
                .map_err(|error| format!("exportMultiInputProgramBundle: state for type 0x{layer_type:02X} id {layer_id}: {error}"))?
        } else {
            Vec::new()
        };
        records.push((layer_type, variant, flags, layer_id, init_payload, state));
    }

    let mut out = Vec::new();
    out.extend_from_slice(MULTI_INPUT_BUNDLE_MAGIC);
    push_u32(&mut out, BUNDLE_SCHEMA_VERSION);
    push_u32(
        &mut out,
        if include_state {
            BUNDLE_FLAG_STATE_INCLUDED
        } else {
            0
        },
    );
    push_u32(
        &mut out,
        checked_u32(plan.len(), "exportMultiInputProgramBundle plan")?,
    );
    push_u32(
        &mut out,
        checked_u32(identity.len(), "exportMultiInputProgramBundle identity")?,
    );
    push_u32(
        &mut out,
        checked_u32(records.len(), "exportMultiInputProgramBundle layer count")?,
    );
    out.extend_from_slice(&plan);
    out.extend_from_slice(identity.as_bytes());
    for (layer_type, variant, flags, layer_id, init_payload, state) in records {
        out.push(layer_type);
        out.push(variant);
        out.push(flags);
        out.push(0);
        push_u32(&mut out, layer_id);
        push_u32(
            &mut out,
            checked_u32(
                init_payload.len(),
                "exportMultiInputProgramBundle init payload",
            )?,
        );
        push_u32(
            &mut out,
            checked_u32(state.len(), "exportMultiInputProgramBundle layer state")?,
        );
        out.extend_from_slice(&init_payload);
        out.extend_from_slice(&state);
    }
    Ok(out)
}

#[wasm_bindgen(js_name = importMultiInputProgramBundle)]
pub fn import_multi_input_program_bundle(
    registry: &mut LayerRegistry,
    bundle: &[u8],
) -> Result<crate::graph::CompiledMultiInputGraph, String> {
    let decoded = decode_multi_input_program_bundle(bundle)
        .map_err(|error| format!("importMultiInputProgramBundle: {error}"))?;
    let input_plan = MultiInputGraphPlan::from_bytes(&decoded.plan)
        .map_err(|error| format!("importMultiInputProgramBundle: {error}"))?;

    let mut staged = LayerRegistry::new();
    for (index, layer) in decoded.layers.iter().enumerate() {
        let payload_len = checked_u32(
            layer.init_payload.len(),
            "importMultiInputProgramBundle init payload",
        )?;
        let header = PacketHeader {
            opcode: OP_INIT,
            layer_type: layer.layer_type,
            variant: layer.variant,
            flags: layer.flags,
            payload_len,
        };
        staged
            .init_layer(&header, &layer.init_payload)
            .map_err(|error| format!("importMultiInputProgramBundle: layer {index} init failed: {error}"))?;
        if decoded.state_included {
            staged
                .load_layer_state(layer.layer_id, layer.layer_type, &layer.state)
                .map_err(|error| format!("importMultiInputProgramBundle: layer {index} state failed: {error}"))?;
        }
    }

    let graph = staged
        .compile_multi_input_graph(&input_plan)
        .map_err(|error| format!("importMultiInputProgramBundle: compile failed: {error}"))?;
    let actual_identity = graph.program_identity();
    if actual_identity != decoded.expected_identity {
        return Err(format!(
            "importMultiInputProgramBundle: structural identity mismatch: expected {}, got {}",
            decoded.expected_identity, actual_identity
        ));
    }
    graph
        .validate_registry_binding(&staged)
        .map_err(|error| format!("importMultiInputProgramBundle: staged binding invalid: {error}"))?;

    *registry = staged;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::{decode_multi_input_program_bundle, decode_program_bundle, export_multi_input_program_bundle,
        export_program_bundle, import_multi_input_program_bundle, import_program_bundle,
        multi_input_program_bundle_capabilities, parse_init_fingerprint, program_bundle_capabilities};
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::graph::CompiledMultiInputGraph;
    use crate::multi_input_graph::{MultiInputGraphPlan, MultiInputInputBundle};
    use crate::protocol::LAYER_LINEAR;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn linear_graph(registry: &mut LayerRegistry) -> (AgentLayerSpec, AgentGraphBuilder, crate::graph::CompiledGraph) {
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
    ) -> (AgentLayerSpec, AgentLayerSpec, AgentGraphBuilder, MultiInputGraphPlan, CompiledMultiInputGraph) {
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
        assert_eq!(u32::from_le_bytes(payload[0..4].try_into().unwrap()), spec.layer_id());
        assert!(parse_init_fingerprint("future-format", spec.layer_type(), spec.layer_id()).is_err());
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
        let original_steps = u32::from_le_bytes(
            corrupt[plan_start..plan_start + 4]
                .try_into()
                .unwrap(),
        );
        corrupt[plan_start..plan_start + 4]
            .copy_from_slice(&(original_steps + 1).to_le_bytes());

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
        assert_eq!(graph.program_identity(), identity, "mutable weights changed structural identity");
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
        assert_ne!(stateful, changed_state, "state checkpoint bytes did not track changed weights");
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
}
