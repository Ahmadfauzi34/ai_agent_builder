use crate::graph::CompiledGraph;
use crate::graph_plan::decode_graph_plan;
use crate::protocol::{
    ACT_GELU, ACT_GLU, ACT_HARDSIGMOID, ACT_HARDSWISH, ACT_LEAKYRELU, ACT_LOGSOFTMAX,
    ACT_MISH, ACT_PRELU, ACT_RELU, ACT_SIGMOID, ACT_SOFTMAX, ACT_SOFTPLUS, ACT_SWIGLU,
    ACT_TANH, LAYER_ACTIVATION, LAYER_BINARY, LAYER_CONV, LAYER_EMBEDDING,
    LAYER_FEATURE_NORM, LAYER_GHOST, LAYER_LINEAR, LAYER_NORM, LAYER_POOL, LAYER_SEBLOCK,
    LAYER_SHIFT,
};
use crate::registry::LayerRegistry;

const BINDING_SCHEMA: &str = "burn-research.graph-parameter-binding.v1";
const LAYOUT_SCHEMA: &str = "burn-research.graph-parameter-layout.v1";
const ORDERING: &str = "unique_first_use_graph_plan";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphParameterOwner {
    layer_type: u8,
    layer_id: u32,
    offset: usize,
    len: usize,
    weight_layout: String,
    init_fingerprint: String,
}

impl GraphParameterOwner {
    pub fn layer_type(&self) -> u8 {
        self.layer_type
    }

    pub fn layer_id(&self) -> u32 {
        self.layer_id
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn weight_layout(&self) -> &str {
        &self.weight_layout
    }

    pub fn init_fingerprint(&self) -> &str {
        &self.init_fingerprint
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphParameterBinding {
    program_identity: String,
    owners: Vec<GraphParameterOwner>,
    total_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterClass {
    FlatTrainable,
    Stateless,
}

// Structural graph step order, deduplicated by (layer_type, layer_id) on first use.
// Candidate coordinates must never be derived from raw graph step count because one
// trainable owner may be referenced by multiple steps.
fn referenced_layer_keys(plan: &[u8]) -> Result<Vec<(u8, u32)>, String> {
    decode_graph_plan(plan)
        .map(|decoded| decoded.unique_first_use_layer_keys())
        .map_err(|error| format!("graph parameter binding: {error}"))
}

fn fingerprint_variant(fingerprint: &str) -> Result<u8, String> {
    let raw = fingerprint
        .split(';')
        .find_map(|part| part.strip_prefix("variant="))
        .ok_or_else(|| {
            "graph parameter binding: layer init fingerprint is missing variant".to_string()
        })?;
    u8::from_str_radix(raw, 16).map_err(|_| {
        format!(
            "graph parameter binding: invalid variant in layer init fingerprint {fingerprint:?}"
        )
    })
}

fn classify_owner(layer_type: u8, fingerprint: &str) -> Result<ParameterClass, String> {
    match layer_type {
        LAYER_LINEAR | LAYER_CONV | LAYER_EMBEDDING | LAYER_NORM => {
            Ok(ParameterClass::FlatTrainable)
        }
        LAYER_POOL | LAYER_SHIFT | LAYER_BINARY | LAYER_FEATURE_NORM => {
            Ok(ParameterClass::Stateless)
        }
        LAYER_ACTIVATION => {
            let variant = fingerprint_variant(fingerprint)?;
            match variant {
                ACT_GELU | ACT_RELU | ACT_SIGMOID | ACT_TANH | ACT_HARDSWISH
                | ACT_LEAKYRELU | ACT_HARDSIGMOID | ACT_SOFTPLUS | ACT_MISH
                | ACT_SOFTMAX | ACT_LOGSOFTMAX | ACT_GLU => Ok(ParameterClass::Stateless),
                ACT_PRELU | ACT_SWIGLU => Err(format!(
                    "graph parameter binding: parameterized activation variant 0x{variant:02X} has no flat-weight bridge"
                )),
                _ => Err(format!(
                    "graph parameter binding: unknown activation variant 0x{variant:02X}; refusing to assume it is stateless"
                )),
            }
        }
        LAYER_GHOST | LAYER_SEBLOCK => Err(format!(
            "graph parameter binding: referenced parameterized layer type 0x{layer_type:02X} has no flat-weight bridge"
        )),
        _ => Err(format!(
            "graph parameter binding: unsupported referenced layer type 0x{layer_type:02X}"
        )),
    }
}

fn checked_total_len(current: usize, next: usize) -> Result<usize, String> {
    current
        .checked_add(next)
        .ok_or_else(|| "graph parameter binding: total parameter length overflow".to_string())
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{1f}' => {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "\\u{:04x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn owner_layout_json(owner: &GraphParameterOwner, include_fingerprint: bool) -> String {
    let mut out = format!(
        "{{\"layer_type\":{},\"layer_id\":{},\"offset\":{},\"len\":{},\"weight_layout\":{}",
        owner.layer_type, owner.layer_id, owner.offset, owner.len, owner.weight_layout
    );
    if include_fingerprint {
        out.push_str(",\"init_fingerprint\":");
        out.push_str(&json_string(&owner.init_fingerprint));
    }
    out.push('}');
    out
}

impl GraphParameterBinding {
    pub fn build(graph: &CompiledGraph, registry: &LayerRegistry) -> Result<Self, String> {
        graph
            .validate_registry_binding(registry)
            .map_err(|error| format!("graph parameter binding: {error}"))?;

        let plan = graph.program_plan();
        let keys = referenced_layer_keys(&plan)?;
        let mut owners = Vec::new();
        let mut total_len = 0usize;

        for (layer_type, layer_id) in keys {
            let fingerprint = registry
                .layer_init_fingerprint(layer_type, layer_id)
                .map_err(|error| format!("graph parameter binding: {error}"))?;
            match classify_owner(layer_type, &fingerprint)? {
                ParameterClass::Stateless => continue,
                ParameterClass::FlatTrainable => {
                    let weights = registry
                        .get_weights_flat(layer_id, layer_type)
                        .map_err(|error| format!("graph parameter binding: {error}"))?;
                    let layout = registry
                        .weight_layout(layer_id, layer_type)
                        .map_err(|error| format!("graph parameter binding: {error}"))?;
                    if weights.is_empty() {
                        continue;
                    }
                    let len = weights.len();
                    let offset = total_len;
                    total_len = checked_total_len(total_len, len)?;
                    owners.push(GraphParameterOwner {
                        layer_type,
                        layer_id,
                        offset,
                        len,
                        weight_layout: layout,
                        init_fingerprint: fingerprint,
                    });
                }
            }
        }

        Ok(Self {
            program_identity: graph.program_identity(),
            owners,
            total_len,
        })
    }

    pub fn owners(&self) -> &[GraphParameterOwner] {
        &self.owners
    }

    pub fn total_len(&self) -> usize {
        self.total_len
    }

    pub fn layout_json(&self) -> String {
        let owners = self
            .owners
            .iter()
            .map(|owner| owner_layout_json(owner, false))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{LAYOUT_SCHEMA}\",\"ordering\":\"{ORDERING}\",\"total_len\":{},\"owners\":[{}]}}",
            self.total_len, owners
        )
    }

    pub fn identity_json(&self) -> String {
        let owners = self
            .owners
            .iter()
            .map(|owner| owner_layout_json(owner, true))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schema\":\"{BINDING_SCHEMA}\",\"ordering\":\"{ORDERING}\",\"program_identity\":{},\"total_len\":{},\"owners\":[{}]}}",
            self.program_identity, self.total_len, owners
        )
    }

    fn validate_current(
        &self,
        graph: &CompiledGraph,
        registry: &LayerRegistry,
        context: &str,
    ) -> Result<(), String> {
        if graph.program_identity() != self.program_identity {
            return Err(format!(
                "{context}: graph structural identity differs from the binding identity"
            ));
        }
        graph
            .validate_registry_binding(registry)
            .map_err(|error| format!("{context}: {error}"))?;

        let rebuilt = Self::build(graph, registry)
            .map_err(|error| format!("{context}: cannot rebuild binding: {error}"))?;
        if rebuilt.owners != self.owners || rebuilt.total_len != self.total_len {
            return Err(format!(
                "{context}: current graph parameter layout differs from the binding"
            ));
        }
        Ok(())
    }

    pub fn read_flat(
        &self,
        graph: &CompiledGraph,
        registry: &LayerRegistry,
    ) -> Result<Vec<f32>, String> {
        self.validate_current(graph, registry, "graph parameter read")?;
        let mut out = Vec::with_capacity(self.total_len);
        for owner in &self.owners {
            let weights = registry
                .get_weights_flat(owner.layer_id, owner.layer_type)
                .map_err(|error| format!("graph parameter read: {error}"))?;
            if weights.len() != owner.len {
                return Err(format!(
                    "graph parameter read: layer type 0x{:02X} id {} length changed from {} to {}",
                    owner.layer_type,
                    owner.layer_id,
                    owner.len,
                    weights.len()
                ));
            }
            out.extend_from_slice(&weights);
        }
        if out.len() != self.total_len {
            return Err(format!(
                "graph parameter read: internal total length mismatch: expected {}, got {}",
                self.total_len,
                out.len()
            ));
        }
        Ok(out)
    }

    pub fn apply_flat(
        &self,
        graph: &CompiledGraph,
        registry: &mut LayerRegistry,
        candidate: &[f32],
    ) -> Result<(), String> {
        self.validate_current(graph, registry, "graph parameter apply")?;
        if candidate.len() != self.total_len {
            return Err(format!(
                "graph parameter apply: expected {} floats, got {}",
                self.total_len,
                candidate.len()
            ));
        }

        // graph-parameter-binding.v1 treats finite-only candidate values as a safety
        // invariant. Reject the whole candidate before any owner setter can run.
        if let Some((index, _)) = candidate
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "graph parameter apply: candidate contains non-finite value at index {index}"
            ));
        }

        // Full cross-layer prevalidation happens before the first mutation. The current
        // LayerRegistry flat setters for Linear/Conv/Embedding/Norm are length-validated
        // record replacements; after exact lengths/support have been proven here they have
        // no remaining candidate-data-dependent rejection path.
        for owner in &self.owners {
            let end = owner
                .offset
                .checked_add(owner.len)
                .ok_or_else(|| "graph parameter apply: slice boundary overflow".to_string())?;
            if end > candidate.len() {
                return Err(format!(
                    "graph parameter apply: layer type 0x{:02X} id {} slice {}..{} exceeds candidate length {}",
                    owner.layer_type,
                    owner.layer_id,
                    owner.offset,
                    end,
                    candidate.len()
                ));
            }
            let current_len = registry
                .get_weights_flat(owner.layer_id, owner.layer_type)
                .map_err(|error| format!("graph parameter apply: {error}"))?
                .len();
            if current_len != owner.len {
                return Err(format!(
                    "graph parameter apply: layer type 0x{:02X} id {} expected {} floats, current bridge exposes {}",
                    owner.layer_type, owner.layer_id, owner.len, current_len
                ));
            }
            let current_layout = registry
                .weight_layout(owner.layer_id, owner.layer_type)
                .map_err(|error| format!("graph parameter apply: {error}"))?;
            if current_layout != owner.weight_layout {
                return Err(format!(
                    "graph parameter apply: layer type 0x{:02X} id {} weight layout changed",
                    owner.layer_type, owner.layer_id
                ));
            }
        }

        for owner in &self.owners {
            let end = owner.offset + owner.len;
            registry
                .set_weights_flat(
                    owner.layer_id,
                    owner.layer_type,
                    &candidate[owner.offset..end],
                )
                .map_err(|error| {
                    format!(
                        "graph parameter apply: prevalidated setter failed for layer type 0x{:02X} id {}: {error}",
                        owner.layer_type, owner.layer_id
                    )
                })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_total_len, GraphParameterBinding, BINDING_SCHEMA, LAYOUT_SCHEMA};
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::protocol::LAYER_LINEAR;
    use crate::registry::LayerRegistry;
    use crate::WasmTensor;

    fn init(registry: &mut LayerRegistry, spec: &AgentLayerSpec) {
        registry.init_agent_layer(spec).unwrap();
    }

    fn two_linear_graph() -> (LayerRegistry, crate::graph::CompiledGraph) {
        let mut registry = LayerRegistry::new();
        let first = AgentLayerSpec::linear(11, 2, 2, true).unwrap();
        let relu = AgentLayerSpec::relu(12);
        let second = AgentLayerSpec::linear(13, 2, 1, true).unwrap();
        init(&mut registry, &first);
        init(&mut registry, &relu);
        init(&mut registry, &second);

        let mut builder = AgentGraphBuilder::new(4).unwrap();
        builder.add_unary(&first, 0, 1).unwrap();
        builder.add_unary(&relu, 1, 2).unwrap();
        builder.add_unary(&second, 2, 3).unwrap();
        builder.set_output(3).unwrap();
        let graph = builder.compile(&registry).unwrap();
        (registry, graph)
    }

    #[test]
    fn single_linear_owner_layout_read_and_apply_are_canonical() {
        let mut registry = LayerRegistry::new();
        let linear = AgentLayerSpec::linear(7, 2, 1, true).unwrap();
        init(&mut registry, &linear);
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&linear, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph = builder.compile(&registry).unwrap();

        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let layer_weights = registry.get_weights_flat(7, LAYER_LINEAR).unwrap();
        assert_eq!(binding.total_len(), layer_weights.len());
        assert_eq!(binding.owners().len(), 1);
        assert_eq!(binding.owners()[0].offset(), 0);
        assert_eq!(binding.owners()[0].len(), layer_weights.len());
        assert!(binding.layout_json().contains(LAYOUT_SCHEMA));
        assert!(binding.identity_json().contains(BINDING_SCHEMA));

        let mut candidate = binding.read_flat(&graph, &registry).unwrap();
        for value in &mut candidate {
            *value += 0.25;
        }
        binding
            .apply_flat(&graph, &mut registry, &candidate)
            .unwrap();
        assert_eq!(binding.read_flat(&graph, &registry).unwrap(), candidate);
    }

    #[test]
    fn multi_layer_layout_uses_unique_first_use_trainable_owners_only() {
        let (registry, graph) = two_linear_graph();
        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let first_len = registry.get_weights_flat(11, LAYER_LINEAR).unwrap().len();
        let second_len = registry.get_weights_flat(13, LAYER_LINEAR).unwrap().len();

        assert_eq!(binding.owners().len(), 2);
        assert_eq!(binding.owners()[0].layer_id(), 11);
        assert_eq!(binding.owners()[0].offset(), 0);
        assert_eq!(binding.owners()[0].len(), first_len);
        assert_eq!(binding.owners()[1].layer_id(), 13);
        assert_eq!(binding.owners()[1].offset(), first_len);
        assert_eq!(binding.owners()[1].len(), second_len);
        assert_eq!(binding.total_len(), first_len + second_len);
        assert!(!binding.layout_json().contains("\"layer_id\":12"));
    }

    #[test]
    fn repeated_trainable_layer_reference_deduplicates_candidate_coordinates() {
        let mut registry = LayerRegistry::new();
        let linear = AgentLayerSpec::linear(21, 2, 2, true).unwrap();
        init(&mut registry, &linear);
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        builder.add_unary(&linear, 0, 1).unwrap();
        builder.add_unary(&linear, 1, 2).unwrap();
        builder.set_output(2).unwrap();
        let graph = builder.compile(&registry).unwrap();

        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        assert_eq!(binding.owners().len(), 1);
        assert_eq!(
            binding.total_len(),
            registry.get_weights_flat(21, LAYER_LINEAR).unwrap().len()
        );
    }

    #[test]
    fn malformed_total_length_fails_before_any_layer_mutation() {
        let (mut registry, graph) = two_linear_graph();
        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let before_first = registry.get_weights_flat(11, LAYER_LINEAR).unwrap();
        let before_second = registry.get_weights_flat(13, LAYER_LINEAR).unwrap();
        let mut malformed = binding.read_flat(&graph, &registry).unwrap();
        malformed.pop();

        assert!(binding.apply_flat(&graph, &mut registry, &malformed).is_err());
        assert_eq!(registry.get_weights_flat(11, LAYER_LINEAR).unwrap(), before_first);
        assert_eq!(registry.get_weights_flat(13, LAYER_LINEAR).unwrap(), before_second);
    }

    #[test]
    fn structural_registry_mismatch_fails_before_any_layer_mutation() {
        let (mut registry, graph) = two_linear_graph();
        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let candidate = binding.read_flat(&graph, &registry).unwrap();
        let before_first = registry.get_weights_flat(11, LAYER_LINEAR).unwrap();

        let replacement = AgentLayerSpec::linear(13, 2, 2, true).unwrap();
        init(&mut registry, &replacement);
        assert!(binding.apply_flat(&graph, &mut registry, &candidate).is_err());
        assert_eq!(registry.get_weights_flat(11, LAYER_LINEAR).unwrap(), before_first);
    }

    #[test]
    fn parameterized_activation_without_flat_bridge_fails_closed() {
        let mut registry = LayerRegistry::new();
        let prelu = AgentLayerSpec::prelu(31, 2, 0.25).unwrap();
        init(&mut registry, &prelu);
        let mut builder = AgentGraphBuilder::new(2).unwrap();
        builder.add_unary(&prelu, 0, 1).unwrap();
        builder.set_output(1).unwrap();
        let graph = builder.compile(&registry).unwrap();

        let err = GraphParameterBinding::build(&graph, &registry).unwrap_err();
        assert!(err.contains("parameterized activation"));
        assert!(err.contains("no flat-weight bridge"));
    }

    #[test]
    fn successful_multi_layer_apply_changes_execution_but_not_binding_identity() {
        let (mut registry, graph) = two_linear_graph();
        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let program_identity = graph.program_identity();
        let binding_identity = binding.identity_json();
        let layout = binding.layout_json();
        let input = WasmTensor::new(&[1.0, 2.0], &[1, 2, 1, 1]);

        let candidate = vec![1.0; binding.total_len()];
        binding
            .apply_flat(&graph, &mut registry, &candidate)
            .unwrap();
        let output = graph.run(&registry, &input).unwrap().to_array();
        assert_eq!(output.len(), 1);
        assert!((output[0] - 9.0).abs() < 1e-5);
        assert_eq!(graph.program_identity(), program_identity);

        let rebuilt = GraphParameterBinding::build(&graph, &registry).unwrap();
        assert_eq!(rebuilt.identity_json(), binding_identity);
        assert_eq!(rebuilt.layout_json(), layout);
        assert_eq!(rebuilt.read_flat(&graph, &registry).unwrap(), candidate);
    }

    #[test]
    fn exact_read_apply_read_replays_candidate() {
        let (mut registry, graph) = two_linear_graph();
        let binding = GraphParameterBinding::build(&graph, &registry).unwrap();
        let original = binding.read_flat(&graph, &registry).unwrap();
        binding
            .apply_flat(&graph, &mut registry, &original)
            .unwrap();
        assert_eq!(binding.read_flat(&graph, &registry).unwrap(), original);
    }

    #[test]
    fn total_length_accounting_fails_closed_on_overflow() {
        assert!(checked_total_len(usize::MAX, 1).is_err());
        assert_eq!(checked_total_len(7, 5).unwrap(), 12);
    }
}
