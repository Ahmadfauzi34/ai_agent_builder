use wasm_bindgen::prelude::*;

use crate::agent::AgentGraphBuilder;
use crate::graph::CompiledGraph;
use crate::input_port_edge_binding::semantic_graph_identity_json;
use crate::registry::LayerRegistry;
use crate::semantic_lifecycle::semantic_lifecycle_identity_json;

const SEMANTIC_EXECUTION_CONTEXT_V1: &str =
    include_str!("../docs/agent-semantic-execution-context.v1.json");

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SemanticExecutionContext {
    pub(crate) program_identity: String,
    pub(crate) semantic_graph_identity: String,
    pub(crate) semantic_lifecycle_identity: String,
    pub(crate) lifecycle_coverage_complete: bool,
    pub(crate) context_fingerprint: String,
}

impl SemanticExecutionContext {
    pub(crate) fn json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.semantic-execution-context.v1\",",
                "\"projection_only\":true,",
                "\"authority\":{{",
                    "\"execution_identity\":\"CompiledGraph.programIdentity\",",
                    "\"semantic_graph\":\"AgentGraphBuilder.semanticGraphIdentity\",",
                    "\"semantic_lifecycle\":\"AgentGraphBuilder.semanticLifecycleIdentity\",",
                    "\"context\":\"derived_projection\"",
                "}},",
                "\"program_identity\":{},",
                "\"semantic_graph_identity\":{},",
                "\"semantic_lifecycle_identity\":{},",
                "\"lifecycle_coverage_complete\":{},",
                "\"context_fingerprint\":\"{}\",",
                "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
                "\"program_identity_effect\":\"none\",",
                "\"execution_effect\":\"none\"",
                "}}"
            ),
            self.program_identity,
            self.semantic_graph_identity,
            self.semantic_lifecycle_identity,
            if self.lifecycle_coverage_complete { "true" } else { "false" },
            json_escape(&self.context_fingerprint),
        )
    }
}

pub(crate) fn semantic_execution_context_for(
    builder: &AgentGraphBuilder,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<SemanticExecutionContext, String> {
    graph
        .validate_registry_binding(registry)
        .map_err(|error| format!("semanticExecutionContext: graph registry binding invalid: {error}"))?;

    if builder.num_slots() != graph.slot_count() {
        return Err(format!(
            "semanticExecutionContext: builder num_slots {} does not match graph num_slots {}",
            builder.num_slots(),
            graph.slot_count()
        ));
    }

    let replay = builder
        .compile_with_output(registry, graph.output_slot())
        .map_err(|error| format!("semanticExecutionContext: current builder cannot compile to graph output: {error}"))?;

    let graph_program_identity = graph.program_identity();
    let replay_program_identity = replay.program_identity();
    if replay_program_identity != graph_program_identity {
        return Err(
            "semanticExecutionContext: current builder/registry executable identity does not match supplied CompiledGraph"
                .to_string(),
        );
    }

    if replay.program_plan() != graph.program_plan() {
        return Err(
            "semanticExecutionContext: current builder canonical plan does not match supplied CompiledGraph"
                .to_string(),
        );
    }

    let semantic_graph_identity = semantic_graph_identity_json(builder);
    let semantic_lifecycle_identity = semantic_lifecycle_identity_json(builder);
    let lifecycle_coverage_complete =
        builder.semantic_lifecycle_transitions().len() == builder.num_steps() as usize;

    let canonical = format!(
        "v1|program={}|semantic_graph={}|semantic_lifecycle={}|coverage_complete={}|",
        graph_program_identity,
        semantic_graph_identity,
        semantic_lifecycle_identity,
        lifecycle_coverage_complete,
    );
    let context_fingerprint = fnv1a64(canonical.bytes());

    Ok(SemanticExecutionContext {
        program_identity: graph_program_identity,
        semantic_graph_identity,
        semantic_lifecycle_identity,
        lifecycle_coverage_complete,
        context_fingerprint,
    })
}

#[wasm_bindgen(js_name = semanticExecutionContextCapabilities)]
pub fn semantic_execution_context_capabilities() -> String {
    SEMANTIC_EXECUTION_CONTEXT_V1.to_string()
}

#[wasm_bindgen(js_name = semanticExecutionContext)]
pub fn semantic_execution_context(
    builder: &AgentGraphBuilder,
    graph: &CompiledGraph,
    registry: &LayerRegistry,
) -> Result<String, String> {
    semantic_execution_context_for(builder, graph, registry).map(|context| context.json())
}

#[cfg(test)]
mod tests {
    use super::semantic_execution_context_for;
    use crate::agent::{AgentGraphBuilder, AgentLayerSpec};
    use crate::input_port::workspace_bind_input_port_metadata;
    use crate::input_port_consumer::InputPortConsumerSpec;
    use crate::input_port_edge_binding::bind_input_port_consumer_edge;
    use crate::registry::LayerRegistry;
    use crate::semantic_lifecycle::{
        bind_semantic_lifecycle_transition, SemanticTransitionSpec,
    };
    use crate::workspace::AgentWorkspace;
    use crate::workspace_ops::workspace_init_unary;

    #[test]
    fn context_binds_semantics_to_exact_executable_without_changing_program_identity() {
        let mut workspace = AgentWorkspace::new(3).unwrap();
        let mut builder = AgentGraphBuilder::new(3).unwrap();
        let mut registry = LayerRegistry::new();

        workspace_bind_input_port_metadata(
            &mut workspace,
            "observation".into(),
            "feed".into(),
            1,
            "fp".into(),
        )
        .unwrap();

        let layer_id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
        let spec = AgentLayerSpec::relu(layer_id);
        let output = workspace_init_unary(
            &mut workspace,
            &mut builder,
            &mut registry,
            &spec,
            0,
            "relu".into(),
        )
        .unwrap();

        let graph = builder.compile_with_output(&registry, output).unwrap();
        let program_before = graph.program_identity();

        let before = semantic_execution_context_for(&builder, &graph, &registry).unwrap();

        let consumer = InputPortConsumerSpec::new(
            "feature".into(),
            "[\"observation\"]".into(),
            false,
            false,
            0,
        )
        .unwrap();
        bind_input_port_consumer_edge(&workspace, &mut builder, &consumer, 0).unwrap();
        let transition = SemanticTransitionSpec::new(
            "observation-to-feature".into(),
            "[\"observation\"]".into(),
            "feature".into(),
        )
        .unwrap();
        bind_semantic_lifecycle_transition(&workspace, &mut builder, &transition, 0).unwrap();

        let after = semantic_execution_context_for(&builder, &graph, &registry).unwrap();

        assert_eq!(graph.program_identity(), program_before);
        assert_eq!(before.program_identity, after.program_identity);
        assert_ne!(before.context_fingerprint, after.context_fingerprint);
        assert_ne!(before.semantic_graph_identity, after.semantic_graph_identity);
        assert_ne!(
            before.semantic_lifecycle_identity,
            after.semantic_lifecycle_identity
        );
        assert!(after.lifecycle_coverage_complete);
    }

    #[test]
    fn stale_or_foreign_builder_fails_closed() {
        let mut registry = LayerRegistry::new();
        let first = AgentLayerSpec::relu(1);
        let second = AgentLayerSpec::relu(2);
        registry.init_agent_layer(&first).unwrap();
        registry.init_agent_layer(&second).unwrap();

        let mut original = AgentGraphBuilder::new(3).unwrap();
        original.add_unary(&first, 0, 1).unwrap();
        let graph = original.compile_with_output(&registry, 1).unwrap();

        let mut different = AgentGraphBuilder::new(3).unwrap();
        different.add_unary(&first, 0, 1).unwrap();
        different.add_unary(&second, 1, 2).unwrap();

        let err = semantic_execution_context_for(&different, &graph, &registry).unwrap_err();
        assert!(err.contains("executable identity does not match"));
    }
}
