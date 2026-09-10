use burn_research::agent::AgentLayerSpec;
use burn_research::protocol::{ACT_RELU, LAYER_ACTIVATION, OP_INIT, PacketHeader};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;

#[test]
fn sync_layer_rejects_same_id_type_wrong_variant_without_mutation() {
    let mut registry = LayerRegistry::new();
    let live = AgentLayerSpec::relu(77);
    registry.init_agent_layer(&live).unwrap();

    let mut workspace = AgentWorkspace::new(4).unwrap();
    let before = workspace.snapshot();
    let wrong = AgentLayerSpec::gelu(77);

    let err = workspace
        .sync_layer(&registry, &wrong, "wrong-variant".into())
        .unwrap_err();
    assert!(err.contains("identity mismatch"));
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(workspace.get("_layers".into(), "77".into()), "null");
}

#[test]
fn sync_layer_rejects_same_variant_wrong_payload_without_mutation() {
    let mut registry = LayerRegistry::new();
    let live = AgentLayerSpec::linear(81, 4, 3, true).unwrap();
    registry.init_agent_layer(&live).unwrap();

    let mut workspace = AgentWorkspace::new(4).unwrap();
    let before = workspace.snapshot();
    let wrong = AgentLayerSpec::linear(81, 4, 5, true).unwrap();

    let err = workspace
        .sync_layer(&registry, &wrong, "wrong-payload".into())
        .unwrap_err();
    assert!(err.contains("identity mismatch"));
    assert_eq!(workspace.snapshot(), before);
    assert_eq!(workspace.get("_layers".into(), "81".into()), "null");
}

#[test]
fn raw_protocol_layer_still_reconciles_when_identity_matches() {
    let layer_id = 90u32;
    let payload = layer_id.to_le_bytes();
    let header = PacketHeader {
        opcode: OP_INIT,
        layer_type: LAYER_ACTIVATION,
        variant: ACT_RELU,
        flags: 0,
        payload_len: payload.len() as u32,
    };

    let mut registry = LayerRegistry::new();
    registry.init_layer(&header, &payload).unwrap();

    let mut workspace = AgentWorkspace::new(4).unwrap();
    let matching_spec = AgentLayerSpec::relu(layer_id);
    workspace
        .sync_layer(&registry, &matching_spec, "raw-relu".into())
        .unwrap();

    let row = workspace.get("_layers".into(), layer_id.to_string());
    assert!(row.contains("initialized"));
    assert!(row.contains("raw-relu"));
    assert!(registry.layer_exists(LAYER_ACTIVATION, layer_id));
}
