# Interactive multi-input WASM ingress

The packaged Node artifact includes `interactive_multi_input_ingress.mjs`. It loads the local WASM through the verified `node.mjs` adapter and keeps one graph session alive while an agent or a human sends JSON Lines requests. Each request produces one JSON response with the same `request_id`.

After downloading and extracting the `burn-wasm-output` artifact, run:

```bash
node interactive_multi_input_ingress.mjs
```

From a repository checkout with `pkg/` built, run `node scripts/interactive_multi_input_ingress.mjs pkg`. Send one command per line. The following example uses the typed `add` layer, two numeric inputs, and an exact source declaration for each logical port:

```jsonl
{"request_id":1,"op":"create","numSlots":3,"layers":[{"constructor":"add","args":[31]}],"steps":[{"kind":"binary","layer":0,"slots":[0,1,2]}],"outputSlot":2,"ports":[{"slot":0,"role":"observation","shape":[1,2,1,1],"layout":"feature_axis1_singleton","requireFingerprint":true,"minimumRevision":2},{"slot":1,"role":"state","shape":[1,2,1,1],"layout":"feature_axis1_singleton","requireFingerprint":true,"minimumRevision":3}],"logicalPorts":[{"id":"observation","slot":0,"source":"sensor-a"},{"id":"memory","slot":1,"source":"memory-b"}]}
{"request_id":2,"op":"bind","slot":0,"values":[1,2],"shape":[1,2,1,1],"role":"observation","layout":"feature_axis1_singleton","source":"sensor-a","revision":2,"fingerprint":"obs"}
{"request_id":3,"op":"bind","slot":1,"values":[3,4],"shape":[1,2,1,1],"role":"state","layout":"feature_axis1_singleton","source":"memory-b","revision":3,"fingerprint":"state"}
{"request_id":4,"op":"inspect"}
{"request_id":5,"op":"consumer","slot":1,"id":"state-reader","acceptedRoles":["state"],"requireFingerprint":true,"minimumRevision":3}
{"request_id":6,"op":"run"}
{"request_id":7,"op":"verify","candidate":[4,6]}
{"request_id":8,"op":"close"}
```

`run` returns `values: [4, 6]`; `verify` returns the ingress status and the numerical verification report. To inspect the contract before binding, send `{"op":"capabilities"}` or `{"op":"port","slot":1}`. `map`, `defer`, and `clear` let the caller revise logical mappings and bindings within the session. The `create` command accepts typed layer constructors listed by `agentCapabilities()`, unary/binary graph steps, and all input ports admitted by `MultiInputGraphPlan.v1`.

The bridge status checks exact graph and bundle plan bytes, source equality, input contracts, deferred required ports, and registry structural binding. `status.ready` describes current coverage; `execution_authorized` stays false. The bridge's `run` method checks status again and delegates execution to `CompiledMultiInputGraph.run`, which repeats graph preflight. `source` and `fingerprint` are caller-declared metadata; this version does not authenticate their origin. Burn remains the final authority for tensor and operator compatibility.

## Signed host provenance

For a host that controls trusted issuer keys, launch the same packaged runner with a second argument:

```bash
node interactive_multi_input_ingress.mjs . /trusted/ingress-policy.json
```

The policy file uses `burn-research.ingress-trust-policy.v1` and pins issuer public keys and allowed subjects. Example structure:

```json
{
  "schema": "burn-research.ingress-trust-policy.v1",
  "issuers": [
    {"source": "sensor-a", "key_id": "sensor-key-1", "subjects": ["run-42"], "public_key_pem": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n"}
  ]
}
```

The producer constructs a claim with `canonicalInputClaim(binding, context, keyId, subject, nonce)` from `ingress_provenance.mjs`. `context` contains `plan_hex`, `manifest_fingerprint`, `manifest_sha256: manifestDigest(manifest)`, and the mapped `logical_port_id` returned by `create`. The issuer signs `Buffer.from(JSON.stringify(claim))` with its Ed25519 private key and sends `proof: {claim, signature: base64}` in the existing `bind` command. See `scripts/audit_signed_ingress_provenance.mjs` for a complete executable producer and host session.

In this mode every `bind` requires a signature from a configured issuer for its source and subject. The signed claim covers the exact plan bytes, full manifest SHA-256, port, metadata, nonce, and SHA-256 of the little-endian f32 values passed to WASM. The runner checks nonces and increasing revisions for its process lifetime, with a 50,000-claim fail-closed cap. `run` and `verify` require signed coverage for every runtime input, and a manifest change makes previous claims stale. `inspect.host_provenance` shows these checks separately from the WASM ingress status.

The trust policy and producer private keys must be controlled outside the JSON Lines caller. This gate applies to this Node runner; a direct caller of the WASM API can still bind caller-declared metadata. A signature proves that the configured issuer made the claim, not that the observed world state is true. Durable replay protection across process restarts requires an external store. The complete machine-readable scope is in `ingress-provenance.v1.json`.

### Host subject and replay across restarts

Provide a private ledger file and a subject fixed by the trusted host as additional startup arguments:

```bash
node init_ingress_replay_ledger.mjs /private/ingress-ledger.json run-42
node interactive_multi_input_ingress.mjs . /trusted/ingress-policy.json /private/ingress-ledger.json run-42
```

Run initialization once under the trusted host deployment before accepting any input. It refuses to overwrite an existing ledger. The ledger parent must exist, be owned by the runner user, and not be writable by other users. Every runner startup requires that existing private file; a missing or deleted ledger fails closed. A later process must present the exact same host subject to use that ledger. Each signed claim must also use that subject and a public key allowed for it by the host trust policy. JSON Lines requests cannot change the host subject or ledger path.

A successful `bind` records its nonce and revision atomically on disk before replying. If the disk commit fails, the runner clears the newly bound tensor. On every `run` and `verify`, it checks that each in-session claim is recorded and remains the latest revision, while holding a filesystem lock. Reusing a nonce or lowering a revision after restarting is rejected. Two runner processes sharing the ledger serialize commits through that lock; a busy or abandoned lock fails closed. After a crash, an operator must confirm no process holds the lock before removing it.

This mode reports `host_provenance.mode = "ed25519_host_durable"` and `replay_scope = "host_file_across_restarts"`. At 50,000 accepted claims or a 32 MiB file, new binds stop instead of evicting evidence. A privileged writer of the ledger can rewrite its history; protecting the host file is part of the trust boundary. See `ingress-replay-ledger.v1.json` and `scripts/audit_durable_signed_ingress.mjs` for the executable proof.
