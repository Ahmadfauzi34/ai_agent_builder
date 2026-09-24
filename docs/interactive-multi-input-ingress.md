# Interactive multi-input WASM ingress

The packaged Node artifact includes `interactive_multi_input_ingress.mjs`. It loads the local WASM through the verified `node.mjs` adapter and keeps one graph session alive while an agent or a human sends JSON Lines requests. Each request produces one JSON response with the same `request_id`.

After downloading and extracting the `burn-wasm-output` artifact, run:

```bash
node interactive_multi_input_ingress.mjs
```

From a repository checkout with `pkg/` built, run `node scripts/interactive_multi_input_ingress.mjs pkg`. Send one command per line. The following example uses the typed `add` layer, two numeric inputs, and an exact source declaration for each logical port:

```jsonl
{"request_id":1,"op":"create","numSlots":3,"layers":[{"constructor":"add","args":[31]}],"steps":[{"kind":"binary","layer":0,"slots":[0,1,2]}],"outputSlot":2,"ports":[{"slot":0,"role":"observation","shape":[1,2,1,1],"layout":"feature_axis1_singleton","requireFingerprint":true,"minimumRevision":2},{"slot":1,"role":"state","shape":[1,2,1,1],"layout":"feature_axis1_singleton","requireFingerprint":true,"minimumRevision":3}],"logicalPorts":[{"id":"observation","slot":0,"source":"sensor-a"},{"id":"memory","slot":1,"source":"memory-b"}]}
{"request_id":2,"op":"explain"}
{"request_id":3,"op":"bind","slot":0,"values":[1,2],"shape":[1,2,1,1],"role":"observation","layout":"feature_axis1_singleton","source":"sensor-a","revision":2,"fingerprint":"obs"}
{"request_id":4,"op":"bind","slot":1,"values":[3,4],"shape":[1,2,1,1],"role":"state","layout":"feature_axis1_singleton","source":"memory-b","revision":3,"fingerprint":"state"}
{"request_id":5,"op":"inspect"}
{"request_id":6,"op":"consumer","slot":1,"id":"state-reader","acceptedRoles":["state"],"requireFingerprint":true,"minimumRevision":3}
{"request_id":7,"op":"trace","startStep":0,"maxSteps":1,"maxTensorBytes":1048576}
{"request_id":8,"op":"run"}
{"request_id":9,"op":"verify","candidate":[4,6]}
{"request_id":10,"op":"close"}
```

`run` returns `values: [4, 6]`; `verify` returns the ingress status and the numerical verification report. `explain` reports the compiled plan's ordered steps and partial static shapes before any input binding; it does not execute Burn or authorize a run. To inspect the contract before binding, send `{"op":"capabilities"}` or `{"op":"port","slot":1}`. `map`, `defer`, and `clear` let the caller revise logical mappings and bindings within the session. The `create` command accepts typed layer constructors listed by `agentCapabilities()`, unary/binary graph steps, and all input ports admitted by `MultiInputGraphPlan.v1`.

`trace` executes the graph once and returns normal output alongside `execution_trace`. It observes selected step indices from `startStep` for up to `maxSteps` (maximum 256), while the whole graph still executes. The report binds the structural `program_identity`, ordered input and step shapes, and SHA-256 of captured f32 little-endian values. The per-tensor limit is `maxTensorBytes` (maximum 16 MiB); all input, step and terminal captures share a 64 MiB total limit. A skipped digest is `null` with a `capture_status`; `trace_complete` describes step coverage and completed execution, not digest coverage. The packaged `multi-input-execution-trace.v1.json` defines the fields and bounds. A failed operator returns `ok:false` with `execution_trace`, fault index and no output or receipt. A preflight or bound failure does not execute the graph and returns no trace. Each `trace` and `run` command is a separate execution, which matters for mutable state.

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

The producer constructs a v1 claim with `canonicalInputClaim(binding, context, keyId, subject, nonce)` from `ingress_provenance.mjs`. `context` contains `plan_hex`, `manifest_fingerprint`, `manifest_sha256: manifestDigest(manifest)`, and the mapped `logical_port_id` returned by `create`. The issuer signs `Buffer.from(JSON.stringify(claim))` with its Ed25519 private key and sends `proof: {claim, signature: base64}` in the existing `bind` command. See `scripts/audit_signed_ingress_provenance.mjs` for a complete executable producer and host session.

In this mode every `bind` requires a signature from a configured issuer for its source and subject. The signed claim covers the exact plan bytes, full manifest SHA-256, port, metadata, nonce, and SHA-256 of the little-endian f32 values passed to WASM. The runner checks nonces and increasing revisions for its process lifetime, with a 50,000-claim fail-closed cap. `run` and `verify` require signed coverage for every runtime input, and a manifest change makes previous claims stale. `inspect.host_provenance` shows these checks separately from the WASM ingress status.

The trust policy and producer private keys must be controlled outside the JSON Lines caller. This gate applies to this Node runner; a direct caller of the WASM API can still bind caller-declared metadata. A signature proves that the configured issuer made the claim, not that the observed world state is true. Durable replay protection across process restarts requires an external store. The v1 machine-readable scope is in `ingress-provenance.v1.json`.

### State-bound claim v2

An issuer opts into v2 with an explicit `claim_schemas` allowlist:

```json
{"source":"sensor-a","key_id":"sensor-key-1","subjects":["run-42"],"claim_schemas":["burn-research.signed-input-claim.v1","burn-research.signed-input-claim.v2"],"public_key_pem":"..."}
```

When any configured issuer allows v2, graph creation, `inspect`, and `restore` include `host_provenance.active_state_checkpoint_bytes_sha256`. It is the exact checkpoint-byte digest for the active graph state; the host does not return raw checkpoint bytes through this field. The producer passes that value in the claim context and constructs a v2 claim with `canonicalStateBoundInputClaim(binding, context, keyId, subject, nonce)`. The signature covers the digest as well as the existing v1 fields. A trust policy without `claim_schemas` remains v1-only.

The host rejects a v2 claim whose digest does not match the active checkpoint at `bind`. It checks the digest again under the durable ledger lock before `run`, `verify`, or `checkpoint`. A restored or otherwise changed state needs freshly signed inputs for its new digest. To make v2 mandatory, start the runner with `--require-state-bound-inputs`; every configured issuer must allow v2 and every runtime input must use a current v2 claim. This option is trusted host configuration, not a JSON Lines command.

Durable receipt v2 records `claim_schema` and each v2 input's `active_state_checkpoint_bytes_sha256`. When every input binds the same checkpoint, the receipt also has `input_state_checkpoint_bytes_sha256`; `state_checkpoint_bytes_sha256` continues to report the post-run state. If execution leaves that checkpoint unchanged, the still-current claim set can be run again: a signed input nonce prevents rebinding, but does not authorize exactly one execution. These hashes bind exact bytes and freshness, not checkpoint origin, semantic equivalence, or action authority. See `ingress-provenance.v2.json`.

### Host subject and replay across restarts

Provide a private ledger file and a subject fixed by the trusted host as additional startup arguments. Optional startup flags may include state-bound input enforcement:

```bash
node init_ingress_replay_ledger.mjs /private/ingress-ledger.json run-42
node interactive_multi_input_ingress.mjs . /trusted/ingress-policy.json /private/ingress-ledger.json run-42 --allow-state-checkpoint-export --allow-checkpoint-restore
```

The checkpoint flags grant independent capabilities: `--allow-state-checkpoint-export` allows `op=checkpoint` to return raw mutable state, while `--allow-checkpoint-restore` permits `op=restore` by a committed receipt ID. `--require-state-bound-inputs` requires v2 for every runtime input and does not grant raw checkpoint export. A host may combine these flags according to its policy. Durable runs retain their checkpoint and bind its byte digest to the receipt regardless of those client-facing flags.

Run initialization once under the trusted host deployment before accepting any input. It refuses to overwrite an existing ledger. The ledger parent must exist, be owned by the runner user, and not be writable by other users. Every runner startup requires that existing private file; a missing or deleted ledger fails closed. A later process must present the exact same host subject to use that ledger. Each signed claim must also use that subject and a public key allowed for it by the host trust policy. JSON Lines requests cannot change the host subject or ledger path.

A successful `bind` records its nonce and revision atomically on disk before replying. If the disk commit fails, the runner clears the newly bound tensor. On every `run` and `verify`, it checks that each in-session claim is recorded and remains the latest revision, while holding a filesystem lock. Reusing a nonce or lowering a revision after restarting is rejected. Two runner processes sharing the ledger serialize commits through that lock; a busy or abandoned lock fails closed. After a crash, an operator must confirm no process holds the lock before removing it.

This mode reports `host_provenance.mode = "ed25519_host_durable"` and `replay_scope = "host_file_across_restarts"`. At 50,000 accepted claims or a 32 MiB file, new binds stop instead of evicting evidence. A privileged writer of the ledger can rewrite its history; protecting the host file is part of the trust boundary. See `ingress-replay-ledger.v1.json` and `scripts/audit_durable_signed_ingress.mjs` for the executable proof.

Durable runs return an execution receipt and `output_f32_le_base64`. To feed that exact output into a later graph tick, bind it to a `role: "state"` port using `values_f32_le_base64`, the receipt's output shape, and `handoff_receipt_id`. The child state claim still needs a valid issuer signature for the child manifest and its configured source. The host records the claim and handoff together; later receipts include the resulting `handoff_id`. The default `handoff_branch_id` is `main`; choose another branch ID to fork an independent lineage. Within each subject/source/logical-port/branch lane, parent receipt sequence must move forward. See `host-state-handoff.v1.json` for the complete boundary and limits.

`{"op":"checkpoint"}` exports the current session's multi-input `ProgramBundle` as `bundle_f32le_base64` and reports `checkpoint_bytes_sha256`. This is a separate internal layer-state snapshot from an external state handoff. Durable run receipts use schema v2 and bind that exact bundle byte digest as `state_checkpoint_bytes_sha256`. Hashes correlate exact serialized bytes only: ProgramBundle state has no signature, the digest is not semantic state identity, and neither grants action authority. Historical schema v1 receipts remain readable.

Each new durable `run` also retains its exact ProgramBundle bytes and manifest in a private host directory beside the ledger (`<ledger>.checkpoints`). The host fsyncs the checkpoint before committing the receipt; a failed receipt commit removes the newly written checkpoint and returns an error. A process crash between those writes can leave an unreferenced checkpoint, which cannot be restored through the host API. Each checkpoint is limited to 8 MiB of bundle bytes; the store is limited to 512 MiB. Existing v2 receipts created before retention remain readable, but cannot be restored without retained bytes.

With `--allow-checkpoint-restore` enabled by the trusted host, `{"op":"restore","receipt_id":"sha256:..."}` loads only a checkpoint associated with a committed receipt for this subject. The host checks the private file, receipt ID, exact byte digest, and manifest digest; WASM imports into a new registry and verifies structural identity. Before replacing the active session, the host commits a restore event anchored to the parent receipt. The response returns `restore_event_id`, manifest, and program identity. Restore never brings back external input tensors, signed proofs, or action authority. Bind all runtime ports again using new signed nonces and increasing revisions before `run`. Missing or tampered checkpoints, unsupported historical receipts, and failed event commits leave the prior session intact. A privileged host file writer remains inside the trust boundary.

The first run after restore records `state_parent_receipt_id` pointing to the restored receipt and `restore_event_id` pointing to the host's restore event. Each subsequent successful run in that session advances its immediate state parent to the previous receipt. A fresh `create` starts a new graph without inferred ancestry. If numerical execution starts but no receipt can be committed, the host discards the session; its possibly changed state cannot produce a misleading descendant receipt. These fields document causal checkpoint selection, including an explicit fork from an older receipt. The existing state handoff rule still orders parent receipts within a lane; it does not claim internal model state is semantically monotonic. Signed input claim v1 authenticates the input and structural manifest but does not bind active mutable checkpoint bytes. The restore flag is a host capability, not a producer signature for each restore. See `host-checkpoint-restore.v1.json`.

### Receipt for a durable multi-input run

A successful durable `trace` also commits a receipt for its own execution and returns `output_f32_le_base64`. When the terminal digest was captured, compare `execution_trace.terminal_output.value_sha256` with the receipt output digest. A failed trace issues no receipt; after numerical execution starts, the durable runner discards that session.

In durable mode, a successful `run` returns `execution_receipt` and `output_f32_le_base64` alongside `values`. The host records the exact graph `programIdentity`, manifest SHA-256, subject, signed claim SHA-256 per input slot, observed output shape and SHA-256 of little-endian f32 values, and a SHA-256 digest of the exact stateful multi-input ProgramBundle bytes captured after the run. It assigns a monotonic receipt sequence and stores the receipt in the same ledger while the input freshness lock is held. A failed disk commit returns an error, even if the numerical run already occurred.

After restarting the runner with the same ledger and subject, send `{"op":"receipt","receipt_id":"sha256:..."}` to retrieve the committed record. Supply `shape` and either `values` or `output_f32_le_base64` with that command to receive `output_matches: true` or `false`. Use the byte encoding returned by `run` for exact comparisons: JSON renders `-0` as `0`, despite their different f32 bytes. This lookup does not execute the graph again; the separate `verify` command still provides Burn numerical comparison against a candidate. Receipt lookup fails if the record is absent or ledger validation fails. Older ledger snapshots without a receipt list retain their signed claim history and accept new receipts.

The receipt is a host observation, not a WASM-signed certificate or permission for an agent action. Its SHA-256 ID detects changed content within the trusted ledger; a privileged host file writer can rewrite both content and ID. `programIdentity` remains structural and omits mutable layer state; `state_checkpoint_bytes_sha256` separately correlates the exact serialized bundle bytes. The bundle bytes have no signature, and the digest does not claim semantic state identity or state authenticity. Receipt storage has a 50,000-record limit and shares the existing 32 MiB ledger cap. The packaged `host-execution-receipt.v2.json` specifies the proof boundary; v1 history remains accepted.
