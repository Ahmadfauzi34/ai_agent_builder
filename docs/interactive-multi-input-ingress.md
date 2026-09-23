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
