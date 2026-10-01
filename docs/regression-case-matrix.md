# Regression case matrix: R-01..R-20

Source of the case definitions: "Daftar Keluhan dan Usulan Pengembangan Core WASM",
table "Regression cases wajib menjaga perilaku yang sudah benar". Each case below
names the script and assertion that pins the behavior in CI, so a future change
that breaks the case fails loudly instead of silently.

| ID | Case (expected behavior) | Pinned by |
|----|--------------------------|-----------|
| R-01 | Program equivalence, same operator (`equivalent=true`) | `scripts/audit_wasm_artifact.mjs` — `equal.equivalent===true` on identical-structure programs with a verified receipt digest |
| R-02 | Program equivalence, changed operator (`equivalent=false`) | `scripts/audit_wasm_artifact.mjs` — `unequal.equivalent===false`, `first_failure_case_index===0`, `max_abs_error===8` |
| R-03 | Loose tolerance changes the verdict | `scripts/audit_interactive_multi_input_ingress.mjs` — `vfail` returns `verification_summary.verdict==='fail'`, `reason==='tolerance_exceeded'`, `first_failure===1` |
| R-04 | Shape mismatch is an explicit verdict reason | `scripts/audit_interactive_multi_input_ingress.mjs` — `vshape`/`vempty` return `verdict==='fail'`, `reason==='shape_mismatch'` (a verdict, not an op-level `internal_error`) |
| R-05 | Stale binding never reaches execution | `scripts/audit_signed_ingress_provenance.mjs` — stale manifest signatures and stale revisions are rejected before `run`; `scripts/audit_durable_signed_ingress.mjs` — stale revision rejected after restart |
| R-06 | Weight mutation changes the revision | `scripts/audit_weight_tracking_restore.mjs` — `setWeights` bumps `registry_revision` and per-layer `weight_revision`; `state_digest` (sha256 over live weights) changes while ReLU clamps outputs |
| R-07 | Missing ports name the field | `scripts/audit_interactive_multi_input_ingress.mjs` — one-port `create` fails with `plan_minimum_ports` and `execution_started===false` |
| R-08 | Unused port is a specific error | `scripts/audit_interactive_multi_input_ingress.mjs` — unused-port `create` fails with `plan_unused_ports`, a different code from `plan_minimum_ports`, naming `unused_ports` |
| R-09 | Invalid role names the allowed values | `scripts/audit_interactive_multi_input_ingress.mjs` — bad role/layout in `create`/`bind`/`defer` fails as `invalid_enum` with `actual_value`, `allowed_values`, `capability_ref` |
| R-10 | Valid signed claim → `ready=true` | `scripts/audit_signed_ingress_provenance.mjs` — honest signed bind/run succeeds, `host_provenance.ready===true`, `host_authority_attested===true`, `denial_reason===null` |
| R-11 | Invalid signed claim → `ready=false` + reason | `scripts/audit_signed_ingress_provenance.mjs` — 8 adversarial attacks (nonce replay, stale revision, payload tampering, wrong-key signature, cross-slot claim, unknown key_id, broken encoding, field tampering) all rejected with precise errors; `ready===false` with a non-empty `denial_reason` |
| R-12 | Empty fingerprint contract is consistent | `scripts/audit_interactive_multi_input_ingress.mjs` — caller-declared `bind` accepts an empty fingerprint; `capabilities` discloses `fingerprint_required: false` for `caller_declared` and `non_empty` for both signed modes |
| R-13 | Bundle corruption detected at every byte | `scripts/audit_interactive_multi_input_ingress.mjs` — single-bit flip at every byte position (including the last, which the structural parser ignores) changes the digest; `verifyCheckpointIntegrity` is session-free and strict: non-canonical base64 is rejected with `invalid_argument` / `canonical_base64` |
| R-14 | Durable restore requires the receipt digest | `scripts/audit_durable_signed_ingress.mjs` — `receipt_bound_restore`: restore re-hashes retained bytes and refuses when they differ from `state_checkpoint_bytes_sha256`; tampered/missing checkpoints rejected |
| R-15 | README local links all resolve | `scripts/check_distribution_links.mjs` (CI step after packaging) — every local markdown link in the packaged README resolves inside the package; negative cases (deleted target, escaping link) pinned in `scripts/audit_node_host.mjs` |
| R-16 | Ledger permissions: 0700/0755 accepted | `scripts/audit_durable_signed_ingress.mjs` — `ledger_parent_permission_matrix_enforced`: 0700/0755 accepted, 0775/0707/0777 fail closed; copy/restore + `stat` round-trip: the host enforces exactly the stat-reported mode; ledger file owner-only (0600) accepted, any group/other bit rejected |
| R-17 | Distribution smoke: quick starts succeed | `scripts/smoke_distribution.mjs` (CI step after packaging) — usage-guide quick starts 1–3 against the packaged runner: `caller_declared`, `ed25519_host_enforced`, `ed25519_host_durable`, each ending in `run → [4, 6]` |
| R-18 | Corrupt bundle bytes never trap; instance recovers | `scripts/audit_wasm_artifact.mjs` — `R-18 corrupt bundle state bytes never trap; instance recovers`: single-byte-flip sweep over a multi-input bundle with state; every corrupt import either throws a per-call error (never a wasm `unreachable` trap) or is inert; an honest import plus run in the same instance still succeeds afterwards. Rust unit test `program_bundle::tests::r18_corrupt_bundle_import_never_panics` pins the same at the native level (no offset may panic; byte 973 → `0xFF` is a per-call `Err`) |
| R-19 | Fresh layers start from deterministic zero weights | Rust unit test `program_bundle::tests::r19_fresh_layers_have_deterministic_zero_weights` — two fresh registries return identical weights (linear/conv/embedding all-zero, norm gamma=1/beta=0) and identical run outputs for layers without a weight accessor; documented in `docs/agent-facade.md` ("Initial weight contract") |
| R-20 | Session-free capabilities expose the canonical execution gate | `scripts/audit_interactive_multi_input_ingress.mjs` — session-free `capabilities` `host_provenance.execution_gate` equals the session surface gate (`field==='ready'`, same `meaning`/`supersedes`/`enforcement_effect`) |

## Notes

- CI invokes the packager, then `check_distribution_links.mjs`, then
  `smoke_distribution.mjs`, then the `audit_*.mjs` scripts in order; the matrix
  above follows that pipeline.
- `verifyCheckpointIntegrity` requires canonical base64 (standard alphabet,
  length multiple of 4, correct padding, no whitespace). `op=checkpoint` always
  emits canonical base64, so strictness never rejects host-produced bundles.
- New regression cases must land in a script CI already runs
  (`.github/workflows/main.yml`); the GitHub App used for pushes cannot modify
  workflow files.
