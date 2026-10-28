# Contracts (normative)

What this page is for: telling you which machine-readable contracts
define what the engine can do, and which one to consult for which
question. The contracts themselves live in `engine-audit/docs/`;
you do not read them directly — you query them via `bin/find-op.mjs`
and `bin/surface.mjs`, or scan `contracts-index/*.jsonl`.

## The contracts that matter to an agent

- **Layer catalog** (`agent-layer-catalog.v1.json`, 48 constructors).
  Use when: you need to know whether an operation exists and what
  its arguments are. Query: `find-op <name>`, list: `find-op --list`.
  Do NOT use when: you want runtime behavior or performance — the
  catalog declares constructibility, not cost or speed.
- **Layout contracts** — do NOT open
  `agent-layout-contracts.v1.json` directly (it lives in
  `engine-audit/`). Query it with `node bin/layout.mjs <name>`
  (list: `--list`) or scan `contracts-index/layout-index.jsonl`.
  Use when: a shape or layout error appears, or you need to know
  which layouts a layer accepts. Example: `layerNorm` input/output
  is `feature_last`, not `feature_axis1_singleton`. The catalog
  entry's `layout_contract_key` points at the relevant contract.
- **Input ports** (`agent-input-port.v1.json`).
  Use when: you declare ports in a spec. Rules that matter: at
  least 2 ports, every port consumed, shapes are 4-D, roles and
  layouts come from the canonical vocabularies (see `spec-schema.md`).
- **Fault contract** (`agent-fault-contract.v1.json`).
  Use when: you get a structured error and want to know what its
  code means. For run-phase engine errors, branch on
  `error_envelope.code` and apply `remediation_hint` first; for
  validate-layer failures the code is top-level (`code`/`stage`),
  and tool-level failures are a bare `{ok:false, error}` — see
  `AGENT.md` for all three shapes. Read this contract only if the
  hint is unclear.
- **Wasm surface** (`wasm-surface.v1.json`, 115 functions + 54 classes).
  Use when: you need a wasm-level function, class, or method.
  Query: `surface <name>`, list: `surface --list`.
  Do NOT use when: a spec-level tool already does the job — prefer
  `validate`/`run`/`verify`/`checkpoint` over calling wasm directly.
- **Runtime surface** (`runtime-surface.v1.json`, plus the generated
  `wasm-surface.actual.json` snapshot).
  Use when: you need to prove which build you are running or what
  capability group an operation belongs to. The snapshot is a
  generated artifact (5,766 lines); never read it directly. Its
  sha256 and the wasm sha256 are pinned in `MANIFEST.json`.

## Not normative

Anything in `reference/history.md` is evidence or research, not a
contract. If a history document and a contract above disagree, the
contract wins, and the disagreement is a bug to report.
