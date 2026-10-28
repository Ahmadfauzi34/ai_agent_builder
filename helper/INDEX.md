# Index: question -> where to go

Start with `AGENT.md`. It is the only file you must read in full.
(Repo note: this index lives at `helper/INDEX.md`; the built pack carries the same map.)
This page is a map, not a manual: find your question, follow the link.

## Do something

- Run a computation -> `node bin/run.mjs <spec.json>`
  (spec format: `spec-schema.md`; first try `examples/linear-relu.json`).
- Check a plan before running it -> `node bin/validate.mjs <spec.json>`.
- Compare two graphs -> `node bin/verify.mjs <baseline.json> <candidate.json>`.
- Save / check / restore a program bundle ->
  `node bin/checkpoint.mjs export|verify|restore ...` (see `AGENT.md`).
- See what a layer constructor takes ->
  `node bin/find-op.mjs <name>` (list all: `--list`).
- See what a wasm function, class, or method is ->
  `node bin/surface.mjs <name>` (list all: `--list`).
- Scan all constructors at a glance ->
  `contracts-index/catalog-index.jsonl` (48 lines, one per constructor).
- Scan the whole wasm surface at a glance ->
  `contracts-index/surface-index.jsonl` (169 lines, one per entry).
- Re-run the release regression gate ->
  `node bin/regression.mjs` (cases: `reference/regression.md`).
- See what layout a constructor requires ->
  `node bin/layout.mjs <name>` (list all: `--list`), or scan
  `contracts-index/layout-index.jsonl` (48 lines, one per constructor).

## Understand something

- What a contract means, and which contracts are normative ->
  `reference/contracts.md`.
- Which hosts are supported (Node / Python / Rust) and what each
  host's adapter and limits are -> `reference/hosts.md`.
- Provenance, replay ledgers, execution receipts, checkpoint branches
  (when each matters, when to ignore it) -> `reference/provenance.md`.
- Old research and evidence documents (historical, NOT normative) ->
  `reference/history.md`.
- Which engine build this is, its sha256, and its audit result ->
  `MANIFEST.json`.

## Do NOT go here

- `engine-audit/` — the verbatim CI build. It exists so the pack can
  be audited (sha256 in `MANIFEST.json`) and so the query tools have
  something true to read. It is not a reading surface. If you catch
  yourself opening a file there, stop and use `find-op` or `surface`
  instead. Exception (see `AGENT.md`): a contract file under
  `engine-audit/docs/` that a `reference/` page names explicitly.
- `lib/` — implementation of the tools (only the modules the tools
  actually import: runner, spec, wasm_direct, shapecore, shapes).
  You run the tools; you do not need to read their code to use
  the pack.
- `builder/` — the script that regenerates `contracts-index/` and
  `MANIFEST.json` from `engine-audit/`. Relevant only if you rebuild
  the pack for a new engine build.
