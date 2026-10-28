# History and evidence (NOT normative)

What this page is for: stopping you from mistaking old research for
current rules. Everything in this category is evidence about how the
engine got here, not a definition of what it does now.

## How to tell history from contract

- Contracts live in `reference/contracts.md` and are machine-readable
  JSON under `engine-audit/docs/*.v1.json`. They are versioned, and
  the engine is checked against them (see `MANIFEST.json`).
- History lives in `engine-audit/docs/` as prose research documents
  (performance slices, rollout studies, decomposition research) and
  as older contract versions kept for provenance (for example,
  execution receipt v1 alongside v2).
- If a history document and a current contract disagree, the contract
  wins. Report the disagreement; do not "resolve" it by following the
  older document.

## When history IS useful

- Debugging a surprising behavior: an old research note may explain
  why a design choice was made (for example, why a dependency is
  restricted to non-wasm targets, or why a surface snapshot exists).
- Understanding the cost of a change: performance and scaling studies
  tell you what was measured, on what, and what the limits were —
  useful context before you claim a speedup.
- Auditing provenance: older receipts and ledgers are still valid
  historical data, even though new work should use the current
  versions.

## When history is a trap

- Copying an example from a research document into a spec without
  checking it against the current catalog (`find-op`) and current
  surface (`surface`). Research examples predate contract changes.
- Treating a measured number from an old study as a current
  guarantee (speed, size, error). Re-measure with `run`/`verify`
  on the pinned build if the number matters to your decision.
- Reading history before reading `AGENT.md`. History is background;
  `AGENT.md` + `INDEX.md` + the query tools are the working surface.
